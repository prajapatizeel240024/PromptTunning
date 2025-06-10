# ----------------- compat shim for transformers 4.52 -----------------
from transformers import modeling_utils
if not getattr(modeling_utils, "ALL_PARALLEL_STYLES", None):
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

# ----------------------------- imports -------------------------------
import json, random, re, torch, numpy as np
from pathlib import Path
from typing import TypedDict, List, Dict, Any
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from langchain_community.llms import HuggingFacePipeline
from langchain_core.messages import AIMessage
from langchain_core.documents import Document
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    pipeline,
)
from langchain_huggingface import HuggingFacePipeline

# Direct imports to avoid LangChain community metaclass conflicts
from sentence_transformers import SentenceTransformer
import faiss

# ------------------------- Optimized RAG Implementation ---------------
class OptimizedEmbeddings:
    """Optimized embedding implementation with caching"""
    def __init__(self, model_name: str):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.cache = {}  # Cache embeddings to avoid recomputation
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents with caching"""
        embeddings = []
        new_texts = []
        new_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            if text in self.cache:
                embeddings.append(self.cache[text])
            else:
                new_texts.append(text)
                new_indices.append(i)
                embeddings.append(None)  # Placeholder
        
        # Compute new embeddings in batch
        if new_texts:
            print(f"Computing embeddings for {len(new_texts)} new documents")
            new_embeddings = self.model.encode(new_texts).tolist()
            
            # Cache and fill placeholders
            for idx, text, embedding in zip(new_indices, new_texts, new_embeddings):
                self.cache[text] = embedding
                embeddings[idx] = embedding
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query with caching"""
        if text in self.cache:
            return self.cache[text]
        
        embedding = self.model.encode([text])[0].tolist()
        self.cache[text] = embedding
        return embedding

class OptimizedFAISS:
    """Optimized FAISS implementation with better indexing"""
    def __init__(self, embeddings, documents):
        self.embeddings = embeddings
        self.documents = documents
        
        # Only embed unique document contents
        unique_contents = list(set(doc.page_content for doc in documents))
        print(f"Creating embeddings for {len(unique_contents)} unique documents (original: {len(documents)})")
        
        self.doc_embeddings = embeddings.embed_documents(unique_contents)
        
        # Create mapping from content to embedding
        self.content_to_embedding = dict(zip(unique_contents, self.doc_embeddings))
        
        # Create FAISS index
        dim = len(self.doc_embeddings[0])
        print(f"Creating FAISS index with dimension {dim}")
        
        # Use more efficient index for small datasets
        if len(unique_contents) < 1000:
            self.index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
        else:
            self.index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, min(100, len(unique_contents)//10))
        
        # Move to GPU if available
        if faiss.get_num_gpus() > 0:
            print("Using GPU for FAISS")
            gpu_res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(gpu_res, 0, self.index)
        
        # Normalize embeddings for cosine similarity
        embeddings_array = np.array(self.doc_embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        
        self.index.add(embeddings_array)
        print(f"Added {len(unique_contents)} unique documents to FAISS index")
        
        # Create reverse mapping for retrieval
        self.embedding_to_docs = {}
        for i, content in enumerate(unique_contents):
            self.embedding_to_docs[i] = [doc for doc in documents if doc.page_content == content]
    
    @classmethod
    def from_documents(cls, documents, embeddings):
        return cls(embeddings, documents)
    
    def as_retriever(self, search_type="similarity", search_kwargs=None):
        if search_kwargs is None:
            search_kwargs = {"k": 3}
        
        def retrieve(query: str) -> List[Document]:
            query_embedding = np.array([self.embeddings.embed_query(query)], dtype=np.float32)
            faiss.normalize_L2(query_embedding)  # Normalize for cosine similarity
            
            k = min(search_kwargs.get("k", 3), len(self.embedding_to_docs))
            
            # Search in FAISS
            similarities, indices = self.index.search(query_embedding, k)
            
            # Return documents with scores
            results = []
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx in self.embedding_to_docs:
                    # Get the first document (they're all the same content)
                    doc = self.embedding_to_docs[idx][0]
                    
                    # Create a copy with similarity score
                    result_doc = Document(
                        page_content=doc.page_content,
                        metadata={
                            **doc.metadata,
                            'similarity_score': float(similarity),
                            'query': query
                        }
                    )
                    results.append(result_doc)
            
            return results
        
        return retrieve

# ------------------------- model loading -----------------------------
BASE   = "google/gemma-2-27b-it"
REWARD = "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load base model WITHOUT quantization
base_tok = AutoTokenizer.from_pretrained(BASE)
if base_tok.pad_token is None:
    base_tok.pad_token = base_tok.eos_token

print("Loading base model without quantization...")
base_raw = AutoModelForCausalLM.from_pretrained(
    BASE, 
    device_map="auto",
    torch_dtype=torch.bfloat16,
    max_memory={0: "40GB", "cpu": "50GB"},
)
hf_pipe = pipeline(
    "text-generation",
    model=base_raw,
    tokenizer=base_tok,
    device_map="auto",
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    repetition_penalty=1.1,
    pad_token_id=base_tok.eos_token_id,
    return_full_text=False,
)
base_llm = HuggingFacePipeline(pipeline=hf_pipe)

# Load reward model WITHOUT quantization
print("Loading reward model without quantization...")
rew_tok = AutoTokenizer.from_pretrained(REWARD)
rew_model = AutoModelForSequenceClassification.from_pretrained(
    REWARD, 
    device_map="auto",
    torch_dtype=torch.bfloat16,
    max_memory={0: "40GB", "cpu": "50GB"},
)

# Load embedding model for RAG using optimized implementation
embeddings = OptimizedEmbeddings(EMBEDDING_MODEL)

def reward(text: str) -> float:
    try:
        if not text or len(text.strip()) == 0:
            return -1.0
        
        text = text[:2048]
        ids = rew_tok(text, return_tensors="pt", truncation=True,
                      max_length=1024, padding=True)
        
        ids = {k: v.to(rew_model.device) for k, v in ids.items()}
        
        with torch.no_grad():
            outputs = rew_model(**ids)
            logits = outputs.logits
            
            if logits.dim() > 1:
                logits = logits.squeeze()
            
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                return -1.0
            
            if logits.dim() == 0:
                score = logits.item()
            else:
                score = logits[0].item() if logits.numel() > 1 else logits.item()
            
            if torch.isnan(torch.tensor(score)) or torch.isinf(torch.tensor(score)):
                return -1.0
                
            return score
            
    except Exception as e:
        print(f"Error in reward function: {e}")
        return -1.0

# --------------------------- Optimized RAG Components ----------------
class OptimizedRAGProcessor:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.vector_store = None
        self.retriever = None
        self.setup_rag()
    
    def load_and_process_data(self) -> List[Document]:
        """Load question.json and create documents for RAG - NO TEXT SPLITTING for math problems"""
        documents = []
        
        if not self.data_path.exists():
            print(f"Creating sample {self.data_path} file...")
            sample_problems = [
                {
                    "problem": "What is 15 + 27?",
                    "solution": "To add 15 + 27, we align the numbers and add: 15 + 27 = 42",
                    "category": "arithmetic",
                    "difficulty": "easy"
                },
                {
                    "problem": "If a rectangle has length 8 and width 5, what is its area?",
                    "solution": "Area = length × width = 8 × 5 = 40 square units",
                    "category": "geometry",
                    "difficulty": "easy"
                },
                {
                    "problem": "Solve for x: 2x + 3 = 11",
                    "solution": "2x + 3 = 11, subtract 3: 2x = 8, divide by 2: x = 4",
                    "category": "algebra",
                    "difficulty": "medium"
                },
                {
                    "problem": "What is the square root of 144?",
                    "solution": "√144 = 12, because 12 × 12 = 144",
                    "category": "arithmetic",
                    "difficulty": "easy"
                },
                {
                    "problem": "If there are 24 students and 3 students per group, how many groups are there?",
                    "solution": "Number of groups = Total students ÷ Students per group = 24 ÷ 3 = 8 groups",
                    "category": "word_problems",
                    "difficulty": "easy"
                },
                {
                    "problem": "What is 7 × 9?",
                    "solution": "7 × 9 = 63",
                    "category": "arithmetic",
                    "difficulty": "easy"
                },
                {
                    "problem": "Convert 0.75 to a fraction.",
                    "solution": "0.75 = 75/100 = 3/4 (simplified)",
                    "category": "fractions",
                    "difficulty": "medium"
                },
                {
                    "problem": "What is the perimeter of a square with side length 6?",
                    "solution": "Perimeter = 4 × side length = 4 × 6 = 24 units",
                    "category": "geometry",
                    "difficulty": "easy"
                },
                {
                    "problem": "If a car travels 60 miles in 2 hours, what is its speed?",
                    "solution": "Speed = Distance ÷ Time = 60 miles ÷ 2 hours = 30 mph",
                    "category": "word_problems",
                    "difficulty": "easy"
                },
                {
                    "problem": "What is 100 - 37?",
                    "solution": "100 - 37 = 63",
                    "category": "arithmetic",
                    "difficulty": "easy"
                }
            ]
            with self.data_path.open('w') as f:
                json.dump(sample_problems, f, indent=2)
        
        try:
            with self.data_path.open() as f:
                data = json.load(f)
                
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict) and "problem" in item:
                        content = f"Problem: {item['problem']}"
                        if 'solution' in item:
                            content += f"\nSolution: {item['solution']}"
                        
                        metadata = {
                            "problem_id": i,
                            "category": item.get("category", "general"),
                            "difficulty": item.get("difficulty", "unknown"),
                            "original_problem": item["problem"]
                        }
                        
                        documents.append(Document(page_content=content, metadata=metadata))
                        
            elif isinstance(data, dict) and "problem" in data:
                content = f"Problem: {data['problem']}"
                if 'solution' in data:
                    content += f"\nSolution: {data['solution']}"
                
                metadata = {
                    "problem_id": 0,
                    "category": data.get("category", "general"),
                    "difficulty": data.get("difficulty", "unknown"),
                    "original_problem": data["problem"]
                }
                documents.append(Document(page_content=content, metadata=metadata))
                
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading data: {e}")
            
        return documents
    
    def setup_rag(self):
        """Set up the optimized RAG pipeline"""
        print("Setting up optimized RAG pipeline...")
        
        documents = self.load_and_process_data()
        
        if not documents:
            print("No documents loaded for RAG")
            return
            
        print(f"Loaded {len(documents)} documents for RAG")
        
        # NO TEXT SPLITTING - math problems are already small and complete
        print("Skipping text splitting for math problems...")
        
        # Create optimized vector store
        print("Creating optimized vector store...")
        self.vector_store = OptimizedFAISS.from_documents(documents, embeddings)
        
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}  # Reduce to 2 most similar problems
        )
        
        print("Optimized RAG pipeline setup complete!")
    
    def retrieve_similar_problems(self, problem: str) -> List[Document]:
        """Retrieve similar problems for a given problem"""
        if self.retriever is None:
            return []
        return self.retriever(problem)
    
    def enhance_prompt_with_rag(self, problem: str, base_prompt: str) -> str:
        """Enhance the base prompt with RAG context"""
        try:
            similar_docs = self.retrieve_similar_problems(problem)
            
            if not similar_docs:
                return base_prompt
            
            # Show similarity scores
            print(f"Retrieved {len(similar_docs)} similar problems:")
            for i, doc in enumerate(similar_docs):
                score = doc.metadata.get('similarity_score', 0)
                print(f"  {i+1}. Similarity: {score:.3f} - {doc.metadata.get('original_problem', 'N/A')[:50]}...")
            
            context = "\n\n".join([
                f"Similar Problem {i+1} (similarity: {doc.metadata.get('similarity_score', 0):.3f}):\n{doc.page_content}"
                for i, doc in enumerate(similar_docs)
            ])
            
            enhanced_prompt = f"""You are an expert tutor. Here are some similar problems and their solutions for reference:

{context}

Now solve this problem using similar reasoning patterns:

Problem: {{problem}}

Solve step-by-step, ending with 'Answer:' and the result.

Thought process:"""
            
            return enhanced_prompt
            
        except Exception as e:
            print(f"Error in RAG enhancement: {e}")
            return base_prompt

# ------------------------- state schema (updated) --------------------
class PState(TypedDict):
    problem: str
    cand_prompt: str
    cand_reasoning: str
    cand_answer: str
    cand_reward: float
    best_prompt: str
    best_reward: float
    iter: int
    patience_left: int
    rag_context: str

# Initialize optimized RAG processor
DATA = Path("question.json")
rag_processor = OptimizedRAGProcessor(DATA)

# ------------------------------ nodes (updated) ---------------------
def generate(state: PState) -> PState:
    try:
        prompt = state["cand_prompt"].format(problem=state["problem"])
        print(f"Generating for prompt length: {len(prompt)}")
        
        llm_out = base_llm.invoke(prompt)
        full_response = llm_out.content if isinstance(llm_out, AIMessage) else llm_out
        
        if prompt in full_response:
            actual_response = full_response.replace(prompt, "").strip()
        else:
            actual_response = full_response
        
        if "Answer:" in actual_response:
            cot, ans = re.split(r"\bAnswer:\s*", actual_response, maxsplit=1)
        else:
            cot, ans = actual_response, "N/A"
            
        reasoning = cot.strip()
        answer = ans.strip() or "N/A"
        
        print(f"Generated answer: {answer}")
        
        state.update(cand_reasoning=reasoning, cand_answer=answer)
        return state
        
    except Exception as e:
        print(f"Error in generate: {e}")
        state.update(cand_reasoning="Error in generation", cand_answer="N/A")
        return state

def score(state: PState) -> PState:
    text = state["cand_reasoning"] + "\nAnswer: " + state["cand_answer"]
    reward_score = reward(text)
    print(f"Reward score: {reward_score:.3f}")
    
    state["cand_reward"] = reward_score
    return state

def mutate_prompt(state: PState, stagnation_patience: int = 2) -> PState:
    if state["cand_reward"] > state["best_reward"]:
        state["best_reward"]   = state["cand_reward"]
        state["best_prompt"]   = state["cand_prompt"]
        state["patience_left"] = stagnation_patience
        print(f"New best score: {state['best_reward']:.3f}")
    else:
        state["patience_left"] -= 1
        
        base_prompt = (
            "You are an expert tutor.\n{filler}\n"
            "Problem: {problem}\nThought process:"
        )
        
        filler = random.choice(
            ["Carefully think step-by-step.", "Show your working.",
             "Explain each step.", "Use similar problem-solving patterns."]
        )
        
        enhanced_prompt = rag_processor.enhance_prompt_with_rag(
            state["problem"], 
            base_prompt.format(filler=filler, problem="{problem}")
        )
        
        state["cand_prompt"] = enhanced_prompt
        
    state["iter"] += 1
    return state

# --------------------------- controllers -----------------------------
def should_continue(state: PState) -> str:
    if (state["iter"] < 5) and (state["patience_left"] > 0):
        return "generate"
    else:
        return END

# --------------------------- main graph ------------------------------
graph = StateGraph(PState)
graph.add_node("generate", generate)
graph.add_node("score", score)
graph.add_node("mutate", mutate_prompt)

graph.set_entry_point("generate")
graph.add_edge("generate", "score")
graph.add_edge("score", "mutate")

graph.add_conditional_edges("mutate", should_continue)
graph = graph.compile()

# --------------------------- run loop -------------------------------
problems = []
try:
    with DATA.open() as f:
        data = json.load(f)
        
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict) and "problem" in item:
                    problems.append(item["problem"])
                if len(problems) >= 3:
                    break
        elif isinstance(data, dict) and "problem" in data:
            problems.append(data["problem"])
            
except (json.JSONDecodeError, FileNotFoundError) as e:
    print(f"Error loading problems: {e}")
    exit(1)

if not problems:
    print("No valid problems found")
    exit(1)

# Test the reward model
print("Testing reward model...")
test_score = reward("This is a test. The answer is 42.")
print(f"Test reward score: {test_score}")

if test_score == -1.0:
    print("ERROR: Reward model is not working properly!")
    exit(1)
else:
    print("Reward model test passed!")

print(f"Loaded {len(problems)} problems from {DATA}")

results = []
for prob in problems:
    print(f"\n{'='*60}")
    print(f"Processing problem: {prob}")
    print(f"{'='*60}")
    
    init_prompt = rag_processor.enhance_prompt_with_rag(
        prob,
        "You are an expert tutor.\nSolve the problem step-by-step, ending with 'Answer:' and the result.\nProblem: {problem}\nThought process:"
    )
    
    similar_problems = rag_processor.retrieve_similar_problems(prob)
    rag_context = "\n".join([doc.page_content for doc in similar_problems]) if similar_problems else "No similar problems found"
    
    state0: PState = dict(
        problem=prob,
        cand_prompt=init_prompt,
        cand_reasoning="",
        cand_answer="",
        cand_reward=float("-inf"),
        best_prompt=init_prompt,
        best_reward=float("-inf"),
        iter=0,
        patience_left=2,
        rag_context=rag_context
    )
    
    final = graph.invoke(state0)
    results.append(
        dict(
            problem=prob,
            best_prompt=final["best_prompt"],
            reward=final["best_reward"],
            answer=final["cand_answer"],
            reasoning=final["cand_reasoning"],
            rag_context=final["rag_context"]
        )
    )
    print(f"Final Score: {final['cand_reward']:.3f} → Answer: {final['cand_answer']}")

print(f"\n{'='*50}")
print("FINAL RESULTS")
print(f"{'='*50}")
for i, result in enumerate(results, 1):
    print(f"\nProblem {i}: {result['problem']}")
    print(f"Answer: {result['answer']}")
    print(f"Reward Score: {result['reward']:.3f}")
    print(f"RAG Context Used: {len(result['rag_context'])} characters")

print(f"\nTotal embeddings in cache: {len(embeddings.cache)}")
print("Done – Optimized RAG-enhanced prompts and scores stored in `results`.")
