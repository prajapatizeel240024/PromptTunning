# ----------------- compat shim for transformers 4.52 -----------------
from transformers import modeling_utils
if not getattr(modeling_utils, "ALL_PARALLEL_STYLES", None):
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

# ----------------------------- imports -------------------------------
import json, random, re, torch
from pathlib import Path
from typing import TypedDict
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.llms import HuggingFacePipeline
from langchain_core.messages import AIMessage
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    pipeline,
)
from langchain_huggingface import HuggingFacePipeline

# ------------------------- model loading -----------------------------
# Remove quantization entirely for stability
BASE   = "google/gemma-2-27b-it"
REWARD = "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"

# Load base model WITHOUT quantization
base_tok = AutoTokenizer.from_pretrained(BASE)
if base_tok.pad_token is None:
    base_tok.pad_token = base_tok.eos_token

print("Loading base model without quantization...")
base_raw = AutoModelForCausalLM.from_pretrained(
    BASE, 
    device_map="auto",
    torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
    max_memory={0: "40GB", "cpu": "50GB"},  # Adjust memory limits
)
hf_pipe = pipeline(
    "text-generation",
    model=base_raw,
    tokenizer=base_tok,
    device_map="auto",
    max_new_tokens=512,            # Increase token limit
    temperature=0.7,               # Add some randomness
    do_sample=True,                # Enable sampling
    top_p=0.9,                     # Nucleus sampling
    repetition_penalty=1.1,        # Prevent repetition
    pad_token_id=base_tok.eos_token_id,
    return_full_text=False,        # Only return generated text, not input
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

def reward(text: str) -> float:
    try:
        # Ensure text is not empty and has reasonable length
        if not text or len(text.strip()) == 0:
            print("Warning: Empty text passed to reward function")
            return -1.0
        
        print(f"Reward function input text: '{text[:200]}...'")
        print(f"Text length: {len(text)}")
        
        # Truncate very long texts
        text = text[:2048]
        
        # Tokenize with proper error handling
        ids = rew_tok(text, return_tensors="pt", truncation=True,
                      max_length=1024, padding=True)
        
        print(f"Tokenized input shape: {ids['input_ids'].shape}")
        print(f"Moving to device: {rew_model.device}")
        
        # Move to device
        ids = {k: v.to(rew_model.device) for k, v in ids.items()}
        
        # Get model output
        with torch.no_grad():
            outputs = rew_model(**ids)
            logits = outputs.logits
            
            print(f"Raw logits shape: {logits.shape}")
            print(f"Raw logits: {logits}")
            
            # Handle different output shapes
            if logits.dim() > 1:
                logits = logits.squeeze()
            
            # Check for nan or inf values
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"Warning: Model returned nan/inf logits: {logits}")
                return -1.0
            
            # Convert to scalar if needed
            if logits.dim() == 0:
                score = logits.item()
            else:
                # If multiple outputs, take the first or mean
                score = logits[0].item() if logits.numel() > 1 else logits.item()
            
            print(f"Final score before checks: {score}")
            
            # Additional check for the final score
            if torch.isnan(torch.tensor(score)) or torch.isinf(torch.tensor(score)):
                print(f"Warning: Final score is nan/inf: {score}")
                return -1.0
                
            print(f"Returning score: {score}")
            return score
            
    except Exception as e:
        print(f"Error in reward function: {e}")
        print(f"Text length: {len(text) if text else 0}")
        import traceback
        traceback.print_exc()
        return -1.0

# --------------------------- state schema ----------------------------
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

# ------------------------------ nodes --------------------------------
def generate(state: PState) -> PState:
    try:
        prompt = state["cand_prompt"].format(problem=state["problem"])
        print(f"Generating for prompt length: {len(prompt)}")
        print(f"Input prompt: {prompt[:200]}...")
        
        # Generate response
        llm_out = base_llm.invoke(prompt)
        full_response = llm_out.content if isinstance(llm_out, AIMessage) else llm_out
        
        print(f"Raw LLM response: {full_response[:300]}...")
        
        # Remove the original prompt from the response if it's included
        if prompt in full_response:
            actual_response = full_response.replace(prompt, "").strip()
            print(f"Cleaned response: {actual_response[:200]}...")
        else:
            actual_response = full_response
        
        print(f"Generated text length: {len(actual_response) if actual_response else 0}")
        
        if "Answer:" in actual_response:
            cot, ans = re.split(r"\bAnswer:\s*", actual_response, maxsplit=1)
        else:
            cot, ans = actual_response, "N/A"
            
        reasoning = cot.strip()
        answer = ans.strip() or "N/A"
        
        print(f"Parsed reasoning: {reasoning[:100]}...")
        print(f"Parsed answer: {answer}")
        
        state.update(cand_reasoning=reasoning, cand_answer=answer)
        return state
        
    except Exception as e:
        print(f"Error in generate: {e}")
        state.update(cand_reasoning="Error in generation", cand_answer="N/A")
        return state

def score(state: PState) -> PState:
    text = state["cand_reasoning"] + "\nAnswer: " + state["cand_answer"]
    print(f"Scoring text: {text[:200]}...")
    
    reward_score = reward(text)
    print(f"Reward score: {reward_score}")
    
    state["cand_reward"] = reward_score
    return state

def mutate_prompt(state: PState, stagnation_patience: int = 2) -> PState:
    if state["cand_reward"] > state["best_reward"]:
        state["best_reward"]   = state["cand_reward"]
        state["best_prompt"]   = state["cand_prompt"]
        state["patience_left"] = stagnation_patience
    else:
        state["patience_left"] -= 1
        filler = random.choice(
            ["Carefully think step-by-step.", "Show your working.",
             "Explain each step."]
        )
        state["cand_prompt"] = (
            "You are an expert tutor.\n{filler}\n"
            "Problem: {problem}\nThought process:"
        ).format(filler=filler, problem="{problem}")
    state["iter"] += 1
    return state

# --------------------------- controllers -----------------------------
def should_continue(state: PState) -> str:
    """Return the next node name based on state"""
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

# Fixed conditional edges - use a single function that returns the next node
graph.add_conditional_edges(
    "mutate",
    should_continue,  # This function returns either "generate" or END
)

# Compile without checkpointer for simplicity
graph = graph.compile()

# --------------------------- run loop -------------------------------
DATA = Path("question.json")            # 10 JSON-lines with {"problem": "..."}

# Check if the data file exists, if not create sample data
if not DATA.exists():
    print(f"Creating sample {DATA} file...")
    sample_problems = [
        {"problem": "What is 15 + 27?"},
        {"problem": "If a rectangle has length 8 and width 5, what is its area?"},
        {"problem": "Solve for x: 2x + 3 = 11"},
        {"problem": "What is the square root of 144?"},
        {"problem": "If there are 24 students and 3 students per group, how many groups are there?"},
        {"problem": "What is 7 × 9?"},
        {"problem": "Convert 0.75 to a fraction."},
        {"problem": "What is the perimeter of a square with side length 6?"},
        {"problem": "If a car travels 60 miles in 2 hours, what is its speed?"},
        {"problem": "What is 100 - 37?"}
    ]
    with DATA.open('w') as f:
        for problem in sample_problems:
            f.write(json.dumps(problem) + '\n')

# Read problems with error handling
problems = []
try:
    with DATA.open() as f:
        data = json.load(f)  # Load the entire JSON array
        
        # Handle both array format and single object format
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict) and "problem" in item:
                    problems.append(item["problem"])
                else:
                    print(f"Warning: Item {i+1} missing 'problem' key or not a dict")
                if len(problems) >= 3:  # Limit to 3 problems
                    break
        elif isinstance(data, dict) and "problem" in data:
            problems.append(data["problem"])
        else:
            print("Error: JSON format not recognized")
            exit(1)
            
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON format: {e}")
    exit(1)
except FileNotFoundError:
    print(f"Error: {DATA} file not found")
    exit(1)

if not problems:
    print("No valid problems found in the data file")
    exit(1)

# Test the reward model before starting
print("Testing reward model...")
test_text = "This is a test. The answer is 42."
test_score = reward(test_text)
print(f"Test reward score: {test_score}")

if test_score == -1.0:
    print("ERROR: Reward model is not working properly!")
    exit(1)
else:
    print("Reward model test passed!")

print(f"Loaded {len(problems)} problems from {DATA}")

results = []
for prob in problems:
    init_prompt = (
        "You are an expert tutor.\n"
        "Solve the problem step-by-step, ending with 'Answer:' and the result.\n"
        "Problem: {problem}\nThought process:"
    )
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
    )
    final = graph.invoke(state0)
    results.append(
        dict(
            problem=prob,
            best_prompt=final["best_prompt"],
            reward=final["best_reward"],
            answer=final["cand_answer"],
            reasoning=final["cand_reasoning"],
        )
    )
    print(f"{final['cand_reward']:.2f} → {final['cand_answer']}")

print("\nDone – best prompts and scores stored in `results`.")