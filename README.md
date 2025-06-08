# Prompt Tuning with LangGraph

This project implements prompt optimization using LangGraph and reward models to automatically improve prompt templates for mathematical problem solving.

![image](https://github.com/user-attachments/assets/930c9ad8-431f-4a14-a05e-ff443b270b77)



## Features

- **Automated Prompt Optimization**: Uses evolutionary approach to find better prompts
- **Reward Model Scoring**: Evaluates response quality using Skywork reward model
- **LangGraph Workflow**: Implements generate → score → mutate cycle
- **Multi-Problem Testing**: Tests prompts across multiple mathematical problems

## Models Used

- **Base Model**: google/gemma-2-27b-it
- **Reward Model**: Skywork/Skywork-Reward-Gemma-2-27B-v0.2

## Prerequisites

- NVIDIA GPU with 80GB+ VRAM (H100, A100, etc.)
- CUDA 11.8 or 12.1
- Python 3.10+

## Installation

### Step 1: Create Virtual Environment

```bash
# Create new virtual environment
python -m venv gemma27b

# Activate virtual environment
source gemma27b/bin/activate
```

### Step 2: Install PyTorch with CUDA

```bash
# Check your CUDA version first
nvidia-smi

# Install PyTorch with CUDA support (replace cu118 with your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1, use:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install Dependencies

```bash
# Install all other requirements
pip install -r requirements.txt
```

### Step 4: Clean GPU Memory

```bash
# Kill any existing Python processes
pkill -9 python

# Kill any Jupyter processes
pkill -9 jupyter
```

### Step 5: Check GPU Memory

```bash
# Run demo script to check available GPU memory
python demo.py
```

This will show you:
- Total GPU memory available
- Currently allocated memory  
- Free memory for model loading

### Step 6: Verify Installation

```bash
# Test CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

## Usage

Once installation is complete and GPU memory is verified:

```bash
# Run the prompt tuning system
python promt_tunning.py
```

The system will:
1. Load both models (Base: Gemma-2-27B, Reward: Skywork)
2. Test the reward model functionality
3. Process 3 sample problems from question.json
4. Optimize prompts through iterative generation-scoring-mutation
5. Display results and best prompts found

## Architecture

The system uses a state graph with three main nodes:

- **Generate**: Creates responses using current prompt template
- **Score**: Evaluates response quality with reward model  
- **Mutate**: Updates prompt based on performance and patience

## Memory Requirements

- **Without Quantization**: ~50-60GB GPU memory
- **With Quantization**: ~25-30GB GPU memory (may be unstable)
- **Recommended**: H100 (94GB) or A100 (80GB) for stable operation

## Troubleshooting

### GPU Memory Issues
```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Check memory usage
nvidia-smi
```

### Virtual Environment Issues
```bash
# Deactivate and recreate if needed
deactivate
rm -rf gemma27b
python -m venv gemma27b
source gemma27b/bin/activate
```

### Model Loading Issues
- Ensure sufficient GPU memory (80GB+)
- Try running demo.py first to check memory
- Consider using smaller models if memory constrained

## File Structure

```
PromptTunning/
├── promt_tunning.py        # Main prompt tuning script
├── demo.py                 # GPU memory check script
├── question.json           # Test problems dataset
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── .gitignore            # Git ignore patterns
```
