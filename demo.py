import torch
import gc
import os

# Set memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# More aggressive cleanup
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# Reset CUDA context (this is more aggressive)
torch.cuda.reset_peak_memory_stats()

# Check what's actually available
print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
print(f"Currently allocated: {torch.cuda.memory_allocated() / (1024**3):.1f} GB")
print(f"Currently reserved: {torch.cuda.memory_reserved() / (1024**3):.1f} GB")