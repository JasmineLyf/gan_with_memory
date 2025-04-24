# release_cpu_memory.py 或 release_memory.py

import gc
import torch

gc.collect()  # ✅ 回收 Python 层对象
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # ✅ 如果用的是 GPU

print("✅ Memory cleanup complete (CPU + optional GPU).")
