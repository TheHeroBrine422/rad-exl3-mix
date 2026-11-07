#!/usr/bin/env python3
"""capq.py OURQ ARGS... -- run ourq.py (as __main__, same __file__: same code stamp) with this process's CUDA
allocator capped at VRAM_CAP_GB (the 4070 is shared: gpu.lock rules)."""
import os, runpy, sys
import torch

cap = float(os.environ.get("VRAM_CAP_GB", "7.2"))
tot = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 2**30 / tot), 0)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
print(f"capq: peak allocated {torch.cuda.max_memory_allocated(0) / 2**30:.2f} GiB (cap {cap} GiB)", flush=True)
