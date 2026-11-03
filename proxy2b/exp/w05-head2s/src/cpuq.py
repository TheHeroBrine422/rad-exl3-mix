"""cpuq.py -- exllamav3's exl3_lib/quantize.py on the CPU, for self-tests: the module source is exec'd with its package
imports replaced by stubs, the CUDA stream / sync calls made no-ops, and the trellis search (ext.quantize_tiles)
replaced by a scalar stand-in: round to the nearest of 2^K uniform levels on [-CLIP_K, CLIP_K] (index = level).
Everything else (regularize, scale search, rotations, block LDL, LDLQ / two-sided LDLQ, scale refit) is the module's
own code. load(path) -> module."""
import math, types, torch

CLIP = {1: 1.0, 2: 1.7, 3: 2.2, 4: 2.7, 5: 3.1, 6: 3.5, 7: 3.9, 8: 4.2}   # ~ Gaussian-optimal uniform clips

def rtn_tiles(tiles, q_tiles, q_idx, temp_costs, temp_edges, K, mcg=False, mul1=False):
    n = 1 << K; a = CLIP[K]; step = 2 * a / n
    i = torch.clamp(torch.floor((tiles + a) / step), 0, n - 1)
    q_tiles.copy_(-a + (i + 0.5) * step); q_idx.copy_(i.to(torch.short))

def hadamard(n, device, dtype, scale):
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n: h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h * scale).to(device=device, dtype=dtype)

class _PB:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def update(self, *a): pass
    def new_task(self, *a): pass

def load(path):
    src = open(path).read()
    for line in src.splitlines():
        if line.startswith("from ...."):
            src = src.replace(line, "")
    # block_ldl parks H with H.cpu() before overwriting the device copy with L: on the CPU .cpu() is the same tensor
    # (H would become L). The harness's one source change, made identically to every module it loads
    assert src.count("H_cpu = H.cpu()\n") == 2
    src = src.replace("H_cpu = H.cpu()\n", "H_cpu = H.cpu().clone()\n")
    ext = types.SimpleNamespace(quantize_tiles=rtn_tiles, pack_trellis=lambda p, e, K: None,
                                pack_trellis_frac=lambda *a: None)
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.Stream = lambda *a, **k: None
    m = types.ModuleType("q_" + str(abs(hash(path))))
    m.__dict__.update(ext=ext, ProgressBar=_PB, free_mem=lambda: None, list_gpu_tensors=lambda *a: None,
                      get_hadamard_dt=hadamard, cuda_sync_active=lambda: None, save_tensor_image=lambda *a: None)
    exec(compile(src, path, "exec"), m.__dict__)
    m.get_temp_buffers = lambda *a, **k: (None, None)
    return m

def hdata(H, rows=1000):
    return {"H": H.clone(), "first_key": "t", "count": rows, "finalized": False, "num_total": rows,
            "inf_nan": torch.zeros(2, dtype=torch.long), "device": torch.device("cpu")}

def quant(m, W, H, K=3, seed=7, **extra):
    """quantize_exl3 on CPU: W (in, out) float, H (in, in) summed Hessian (count 1000). -> (weight_q, out_tensors, qa)"""
    qa = {"seed": seed, "K": K, "devices": ["cpu"], "device_ratios": None, "apply_out_scales": True, "mul1": True}
    qa.update(extra)
    wq, perr, t = m.quantize_exl3(W.clone(), hdata(H), qa, True)
    return wq, t, qa
