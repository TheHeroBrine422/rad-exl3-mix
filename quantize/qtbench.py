import torch, time, sys
import exllamav3.modules.quant.exl3_lib.quantize as Q
dev = torch.device("cuda:0"); torch.manual_seed(0)
t = torch.randn(2176, 256, device=dev)
for K in range(1, 9):
    qa = {"K": K, "devices": [0], "device_ratios": None, "mul1": True}
    Q.quantize_tiles(t, qa); torch.cuda.synchronize()
    s = time.perf_counter()
    for _ in range(4): Q.quantize_tiles(t, qa)
    torch.cuda.synchronize(); print(f"K{K} {(time.perf_counter()-s)/4/2176*1e6:.2f}", flush=True)
