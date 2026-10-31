import torch, sys
import exllamav3.modules.quant.exl3_lib.quantize as Q
dev = torch.device("cuda:0"); torch.manual_seed(0)
t = torch.randn(2176, 256, device=dev)
out = {}
for K in range(1, 9):
    qa = {"K": K, "devices": [0], "device_ratios": None, "mul1": True}
    w, i = Q.quantize_tiles(t, qa); torch.cuda.synchronize()
    out[K] = (w.cpu(), i.cpu())
torch.save(out, sys.argv[1])
