import json, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/workspace/rad-exl3-mix/proxy2b")
from common import load_model
torch.cuda.set_per_process_memory_fraction(7.2 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
print("allow_bf16_reduced_precision_reduction", torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
gd = "/data/g08-fit1"; m = json.load(open(f"{gd}/kld.json")); toks = np.memmap(f"{gd}/tokens.i32", np.int32, "r")
off = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
seqs = [torch.from_numpy(np.array(toks[off[i]:off[i] + 1024], np.int64)) for i, d in enumerate(m["docs"]) if d["tokens"] >= 1024][:8]
A, _, _ = load_model("/data/models/qwen35-08b"); B, _, _ = load_model("/data/yb/w75/m08perm")
def klrows(a, b):
    out = []
    for i in range(0, a.shape[0], 128):
        pa, pb = F.log_softmax(a[i:i+128].float(), -1), F.log_softmax(b[i:i+128].float(), -1)
        out.append((pa.exp() * (pa - pb)).sum(-1).clamp(min=0).cpu())
    return torch.cat(out)
for flag in (True, False):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
    ks = []
    with torch.no_grad():
        for s in seqs:
            x = s[None].cuda(); ks.append(klrows(A(x).logits[0], B(x).logits[0]))
    k = torch.cat(ks).double()
    q = torch.quantile(k, torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=torch.double))
    print(f"reduced_precision_reduction={flag}: KL(A||B) mean {k.mean():.2e} median {q[0]:.2e} p90 {q[1]:.2e} p99 {q[2]:.2e} p99.9 {q[3]:.2e} max {k.max():.2e}; by position quartile:",
          [f"{float(k.view(8, 1024)[:, a:a+256].mean()):.1e}" for a in range(0, 1024, 256)])
