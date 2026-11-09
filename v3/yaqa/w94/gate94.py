#!/usr/bin/env python3
"""gate94.py DUMPDIR GOLDEN CKPT OUT.json [MAXROWS] -- G1 of #124: are the dumped head inputs the golden's rows?

Logits of the dumped rows through the bf16 lm_head (f32 accumulate) against the golden's top-512 log-probabilities of
the same golden rows (top.bin): KL_top = sum over the golden's top-512 of p_g (log p_g - log q) (the tail beyond 512
left out), and top-1 agreement. A row off by one position gives KL of several nats. PASS (rule RULES.md #124 G1): on
the gen rows, mean KL_top <= 0.01 and top-1 agreement >= 0.97. GPU if visible, else CPU (MAXROWS evenly spaced rows)."""
import json, sys
import numpy as np, torch
from safetensors import safe_open

DUMP, G, CK, OUT = sys.argv[1:5]; MAXR = int(sys.argv[5]) if len(sys.argv) > 5 else 4096
torch.set_grad_enabled(False)
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
H, bf, N = open(f"{DUMP}/336.meta").read().split(); H, N = int(H), int(N)
X = torch.from_numpy(np.fromfile(f"{DUMP}/336.x", np.int16).reshape(N, H)).view(torch.bfloat16)
rows = np.fromfile(f"{DUMP}/rows.i64", np.int64); gen = np.fromfile(f"{DUMP}/gen.u8", np.uint8).astype(bool)
sel = np.unique(np.linspace(0, N - 1, min(N, MAXR)).astype(int))
m = json.load(open(f"{G}/kld.json")); TOP = m["top"]
tb = np.memmap(f"{G}/top.bin", np.uint8, "r", shape=(m["rows"], TOP * 8))
idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]["lm_head.weight"]
with safe_open(f"{CK}/{idx}", "pt") as f: W = f.get_tensor("lm_head.weight").to(dev)
kl, hit = np.zeros(len(sel)), np.zeros(len(sel), bool)
for c in range(0, len(sel), 256):
    s = sel[c:c + 256]
    x = X[s].to(dev)
    z = (torch.mm(x, W.T, out_dtype=torch.float32) if dev.type == "cuda" else x.float() @ W.float().T)
    lq = torch.log_softmax(z, -1)
    g = np.asarray(tb[rows[s]]).view(np.int32).reshape(len(s), TOP, 2)
    ids = g[..., 0]; lp = g[..., 1].view(np.float32).astype(np.float64)
    p = np.exp(lp)
    lqs = torch.gather(lq, 1, torch.from_numpy(ids.astype(np.int64)).to(dev)).double().cpu().numpy()
    kl[c:c + len(s)] = (p * (lp - lqs)).sum(1)
    hit[c:c + len(s)] = lq.argmax(1).cpu().numpy() == ids[np.arange(len(s)), lp.argmax(1)]
gs = gen[sel]
r = {"dump": DUMP, "rows_checked": int(len(sel)), "gen_checked": int(gs.sum()),
     "kl_top_gen": float(kl[gs].mean()), "top1_gen": float(hit[gs].mean()),
     "kl_top_all": float(kl.mean()), "top1_all": float(hit.mean()), "kl_top_p99_gen": float(np.percentile(kl[gs], 99))}
r["pass"] = bool(r["kl_top_gen"] <= 0.01 and r["top1_gen"] >= 0.97)
json.dump(r, open(OUT, "w"), indent=1); print(json.dumps(r))
