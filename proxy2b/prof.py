#!/usr/bin/env python3
"""prof.py MODEL GOLDEN [DOC] -- where a bf16 2B scoring pass spends its time on one document (after warmup), and the
achieved matmul TFLOPS against the card's peak: decoder forward, LM head, log-softmax (f64), scoring. FLOPs counted
as 2 x tokens x (decoder linear params) + the full-attention layers' QK/PV + 2 x rows x head params (GDN recurrence
not counted: a lower bound on work done)."""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model, Golden, score_rows
mp, gd = sys.argv[1:3]; di = int(sys.argv[3]) if len(sys.argv) > 3 else 0
PEAK = float(os.environ.get("PEAK_TFLOPS", 58.3))   # RTX 4070 bf16 tensor, fp32 accumulate, dense
model, dec, head = load_model(mp); g = Golden(gd)
d = g.docs[di]; tk = g.doc_tokens(di); t = torch.from_numpy(tk.astype(np.int64))[None].cuda(); n = len(tk) - 1; r0 = d["row0"]
top = g.top[r0:r0 + n]; ids = torch.from_numpy(np.ascontiguousarray(top["id"])).cuda()
lp = torch.from_numpy(np.ascontiguousarray(top["lp"])).cuda(); pt = torch.from_numpy(np.asarray(g.rows[r0:r0 + n, 2], np.float64)).cuda()
def T(f, k=3):
    f(); torch.cuda.synchronize(); s = time.time()
    for _ in range(k): f()
    torch.cuda.synchronize(); return (time.time() - s) / k
C = int(os.environ.get("P2B_CHUNK", 256))
with torch.no_grad():
    h = dec(input_ids=t).last_hidden_state[0, :n]
    tf = T(lambda: dec(input_ids=t).last_hidden_state)
    th = T(lambda: [head(h[j:j + C]) for j in range(0, n, C)])
    DT = torch.float64 if os.environ.get("P2B_F64") else torch.float32
    ts = T(lambda: [torch.log_softmax(head(h[j:j + C]).to(DT), 1) for j in range(0, n, C)])
    tall = T(lambda: [score_rows(torch.log_softmax(head(h[j:j + C]).to(DT), 1), ids[j:j + C], lp[j:j + C], pt[j:j + C]) for j in range(0, n, C)])
lin = sum(m.weight.numel() for nme, m in dec.named_modules() if isinstance(m, torch.nn.Linear))
cfg = model.config.get_text_config(); nfull = sum(1 for x in cfg.layer_types if x == "full_attention")
attn = nfull * 2 * 2 * (n * n / 2) * cfg.num_attention_heads * cfg.head_dim
fd = 2 * (n + 1) * lin + attn; fh = 2 * n * head.weight.numel()
print(f"doc {di}: {n + 1} tokens; decoder linears {lin / 1e9:.2f}B params, head {head.weight.numel() / 1e9:.2f}B")
print(f"forward {tf * 1e3:.0f} ms = {fd / tf / 1e12:.1f} TFLOPS ({100 * fd / tf / 1e12 / PEAK:.0f}% of {PEAK:g})")
print(f"head    {th * 1e3:.0f} ms = {fh / th / 1e12:.1f} TFLOPS ({100 * fh / th / 1e12 / PEAK:.0f}%)")
print(f"[{DT}, chunk {C}] head + log-softmax {ts * 1e3:.0f} ms; + scoring {tall * 1e3:.0f} ms")
tot = tf + tall
print(f"scoring pass: {tot * 1e3:.0f} ms = {n / tot:.0f} rows/s; matmul work {(fd + fh) / tot / 1e12:.1f} TFLOPS = "
      f"{100 * (fd + fh) / tot / 1e12 / PEAK:.0f}% of peak; time shares: forward {100 * tf / tot:.0f}%, head {100 * th / tot:.0f}%, "
      f"softmax+score {100 * (tall - th) / tot:.0f}%")
