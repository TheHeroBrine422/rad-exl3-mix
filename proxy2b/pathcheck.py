#!/usr/bin/env python3
"""pathcheck.py MODEL META DOC -- the final hidden states of one document under: whole forward twice (determinism),
use_cache False vs True, and the same first 512 tokens alone (vs the whole doc's first 512 rows: causality)."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model
mp, meta, di = sys.argv[1], sys.argv[2], int(sys.argv[3])
_, dec, _ = load_model(mp)
m = json.load(open(f"{meta}/kld.json")); toks = np.memmap(f"{meta}/tokens.i32", np.int32, "r")
off = sum(d["tokens"] for d in m["docs"][:di]); n = m["docs"][di]["tokens"]
tk = torch.from_numpy(np.asarray(toks[off:off + n], np.int64))[None].cuda()
def H(**kw):
    with torch.no_grad(): return dec(input_ids=kw.pop("t", tk), **kw).last_hidden_state[0].float()
def r(a, b): return f"{((a - b).norm(dim=1) / b.norm(dim=1)).mean().item():.2e}"
a0 = H(use_cache=False)
print("repeat           ", r(H(use_cache=False), a0))
print("use_cache True   ", r(H(use_cache=True), a0))
print("first 512 alone  ", r(H(t=tk[:, :512], use_cache=False), a0[:512]))
print("first 64 alone   ", r(H(t=tk[:, :64], use_cache=False), a0[:64]))
# the distribution: KL(whole-doc rows || same rows from a shorter forward), per cut length
_, _, head = load_model(mp) if False else (None, None, None)
import torch.nn.functional as F
from common import load_model as _lm
model, _, hd = _lm(mp)
def lsm(h): return torch.log_softmax(hd(h.to(torch.bfloat16)).float(), 1)
with torch.no_grad():
    for cut in (64, 512, 2048):
        if cut >= n: continue
        p = lsm(a0[:cut]); q = lsm(H(t=tk[:, :cut], use_cache=False))
        kl = (p.exp() * (p - q)).sum(1)
        print(f"KL whole vs first-{cut}-alone: mean {kl.mean().item():.2e}, p99 {kl.quantile(.99).item():.2e}, top1 agree {(p.argmax(1) == q.argmax(1)).float().mean().item():.4f}")
