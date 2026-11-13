#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""mkfake.py MODEL LEVEL SRC OUT --layers N --docs N --len L -- a test golden for qad.py on a small card: the bf16
model TRUNCATED to its first N layers (+ final norm + lm_head) scored on SRC's first --docs documents (each cut to its
first L tokens; SRC = a golden dir: kld.json, tokens.i32, gen.u8), in the radiance-kld-top-1 layout qad.py reads
(top.bin [rows, 512] {id, log p}, rows.f32 [rows, 3] = nll, entropy, tail; gen.u8 copied from SRC's rows). With the
same --layers, qad.py's student is the quantized version of this truncated model, so the whole loop (loss, backward,
monitor, early stop, export) runs end to end without the 27B. Also writes OUT/pools.json (SRC docs' pools-v3 entries)."""
import argparse, json, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import qmodel  # noqa: E402
from qad import REC, TOP  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("level"); ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--layers", type=int, default=4); ap.add_argument("--docs", type=int, default=200)
ap.add_argument("--len", type=int, default=2048); ap.add_argument("--dev", default="cuda:0")
ap.add_argument("--pools", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "eval", "pools", "pools-v3.json"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.set_grad_enabled(False)
m = json.load(open(f"{a.src}/kld.json")); docs = m["docs"][:a.docs]
tok = np.memmap(f"{a.src}/tokens.i32", np.int32, "r"); gen = np.memmap(f"{a.src}/gen.u8", np.uint8, "r")
off = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
stu = qmodel.Student(a.model, a.level, [a.dev], layers=a.layers, quant_layers=[])
out_docs, toks, gens, tops, rows = [], [], [], [], []
R = 0
for i, d in enumerate(docs):
    n = min(a.len, d["tokens"])
    ids = np.asarray(tok[off[i]:off[i] + n], np.int64)
    sf = d["score_from"]
    if n - 1 <= sf: out_docs.append({"tokens": int(n), "score_from": int(sf), "row0": R, "orig": i}); toks.append(ids.astype(np.int32)); continue
    h = stu.run(ids, ckpt=False)[sf:n - 1]
    nxt = torch.from_numpy(ids[sf + 1:n]).to(h.device)
    t = np.empty((n - 1 - sf, TOP), REC); rr = np.empty((n - 1 - sf, 3), np.float32)
    for c in range(0, len(h), 256):
        lq = torch.log_softmax(stu.logits(h[c:c + 256]), -1)
        lp, ix = lq.topk(TOP, dim=1)
        nll = -lq.gather(1, nxt[c:c + 256, None])[:, 0]
        e = lq.exp(); ent = -(e * lq).sum(1, dtype=torch.float64); e.scatter_(1, ix, 0.0); tail = e.sum(1, dtype=torch.float64)
        t[c:c + 256]["id"] = ix.int().cpu().numpy(); t[c:c + 256]["lp"] = lp.float().cpu().numpy()
        rr[c:c + 256] = np.stack([nll.float().cpu().numpy(), ent.float().cpu().numpy(), tail.float().cpu().numpy()], 1)
    tops.append(t); rows.append(rr)
    r0 = d["row0"]; gens.append(np.asarray(gen[r0:r0 + n - 1 - sf]))
    out_docs.append({"tokens": int(n), "score_from": int(sf), "row0": R, "orig": i, "source": d.get("source")})
    toks.append(ids.astype(np.int32)); R += n - 1 - sf
    print(f"doc {i}: {n} tokens, {R} rows", flush=True)
np.concatenate(toks).tofile(f"{a.out}/tokens.i32")
np.concatenate(gens).astype(np.uint8).tofile(f"{a.out}/gen.u8")
np.concatenate(tops).tofile(f"{a.out}/top.bin")
np.concatenate(rows).astype(np.float32).tofile(f"{a.out}/rows.f32")
json.dump({"format": "radiance-kld-top-1", "model": f"{a.model} first {a.layers} layers (qad test)", "top": TOP, "rows": int(R),
           "docs": out_docs}, open(f"{a.out}/kld.json", "w"), indent=0)
pools = json.load(open(a.pools))["pool"]
json.dump([pools[d["orig"]] for d in out_docs], open(f"{a.out}/pools.json", "w"))
print(f"FAKE GOLDEN {len(out_docs)} docs, {R} rows -> {a.out}")
