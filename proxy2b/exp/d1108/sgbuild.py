#!/usr/bin/env python3
"""sgbuild.py PREPDIR GENDIR OUT KIND -- d1108 step 3 (CPU): a golden-format corpus directory (kld.json docs with
tokens / score_from 0 / row0, tokens.i32, gen.u8 row mask; plus pools.json with every document "qad") that collect.py
(RXM_WORK/challenge/golden/calib), hcollect.py, heal.py (--train-meta) and mkgold.py (FULL) read.
  KIND selfgen-train: prompt + the model's continuation (prompts-train), the last 2048 tokens of each (one heal /
                      collect window a document); gen = the rows predicting a generated token.
  KIND selfgen-eval:  prompts-eval + continuation, whole; gen = generated tokens (the agentic eval golden).
  KIND raw-train:     sgprep's raw 2048-token windows; gen = the other model's assistant-turn tokens.
  KIND td-trace:      GENDIR = turboderp's cal_trace.json (Qwen3.8-27B 6.00 bpw sampling ~140 conversations; his SC
                      calibration data): input_ids + response_ids per row, the last 2048 tokens; gen = response tokens.
Self-generated items are filtered: fewer than 32 generated tokens, or degenerate (a 16-gram repeated > 6 times)."""
import json, os, sys, glob
import numpy as np

prep, gdir, out, kind = sys.argv[1:5]
docs, toks, gens, stats = [], [], [], {"kept": 0, "short": 0, "degenerate": 0, "unfinished_kept": 0}

def degenerate(g):
    if len(g) < 96: return False
    c = {}
    for k in range(len(g) - 16):
        t = tuple(g[k:k + 16]); c[t] = c.get(t, 0) + 1
        if c[t] > 6: return True
    return False

def add(seq, mask_next):                      # mask_next[p] = 1: the token at p is scored (predicted by row p - 1)
    seq = np.asarray(seq, np.int32); g = np.asarray(mask_next[1:], np.uint8)
    docs.append({"source": kind, "tokens": int(len(seq)), "score_from": 0, "row0": int(sum(len(x) for x in gens))})
    toks.append(seq); gens.append(g)

if kind.startswith("selfgen"):
    P = np.load(f"{prep}/prompts-{'train' if kind == 'selfgen-train' else 'eval'}.npy")
    k = 0
    for f in sorted(glob.glob(f"{gdir}/b*.npz")):
        z = np.load(f)
        for G, n, fin in zip(z["gen"], z["len"], z["finished"]):
            p = P[k]; k += 1; g = G[:n]
            if n < 32: stats["short"] += 1; continue
            if degenerate(g.tolist()): stats["degenerate"] += 1; continue
            stats["kept"] += 1; stats["unfinished_kept"] += int(not fin)
            seq = np.concatenate([p, g]); m = np.zeros(len(seq), np.uint8); m[len(p):] = 1
            if kind == "selfgen-train": seq, m = seq[-2048:], m[-2048:]
            add(seq, m)
elif kind == "td-trace":
    for r in json.load(open(gdir))["rows"]:
        p, g = np.asarray(r["input_ids"], np.int32), np.asarray(r["response_ids"], np.int32)
        if len(g) < 32: stats["short"] += 1; continue
        stats["kept"] += 1; stats["unfinished_kept"] += int(r.get("eos_reason") != "stop_token")
        seq = np.concatenate([p, g]); m = np.zeros(len(seq), np.uint8); m[len(p):] = 1
        add(seq[-2048:], m[-2048:])
else:
    R, M = np.load(f"{prep}/raw-train.npy"), np.load(f"{prep}/raw-train-mask.npy")
    for s, m in zip(R, M): add(s, m); stats["kept"] += 1
os.makedirs(out, exist_ok=True)
np.concatenate(toks).astype(np.int32).tofile(f"{out}/tokens.i32")
np.concatenate(gens).astype(np.uint8).tofile(f"{out}/gen.u8")
rows = int(sum(len(x) for x in gens))
json.dump({"format": "d1108", "rows": rows, "docs": docs}, open(f"{out}/kld.json", "w"))
json.dump({"pool": ["qad"] * len(docs)}, open(f"{out}/pools.json", "w"))
stats.update({"docs": len(docs), "tokens": int(sum(len(t) for t in toks)), "scored_rows": int(sum(int(g.sum()) for g in gens))})
json.dump(stats, open(f"{out}/build.json", "w"))
print("SGBUILD DONE", kind, json.dumps(stats))
