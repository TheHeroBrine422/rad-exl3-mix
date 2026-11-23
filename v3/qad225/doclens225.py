#!/usr/bin/env python3
"""w225: token lengths of the training documents (w214's chunks + the calib train214 pool [+ the ext plan's new goldens]) at a cap,
and the plan's concurrent depth: per step, how many of the V streams sit at depth >= 32k / 64k / 128k (host-RAM / VRAM sizing of
the carried KV, 64 KB a token). Metadata only (kld.json), no golden rows read beyond gen masks."""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "qad181"))
import data181 as D
ap = argparse.ArgumentParser(); ap.add_argument("--chunks", required=True); ap.add_argument("--calib"); ap.add_argument("--pools")
ap.add_argument("--pool", default="train214"); ap.add_argument("--genmask"); ap.add_argument("--cap", type=int, default=262144)
ap.add_argument("--V", type=int, default=24); ap.add_argument("--Q", type=int, default=1024); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cap-keep"); ap.add_argument("--cap-keep-len", type=int, default=8192); ap.add_argument("--steps", type=int, default=120)
a = ap.parse_args()
chunks = [x.strip() for x in (open(a.chunks[1:]).read().split() if a.chunks.startswith("@") else a.chunks.split(",")) if x.strip()]
C = D.Corpus(chunks, a.cap, calib=a.calib, pools=a.pools, pool=a.pool, genmask=a.genmask)
keep = set(json.load(open(a.cap_keep))) if a.cap_keep else set(); cg = len(C.g) - 1 if a.calib else None
C.items = [(g, i, a.cap_keep_len, C.g[g].nrows(i, a.cap_keep_len)) if (g == cg and i in keep and n > a.cap_keep_len) else (g, i, n, r) for g, i, n, r in C.items]
n = np.array([x[2] for x in C.items]); r = np.array([x[3] for x in C.items])
print(json.dumps(C.summary()))
for t in (8192, 32768, 65536, 131072, 262144):
    print(f"docs > {t}: {(n > t).sum()}  tokens beyond {t}: {np.clip(n - t, 0, None).sum()}  ({np.clip(n - t, 0, None).sum() / n.sum():.3f})")
order = np.random.default_rng(a.seed).permutation(len(C.items)); steps = D.plan(C.items, order, a.V, a.Q, max_steps=a.steps)
rows = [sum(C.items[it][3] if False else 0 for _ in [0]) for _ in []]
deep = {k: [] for k in (32768, 65536, 131072)}; kvsum = []
for st in steps:
    ds = [max(b for (_, _, b) in segs) if segs else 0 for segs in st]
    for k in deep: deep[k].append(sum(d >= k for d in ds))
    kvsum.append(sum(ds) * 65536 / 2**30)
for k, v in deep.items(): print(f"streams at depth >= {k}: mean {np.mean(v):.2f} max {max(v)} over {len(steps)} steps")
print(f"carried KV of all streams: mean {np.mean(kvsum):.1f} GiB, max {max(kvsum):.1f} GiB (64 KB a token)")
