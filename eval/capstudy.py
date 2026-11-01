#!/usr/bin/env python3
"""capstudy.py GOLDEN RESULTS.jsonl SCORESDIR [CAPS] -- how the noise of a paired comparison depends on the per-document
position cap (the stream's eval design, 2026-11-01). Uses every mix scored on GOLDEN (per-row .f32 scores, matched by
row count), and pairs of mixes within 3% in bytes (the comparisons the search makes).
Per cap c (score only positions < c of each document; the eval runs min(len, c) tokens a document):
  se     median over pairs of the document-bootstrap SE of log(mean_A / mean_B)
  tok    tokens the eval processes; t = T0 + tok / RATE (fitted on the slice's wall times)
  eff    se^2 x t relative to the largest cap: < 1 = less noise per GPU-second (if the pool has the documents to
         spend the saved time on more of them)
  agree  correlation over pairs of log(mean_A / mean_B) at cap c with the full-slice value (does a cap change the call)
Also: early (< 2k) vs late (2k..8k) positions agreement, and the intraclass correlation of per-position paired diffs."""
import glob, json, math, os, re, sys
import numpy as np

G, RES, SD = sys.argv[1:4]
CAPS = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "256,512,1024,2048,4096,8192").split(",")]
m = json.load(open(f"{G}/kld.json")); R = m["rows"]
gen = np.fromfile(f"{G}/gen.u8", np.uint8).astype(bool)
docs = m["docs"]; D = len(docs)
doc = np.zeros(R, np.int32); pos = np.zeros(R, np.int64)
for i, d in enumerate(docs):
    n = d["tokens"] - 1 - d["score_from"]; doc[d["row0"]:d["row0"] + n] = i
    pos[d["row0"]:d["row0"] + n] = d["score_from"] + np.arange(n)
dl = np.array([d["tokens"] for d in docs])
gname = os.path.basename(G.rstrip("/"))
meta = {}
for l in open(RES):
    try: r = json.loads(l)
    except ValueError: continue
    if r.get("golden") == gname and "kld" in r and "bytes" in r:
        w = re.search(r"wall=(\d+)s", r.get("run", "")); meta[r["tag"]] = (r["bytes"], int(w.group(1)) if w else None)
K = {}
for t, (b, w) in meta.items():
    p = f"{SD}/{t}.f32"
    if os.path.exists(p) and os.path.getsize(p) == R * 16:
        s = np.fromfile(p, np.float32).reshape(R, 4)[:, 0].astype(np.float64)
        if (s[gen] > 0).mean() > 0.99: K[t] = s
tags = sorted(K, key=lambda t: meta[t][0])
print(f"{gname}: {D} docs, {gen.sum()} scored rows, {len(tags)} mixes with per-row scores")
pairs = [(a, b) for i, a in enumerate(tags) for b in tags[i + 1:i + 8] if abs(meta[b][0] / meta[a][0] - 1) < 0.03]
rng = np.random.default_rng(0)
if len(pairs) > 400: pairs = [pairs[i] for i in rng.choice(len(pairs), 400, replace=False)]
print(f"pairs within 3% bytes: {len(pairs)}")
# wall time model t = T0 + tok / RATE over this slice's evals (all at the full slice)
walls = [meta[t][1] for t in tags if meta[t][1]]
T0 = 15.0; RATE = dl.sum() / max(np.median(walls) - T0, 1)
print(f"median wall {np.median(walls):.0f} s for {dl.sum()} tokens -> assumed T0 {T0:.0f} s, {RATE:.0f} tok/s")
B = 400; boot = rng.integers(0, D, (B, D))
def docsums(s, c):
    k = gen & (pos < c)
    return np.bincount(doc[k], s[k], D), np.bincount(doc[k], None, D)
full = {}; rows = []
for c in CAPS + [10 ** 9]:
    ses, lr = [], []
    for a, b in pairs:
        sa, n = docsums(K[a], c); sb, _ = docsums(K[b], c)
        lr.append(math.log(sa.sum() / sb.sum()))
        bs = np.log(sa[boot].sum(1) / sb[boot].sum(1)); ses.append(bs.std())
    tok = np.minimum(dl, c).sum(); t = T0 + tok / RATE
    rows.append((c, float(np.median(ses)), int(tok), t, np.array(lr)))
cF, seF, tokF, tF, lrF = rows[-1]
print(f"{'cap':>8} {'se':>7} {'tokens':>9} {'t (s)':>6} {'eff':>6} {'agree':>6}")
for c, se, tok, t, lr in rows:
    eff = se ** 2 * t / (seF ** 2 * tF); ag = np.corrcoef(lr, lrF)[0, 1]
    print(f"{'all' if c == 10 ** 9 else c:>8} {se:7.4f} {tok:9d} {t:6.0f} {eff:6.2f} {ag:6.3f}")
early, late = [], []
for a, b in pairs:
    for lo, hi, out in ((0, 2048, early), (2048, 8192, late)):
        k = gen & (pos >= lo) & (pos < hi)
        if k.sum(): out.append(math.log(K[a][k].sum() / K[b][k].sum()))
if len(early) == len(late) and len(late) > 2:
    print(f"early (< 2k) vs late (2k-8k) log-ratio correlation over pairs: {np.corrcoef(early, late)[0, 1]:.3f} "
          f"(late positions: {(gen & (pos >= 2048)).sum()} rows in {len(set(doc[gen & (pos >= 2048)]))} docs)")
icc = []
for a, b in pairs[:100]:
    dlt = (K[a] - K[b])[gen]; dd = doc[gen]; n = np.bincount(dd, None, D); mu = np.bincount(dd, dlt, D) / np.maximum(n, 1)
    k = n > 1; vb = np.average((mu[k] - dlt.mean()) ** 2, weights=n[k]); vt = dlt.var()
    icc.append(vb / vt)
print(f"between-document share of per-position paired variance (median over pairs): {np.median(icc):.4f}")
