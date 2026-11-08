#!/usr/bin/env python3
"""test_paired.py ORIG_PAIRED NEW_PAIRED SCRATCH -- synthetic tests of paired.py's --cluster / --block (eval/traces/README.md):
1. without flags NEW's output is byte-identical to ORIG's (several synthetic goldens, gen masks, score_from);
2. --cluster on a key unique per document (and --block 1 on any key) = the per-document bootstrap (same se);
3. --cluster / --block = an independent implementation of the cluster bootstrap (seed 0, 2000 resamples);
4. chunks of a trace that share a per-trace shift: the cluster se is larger than the per-document se (what it is for).
Prints one line per check; exit 1 on any failure."""
import json, os, subprocess, sys
import numpy as np

orig, new, scr = sys.argv[1:4]
os.makedirs(scr, exist_ok=True)
fails = 0


def run(p, *args):
    return subprocess.run([sys.executable, "-I", p, *args], capture_output=True, text=True, check=True).stdout


def check(name, ok, info=""):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} {name} {info}")


def make(g, ndocs, seed, trace_shift=0.0, per_trace=6):
    r = np.random.default_rng(seed)
    docs, row = [], 0
    for d in range(ndocs):
        n = int(r.integers(50, 400)); sf = int(r.integers(0, 20))
        docs.append({"source": f"agent-trace/t{d // per_trace}", "tokens": n, "score_from": sf, "row0": row,
                     "orig": 1000 + d // per_trace, "chunk": d % per_trace}); row += n - 1 - sf
    os.makedirs(g, exist_ok=True)
    json.dump({"format": "radiance-kld-top-1", "rows": row, "docs": docs}, open(f"{g}/kld.json", "w"))
    (r.random(row) < 0.55).astype(np.uint8).tofile(f"{g}/gen.u8")
    base = r.exponential(0.05, row)
    # a per-trace shift: the same for all chunks of a trace
    tid = np.concatenate([np.full(d["tokens"] - 1 - d["score_from"], d["orig"]) for d in docs])
    ts = {t: r.normal(0, trace_shift) for t in np.unique(tid)}
    shift = np.array([ts[t] for t in tid]) * base
    for name, x in (("A", base), ("B", base * 1.02 + shift + r.normal(0, 0.01, row) * base)):
        rows = np.zeros((row, 4), np.float32); rows[:, 0] = x; rows[:, 3] = 1
        rows.tofile(f"{g}/{name}.rows")
    return docs


def ref_cluster(g, labels, blk):
    m = json.load(open(f"{g}/kld.json")); gen = np.fromfile(f"{g}/gen.u8", np.uint8).astype(bool)
    A = np.fromfile(f"{g}/A.rows", np.float32).reshape(-1, 4)[:, 0].astype(np.float64)
    B = np.fromfile(f"{g}/B.rows", np.float32).reshape(-1, 4)[:, 0].astype(np.float64)
    units, order, cnt = {}, [], {}
    for d, lab in zip(m["docs"], labels):
        k = d["tokens"] - 1 - d["score_from"]; s = slice(d["row0"], d["row0"] + k); mk = gen[s]
        j = cnt.get(lab, 0); cnt[lab] = j + 1; u = (lab, j // blk if blk else 0)
        if u not in units: units[u] = [0.0, 0.0, 0]; order.append(u)
        units[u][0] += A[s][mk].sum(); units[u][1] += B[s][mk].sum(); units[u][2] += mk.sum()
    sa, sb, n = (np.array([units[u][c] for u in order], np.float64) for c in range(3))
    rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(n), len(n)); bs.append((sb[i].sum() - sa[i].sum()) / n[i].sum())
    return float(np.std(bs)), len(order)


for t, (nd, seed, sh) in enumerate([(18, 1, 0.0), (36, 2, 0.3), (7, 3, 0.1), (103, 4, 0.0)]):
    g = f"{scr}/g{t}"; docs = make(g, nd, seed, sh)
    o, n0 = run(orig, g, f"{g}/A.rows", f"{g}/B.rows"), run(new, g, f"{g}/A.rows", f"{g}/B.rows")
    check(f"g{t} default byte-identical", o == n0, o.strip()[:80])
    base = json.loads(n0)
    json.dump([f"d{k}" for k in range(nd)], open(f"{g}/uniq.json", "w"))
    for args in (["--cluster", f"{g}/uniq.json"], ["--cluster", "source", "--block", "1"]):
        x = json.loads(run(new, g, f"{g}/A.rows", f"{g}/B.rows", *args))
        check(f"g{t} {' '.join(a.replace(g + '/', '') for a in args)} = per-document", x["se"] == base["se"] and x["units"] == nd
              and all(x[k] == base[k] for k in ("kld_a", "kld_b", "diff", "positions", "docs")), f"se {x['se']:.3e} vs {base['se']:.3e}")
    for key, blk in (("source", 0), ("orig", 0), ("source", 2), ("source", 3)):
        args = ["--cluster", key] + (["--block", str(blk)] if blk else [])
        x = json.loads(run(new, g, f"{g}/A.rows", f"{g}/B.rows", *args))
        se, nu = ref_cluster(g, [d[key] for d in docs], blk)
        check(f"g{t} {' '.join(args)} = reference", abs(x["se"] - se) <= 1e-12 * max(se, 1e-30) and x["units"] == nu,
              f"se {x['se']:.4e} ref {se:.4e}, {nu} units")
    if sh >= 0.3:
        x = json.loads(run(new, g, f"{g}/A.rows", f"{g}/B.rows", "--cluster", "source"))
        check(f"g{t} trace-level shift: cluster se > per-document se", x["se"] > base["se"], f"{x['se']:.4e} > {base['se']:.4e}")
print("ALL PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
