import json, sys, numpy as np, collections
g, fa, fb = sys.argv[1:4]
m = json.load(open(f"{g}/kld.json"))
A = np.fromfile(fa, np.float32).reshape(-1, 4); B = np.fromfile(fb, np.float32).reshape(-1, 4)
gen = A[:, 2] > 0; rows = []
for d in m["docs"]:
    k = d["tokens"] - 1 - d["score_from"]; s = slice(d["row0"], d["row0"] + k); gg = gen[s]
    rows.append((d["source"], d["tokens"], A[s, 0][gg].sum(), B[s, 0][gg].sum(), gg.sum()))
print("max tokens", max(r[1] for r in rows))
tot = sum(r[3] - r[2] for r in rows)
rows.sort(key=lambda r: -(r[3] - r[2]))
print("total diff sum", tot, "top-5 docs share", sum(r[3] - r[2] for r in rows[:5]) / tot)
for r in rows[:6]: print(r[0], r[1], round(r[2] / r[4], 5), round(r[3] / r[4], 5))
src = collections.defaultdict(lambda: [0, 0, 0])
for r in rows: s = src[r[0].split("/")[0]]; s[0] += r[2]; s[1] += r[3]; s[2] += r[4]
for k, v in sorted(src.items(), key=lambda x: -x[1][2]): print(f"{k:24s} n {v[2]:7d} ratio {v[1] / v[0]:.3f}")
