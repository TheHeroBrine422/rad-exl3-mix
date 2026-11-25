import json, numpy as np, collections
Q = "/data/q27/queue"
jgm = [json.loads(l) for l in open(f"{Q}/slices/golden-jgap/corpus.meta.jsonl")]
kg = lambda i: jgm[i]["cell"] + (" (WC)" if jgm[i]["cell"] == "long-chat" and "wildchat" in jgm[i]["source"].lower() else "")
def per(g, nm, key):
    m = json.load(open(f"{Q}/slices/{g}/kld.json")); ge = np.fromfile(f"{Q}/slices/{g}/gen.u8", np.uint8).astype(bool)
    K = np.fromfile(f"{Q}/scores/{nm}.json.rows", np.float32).reshape(-1, 4)[:, 0].astype(float); s = collections.defaultdict(list)
    for i, d in enumerate(m["docs"]):
        k = d["tokens"] - 1 - d["score_from"]; sl = slice(d["row0"], d["row0"] + k); s[key(i)].append(K[sl][ge[sl]].sum())
    return {c: np.array(v) for c, v in s.items()}
rng = np.random.default_rng(0)
def rat(b, x):
    bs = []
    for _ in range(2000):
        j = rng.integers(0, len(b), len(b)); bs.append(x[j].sum() / b[j].sum())
    return f"{x.sum()/b.sum():.3f} [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]"
# 4.50 drift vs FP8, compared with 7.00 / IQ8R
F = per("golden-jgap", "JG-REF-FP8", kg); Fc = per("golden-jc228", "JC-REF-FP8", lambda i: "c")["c"].sum()
print("4.50 / FP8 drift (ratio on cell / ratio on jc228):")
b450 = per("golden-jc228", "JC-UN4.50", lambda i: "c")["c"].sum() / Fc
U = per("golden-jgap", "JG-UN4.50", kg)
for c in sorted(F) + ["ALL"]:
    f = np.concatenate(list(F.values())) if c == "ALL" else F[c]; u = np.concatenate(list(U.values())) if c == "ALL" else U[c]
    print(f"  {c:22s} 4.50/FP8 {u.sum()/f.sum():.3f}  drift {u.sum()/f.sum()/b450:.2f}")
print(f"  jc228 4.50/FP8 {b450:.3f}")
# mid-run arms (+0.5M rows) vs the 10M end
for J, g, key in [("JC", "golden-jc228", lambda i: "jc228"), ("JG", "golden-jgap", kg)]:
    B = per(g, f"{J}-I25-step-10M-ckpt-1e+07", key)
    for a in ["C229-xg", "C229-xgc"]:
        X = per(g, f"{J}-{a}-ckpt-raw-1.05e+07", key)
        cells = sorted(B) + (["ALL"] if J == "JG" else [])
        out = []
        for c in cells:
            b = np.concatenate(list(B.values())) if c == "ALL" else B[c]; x = np.concatenate(list(X.values())) if c == "ALL" else X[c]
            out.append(f"{c}: {rat(b, x)}")
        print(J, a, "+0.5M vs 10M end |", " | ".join(out))
