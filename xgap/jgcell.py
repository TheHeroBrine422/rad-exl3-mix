import json, numpy as np, collections, sys
Q = "/data/q27/queue"; g = f"{Q}/slices/golden-jgap"
m = json.load(open(f"{g}/kld.json")); ge = np.fromfile(f"{g}/gen.u8", np.uint8).astype(bool)
meta = [json.loads(l) for l in open(f"{g}/corpus.meta.jsonl")]
def by(nm, key):
    K = np.fromfile(f"{Q}/scores/{nm}.json.rows", np.float32).reshape(-1, 4)[:, 0].astype(float); s = collections.defaultdict(float); n = collections.Counter()
    for i, d in enumerate(m["docs"]):
        k = d["tokens"] - 1 - d["score_from"]; sl = slice(d["row0"], d["row0"] + k); x = K[sl][ge[sl]]; c = key(i); s[c] += x.sum(); n[c] += len(x)
    return s, n
names = sys.argv[1:]
key = lambda i: meta[i]["cell"]
R = {nm: by(nm, key) for nm in names}
s0, n0 = R[names[0]]
print("cell | rows | " + " | ".join(names))
for c in sorted(n0, key=lambda c: -n0[c]):
    print(f"{c:18s} | {n0[c]:6d} | " + " | ".join(f"{R[nm][0][c]/n0[c]:.4f}" for nm in names))
print(f"{'all':18s} | {sum(n0.values()):6d} | " + " | ".join(f"{sum(R[nm][0].values())/sum(n0.values()):.4f}" for nm in names))
