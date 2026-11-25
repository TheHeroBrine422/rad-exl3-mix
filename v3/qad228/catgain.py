import json, numpy as np, collections, sys
Q = sys.argv[1]; C = sys.argv[2]
cat = [json.loads(l)["category"] for l in open(f"{C}/corpus/calib.jsonl")]
g = f"{Q}/slices/golden-jc228"; m = json.load(open(f"{g}/kld.json")); ge = np.fromfile(f"{g}/gen.u8", np.uint8).astype(bool)
def by(nm):
    K = np.fromfile(f"{Q}/scores/{nm}.json.rows", np.float32).reshape(-1, 4)[:, 0].astype(float); s = collections.defaultdict(float); n = collections.Counter()
    for d in m["docs"]:
        k = d["tokens"] - 1 - d["score_from"]; sl = slice(d["row0"], d["row0"] + k); x = K[sl][ge[sl]]; c = cat[d["orig"]]; s[c] += x.sum(); n[c] += len(x)
    return s, n
u, n = by("JC-UN2.50"); t, _ = by("JC-I25-step-10M-ckpt-1e+07")
print("category | rows | unhealed KL | 10M KL | 10M / unhealed")
for c in sorted(u, key=lambda c: t[c] / u[c]):
    print(f"{c:14s} | {n[c]:6d} | {u[c]/n[c]:.4f} | {t[c]/n[c]:.4f} | {t[c]/u[c]:.3f}")
print(f"{'all':14s} | {sum(n.values()):6d} | {sum(u.values())/sum(n.values()):.4f} | {sum(t.values())/sum(n.values()):.4f} | {sum(t.values())/sum(u.values()):.3f}")
