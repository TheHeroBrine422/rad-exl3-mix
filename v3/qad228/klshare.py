import json, numpy as np, collections
Q = "/data/q27/queue"; C = "/data/q27/challenge"
cat = [json.loads(l)["category"] for l in open(f"{C}/corpus/calib.jsonl")]
# scored rows per category in calib
m = json.load(open(f"{C}/golden/calib/kld.json")); gen = np.fromfile(f"{C}/golden/calib/gen.u8", np.uint8)
rows = collections.Counter()
for i, d in enumerate(m["docs"]):
    k = d["tokens"] - 1 - d["score_from"]; rows[cat[i]] += int(gen[d["row0"]:d["row0"] + k].sum())
def kl_by(g, nm, key):
    mm = json.load(open(f"{Q}/slices/{g}/kld.json")); ge = np.fromfile(f"{Q}/slices/{g}/gen.u8", np.uint8).astype(bool)
    K = np.fromfile(f"{Q}/scores/{nm}.json.rows", np.float32).reshape(-1, 4)[:, 0].astype(float); s = collections.defaultdict(list)
    for d in mm["docs"]:
        k = d["tokens"] - 1 - d["score_from"]; sl = slice(d["row0"], d["row0"] + k); s[key(d)].append(K[sl][ge[sl]])
    return {c: np.concatenate(v) for c, v in s.items()}
jc = kl_by("golden-jc228", "JC-I25-step-10M-ckpt-1e+07", lambda d: cat[d["orig"]])
jl = kl_by("golden-jlong", "JL-I25-step-10M-ckpt-1e+07", lambda d: d["source"])
jlc = kl_by("golden-jlong", "JL-I25-step-10M-ckpt-1e+07", lambda d: cat[d["orig"]] if "orig" in d else "?")
kl = {c: v.mean() for c, v in jc.items()}; n = {c: len(v) for c, v in jc.items()}
kl["long-agentic"] = np.concatenate([jl["agent-trace/attrs"], jl["agent-trace/click"]]).mean(); n["long-agentic"] = sum(len(jl[k]) for k in ("agent-trace/attrs", "agent-trace/click"))
for c, v in jlc.items():
    if c not in kl: kl[c] = v.mean(); n[c] = len(v); print("from jlong (positions >= 8192):", c)
w = dict(rows); w["long-agentic"] *= 2      # leaderboard subset: 2 of 8 traces vs every 8th doc
W = sum(w.values()); tot = sum(w[c] / W * kl[c] for c in w)
print(f"leaderboard-weighted mean KL (10M 2.50 model, calib judges) {tot:.4f}")
print("category | share of scored positions | mean KL/row | share of total KL | judge rows")
for c in sorted(w, key=lambda c: -w[c] / W * kl[c]):
    print(f"{c:14s} | {w[c]/W*100:5.1f}% | {kl[c]:.4f} | {w[c]/W*kl[c]/tot*100:5.1f}% | {n.get(c,0)}")
