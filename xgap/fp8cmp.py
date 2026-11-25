import json, numpy as np, collections
Q = "/data/q27/queue"
cat = [json.loads(l)["category"] for l in open("/data/q27/challenge/corpus/calib.jsonl")]
def by(g, nm, key):
    m = json.load(open(f"{Q}/slices/{g}/kld.json")); ge = np.fromfile(f"{Q}/slices/{g}/gen.u8", np.uint8).astype(bool)
    K = np.fromfile(f"{Q}/scores/{nm}.json.rows", np.float32).reshape(-1, 4)[:, 0].astype(float); s = collections.defaultdict(float); n = collections.Counter()
    for i, d in enumerate(m["docs"]):
        k = d["tokens"] - 1 - d["score_from"]; sl = slice(d["row0"], d["row0"] + k); x = K[sl][ge[sl]]; c = key(i, d); s[c] += x.sum(); n[c] += len(x)
    return s, n
jgm = [json.loads(l) for l in open(f"{Q}/slices/golden-jgap/corpus.meta.jsonl")]
def jgkey(i, d):
    c = jgm[i]["cell"]
    return c + (" (WildChat)" if c == "long-chat" and "wildchat" in jgm[i]["source"].lower() else "")
M = [("unhealed", "UN2.50"), ("10M", "I25-step-10M-ckpt-1e+07"), ("E700", "REF-E700"), ("IQ8R", "REF-IQ8R"), ("FP8", "REF-FP8")]
for J, g, key in [("JC", "golden-jc228", lambda i, d: "calib: " + cat[d["orig"]]), ("JG", "golden-jgap", jgkey)]:
    R = {lab: by(g, f"{J}-{nm}", key) for lab, nm in M}
    s0, n0 = R["FP8"]
    tot = {lab: sum(R[lab][0].values()) / sum(n0.values()) for lab, _ in M}
    print(f"\n{J} | rows | FP8 KL | " + " | ".join(f"{lab}/FP8" for lab, _ in M[:4]))
    print(f"{'ALL':32s} | {sum(n0.values()):6d} | {tot['FP8']:.5f} | " + " | ".join(f"{tot[lab]/tot['FP8']:.2f}" for lab, _ in M[:4]))
    for c in sorted(n0, key=lambda c: -n0[c]):
        f = s0[c] / n0[c]
        print(f"{c:32s} | {n0[c]:6d} | {f:.5f} | " + " | ".join(f"{(R[lab][0][c]/n0[c])/f:.2f}" for lab, _ in M[:4]))
