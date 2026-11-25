import json, numpy as np, collections
Q = "/data/q27/queue"
def per(g, nm, key):
    m = json.load(open(f"{Q}/slices/{g}/kld.json")); ge = np.fromfile(f"{Q}/slices/{g}/gen.u8", np.uint8).astype(bool)
    K = np.fromfile(f"{Q}/scores/{nm}.json.rows", np.float32).reshape(-1, 4)[:, 0].astype(float); s = collections.defaultdict(list)
    for i, d in enumerate(m["docs"]):
        k = d["tokens"] - 1 - d["score_from"]; sl = slice(d["row0"], d["row0"] + k); s[key(i)].append(K[sl][ge[sl]].sum())
    return {c: np.array(v) for c, v in s.items()}
jgm = [json.loads(l) for l in open(f"{Q}/slices/golden-jgap/corpus.meta.jsonl")]
kg = lambda i: jgm[i]["cell"] + (" (WC)" if jgm[i]["cell"] == "long-chat" and "wildchat" in jgm[i]["source"].lower() else "")
F = {"JC": per("golden-jc228", "JC-REF-FP8", lambda i: "jc228"), "JG": per("golden-jgap", "JG-REF-FP8", kg)}
E = {"JC": per("golden-jc228", "JC-REF-E700", lambda i: "jc228"), "JG": per("golden-jgap", "JG-REF-E700", kg)}
I = {"JC": per("golden-jc228", "JC-REF-IQ8R", lambda i: "jc228"), "JG": per("golden-jgap", "JG-REF-IQ8R", kg)}
rng = np.random.default_rng(0)
base_e = E["JC"]["jc228"].sum() / F["JC"]["jc228"].sum(); base_i = I["JC"]["jc228"].sum() / F["JC"]["jc228"].sum()
print(f"jc228: E700/FP8 {base_e:.4f}  IQ8R/FP8 {base_i:.4f}")
print("cell | docs | E700 drift (ratio / own jc228 ratio) | IQ8R drift | E700 drift / IQ8R drift [95% CI by doc]")
cells = sorted(F["JG"]) + ["ALL"]
for c in cells:
    if c == "ALL":
        f = np.concatenate(list(F["JG"].values())); e = np.concatenate(list(E["JG"].values())); q = np.concatenate(list(I["JG"].values()))
    else: f, e, q = F["JG"][c], E["JG"][c], I["JG"][c]
    de = e.sum() / f.sum() / base_e; di = q.sum() / f.sum() / base_i; bs = []
    for _ in range(2000):
        j = rng.integers(0, len(f), len(f)); bs.append((e[j].sum() / q[j].sum()) / (base_e / base_i))
    print(f"{c:22s} | {len(f):3d} | {de:.2f} | {di:.2f} | {de/di:.2f} [{np.percentile(bs,2.5):.2f}, {np.percentile(bs,97.5):.2f}]")
