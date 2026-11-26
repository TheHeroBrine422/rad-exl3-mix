import json, numpy as np, collections, os
Q = "/data/q27/queue"; rng = np.random.default_rng(0)
def docs(g, nm):
    m = json.load(open(f"{Q}/slices/{g}/kld.json")); ge = np.fromfile(f"{Q}/slices/{g}/gen.u8", np.uint8).astype(bool)
    K = np.fromfile(f"{Q}/scores/{nm}.json.rows", np.float32).reshape(-1, 4)[:, 0].astype(float); s, n = [], []
    for d in m["docs"]:
        k = d["tokens"] - 1 - d["score_from"]; sl = slice(d["row0"], d["row0"] + k); x = K[sl][ge[sl]]; s.append(x.sum()); n.append(max(len(x), 1))
    return np.array(s), np.array(n), m
def cmp(b, x, keep=None):
    sb, nb, _ = b; sx = x[0]
    if keep is not None: sb, nb, sx = sb[keep], nb[keep], sx[keep]
    bs = [sx[j].sum() / sb[j].sum() for j in (rng.integers(0, len(sb), len(sb)) for _ in range(1500))]
    return f"{sx.sum()/sb.sum():.3f} [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}] m{np.median(sx/np.maximum(sb,1e-12)):.3f}"
jgm = [json.loads(l) for l in open(f"{Q}/slices/golden-jgap/corpus.meta.jsonl")]
nowc = np.array([not (j["cell"] == "long-chat" and "wildchat" in j["source"].lower()) for j in jgm])
ml = np.array([j["cell"] == "multilingual-text" for j in jgm])
arms = [("ctl 100/0/0", "C228-ctl"), ("xgc 50/50/0", "C229-xgc"), ("opmix 0/0/100(onpol)", "C228-opmix"),
        ("p701515", "C230-p701515"), ("p602515", "C230-p602515"), ("p502525", "C230-p502525"), ("p403030", "C230-p403030")]
B = {J: docs(g, f"{J}-I25-step-10M-ckpt-1e+07") for J, g in [("JC", "golden-jc228"), ("JG", "golden-jgap"), ("JX", "golden-jt2")]}
print("arm | jc228 | gap w/o WildChat | gap multilingual text | teacher set-2 judge (JX)")
for lab, a in arms:
    row = [lab]
    for J in ("JC", "JG", "JX"):
        nm = f"{J}-{a}-ckpt-1e+07"
        if J == "JX" and a == "C230-p602515": nm = "JX-C230-p602515r-ckpt-1e+07"
        if not os.path.exists(f"{Q}/scores/{nm}.json"): row.append("-"); continue
        X = docs({"JC": "golden-jc228", "JG": "golden-jgap", "JX": "golden-jt2"}[J], nm)
        if J == "JG": row += [cmp(B[J], X, nowc), cmp(B[J], X, ml)]
        else: row.append(cmp(B[J], X))
    print(" | ".join(row))
