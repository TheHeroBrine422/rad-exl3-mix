import json, numpy as np
Q = "/data/q27/queue"
A = ["C228-ctl", "C228-optr", "C228-opcal", "C228-opmix", "C228-stud", "C228-studcal"]
def docs(g):
    m = json.load(open(f"{Q}/slices/{g}/kld.json")); ge = np.fromfile(f"{Q}/slices/{g}/gen.u8", np.uint8).astype(bool); return m, ge
def per_doc(g, nm, m, ge):
    K = np.fromfile(f"{Q}/scores/{nm}.json.rows", np.float32).reshape(-1, 4)[:, 0].astype(float); s, n = [], []
    for d in m["docs"]:
        k = d["tokens"] - 1 - d["score_from"]; sl = slice(d["row0"], d["row0"] + k); s.append(K[sl][ge[sl]].sum()); n.append(ge[sl].sum())
    return np.array(s)
def ratio(b, x, keep=None, B=2000):
    if keep is not None: b, x = b[keep], x[keep]
    r = x.sum() / b.sum(); rng = np.random.default_rng(0); bs = []
    for _ in range(B):
        i = rng.integers(0, len(b), len(b)); bs.append(x[i].sum() / b[i].sum())
    return f"{r:.3f} [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]"
# cuts used by any arm -> restrict JT / JU to cuts no arm used
used = {}
for gold, conts in [("onpol-b1", ["onpol-b1", "onpol-mix"]), ("onpol-calib-b1", ["onpol-calib-b1", "onpol-mix"])]:
    pass
def cuts(name): return [json.loads(l)["cut"] for l in open(f"{Q}/ext/gold/{name}/corpus.meta.jsonl")]
uc = set()
for g, c in [("onpol-b1", "onpol-b1"), ("onpol-calib-b1", "onpol-calib-b1"), ("stud-b1", "stud-b1"), ("stud-calib-b1", "stud-calib-b1")]:
    cc = cuts(g); uc |= {(g.replace("stud", "X").replace("onpol", "X"), cc[x[1]]) for x in json.load(open(f"{Q}/cont228/cont-{c}.json"))["items"]}
mix = json.load(open(f"{Q}/cont228/cont-onpol-mix.json"))["items"]
for x in mix:
    cc = cuts(x[0]); uc.add((x[0].replace("onpol", "X"), cc[x[1]]))
rows = {}
for p, g, src in [("JC", "golden-jc228", None), ("JT", "golden-jteach", "onpol-b1"), ("JU", "golden-jteachc", "onpol-calib-b1"), ("JS", "golden-jstud", "stud-b1"), ("JV", "golden-jstudc", "stud-calib-b1")]:
    m, ge = docs(g); keep = None
    if p in ("JT", "JU"):
        cc = cuts(src); keep = np.array([("X" + src[len("onpol"):] if True else "", cc[d["orig"]]) not in uc for d in m["docs"]])
        keep = np.array([(src.replace("onpol", "X"), cc[d["orig"]]) not in uc for d in m["docs"]])
    b = per_doc(g, f"{p}-I25-step-10M-ckpt-1e+07", m, ge)
    for a in A:
        x = per_doc(g, f"{p}-{a}-ckpt-1e+07", m, ge); rows.setdefault(a, {})[p] = ratio(b, x, keep)
    rows.setdefault("_n", {})[p] = int(keep.sum()) if keep is not None else len(b)
# agent traces from jlong
m, ge = docs("golden-jlong"); bK = np.fromfile(f"{Q}/scores/JL-I25-step-10M-ckpt-1e+07.json.rows", np.float32).reshape(-1, 4)[:, 0]
idx = [i for i, d in enumerate(m["docs"]) if d["source"].startswith("agent-trace")]
def tr(nm):
    K = np.fromfile(f"{Q}/scores/{nm}.json.rows", np.float32).reshape(-1, 4)[:, 0]; out = []
    for i in idx:
        d = m["docs"][i]; k = d["tokens"] - 1 - d["score_from"]; sl = slice(d["row0"], d["row0"] + k); out.append(K[sl][ge[sl]].sum() / bK[sl][ge[sl]].sum())
    return " / ".join(f"{v:.3f}" for v in out)
for a in A: rows[a]["traces"] = tr(f"JL-{a}-ckpt-1e+07")
print("docs per judge:", rows["_n"])
print("arm | jc228 | teacher traces JT | teacher calib JU | STUDENT traces JS | student calib JV | agent traces attrs / click")
for a in A:
    r = rows[a]; print(f"{a} | {r['JC']} | {r['JT']} | {r['JU']} | {r['JS']} | {r['JV']} | {r['traces']}")
