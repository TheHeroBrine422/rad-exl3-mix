#!/usr/bin/env python3
"""pair182.py W BASE TAG... -- #182 paired comparison (w156): each TAG's answer-row KL over BASE's on the image items
and on the text control (same documents, so paired: 4,000 document bootstraps resample documents once for both),
ratios with 95 % CIs, and the image ratio over the text ratio (does the change help or hurt vision more than text?);
per kind too. Rows: W/res/TAG-vis.json.rows."""
import json, sys
import numpy as np

W, BASE, TAGS = sys.argv[1], sys.argv[2], sys.argv[3:]
rng = np.random.default_rng(1823)
meta = [json.loads(l) for l in open(f"{W}/corpus/meta.jsonl")]
gv = json.load(open(f"{W}/gold/vis182/kld.json"))
spans = [(d["row0"] + m["tail"] - m["ans"], d["row0"] + m["tail"]) for d, m in zip(gv["docs"], meta)]
kind = np.array([m["kind"] for m in meta])


def docs(tag):
    r = np.fromfile(f"{W}/res/{tag}-vis.json.rows", dtype=np.float32).reshape(-1, 4)
    return np.array([r[lo:hi, 0].sum(dtype=np.float64) for lo, hi in spans])


b = docs(BASE)
img, txt = np.where(kind != "text")[0], np.where(kind == "text")[0]
bi = img[rng.integers(0, len(img), (4000, len(img)))]; bt = txt[rng.integers(0, len(txt), (4000, len(txt)))]
q = lambda x: [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]
out = {}
print(f"| {BASE} → | image ans ratio | text ans ratio | image / text |")
print("|---|---|---|---|")
for t in TAGS:
    x = docs(t)
    ri, rt = x[img].sum() / b[img].sum(), x[txt].sum() / b[txt].sum()
    bri, brt = x[bi].sum(1) / b[bi].sum(1), x[bt].sum(1) / b[bt].sum(1)
    out[t] = {"img": [float(ri)] + q(bri), "txt": [float(rt)] + q(brt), "img_over_txt": [float(ri / rt)] + q(bri / brt),
              "kinds": {k: float(x[kind == k].sum() / b[kind == k].sum()) for k in sorted(set(kind))}}
    c = lambda v: f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"
    print(f"| {t} | {c(out[t]['img'])} | {c(out[t]['txt'])} | {c(out[t]['img_over_txt'])} |")
print("\nper kind (ratio to " + BASE + "):")
ks = sorted(set(kind))
print("| tag | " + " | ".join(ks) + " |")
for t in TAGS: print(f"| {t} | " + " | ".join(f"{out[t]['kinds'][k]:.3f}" for k in ks) + " |")
json.dump(out, open(f"{W}/res/pair182-{BASE}.json", "w"), indent=1)
