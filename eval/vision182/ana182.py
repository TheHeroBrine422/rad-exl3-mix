#!/usr/bin/env python3
"""ana182.py W OUT.json -- #182 report (w156): per candidate (W/cands.txt), from W/res/TAG-vis.json.rows (the vision set:
image items + the dolly text control, W/corpus/meta.jsonl order) and W/res/TAG-s173x.json.rows (golden-s173e-x, the
xcorp part of s173e; gen-masked by its gen.u8, as w151's stage-B rows):
  - mean KL on the answer rows (model-written, the gen mask) and on the question rows after the image, per kind;
  - vision / text ratios: image answers vs the text control's answers (same generator, template, scoring: the matched
    comparison) and vs s173e-x gen rows (the corpus-style text number), with 95 % CIs from 4,000 document bootstraps
    (each set resampled on its own; position-weighted means);
  - per kind: ratio to the text control with its CI;
  - entropy-matched vision / text ratio: answer rows binned by the bf16 reference's entropy (6 quantile bins of the
    pooled answer rows), the text control's per-bin means reweighted to the image rows' bin shares (does vision differ
    only because its answers are more or less certain?), same bootstrap.
Rows files: [rows, 4] f32 (kl, nll_q, nll_p, top1), the golden's row order."""
import json, os, sys
import numpy as np

W, OUT = sys.argv[1:3]
GOLD = os.environ.get("GOLD", "vis182")          # another reference over the same corpus (e.g. gold/k6f, the A8 test)
CANDS = os.environ.get("CANDS", "cands.txt")
rng = np.random.default_rng(182)
NB = 4000
meta = [json.loads(l) for l in open(f"{W}/corpus/meta.jsonl")]
gv = json.load(open(f"{W}/gold/{GOLD}/kld.json"))
assert len(gv["docs"]) == len(meta)
for d, m in zip(gv["docs"], meta):
    assert d["tokens"] - 1 - d["score_from"] == m["tail"], (d, m)
GX = "/data/q27/w131/slices/golden-s173e-x"
gx = json.load(open(f"{GX}/kld.json"))
genx = np.fromfile(f"{GX}/gen.u8", dtype=np.uint8)
assert len(genx) == gx["rows"]


ent = np.fromfile(f"{W}/gold/{GOLD}/rows.f32", dtype=np.float32).reshape(-1, 3)[:, 1]   # (nll, entropy, tail mass)


def rows(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def docsums(kl, spans):
    """per document (sum KL, n, top1 sum) over the given row spans [(lo, hi)]"""
    return np.array([(kl[lo:hi, 0].sum(dtype=np.float64), hi - lo, kl[lo:hi, 3].sum()) for lo, hi in spans])


def spans_vis(sel, part):
    out = []
    for d, m in zip(gv["docs"], meta):
        if not sel(m): continue
        r0, a = d["row0"], m["ans"]
        lo, hi = (r0 + m["tail"] - a, r0 + m["tail"]) if part == "ans" else (r0, r0 + m["tail"] - a)
        if hi > lo: out.append((lo, hi))
    return out


def gen_rows_x(r):
    """s173e-x per document: (sum KL, n, top1) over its gen-masked rows"""
    out = []
    for d in gx["docs"]:
        lo, hi = d["row0"], d["row0"] + d["tokens"] - 1 - d["score_from"]
        m = genx[lo:hi].astype(bool)
        if m.any(): out.append((r[lo:hi, 0][m].sum(dtype=np.float64), m.sum(), r[lo:hi, 3][m].sum()))
    return np.array(out)


def mean(s): return s[:, 0].sum() / s[:, 1].sum()


def boot_ratio(a, b):
    ia = rng.integers(0, len(a), (NB, len(a))); ib = rng.integers(0, len(b), (NB, len(b)))
    ra = a[ia, 0].sum(1) / a[ia, 1].sum(1); rb = b[ib, 0].sum(1) / b[ib, 1].sum(1)
    q = np.quantile(ra / rb, [0.025, 0.975])
    return [float(mean(a) / mean(b)), float(q[0]), float(q[1])]


def binsums(kl, spans, edges):
    """per document, per entropy bin: (sum KL, n) over its spans"""
    out = np.zeros((len(spans), len(edges) - 1, 2))
    for i, (lo, hi) in enumerate(spans):
        b = np.clip(np.searchsorted(edges, ent[lo:hi], side="right") - 1, 0, len(edges) - 2)
        out[i, :, 0] = np.bincount(b, weights=kl[lo:hi, 0].astype(np.float64), minlength=len(edges) - 1)
        out[i, :, 1] = np.bincount(b, minlength=len(edges) - 1)
    return out


def ent_ratio(a, b):
    """vision / text with the text control reweighted to the image rows' entropy-bin shares; a, b: [docs, bins, 2]"""
    def r(A, B):
        w = A[..., 1].sum(-2); w = w / w.sum(-1, keepdims=True)
        ma = A[..., 0].sum(-2) / np.maximum(A[..., 1].sum(-2), 1); mb = B[..., 0].sum(-2) / np.maximum(B[..., 1].sum(-2), 1)
        return (w * ma).sum(-1) / (w * mb).sum(-1)
    ia = rng.integers(0, len(a), (NB, len(a))); ib = rng.integers(0, len(b), (NB, len(b)))
    q = np.quantile(r(a[ia], b[ib]), [0.025, 0.975])
    return [float(r(a, b)), float(q[0]), float(q[1])]


kinds = sorted({m["kind"] for m in meta if m["kind"] != "text"})
res = {}
for line in open(f"{W}/{CANDS}"):
    if not line.strip() or line.startswith("#"): continue
    t = line.split()[0]
    pv, px = f"{W}/res/{t}-vis.json.rows", f"{W}/res/{t}-s173x.json.rows"
    if not os.path.exists(pv): continue
    rv = rows(pv); assert len(rv) == gv["rows"]
    rx = rows(px) if os.path.exists(px) else np.full((gx["rows"], 4), np.nan, np.float32)   # no text run: NaN columns
    img = lambda m: m["kind"] != "text"
    va, vq = docsums(rv, spans_vis(img, "ans")), docsums(rv, spans_vis(img, "q"))
    ta, tq = docsums(rv, spans_vis(lambda m: m["kind"] == "text", "ans")), docsums(rv, spans_vis(lambda m: m["kind"] == "text", "q"))
    xg = gen_rows_x(rx)
    r = {"vis_ans": float(mean(va)), "vis_q": float(mean(vq)), "txt_ans": float(mean(ta)), "txt_q": float(mean(tq)),
         "s173x_gen": float(mean(xg)), "s173x_all": float(rx[:, 0].mean(dtype=np.float64)),
         "top1": {"vis_ans": float(va[:, 2].sum() / va[:, 1].sum()), "txt_ans": float(ta[:, 2].sum() / ta[:, 1].sum()),
                  "s173x_gen": float(xg[:, 2].sum() / xg[:, 1].sum())},
         "n": {"vis_docs": len(va), "vis_ans_rows": int(va[:, 1].sum()), "vis_q_rows": int(vq[:, 1].sum()),
               "txt_docs": len(ta), "txt_ans_rows": int(ta[:, 1].sum()), "s173x_docs": len(xg), "s173x_gen_rows": int(xg[:, 1].sum())},
         "ratio_vis_txt": boot_ratio(va, ta), "ratio_vis_s173x": boot_ratio(va, xg), "ratio_txt_s173x": boot_ratio(ta, xg),
         "ratio_visq_txtq": boot_ratio(vq, tq), "kinds": {}}
    sv, st = spans_vis(img, "ans"), spans_vis(lambda m: m["kind"] == "text", "ans")
    pooled = np.concatenate([ent[lo:hi] for lo, hi in sv + st])
    edges = np.quantile(pooled, np.linspace(0, 1, 7)); edges[0], edges[-1] = -np.inf, np.inf
    bv, bt = binsums(rv, sv, edges), binsums(rv, st, edges)
    r["ratio_vis_txt_entmatched"] = ent_ratio(bv, bt)
    r["ent_bins"] = {"edges": [float(x) for x in edges[1:-1]], "vis_share": (bv[..., 1].sum(0) / bv[..., 1].sum()).tolist(),
                     "txt_share": (bt[..., 1].sum(0) / bt[..., 1].sum()).tolist(),
                     "vis_mean": (bv[..., 0].sum(0) / bv[..., 1].sum(0)).tolist(), "txt_mean": (bt[..., 0].sum(0) / bt[..., 1].sum(0)).tolist()}
    for k in kinds:
        ka = docsums(rv, spans_vis(lambda m, k=k: m["kind"] == k, "ans"))
        kq = docsums(rv, spans_vis(lambda m, k=k: m["kind"] == k, "q"))
        sk = spans_vis(lambda m, k=k: m["kind"] == k, "ans")
        r["kinds"][k] = {"docs": len(ka), "ans_rows": int(ka[:, 1].sum()), "ans": float(mean(ka)), "q": float(mean(kq)),
                         "top1_ans": float(ka[:, 2].sum() / ka[:, 1].sum()), "ratio_txt": boot_ratio(ka, ta),
                         "ratio_txt_entmatched": ent_ratio(binsums(rv, sk, edges), bt)}
    res[t] = r
json.dump(res, open(OUT, "w"), indent=1)

f = lambda x: f"{x:.4f}" if x >= 0.01 else f"{x:.5f}"
ci = lambda c: f"{c[0]:.2f} [{c[1]:.2f}, {c[2]:.2f}]"
print("| cand | vision ans | text ctrl ans | s173e-x | vis / ctrl | vis / ctrl, entropy-matched | vis / s173x | ctrl / s173x | vis q / ctrl q | top-1 vis / ctrl |")
print("|---|---|---|---|---|---|---|---|---|---|")
for t, r in res.items():
    print(f"| {t} | {f(r['vis_ans'])} | {f(r['txt_ans'])} | {f(r['s173x_gen'])} | {ci(r['ratio_vis_txt'])} | {ci(r['ratio_vis_txt_entmatched'])} | {ci(r['ratio_vis_s173x'])} | "
          f"{ci(r['ratio_txt_s173x'])} | {ci(r['ratio_visq_txtq'])} | {100 * r['top1']['vis_ans']:.2f} / {100 * r['top1']['txt_ans']:.2f} % |")
print()
print("| cand | " + " | ".join(kinds) + " |")
print("|---|" + "---|" * len(kinds))
for t, r in res.items():
    print(f"| {t} | " + " | ".join(f"{f(r['kinds'][k]['ans'])} (×{ci(r['kinds'][k]['ratio_txt'])})" for k in kinds) + " |")
print("\nentropy-matched (per kind vs the text control):")
for t, r in res.items():
    print(f"| {t} | " + " | ".join(f"×{ci(r['kinds'][k]['ratio_txt_entmatched'])}" for k in kinds) + " |")
