#!/usr/bin/env python3
"""swapana.py W OUT.json -- #182 swap test report (w156): for each S-<group> mix (K2.5 with that group's trunk weights from
K6, swap182.sh), the share of the K2.5 -> K6 answer-row KL gap it closes, on the image items and on the text control,
paired document bootstrap (4,000; the same resampled documents for every candidate). A group whose upgrade closes more of
the image gap than of the text gap is where vision is more sensitive than text.
Also per kind (image share closed per kind)."""
import json, sys
import numpy as np

W, OUT = sys.argv[1:3]
rng = np.random.default_rng(1822)
meta = [json.loads(l) for l in open(f"{W}/corpus/meta.jsonl")]
gv = json.load(open(f"{W}/gold/vis182/kld.json"))
spans = [(d["row0"] + m["tail"] - m["ans"], d["row0"] + m["tail"]) for d, m in zip(gv["docs"], meta)]
kind = np.array([m["kind"] for m in meta])


def docs(tag):
    r = np.fromfile(f"{W}/res/{tag}-vis.json.rows", dtype=np.float32).reshape(-1, 4)
    return np.array([r[lo:hi, 0].sum(dtype=np.float64) for lo, hi in spans])


n = np.array([hi - lo for lo, hi in spans], dtype=np.float64)
base, top = docs("K2.5"), docs("K6")   # the ends (1 document in flight; the mixes 8: batching moves KL by ~1e-5)
tags = [l.split()[0] for l in open(f"{W}/cands-swap.txt") if l.strip()]
img, txt = kind != "text", kind == "text"
B = 4000
bi = rng.integers(0, img.sum(), (B, img.sum())); bt = rng.integers(0, txt.sum(), (B, txt.sum()))


def closed(x, sel, idx=None):
    s, b, t = x[sel], base[sel], top[sel]
    f = lambda S, Bs, T: (Bs.sum(-1) - S.sum(-1)) / (Bs.sum(-1) - T.sum(-1))
    v = float(f(s, b, t))
    return v if idx is None else (v, f(s[idx], b[idx], t[idx]))


res = {"base_K2.5": {"img": float(base[img].sum() / n[img].sum()), "txt": float(base[txt].sum() / n[txt].sum())},
       "top_K6": {"img": float(top[img].sum() / n[img].sum()), "txt": float(top[txt].sum() / n[txt].sum())}}
print("| swap (K2.5 + K6 group) | image KL | text KL | image gap closed | text gap closed | image − text [95 % CI] |")
print("|---|---|---|---|---|---|")
for t in tags:
    try: x = docs(t)
    except FileNotFoundError: continue
    vi, bvi = closed(x, img, bi); vt, bvt = closed(x, txt, bt)
    d = np.quantile(bvi - bvt, [0.025, 0.975])
    per = {k: closed(x, kind == k) for k in sorted(set(kind)) if k != "text"}
    res[t] = {"img": float(x[img].sum() / n[img].sum()), "txt": float(x[txt].sum() / n[txt].sum()), "img_closed": vi,
              "txt_closed": vt, "diff": [vi - vt, float(d[0]), float(d[1])], "img_closed_per_kind": per}
    print(f"| {t} | {res[t]['img']:.4f} | {res[t]['txt']:.4f} | {vi:.2f} | {vt:.2f} | {vi - vt:+.2f} [{d[0]:+.2f}, {d[1]:+.2f}] |")
json.dump(res, open(OUT, "w"), indent=1)
print("\nimage gap closed per kind:")
ks = sorted(set(kind) - {"text"})
print("| swap | " + " | ".join(ks) + " |")
for t in tags:
    if t in res: print(f"| {t} | " + " | ".join(f"{res[t]['img_closed_per_kind'][k]:.2f}" for k in ks) + " |")
