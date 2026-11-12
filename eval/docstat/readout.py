#!/usr/bin/env python3
"""readout.py -- the per-document accept / reject readout of a paired A -> B comparison on a two-part 27B search slice
(TODO "Eval readout", user 11-09; w120 11-12). docstat.py's view as a library + JSON, plus the KL-share cap.

  readout.py pair OUT.json --part x GOLD_X A_X.rows B_X.rows --part c GOLD_C A_C.rows B_C.rows [--cap 0.2] [--boot 2000]
  readout.py dens OUT.json --part x GOLD_X M_X.rows [--part c ...] [--caps 8192,4096,2048,1024]

GOLD = a golden dir (kld.json + gen.u8: radiance's scored mask), ROWS = a radiance --kld-rows file (float32 x 4 per row,
KL in column 0), rows aligned with the golden. Only scored positions (gen mask) count, per document from row0 over
tokens - 1 - score_from rows (eval/docstat/docstat.py's convention).

pair, per part and pooled ("both"):
  r            pooled KL ratio sum(B) / sum(A), document bootstrap CI (paired, seed 0) and z = (r - 1) / SE
  r_cap        the same with every document's weight capped so that no document holds more than --cap of A's KL in
               its part (water-filling: the heaviest documents are scaled down to the cap, the rest keep weight 1)
  doc_geo / doc_median / docs_better   per-document ratio statistics (documents with A KL > 0)
  top1pct_share   the share of A's (and B's) KL held by the top 1% of positions (V2 lesson: the tail decides)
  max_doc_share, heavy (documents > cap: share, ratio, source), loo (leave-one-out r range), drop_k (r without the k
  heaviest documents), by_category / by_source: KL ratio per group with its share of A's KL
  flags        human-readable warnings: a document above the cap, r and r_cap on opposite sides of 1, a part whose sign
               differs from the other part's (the two-part rule needs the same sign on both parts)
dens: per document the KL density (doc mean / part mean) and, for each token cap c, the largest share a document would
hold if every document were cut at its first c scored positions (slice sizing; mkslice27.sh)."""
import argparse, collections, json, sys

import numpy as np


def load_part(gold, rows):
    """per document: (KL sums at every scored position as an array, source, category) for one rows file"""
    m = json.load(open(f"{gold}/kld.json")); gen = np.fromfile(f"{gold}/gen.u8", np.uint8).astype(bool)
    K = np.fromfile(rows, np.float32).reshape(-1, 4)[:, 0].astype(np.float64)
    if len(K) != len(gen): raise SystemExit(f"{rows}: {len(K)} rows vs {len(gen)} in {gold}/gen.u8")
    out = []
    for d in m["docs"]:
        s = slice(d["row0"], d["row0"] + d["tokens"] - 1 - d["score_from"])
        out.append({"kl": K[s][gen[s]], "source": str(d.get("source", d.get("name", "")))[:60],
                    "category": str(d.get("category", d.get("source", "?")))})
    return out


def capped_weights(a, cap):
    """weights w <= 1 so that w_i a_i <= cap * sum(w a) for every i (heaviest scaled to the cap, others 1)"""
    w = np.ones(len(a)); tot = a.sum()
    if tot <= 0 or len(a) * cap < 1: return w
    for _ in range(len(a)):
        sh = w * a / (w * a).sum()
        over = sh > cap + 1e-12
        if not over.any(): break
        # fix the over-cap documents at exactly cap of the new total: x_i = cap * T, T = rest / (1 - n_over cap)
        rest = (w * a)[~over].sum(); T = rest / (1 - over.sum() * cap)
        w[over] = cap * T / a[over]
    return w


def boot_ratio(sa, sb, w=None, n=2000, seed=0):
    w = np.ones(len(sa)) if w is None else w
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(sa), (n, len(sa)))
    v = (w[idx] * sb[idx]).sum(1) / (w[idx] * sa[idx]).sum(1)
    return np.percentile(v, [2.5, 97.5]).tolist(), float(np.std(v))


def pair_stats(A, B, cap, nboot):
    sa = np.array([d["kl"].sum() for d in A]); sb = np.array([d["kl"].sum() for d in B])
    npos = np.array([len(d["kl"]) for d in A])
    r = sb.sum() / sa.sum(); ci, se = boot_ratio(sa, sb, n=nboot)
    w = capped_weights(sa, cap); rc = (w * sb).sum() / (w * sa).sum(); cic, sec = boot_ratio(sa, sb, w, n=nboot)
    ok = sa > 0; rr = sb[ok] / sa[ok]; share = sa / sa.sum(); o = np.argsort(-share)
    pa = np.concatenate([d["kl"] for d in A]); pb = np.concatenate([d["kl"] for d in B])
    top = lambda p: float(np.sort(p)[::-1][:max(1, len(p) // 100)].sum() / p.sum()) if p.sum() > 0 else None
    grp = lambda key: {g: {"r": float(sb[ix].sum() / sa[ix].sum()) if sa[ix].sum() > 0 else None,
                           "share_a": float(sa[ix].sum() / sa.sum()), "docs": int(len(ix))}
                       for g, ix in ((g, np.array([i for i, d in enumerate(A) if d[key] == g])) for g in sorted({d[key] for d in A}))}
    loo = [(sb.sum() - sb[i]) / (sa.sum() - sa[i]) for i in range(len(sa))] if len(sa) > 1 else [r]
    return {"docs": int(len(sa)), "positions": int(npos.sum()), "kld_a": float(sa.sum() / npos.sum()), "kld_b": float(sb.sum() / npos.sum()),
            "r": float(r), "ci95": ci, "se": se, "z": float((r - 1) / se) if se > 0 else None,
            "r_cap": float(rc), "ci95_cap": cic, "z_cap": float((rc - 1) / sec) if sec > 0 else None, "cap": cap,
            "docs_capped": int((w < 1).sum()),
            "doc_geo": float(np.exp(np.log(rr).mean())), "doc_median": float(np.median(rr)), "docs_better": float(np.mean(rr < 1)),
            "top1pct_share_a": top(pa), "top1pct_share_b": top(pb), "max_doc_share": float(share.max()),
            "heaviest": [{"share": float(share[i]), "r": float(sb[i] / sa[i]) if sa[i] > 0 else None, "source": A[i]["source"]} for i in o[:5]],
            "heavy": [{"share": float(share[i]), "r": float(sb[i] / sa[i]), "source": A[i]["source"]} for i in o if share[i] > cap],
            "loo": [float(min(loo)), float(max(loo))],
            "drop_k": {str(k): float(sb[o[k:]].sum() / sa[o[k:]].sum()) for k in (0, 1, 2, 3, 5, 10) if k < len(o)},
            "by_category": grp("category"), "by_source": grp("source")}


def pair(a):
    parts = {}
    for name, gold, ra, rb in a.part:
        parts[name] = (load_part(gold, ra), load_part(gold, rb))
    out = {"cap": a.cap, "parts": {}, "flags": []}
    for name, (A, B) in parts.items():
        out["parts"][name] = s = pair_stats(A, B, a.cap, a.boot)
        for h in s["heavy"]:
            out["flags"].append(f"{name}: a document holds {h['share']:.1%} of A's KL (> {a.cap:.0%}; r {h['r']:.3f}, {h['source']})")
        if (s["r"] - 1) * (s["r_cap"] - 1) < 0:
            out["flags"].append(f"{name}: r {s['r']:.4f} and capped r {s['r_cap']:.4f} on opposite sides of 1")
    if len(parts) > 1:
        A = [d for p in parts.values() for d in p[0]]; B = [d for p in parts.values() for d in p[1]]
        out["parts"]["both"] = pair_stats(A, B, a.cap, a.boot)
        sg = {n: np.sign(s["r"] - 1) for n, s in out["parts"].items() if n != "both"}
        if len(set(sg.values())) > 1: out["flags"].append(f"parts disagree in sign: {({n: round(out['parts'][n]['r'], 4) for n in sg})}")
    json.dump(out, open(a.out, "w"), indent=1)
    for n, s in out["parts"].items():
        print(f"== {n}: {s['docs']} docs {s['positions']} pos  r {s['r']:.4f} [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}] z {s['z']:+.2f}  "
              f"capped({a.cap:.0%}, {s['docs_capped']} docs) {s['r_cap']:.4f} z {s['z_cap']:+.2f}  doc geo {s['doc_geo']:.4f} median "
              f"{s['doc_median']:.4f} better {s['docs_better']:.2f}  top-1% pos share {s['top1pct_share_a']:.3f}  max doc {s['max_doc_share']:.1%}")
        print("   heaviest: " + "; ".join(f"{h['share']:.1%} r {h['r']:.3f} {h['source']}" for h in s["heaviest"][:4] if h["r"] is not None))
        print("   by category: " + "  ".join(f"{g} {v['r']:.3f} ({v['share_a']:.0%})" for g, v in s["by_category"].items() if v["r"]))
    for f in out["flags"]: print("FLAG " + f)
    return 0


def dens(a):
    caps = [int(c) for c in a.caps.split(",")]
    out = {"parts": {}}
    for name, gold, rows in a.part:
        D = load_part(gold, rows); tot = sum(d["kl"].sum() for d in D); npos = sum(len(d["kl"]) for d in D)
        mean = tot / npos
        docs = [{"source": d["source"], "category": d["category"], "n": len(d["kl"]),
                 "density": float(d["kl"].mean() / mean) if len(d["kl"]) else 0.0,
                 "kl_at": {c: float(d["kl"][:c].sum()) for c in caps}} for d in D]
        mx = {c: max(x["kl_at"][c] for x in docs) / sum(x["kl_at"][c] for x in docs) for c in caps}
        out["parts"][name] = {"docs": docs, "mean": mean, "positions": npos, "max_share_at_cap": mx}
        print(f"== {name}: {len(D)} docs, {npos} positions, mean KL {mean:.4f}; max doc share at cap " +
              "  ".join(f"{c}: {v:.1%}" for c, v in mx.items()))
        cat = collections.defaultdict(list)
        for x in docs: cat[x["category"]].append(x["density"])
        print("   density by category (median / max): " + "  ".join(f"{k} {np.median(v):.2f}/{max(v):.1f} (n {len(v)})" for k, v in sorted(cat.items())))
    json.dump(out, open(a.out, "w"))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="readout.py"); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("pair"); p.add_argument("out"); p.add_argument("--part", nargs=4, action="append", required=True,
                                                                   metavar=("NAME", "GOLD", "A_ROWS", "B_ROWS"))
    p.add_argument("--cap", type=float, default=0.2); p.add_argument("--boot", type=int, default=2000)
    q = sp.add_parser("dens"); q.add_argument("out"); q.add_argument("--part", nargs=3, action="append", required=True,
                                                                    metavar=("NAME", "GOLD", "ROWS"))
    q.add_argument("--caps", default="8192,4096,2048,1024")
    a = ap.parse_args(argv)
    return pair(a) if a.cmd == "pair" else dens(a)


if __name__ == "__main__":
    sys.exit(main())
