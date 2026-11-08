#!/usr/bin/env python3
"""report32.py GOLDEN ROWSDIR [--json OUT] [--paired PAIRED.py] -- first readout of the K2.5 / K4 mixes on a chunked
agentic slice (eval/traces/README.md; report only, nothing is decided on it). Rows: radiance 1.16 --kld-out,
[rows, 4] f32 = {kl, nll_q, nll_p, top1}, ROWSDIR/<arm>.json.rows. Over the rows GOLDEN's gen.u8 marks:
  per arm   mean / p99 KLD, top-1 agreement, and the mean per trace;
  per pair  B / A, with three bootstraps (seed 0, 2,000 resamples, as paired.py): per chunk (paired.py's default), trace x
            chunk blocks of 3 consecutive chunks (--cluster orig --block 3), per trace (--cluster orig: 2 traces, so
            its interval is shown, not trusted); se, z and the ratio's 95% interval for each.
--paired: also run that paired.py (eval/traces/paired.py) per pair and bootstrap and require the same se (self-check)."""
import json, os, subprocess, sys
import numpy as np

ARMS = ["base", "s1", "baseswj", "s1swj", "base4v4", "base4swj", "bn4v4", "bn4swj"]
PAIRS = [("base", "baseswj", "K2.5 swj, seed 0"), ("s1", "s1swj", "K2.5 swj, seed 1"), ("base", "s1", "K2.5 seed floor"),
         ("base4v4", "base4swj", "K4 swj, head-matched"), ("bn4v4", "bn4swj", "K4 swj on bn, head-matched"),
         ("base4v4", "bn4v4", "K4 bn H_out")]
BOOT = [("chunk", None, 0), ("blocks3", "orig", 3), ("trace", "orig", 0)]


def units(m, key, blk):
    if key is None: return np.arange(len(m["docs"]))
    seen, out = {}, []
    for d in m["docs"]:
        c = seen.setdefault(json.dumps(d[key]), [len(seen), 0]); out.append((c[0], c[1] // blk if blk else 0)); c[1] += 1
    ids = {u: j for j, u in enumerate(dict.fromkeys(out))}
    return np.array([ids[u] for u in out])


def main():
    g, rd = sys.argv[1:3]; opt = sys.argv[3:]
    jout = opt[opt.index("--json") + 1] if "--json" in opt else None
    pp = opt[opt.index("--paired") + 1] if "--paired" in opt else None
    m = json.load(open(f"{g}/kld.json")); gen = np.fromfile(f"{g}/gen.u8", np.uint8).astype(bool)
    arms = [a for a in ARMS if os.path.exists(f"{rd}/{a}.json.rows")]
    rows = {}
    for a in arms:
        s = np.fromfile(f"{rd}/{a}.json.rows", np.float32).reshape(-1, 4).astype(np.float64)
        assert len(s) == m["rows"], (a, len(s), m["rows"])
        assert not (s == 0).all(1).any(), f"{a}: unscored rows"
        assert np.isin(s[:, 3], (0.0, 1.0)).all(), f"{a}: column 3 is not top-1"
        rows[a] = s
    sl = [slice(d["row0"], d["row0"] + d["tokens"] - 1 - d["score_from"]) for d in m["docs"]]
    srcs = list(dict.fromkeys(d["source"] for d in m["docs"]))
    out = {"golden": os.path.basename(g.rstrip("/")), "docs": len(m["docs"]), "gen_rows": int(gen.sum()), "arms": {}, "pairs": []}
    print(f"## {out['golden']}: {len(m['docs'])} chunks of {len(srcs)} traces ({', '.join(s.split('/')[-1] for s in srcs)}), "
          f"{int(gen.sum())} scored (gen) positions")
    print(f"{'arm':9} {'mean KLD':>10} {'p99 KLD':>9} {'top-1':>7}  " + "  ".join(f"{s.split('/')[-1]:>9}" for s in srcs))
    for a, s in rows.items():
        kl = s[gen, 0]; per = {}
        for src in srcs:
            mk = np.zeros(len(gen), bool)
            for d, x in zip(m["docs"], sl):
                if d["source"] == src: mk[x] = gen[x]
            per[src] = float(s[mk, 0].mean())
        st = {"mean": float(kl.mean()), "p99": float(np.quantile(kl, 0.99)), "top1": float(s[gen, 3].mean()), "per_trace": per}
        out["arms"][a] = st
        print(f"{a:9} {st['mean']:10.6f} {st['p99']:9.4f} {st['top1']:7.4f}  " + "  ".join(f"{per[x]:9.6f}" for x in srcs))
    print(f"\n{'pair':22} {'B/A':>7}  " + "  ".join(f"{b:>27}" for b, _, _ in BOOT) + "   (se, z, 95% ratio interval)")
    for a, b, what in PAIRS:
        if a not in rows or b not in rows: continue
        sa = np.array([rows[a][x, 0][gen[x]].sum() for x in sl]); sb = np.array([rows[b][x, 0][gen[x]].sum() for x in sl])
        n = np.array([gen[x].sum() for x in sl], np.float64)
        r = {"a": a, "b": b, "what": what, "kld_a": sa.sum() / n.sum(), "kld_b": sb.sum() / n.sum()}
        r["ratio"] = r["kld_b"] / r["kld_a"]; cells = []
        for bn, key, blk in BOOT:
            u = units(m, key, blk); nu = u.max() + 1
            ua, ub, un = (np.bincount(u, w, nu) for w in (sa, sb, n))
            if key is None: ua, ub, un = sa, sb, n.astype(np.int64)      # paired.py's exact arithmetic
            rng = np.random.default_rng(0); bs, rs = [], []
            for _ in range(2000):
                i = rng.integers(0, nu, nu); bs.append((ub[i].sum() - ua[i].sum()) / un[i].sum()); rs.append(ub[i].sum() / ua[i].sum())
            se = float(np.std(bs)); lo, hi = float(np.quantile(rs, 0.025)), float(np.quantile(rs, 0.975))
            r[bn] = {"units": int(nu), "se": se, "z": (r["kld_b"] - r["kld_a"]) / se, "lo": lo, "hi": hi}
            cells.append(f"{nu:2d}u z{r[bn]['z']:+6.2f} [{lo:.3f},{hi:.3f}]")
            if pp:
                args = [] if key is None else ["--cluster", key] + (["--block", str(blk)] if blk else [])
                x = json.loads(subprocess.run([sys.executable, "-I", pp, g, f"{rd}/{a}.json.rows", f"{rd}/{b}.json.rows", *args],
                                              capture_output=True, text=True, check=True).stdout)
                assert abs(x["se"] - se) <= 1e-9 * se, f"paired.py {args}: se {x['se']} vs {se}"
        out["pairs"].append(r)
        print(f"{a + ' -> ' + b:22} {r['ratio']:7.4f}  " + "  ".join(f"{c:>27}" for c in cells) + f"   {what}")
    print("(u = resampled units; the per-trace interval has 2 units and is shown for completeness only)")
    if jout: json.dump(out, open(jout, "w"), indent=1)


if __name__ == "__main__":
    main()
