#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""healeval.py TAG LEVEL_DIR OUT --host HOSTCONF [--scales RUN] [--recipe R] [--slice NAME=GOLDEN ...]  /  healeval.py cmp OUT TAG_A TAG_B
The judgement path of #165: a level (rxq final dir) with a QAD run's scales applied (rxq.swj apply_level: suh *= r,
svh *= c, rounded to f16 as stored; --scales omitted = the unhealed control), packed with the level's own final.recipe
(src replaced), mixed over HOSTCONF eval's base like a search eval (rxrun.hostd run_eval: every trunk weight from the
pack), scored on each slice with radiance (HOSTCONF eval pack_cmd / mix_cmd / score_cmd, the same locks). Restartable:
every step skips its finished output. cmp: paired per document (mix/paired.py's arithmetic): ratio B / A, its 95%
interval (document bootstrap, 2,000 resamples, seed 0), documents better / worse, per slice.
--head-donor RAD: output.weight from RAD (a container whose output.weight headslab.py build overwrote: an R16 head,
healed or not) instead of the base's; --max-len N: score_cmd's --max-model-len (the 32k trace chunks)."""
import argparse, json, os, subprocess, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "pipeline"))


def sh(cmd, log):
    with open(log, "a") as f:
        f.write(f"$ {' '.join(cmd)}\n"); f.flush()
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)


def fill(tpl, **kw): return [x.format(**kw) for x in tpl]


class Lock:
    def __init__(self, path, excl=True): self.p, self.x = path, excl
    def __enter__(self):
        import fcntl
        self.f = open(self.p, "a"); fcntl.flock(self.f, fcntl.LOCK_EX if self.x else fcntl.LOCK_SH); return self
    def __exit__(self, *a): self.f.close()


def doc_sums(golden, rows):
    """per document (KL sum, scored rows) over gen rows: rxrun.hostd.doc_sums"""
    m = json.load(open(f"{golden}/kld.json")); gen = np.fromfile(f"{golden}/gen.u8", np.uint8).astype(bool)
    K = np.fromfile(rows, np.float32).reshape(-1, 4)
    s, n = [], []
    for d in m["docs"]:
        sl = slice(d["row0"], d["row0"] + d["tokens"] - 1 - d["score_from"]); g = gen[sl]
        s.append(float(K[sl, 0][g].astype(np.float64).sum())); n.append(int(g.sum()))
    return np.array(s), np.array(n)


def run(a):
    H = json.load(open(a.host)); E = H["eval"]
    d = os.path.join(a.out, a.tag); os.makedirs(d, exist_ok=True); log = os.path.join(d, "eval.log")
    lv = a.level.rstrip("/")
    if a.scales:
        from rxq import swj
        dst = os.path.join(d, "level")
        swj.apply_level(lv, a.scales, dst)                   # skips when DST/.done exists
        src = dst
    else:
        src = lv
    rec = os.path.join(d, "final.recipe")
    if not os.path.exists(rec):
        import re
        r = open(a.recipe or os.path.join(os.path.dirname(lv), "final.recipe")).read()
        olds = set(re.findall(r"src=(\S+)", r))
        assert len(olds) == 1, f"recipe names {len(olds)} sources: {sorted(olds)}"
        open(rec + ".part", "w").write(r.replace(f"src={olds.pop()} ", f"src={src} ")); os.replace(rec + ".part", rec)
    pk = os.path.join(d, "trunk.rad")
    if not os.path.exists(pk) and not os.path.exists(os.path.join(d, "mix.rad")):     # the trunk pack may be deleted once mixed
        with Lock(E["conv_lock"]), Lock(E["ram_lock"]):
            if os.path.exists(pk + ".part"): os.remove(pk + ".part")
            rc = sh(fill(E["pack_cmd"], recipe=rec, out=pk + ".part", reuse="", model=E.get("model", "")), log)
            if rc or not os.path.exists(pk + ".part"): raise SystemExit(f"pack rc {rc} ({log})")
            os.replace(pk + ".part", pk)
    from rxq import mix as M
    mf = os.path.join(d, "mix.txt"); mrad = os.path.join(d, "mix.rad")
    open(mf, "w").write("\n".join(f"{w} {pk}" for w in M.trunk_names()) + "\n" + (f"output.weight {a.head_donor}\n" if a.head_donor else ""))
    base = E.get("aliases", {}).get(a.base, a.base)
    if not os.path.exists(mrad):
        with Lock(E["conv_lock"]), Lock(E["ram_lock"]):
            rc = sh(fill(E["mix_cmd"], out=mrad + ".part", base=base, mixfile=mf), log)
            if rc or not os.path.exists(mrad + ".part"): raise SystemExit(f"mix rc {rc} ({log})")
            os.replace(mrad + ".part", mrad)
    card = a.card if a.card is not None else E.get("card", 0)
    card_lock = a.card_lock or E["card_lock"]
    res = {"tag": a.tag, "level": lv, "scales": a.scales, "base": base, "head_donor": a.head_donor, "bytes": os.path.getsize(mrad), "slices": {}}
    scmd = E["score_cmd"]
    if a.max_len:
        scmd = [x.replace("--max-model-len 16384", f"--max-model-len {a.max_len}") for x in scmd]
        assert scmd != E["score_cmd"] or a.max_len == 16384, "score_cmd has no --max-model-len 16384 to replace"
    for sl in a.slice:
        name, g = sl.split("=", 1)
        o = os.path.join(d, f"score-{name}")
        if not os.path.exists(o + ".json.rows"):
            with Lock(card_lock), Lock(E["ram_lock"], False):
                rc = sh(fill(scmd, mix=mrad, golden=g, out=o, card=card), log)
            if rc or not os.path.exists(o + ".json.rows"): raise SystemExit(f"score {name} rc {rc} ({log})")
        s, n = doc_sums(g, o + ".json.rows")
        res["slices"][name] = {"golden": g, "rows": o + ".json.rows", "kld": float(s.sum() / n.sum()), "positions": int(n.sum()), "docs": len(n)}
    json.dump(res, open(os.path.join(d, a.result), "w"), indent=1)
    print(json.dumps(res))


def cmp(out, ta, tb, result="result.json"):
    A = json.load(open(os.path.join(out, ta, result))); B = json.load(open(os.path.join(out, tb, result)))
    rep = {"a": ta, "b": tb, "slices": {}}
    for name, sa in A["slices"].items():
        if name not in B["slices"]: continue
        g = sa["golden"]; xa, n = doc_sums(g, sa["rows"]); xb, nb = doc_sums(g, B["slices"][name]["rows"])
        assert (n == nb).all()
        ka, kb = xa.sum() / n.sum(), xb.sum() / n.sum()
        rng = np.random.default_rng(0); bs = []
        for _ in range(2000):
            i = rng.integers(0, len(n), len(n)); bs.append(xb[i].sum() / xa[i].sum())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ok = n > 0
        rep["slices"][name] = {"kld_a": ka, "kld_b": kb, "ratio": kb / ka, "ci95": [float(lo), float(hi)],
                               "docs_better": int(((xb < xa) & ok).sum()), "docs_worse": int(((xb > xa) & ok).sum()),
                               "docs": int(ok.sum()), "positions": int(n.sum())}
    print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    if sys.argv[1:2] == ["cmp"]:
        cmp(*sys.argv[2:6]); sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("level"); ap.add_argument("out")
    ap.add_argument("--host", required=True); ap.add_argument("--scales", default=None)
    ap.add_argument("--base", default="V2-3.00"); ap.add_argument("--slice", action="append", default=[])
    ap.add_argument("--card", type=int, default=None); ap.add_argument("--card-lock", default=None)
    ap.add_argument("--result", default="result.json", help="the result file's name in OUT/TAG (a later slice set: its own file)")
    ap.add_argument("--head-donor", default=None); ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--recipe", default=None, help="recipe template (default: LEVEL_DIR/../final.recipe); its one src= is replaced")
    run(ap.parse_args())
