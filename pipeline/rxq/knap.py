# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""S7: the knapsack over the library of quantized levels, and its hand-off to the search stream.

  library   the levels that exist: one entry per (codebook, K) named as the knapsack names its options (i8-K2.375,
            i4-K1, m1-K6.5): dir, K, codebook, settings / code stamps, post passes applied (swj), packed donor if any
  env       knap5's switches from the settings file ([knap], outside the stamp: pricing, not quantization) and the
            library: the V3R menu lists KN_I4 / KN_I8 / KN_M1 = the library's levels, KN_SWJ / KN_SWJ_PTS when
            post.swj is on, KN_CD when tile CD is on and [knap].cd_factor is set, KN_NOA4 unless [knap].a4
  solve     knap5.py (rad-exl3-mix v3/levels/reprice, the solver of the 11-07 re-pricing; run in place from a tree that
            holds its data, as run5.py runs it) for targets x lambdas -> one JSON line per solve, picks included
  handoff   a solve line -> the search stream's inputs: MIXDIR/knap-<tag>-<base>-<target>.mix (radmix's format,
            weights from the library's packed donors, act lines for int modes; non-trunk picks from a donor map) and
            the frontier line "<name> bpw X pred KLD Y" (search/auto/streamq.py reads both)
Check (tests/check_knap.py): with the 11-07 ladder as the library and the w63 settings, env + solve reproduce
data/levels/reprice/swj2/out/main-acc-2.5.jsonl byte for byte (the knapsack solve the ladder re-pricing reported).

python -m rxq.knap library LEVELDIR... -o library.json [--packs DIR]
python -m rxq.knap env library.json [--config F] [--accurate]
python -m rxq.knap solve library.json OUT.jsonl [--tree RXM] -t 2.5[,3.0] [-l 0[,3e-7]] [--accurate] [--config F]
python -m rxq.knap handoff SOLVE.jsonl library.json MIXDIR --tag TAG [--base NAME] [--donors donors.json] [--frontier F]"""
import argparse, json, os, subprocess, sys

from . import qwen35
from .mix import mix_text, trunk_names, level_info

CBN = {"i8": "i8", "i4": "i4", "mul1": "m1"}                  # level codebook -> knap5's option prefix


def level_name(codebook, K):
    return f"{CBN[codebook]}-K{float(K):g}"


def scan(dirs, packs=None):
    lib = {}
    for d in dirs:
        K, cb = level_info(d)
        e = json.load(open(os.path.join(d, "stamp.json")))[-1]
        name = level_name(cb, K)
        if name in lib: raise SystemExit(f"two levels named {name}: {lib[name]['dir']} and {d}")
        p = os.path.join(packs, f"pack-{name}.rad") if packs else None
        lib[name] = {"dir": os.path.abspath(d), "K": K, "codebook": cb, "stamp": e.get("stamp"),
                     "rxq_settings": e.get("rxq_settings"), "swj": _swj_applied(d),
                     "pack": p if p and os.path.exists(p) else None}
    return lib


def _swj_applied(d):
    """a level made by rxq.swj apply_level carries "swj27" in its shards' metadata"""
    import struct
    f = next((x for x in sorted(os.listdir(d)) if x.startswith("q-") and x.endswith(".safetensors")), None)
    if not f: return False
    with open(os.path.join(d, f), "rb") as fh: n = struct.unpack("<Q", fh.read(8))[0]; md = json.loads(fh.read(n)).get("__metadata__") or {}
    return "swj27" in md


def _klist(ks): return ",".join(f"{k:g}" for k in sorted(ks))


def env(cfg, lib, accurate=False):
    kn = cfg["knap"]; e = {}
    e.update({k: str(v) for k, v in kn["env"].items()})
    e["KN_V3R"] = "1"; e["KN_PICKS"] = "1"
    by = {"i4": [], "i8": [], "m1": []}
    sub1 = []
    for name, v in lib.items():
        cb = CBN[v["codebook"]]
        if v["K"] < 1: sub1.append(name); continue
        by[cb].append(v["K"])
    for cb, var in (("i4", "KN_I4"), ("i8", "KN_I8"), ("m1", "KN_M1")):
        e[var] = _klist(by[cb])
    if sub1 and not kn.get("sub1_ratios"):
        print(f"knap: {len(sub1)} sub-1 levels in the library ({', '.join(sorted(sub1)[:3])}..) are not priced: set "
              f"[knap].sub1_ratios (knap5 KN_SUB1)", file=sys.stderr)
    if kn.get("sub1_ratios"): e["KN_SUB1"] = kn["sub1_ratios"]
    if cfg["post"]["swj"]["enable"]:
        e["KN_SWJ"] = kn["swj_mode"]; e["KN_SWJ_PTS"] = kn["swj_pts"]
    if cfg["quant"]["tile_cd"]["enable"] and kn.get("cd_factor"):
        e["KN_CD"] = f"{kn['cd_factor']:g}"
    if not kn.get("a4"): e["KN_NOA4"] = "1"
    if accurate: e["KN_MODES"] = "f16"
    return e


def stage_tree(rxm, solver, scratch):
    """knap5 reads data relative to its own file and to the cwd (w58 ran it as tree/v3/nf/knap5.py): a scratch tree of
    symlinks into the checkout with the solver linked in as v3/nf/knap5.py (abspath keeps the link's dir)"""
    os.makedirs(os.path.join(scratch, "v3", "nf"), exist_ok=True)
    def link(src, dst):
        if not os.path.lexists(dst): os.symlink(src, dst)
    for x in os.listdir(rxm):
        if x != "v3": link(os.path.join(rxm, x), os.path.join(scratch, x))
    for x in os.listdir(os.path.join(rxm, "v3")):
        if x != "nf": link(os.path.join(rxm, "v3", x), os.path.join(scratch, "v3", x))
    nf = os.path.join(scratch, "v3", "nf")
    for x in os.listdir(os.path.join(rxm, "v3", "nf")): link(os.path.join(rxm, "v3", "nf", x), os.path.join(nf, x))
    src = os.path.normpath(os.path.join(rxm, "v3", "nf", solver))
    dst = os.path.join(nf, os.path.basename(src))
    if os.path.lexists(dst) and os.path.realpath(dst) != src: os.remove(dst)
    link(src, dst)
    return nf, os.path.basename(src)


def solve(cfg, lib, out, tree, targets, lambdas, accurate=False, extra_env=None):
    kn = cfg["knap"]
    nf, solver = stage_tree(tree, kn["solver"], os.path.join(os.path.dirname(os.path.abspath(out)), ".knaptree"))
    E = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", KN_BK=str(1 << 18))
    for k in list(E):
        if k.startswith("KN_") and k != "KN_BK": del E[k]
    E.update(env(cfg, lib, accurate)); E.update(extra_env or {})
    argv = [sys.executable, solver, kn["model"], kn["aq"], kn["aqfactors"], kn["map"], os.path.abspath(out),
            ",".join(map(str, targets)), ",".join(map(str, lambdas))]
    return subprocess.call(argv, cwd=nf, env=E)


def handoff(line, lib, mixdir, tag, base="v3", donors=None, frontier=None):
    """one solve line -> MIXDIR/knap-<tag>-<base>-<target>.mix + its frontier line"""
    picks = line["picks"]; packs = {n: v["pack"] for n, v in lib.items() if v.get("pack")}
    packs.update(donors or {})
    trunk = set(trunk_names())
    unresolved = sorted(w for w, (o, _) in picks.items() if o not in packs)
    if unresolved:
        raise SystemExit(f"{len(unresolved)} picks without a donor (first {unresolved[0]}: {picks[unresolved[0]][0]}); "
                         "pack the level or give --donors")
    miss = sorted(trunk - set(picks))
    if miss: raise SystemExit(f"{len(miss)} trunk weights without a pick (first {miss[0]})")
    name = f"knap-{tag}-{base}-{line['target']:g}"
    os.makedirs(mixdir, exist_ok=True)
    p = os.path.join(mixdir, f"{name}.mix")
    open(p + ".part", "w").write(mix_text({w: tuple(v) for w, v in picks.items()}, packs)); os.replace(p + ".part", p)
    fl = f"{name}  bpw {line['bpw']:.4f}  pred KLD {line['kld']:.6f}"
    if frontier:
        with open(frontier, "a") as fh: fh.write(fl + "\n")
    return p, fl


def main(argv=None):
    from . import config as C
    ap = argparse.ArgumentParser(prog="rxq.knap")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("library"); s.add_argument("levels", nargs="+"); s.add_argument("-o", required=True); s.add_argument("--packs")
    e = sub.add_parser("env"); e.add_argument("library"); e.add_argument("--config"); e.add_argument("--accurate", action="store_true")
    v = sub.add_parser("solve"); v.add_argument("library"); v.add_argument("out"); v.add_argument("--tree", default=None,
                                                                                      help="a tree with knap5's data (default: RXQ_RXM)")
    v.add_argument("-t", required=True); v.add_argument("-l", default="0"); v.add_argument("--accurate", action="store_true")
    v.add_argument("--config")
    h = sub.add_parser("handoff"); h.add_argument("solve"); h.add_argument("library"); h.add_argument("mixdir")
    h.add_argument("--tag", required=True); h.add_argument("--base", default="v3"); h.add_argument("--donors")
    h.add_argument("--frontier")
    a = ap.parse_args(argv)
    if a.cmd == "library":
        lib = scan(a.levels, a.packs); json.dump(lib, open(a.o, "w"), indent=1); print(f"{len(lib)} levels -> {a.o}"); return 0
    lib = json.load(open(a.library))
    if a.cmd == "env":
        print(" ".join(f"{k}={v}" for k, v in sorted(env(C.load(a.config), lib, a.accurate).items()))); return 0
    if a.cmd == "solve":
        from .engine import rxm_root
        return solve(C.load(a.config), lib, a.out, a.tree or rxm_root(), [float(x) for x in a.t.split(",")],
                     [float(x) for x in a.l.split(",")], a.accurate)
    don = json.load(open(a.donors)) if a.donors else None
    for l in open(a.solve):
        if l.strip():
            p, fl = handoff(json.loads(l), lib, a.mixdir, a.tag, a.base, don, a.frontier); print(p, "|", fl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
