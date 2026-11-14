#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""eval181.py TAG VLEVEL OUT --host HOSTCONF --served SERVED.json [--scales RUN_OR_CKPT] [--slice NAME=GOLDEN ...]
              [--emb base] [--card N --card-lock L]   /   eval181.py cmp OUT TAG_A TAG_B [RESULT]
#181's judgement on the served path: a mix's virtual level (vlevel.py) with a checkpoint's scales applied (rxq.swj
apply_level; none = the control), packed with the virtual level's per-tensor recipe (src replaced), swapped into the
mix's served form and scored with radiance. SERVED.json = {"mix": the incumbent's hostd .mix (donor lines), "base": the
radmix base, "head": output.weight donor, "emb": token_embd.weight donor (bf16), "act": "a8g128"}: the mix file is the
.mix's non-trunk lines + every trunk weight from this pack + "act <w> a8g128" on every trunk weight + the bf16
embedding (--emb base: the base's embedding instead, for the identity gate against rows scored that way). Census
(store/census.py) before any score. Locks as healeval.py (conv + ram exclusive for pack / mix; card + ram shared for
scores). Restartable; the trunk pack is deleted once the mix exists (--keep-pack keeps it). cmp = healeval.cmp."""
import argparse, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qad")); sys.path.insert(0, os.path.join(HERE, "..", "..", "pipeline"))
from healeval import Lock, sh, fill, doc_sums, cmp  # noqa: E402


def main():
    if sys.argv[1:2] == ["cmp"]:
        cmp(*sys.argv[2:6]); return
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("vlevel"); ap.add_argument("out")
    ap.add_argument("--host", required=True); ap.add_argument("--served", required=True); ap.add_argument("--scales")
    ap.add_argument("--slice", action="append", default=[]); ap.add_argument("--emb", default="bf16", choices=("bf16", "base"))
    ap.add_argument("--card", type=int, default=None); ap.add_argument("--card-lock", default=None)
    ap.add_argument("--census", default=os.path.join(HERE, "..", "..", "store", "census.py"))
    ap.add_argument("--keep-pack", action="store_true"); ap.add_argument("--result", default="result.json")
    a = ap.parse_args()
    a.out, a.vlevel = os.path.abspath(a.out), os.path.abspath(a.vlevel)          # the host's pack_cmd cds elsewhere
    a.scales = os.path.abspath(a.scales) if a.scales else None
    from rxq import mix as M, swj
    H = json.load(open(a.host)); E = H["eval"]; S = json.load(open(a.served))
    d = os.path.join(a.out, a.tag); os.makedirs(d, exist_ok=True); log = os.path.join(d, "eval.log")
    vl = a.vlevel.rstrip("/")
    src = vl
    built = os.path.exists(os.path.join(d, "mix.rad")) or os.path.exists(os.path.join(d, "trunk.rad"))
    if a.scales:
        src = os.path.join(d, "level")
        if not built: swj.apply_level(vl, a.scales, src)      # a finished pack / mix needs no level copy
    rec = os.path.join(d, "final.recipe")
    if not os.path.exists(rec):
        r = open(vl + ".recipe").read(); olds = set(re.findall(r"src=(\S+)", r)); assert len(olds) == 1, olds
        open(rec + ".part", "w").write(r.replace(f"src={olds.pop()} ", f"src={os.path.abspath(src)} ")); os.replace(rec + ".part", rec)
    pk, mrad, mf = os.path.join(d, "trunk.rad"), os.path.join(d, "mix.rad"), os.path.join(d, "mix.txt")
    if not os.path.exists(mrad) and not os.path.exists(pk):
        with Lock(E["conv_lock"]), Lock(E["ram_lock"]):
            if os.path.exists(pk + ".part"): os.remove(pk + ".part")
            rc = sh(fill(E["pack_cmd"], recipe=rec, out=pk + ".part", reuse="", model=E.get("model", "")), log)
            if rc or not os.path.exists(pk + ".part"): raise SystemExit(f"pack rc {rc} ({log})")
            os.replace(pk + ".part", pk)
    trunk = M.trunk_names()
    other = [l for l in open(S["mix"]).read().splitlines() if l.strip() and not l.startswith("blk.") and not l.startswith("act ")
             and not l.startswith("token_embd.weight") and not l.startswith("output.weight")]
    lines = [f"{w} {pk}" for w in trunk] + other + [f"output.weight {S['head']}"]
    if a.emb == "bf16": lines.append(f"token_embd.weight {S['emb']}")
    lines += [f"act {w} {S.get('act', 'a8g128')}" for w in trunk]
    open(mf, "w").write("\n".join(lines) + "\n")
    base = E.get("aliases", {}).get(S["base"], S["base"])
    cen = os.path.join(d, "census.txt")
    if not os.path.exists(mrad):
        with Lock(E["conv_lock"]), Lock(E["ram_lock"]):
            if os.path.exists(mrad + ".part"): os.remove(mrad + ".part")
            rc = sh(fill(E["mix_cmd"], out=mrad + ".part", base=base, mixfile=mf), log)
            if rc or not os.path.exists(mrad + ".part"): raise SystemExit(f"mix rc {rc} ({log})")
            ex = ["--expect-act", S.get("act", "a8g128"), "--expect-codebook", "mul1i8", "--max-version", "3"] + \
                 (["--expect-emb", "plain - bf16"] if a.emb == "bf16" else [])
            rc = sh([sys.executable, "-I", a.census, mrad + ".part", "--mix", mf, "--base", base] + ex, cen)
            if rc: raise SystemExit(f"census FAILED ({cen})")
            os.replace(mrad + ".part", mrad)
        if not a.keep_pack and os.path.exists(pk): os.remove(pk)
    card = a.card if a.card is not None else E.get("card", 0); card_lock = a.card_lock or E["card_lock"]
    res = {"tag": a.tag, "vlevel": vl, "scales": a.scales, "served": S, "emb": a.emb, "bytes": os.path.getsize(mrad), "slices": {}}
    for sl in a.slice:
        name, g = sl.split("=", 1); o = os.path.join(d, f"score-{name}")
        if not os.path.exists(o + ".json.rows"):
            with Lock(card_lock), Lock(E["ram_lock"], False):
                rc = sh(fill(E["score_cmd"], mix=mrad, golden=g, out=o, card=card), log)
            if rc or not os.path.exists(o + ".json.rows"): raise SystemExit(f"score {name} rc {rc} ({log})")
        s, n = doc_sums(g, o + ".json.rows")
        res["slices"][name] = {"golden": g, "rows": o + ".json.rows", "kld": float(s.sum() / n.sum()), "positions": int(n.sum()), "docs": len(n)}
    json.dump(res, open(os.path.join(d, a.result), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "served"}))


if __name__ == "__main__":
    main()
