#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""eval196.py (#196: eval191.py + the norm base) TAG OUT --vlevel VL --template MIX --host HOSTCONF [--ckpt CKPT_DIR | --as-is] [--slice NAME=GOLDEN ...]
              [--card N]   /   eval191.py cmp OUT.npz ROWS GOLDEN
#191's served-path read (RULES191: the L-off arms, and torch vs served): eval181.py's path for the 2.5 incumbent's
served form. The checkpoint's scales applied to the virtual level (rxq.swj apply_level), packed with the virtual level's
per-tensor recipe (src replaced), and swapped into the incumbent's own hostd mix file (TEMPLATE: its 336 trunk lines ->
this pack; head / embedding donor lines and the `act ... a8g128` lines as they are) over the search's base (HOSTCONF's
eval.aliases); census with the search's census_cmd; scored with its score_cmd (radiance, bf16 KV, max-model-len 16384).
No --ckpt = the control through the same pack path; --as-is = the template mix itself (the search's packs: the
pack-path gate). Locks as hostd (conv + ram exclusive for pack / mix; card + ram shared for scores). Restartable; the
pack and the level copy are deleted once the mix exists, the mix once every slice is scored (--keep-mix keeps it).
N (#196): a checkpoint with norms.safetensors gets its served base patched (patchnorms196.py: a reflink copy of the
base with the 209 norm planes overwritten in place, read back), used as the mix's and the census's base; deleted after
the mix is written. cmp: torch rows (tscore191 npz) vs served rows on the same golden: per-row max |diff|, mean KL both ways."""
import argparse, json, os, shutil, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qad")); sys.path.insert(0, os.path.join(HERE, "..", "..", "pipeline"))
from healeval import Lock, sh, fill, doc_sums  # noqa: E402


def cmp(npz, rows, golden):
    z = np.load(npz); m = json.load(open(f"{golden}/kld.json")); gen = np.fromfile(f"{golden}/gen.u8", np.uint8).astype(bool)
    K = np.fromfile(rows, np.float32).reshape(-1, 4)[:, 0].astype(np.float64)
    srv = np.concatenate([K[d["row0"]:d["row0"] + d["tokens"] - 1 - d["score_from"]][gen[d["row0"]:d["row0"] + d["tokens"] - 1 - d["score_from"]]]
                          for d in m["docs"]])
    t = z["kl"]; assert len(t) == len(srv), (len(t), len(srv))
    d = np.abs(t - srv)
    r = {"rows": len(t), "torch": float(t.mean()), "served": float(srv.mean()), "rel": float(t.mean() / srv.mean() - 1),
         "max_abs_row": float(d.max()), "median_abs_row": float(np.median(d)), "mean_abs_row": float(d.mean())}
    print(json.dumps(r)); return r


def main():
    if sys.argv[1:2] == ["cmp"]:
        cmp(*sys.argv[2:5]); return
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("out")
    ap.add_argument("--vlevel", required=True); ap.add_argument("--template", required=True); ap.add_argument("--host", required=True)
    ap.add_argument("--base", default="V3-T4"); ap.add_argument("--ckpt"); ap.add_argument("--as-is", action="store_true")
    ap.add_argument("--slice", action="append", default=[]); ap.add_argument("--card", type=int, required=True)
    ap.add_argument("--keep-mix", action="store_true")
    ap.add_argument("--pack-home", default=None, help="RADIANCE_HOME for the pack (box2: the search's home-pack refuses codebook=mad at declare)")
    a = ap.parse_args()
    a.out, a.vlevel = os.path.abspath(a.out), os.path.abspath(a.vlevel)
    from rxq import mix as M, swj
    E = json.load(open(a.host))["eval"]
    if a.pack_home:
        import re
        E["pack_cmd"] = [re.sub(r"RADIANCE_HOME=\S+", f"RADIANCE_HOME={a.pack_home}", x) for x in E["pack_cmd"]]
    d = os.path.join(a.out, a.tag); os.makedirs(d, exist_ok=True); log = os.path.join(d, "eval.log")
    pk, mrad, mf = os.path.join(d, "trunk.rad"), os.path.join(d, "mix.rad"), os.path.join(d, "mix.txt")
    scored = all(os.path.exists(os.path.join(d, f"score-{s.split('=', 1)[0]}.json.rows")) for s in a.slice)
    tpl = open(a.template).read().splitlines()
    trunk = set(M.trunk_names())
    if a.as_is: lines = tpl
    else:
        assert sum(1 for l in tpl if l.split(" ")[0] in trunk) == len(trunk), "template: not one line per trunk weight"
        lines = [(f"{l.split(' ')[0]} {pk}" if l.split(" ")[0] in trunk else l) for l in tpl if l.strip()]
    base = E.get("aliases", {}).get(a.base, a.base)
    nb = None
    if a.ckpt and os.path.exists(os.path.join(a.ckpt, "norms.safetensors")):
        nb = os.path.join(d, "base-N.rad")
        if not os.path.exists(nb) and not scored and not os.path.exists(mrad):
            import subprocess
            rc = subprocess.call([sys.executable, os.path.join(HERE, "patchnorms196.py"), base, nb + ".part", os.path.join(a.ckpt, "norms.safetensors")], stdout=open(log, "a"), stderr=subprocess.STDOUT)
            if rc: raise SystemExit(f"patchnorms196 rc {rc} ({log})")
            os.replace(nb + ".part", nb)
        base = nb
    if not scored and not os.path.exists(mrad):
        if not a.as_is and not os.path.exists(pk):
            src = a.vlevel
            if a.ckpt:
                src = os.path.join(d, "level")
                swj.apply_level(a.vlevel, a.ckpt, src)                     # writes src/.done; skips a finished copy
            rec = os.path.join(d, "final.recipe")
            r = open(a.vlevel + ".recipe").read()
            open(rec, "w").write(r.replace(f"src={a.vlevel} ", f"src={os.path.abspath(src)} "))
            with Lock(E["conv_lock"]), Lock(E["ram_lock"]):
                if os.path.exists(pk + ".part"): os.remove(pk + ".part")
                rc = sh(fill(E["pack_cmd"], recipe=rec, out=pk + ".part"), log)
                if rc or not os.path.exists(pk + ".part"): raise SystemExit(f"pack rc {rc} ({log})")
                os.replace(pk + ".part", pk)
            if a.ckpt: shutil.rmtree(src, ignore_errors=True)
        open(mf, "w").write("\n".join(lines) + "\n")
        with Lock(E["conv_lock"]), Lock(E["ram_lock"]):
            if os.path.exists(mrad + ".part"): os.remove(mrad + ".part")
            rc = sh(fill(E["mix_cmd"], out=mrad + ".part", base=base, mixfile=mf), log)
            if rc or not os.path.exists(mrad + ".part"): raise SystemExit(f"mix rc {rc} ({log})")
            rc = sh(fill(E["census_cmd"], mix=mrad + ".part", mixfile=mf, base=base), os.path.join(d, "census.txt"))
            if rc: raise SystemExit(f"census FAILED ({d}/census.txt)")
            os.replace(mrad + ".part", mrad)
        if nb and os.path.exists(nb): os.remove(nb)      # the mix is written and censused: its copy of the patched base is no longer needed
        if os.path.exists(pk): os.remove(pk)
    res = {"tag": a.tag, "ckpt": a.ckpt, "as_is": a.as_is, "template": a.template, "slices": {}}
    for sl in a.slice:
        name, g = sl.split("=", 1); o = os.path.join(d, f"score-{name}")
        if not os.path.exists(o + ".json.rows"):
            with Lock(f"/data/q27/locks/card{a.card}.lock"), Lock(E["ram_lock"], False):
                rc = sh(fill(E["score_cmd"], mix=mrad, golden=g, out=o, card=a.card), log)
            if rc or not os.path.exists(o + ".json.rows"): raise SystemExit(f"score {name} rc {rc} ({log})")
        s, n = doc_sums(g, o + ".json.rows")
        res["slices"][name] = {"golden": g, "rows": o + ".json.rows", "kld": float(s.sum() / n.sum()), "positions": int(n.sum()), "docs": len(n)}
    json.dump(res, open(os.path.join(d, "result.json"), "w"), indent=1)
    if not a.keep_mix and os.path.exists(mrad) and len(res["slices"]) == len(a.slice): os.remove(mrad)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
