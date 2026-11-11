# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""cmp109.py A B [--common] [NAME=NAME2 ...] -- (--common: only the scale files and swiglu layers both have, e.g. a dev
run's first layers against a production pass; swjfit.json not compared) #144's identity judge between two swjfit OUT dirs: for every source (A's names,
or the given pairs) every scales/<key>.safetensors tensor torch.equal (r, c), every swiglu.jsonl record equal (exact
floats), swjfit.json equal except tool / created / kernels / prof. Prints one line per source + "IDENTICAL" / "DIFFER"."""
import json, os, sys

import torch
from safetensors import safe_open

A, B = sys.argv[1:3]; COMMON = "--common" in sys.argv
pairs = [p.split("=") for p in sys.argv[3:] if p != "--common"] or [[n, n] for n in sorted(os.listdir(A)) if os.path.isdir(f"{A}/{n}")]
bad = 0
for na, nb in pairs:
    a, b = f"{A}/{na}", f"{B}/{nb}"
    ka = sorted(os.listdir(f"{a}/scales")); kb = sorted(os.listdir(f"{b}/scales"))
    if COMMON: ka = kb = sorted(set(ka) & set(kb))
    diff = [] if ka == kb else [f"key sets differ ({len(ka)} vs {len(kb)})"]
    for k in ka:
        if k not in kb: continue
        with safe_open(f"{a}/scales/{k}", "pt") as fa, safe_open(f"{b}/scales/{k}", "pt") as fb:
            if sorted(fa.keys()) != sorted(fb.keys()): diff.append(f"{k}: tensor names"); continue
            for t in fa.keys():
                x, y = fa.get_tensor(t), fb.get_tensor(t)
                if not (x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)):
                    n = int((x != y).sum()) if x.shape == y.shape else -1
                    diff.append(f"{k}:{t} {n} of {x.numel()} differ, max |d| {float((x - y).abs().max()) if n > 0 else 0:.3g}")
    ja = [json.loads(l) for l in open(f"{a}/swiglu.jsonl")]; jb = [json.loads(l) for l in open(f"{b}/swiglu.jsonl")]
    if COMMON:
        lay = {r["layer"] for r in ja} & {r["layer"] for r in jb}
        ja = [{k: v for k, v in r.items() if not k.endswith("_sub") and k != "sub_docs"} for r in ja if r["layer"] in lay]
        jb = [r for r in jb if r["layer"] in lay]
    if ja != jb: diff.append(f"swiglu.jsonl: {sum(x != y for x, y in zip(ja, jb)) + abs(len(ja) - len(jb))} records differ")
    drop = ("tool", "created", "kernels", "prof", "src", "split", "w109")
    sa = {k: v for k, v in json.load(open(f"{a}/swjfit.json")).items() if k not in drop}
    sb = {k: v for k, v in json.load(open(f"{b}/swjfit.json")).items() if k not in drop}
    if sa != sb and not COMMON: diff.append(f"swjfit.json: {sorted(k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))}")
    bad += bool(diff)
    print(f"{na} vs {nb}: {len(ka)} scale files, {len(ja)} layers: " + ("identical" if not diff else "DIFFER " + "; ".join(diff[:6])))
print("IDENTICAL" if not bad else "DIFFER")
sys.exit(1 if bad else 0)
