#!/usr/bin/env python3
"""check_quant.py REF_LEVEL JOB_OUT [KEY ...] -- bit-identity of rxq's per-tensor quant jobs against an ourq.py level
made with the same settings: every tensor of every key (trellis, suh, svh, codebook marker) torch.equal, and the
proxy log's proxy_err / g_scale / sigma_reg / out_scales equal. Keys default to every t-<key> file in JOB_OUT.
The reference must hold each key's Hessian-group head (rxq.qwen35.missing_group_heads): an ourq --only run without it
draws the group's su under another seed and is not the ladder's tensor; such keys print "BADREF" and count as failed.
Prints one line per key and a verdict line "QUANT IDENT PASS|FAIL n/m"."""
import json, os, struct, sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rxq import qwen35  # noqa: E402

ref, out = sys.argv[1], sys.argv[2]
keys = sys.argv[3:] or sorted(f[2:-len(".safetensors")] for f in os.listdir(out) if f.startswith("t-") and f.endswith(".safetensors"))
wm = json.load(open(f"{ref}/model.safetensors.index.json"))["weight_map"]
rlog = {}
for l in open(f"{ref}/proxy.jsonl"):
    if l.strip(): r = json.loads(l); rlog[r["key"]] = r
olog = {}
for l in open(f"{out}/log.jsonl"):
    if l.strip(): r = json.loads(l); olog[r["key"]] = r
badref = qwen35.missing_group_heads({n.rsplit(".", 1)[0] for n in wm})
ok = 0
for k in keys:
    if k in badref:
        print(f"BADREF {k}: the reference lacks its group head {badref[k]} (ourq --only drew the group's su under the "
              f"first listed member's seed, not the head's; rerun the reference with the head in --only)"); continue
    with safe_open(f"{out}/t-{k}.safetensors", "pt") as f: mine = {n: f.get_tensor(n) for n in f.keys()}
    theirs = {}
    for n in mine:
        if n not in wm: theirs = None; break
        with safe_open(f"{ref}/{wm[n]}", "pt") as f: theirs[n] = f.get_tensor(n)
    refn = sorted(n for n in wm if n.startswith(k + "."))
    same_set = theirs is not None and sorted(mine) == refn
    diff = [n for n in mine if not (same_set and mine[n].dtype == theirs[n].dtype and torch.equal(mine[n], theirs[n]))]
    lg = [f for f in ("proxy_err", "g_scale", "sigma_reg", "out_scales", "fallback", "drift") if rlog.get(k, {}).get(f) != olog.get(k, {}).get(f)]
    good = same_set and not diff and not lg
    ok += good
    extra = ""
    if diff and theirs:
        t = diff[0]; a, b = mine[t], theirs[t]
        extra = f" first diff {t}: {(a != b).float().mean().item():.4f} of entries"
    print(f"{'SAME' if good else 'DIFF'} {k}: tensors {len(mine)} differ {len(diff)}; log fields differ {lg}{extra}"
          f"; proxy {olog.get(k, {}).get('proxy_err')} vs {rlog.get(k, {}).get('proxy_err')}")
print(f"QUANT IDENT {'PASS' if ok == len(keys) else 'FAIL'} {ok}/{len(keys)}")
