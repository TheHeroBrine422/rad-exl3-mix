#!/usr/bin/env python3
"""mkmanifest197.py ASSIGN.json OUT.mix [--head PATH|base] [--emb PATH|base] [--act a8g128] [--final-root DIR]
   [--base PATH] -- #197 (w185): a serve manifest for one arm, in the vload form of radiance (w162, #185 (B)):
     base BASE                                  the base container (its own head / embedding stay when no donor line)
     blk.N.W final FINAL_ROOT/LEVEL/final codebook=mad   each trunk linear from its level's final/ (no pack, no mix file)
     output.weight PATH / token_embd.weight PATH       head / embedding donors (plain or trellis .rad), or nothing = base's
     act blk.N.W a8g128                          activation mode on every trunk linear (the search's act)
   The line order follows pipeline/rxrun/hostd.py (trunk, donors, act). ASSIGN is an assign dict {weight: option}; the
   head / embedding donor for an option is the menu's donor (--head/--emb override: 'base' = no line, or a path).
   Writes OUT.mix (the file radiance reads as --model). Nothing else."""
import argparse, json, os

MENU_HEAD_DIR = "/data/q27/w158/heads"                 # hd-u<K>.rad (box2)
MENU_EMB_DIR = "/data/q27/emb/trellis"                 # emb-t-i8-K<K>.rad (box2)

ap = argparse.ArgumentParser()
ap.add_argument("assign"); ap.add_argument("out")
ap.add_argument("--head", default=None); ap.add_argument("--emb", default=None)
ap.add_argument("--act", default="a8g128")
ap.add_argument("--final-root", default="/data/q27/ladder-src-mad")
ap.add_argument("--base", default="/data/q27/srch183/base/base-T4.rad")
a = ap.parse_args()
asg = json.load(open(a.assign))
trunk = sorted((w for w in asg if w.startswith("blk.")), key=lambda w: (int(w.split(".")[1]), w))
assert len(trunk) == 336, len(trunk)

def donor_for(w, opt):
    if w == "output.weight":
        return f"{MENU_HEAD_DIR}/hd-{opt}.rad"
    assert w == "token_embd.weight" and opt.startswith("t") and not opt.endswith("m"), opt
    return f"{MENU_EMB_DIR}/emb-t-i8-K{opt[1:]}.rad"

lines = [f"base {a.base}"]
for w in trunk:
    lines.append(f"{w} final {a.final_root}/{asg[w]}/final codebook=mad")
for w, flag in (("output.weight", a.head), ("token_embd.weight", a.emb)):
    if flag == "base":
        continue
    path = flag if flag else donor_for(w, asg[w])
    lines.append(f"{w} {path}")
if a.act:
    lines += [f"act {w} {a.act}" for w in trunk]
tmp = a.out + ".part"
open(tmp, "w").write("\n".join(lines) + "\n")
os.replace(tmp, a.out)
print(json.dumps({"out": a.out, "lines": len(lines), "trunk": len(trunk),
                  "head": "base" if a.head == "base" else (a.head or donor_for("output.weight", asg["output.weight"])),
                  "emb": "base" if a.emb == "base" else (a.emb or donor_for("token_embd.weight", asg["token_embd.weight"]))}))
