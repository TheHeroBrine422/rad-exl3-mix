#!/usr/bin/env python3
"""mksub.py SRC DST SUBSET -- a --hout dir holding only SUBSET's diagonal files (symlinks into SRC, a full per-linear
H_out dir such as c18/hdiag-snap1024-dy); the other linears get no file, so ourq.py quantizes them exactly as the base
arm (one-sided, same seed). SUBSET: fam:gdn_in | fam:attn_in | fam:mlp_in | fam:writers, or depth:A-B (layers A..B).
Hessian groups (q/k/v, GDN qkv/z, gate/up) are never split by these subsets. Writes DST/subset.txt (the keys) and
DST/.done; refuses an empty subset or a SRC without the whole 0.8B trunk (150 linears)."""
import os, re, sys

FAM = {"gdn_in": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
       "attn_in": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
       "mlp_in": ("mlp.gate_proj", "mlp.up_proj"),
       "writers": ("linear_attn.out_proj", "self_attn.o_proj", "mlp.down_proj")}
PAT = re.compile(r"model\.language_model\.layers\.(\d+)\.(.+)\.safetensors$")

def select(files, subset):
    kind, _, v = subset.partition(":")
    out = []
    for f in files:
        m = PAT.match(f)
        if not m: continue
        l, mod = int(m.group(1)), m.group(2)
        if kind == "fam": ok = mod in FAM[v]
        elif kind == "depth": lo, hi = map(int, v.split("-")); ok = lo <= l <= hi
        else: raise SystemExit(f"bad subset {subset}")
        if ok: out.append(f)
    return sorted(out)

if __name__ == "__main__":
    src, dst, subset = sys.argv[1:4]
    files = sorted(f for f in os.listdir(src) if PAT.match(f))
    assert len(files) == 150, f"{src}: {len(files)} linear files, expected the 0.8B's 150"
    sel = select(files, subset)
    assert sel, f"subset {subset} is empty"
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(dst):                         # rebuild from scratch: no stale links
        if f.endswith(".safetensors"): os.remove(os.path.join(dst, f))
    for f in sel: os.symlink(os.path.abspath(os.path.join(src, f)), os.path.join(dst, f))
    open(os.path.join(dst, "subset.txt"), "w").write("".join(f[:-len(".safetensors")] + "\n" for f in sel))
    open(os.path.join(dst, ".done"), "w").close()
    print(f"{subset}: {len(sel)} of {len(files)} linears -> {dst}")
