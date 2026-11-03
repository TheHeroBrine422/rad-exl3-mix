#!/usr/bin/env python3
"""a9mk.py BASE SRC OUT -- an ourq.py --hout dir: every BASE file (<checkpoint key>.safetensors) linked, except the MLP
gate_proj / up_proj, which are linked from SRC instead ("none": left out, so ourq.py quantizes them one-sided). Checks
the result has a file for every other linear of BASE and, unless none, all 2 x layers gate / up files."""
import glob, os, sys
base, src, out = map(os.path.abspath, sys.argv[1:4]) if sys.argv[2] != "none" else (os.path.abspath(sys.argv[1]), None, os.path.abspath(sys.argv[3]))
gu = lambda f: f.endswith((".mlp.gate_proj.safetensors", ".mlp.up_proj.safetensors"))
os.makedirs(out, exist_ok=True)
for f in glob.glob(f"{out}/*.safetensors"): os.remove(f)
bf = sorted(glob.glob(f"{base}/model.language_model.layers.*.safetensors"))
n_other = n_gu = 0
for f in bf:
    b = os.path.basename(f)
    if not gu(b): os.symlink(f, f"{out}/{b}"); n_other += 1
    elif src:
        s = f"{src}/{b}"; assert os.path.exists(s), f"missing {s}"
        os.symlink(s, f"{out}/{b}"); n_gu += 1
n_base_gu = sum(gu(os.path.basename(f)) for f in bf)
assert n_other == len(bf) - n_base_gu and n_other > 0 and (src is None or n_gu == n_base_gu), (n_other, n_gu, n_base_gu)
print(f"{out}: {n_other} linears from {base}, {n_gu} gate/up from {src}")
