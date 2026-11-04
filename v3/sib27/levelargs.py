#!/usr/bin/env python3
"""levelargs.py LEVEL -- the ourq.py arguments a level was made with, from its stamp.json (the last entry's "args",
every result-affecting argument; ourq.py _stamp), printed as one shell line, so a partial requant (--only) of that
level runs with exactly its settings (and, with the same seeds, reproduces its tensors: the no-op gate).
Refuses levels whose args this driver cannot extend (--hin, --wpre or --only already set; an unknown argument, e.g. a
newer ourq option: add it here first). Device, output dir, layers and head are not result-affecting and not printed."""
import json, shlex, sys

d = sys.argv[1]
st = json.load(open(f"{d}/stamp.json"))
if len({json.dumps(e["args"], sort_keys=True) for e in st}) > 1:
    print(f"warning: {d}/stamp.json has {len(st)} entries with different args; using the last", file=sys.stderr)
A = dict(st[-1]["args"])
for k in ("hin", "wpre", "only"):
    if A.get(k): sys.exit(f"{d}: made with --{k} {A[k]}: not a base this driver extends")
FLAG = {"K": "-K", "frac_ext": "--frac-ext", "codebook": "--codebook", "seed_offset": "--seed-offset", "sigma_reg": "--sigma-reg",
        "hout": "--hout", "hout_diag": "--hout-diag", "sigma_reg_in": "--sigma-reg-in", "sigma_in_sweep": "--sigma-in-sweep",
        "gain": "--gain", "sigma_reg_out": "--sigma-reg-out", "out_scales_off": "--out-scales-off", "head_perm": "--head-perm"}
BOOL = {"scaled_hessian": "--scaled-hessian", "qtf2": "--qtf2", "head_mean_row": "--head-mean-row"}
DEFAULT = {"seed_offset": 0, "codebook": "mul1"}
unknown = set(A) - set(FLAG) - set(BOOL) - {"hin", "wpre", "only"}
if unknown: sys.exit(f"{d}: unknown ourq args {sorted(unknown)}: teach levelargs.py first")
out = []
for k, f in FLAG.items():
    v = A.get(k)
    if v is None or (k in DEFAULT and v == DEFAULT[k]): continue
    out += [f, str(v)]
out += [f for k, f in BOOL.items() if A.get(k)]
print(shlex.join(out))
