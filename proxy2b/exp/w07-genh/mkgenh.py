#!/usr/bin/env python3
"""mkgenh.py CALIB NG OUT W0[,W0..] -- gen-weighted input Hessians for ourq.py --hin: per group
    H(w0) = H_calib - (1 - w0) H_ng        rows = rows_calib - (1 - w0) rows_ng
with H_ng hgen.py's sum over the unscored positions of its documents (non-verify docs outside the eval goldens), i.e.
those positions re-weighted to w0 and every other calib position kept at 1 (the eval documents keep exactly base's
weights, so the comparison with base is not flattered by their own masks). fp64; written as OUT/w<w0>/<group>.safetensors
(challenge layout) for every group NG has. OUT/w<w0>/stats.jsonl: per group the share of tr(H_calib) removed and
the smallest diagonal entry relative to the mean (a starved channel shows as ~0)."""
import glob, json, os, struct, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file
CAL, NG, OUT = sys.argv[1:4]; W0 = [float(x) for x in sys.argv[4].split(",")]
def load(p):
    with open(p, "rb") as fh: n = struct.unpack("<Q", fh.read(8))[0]; meta = json.loads(fh.read(n))["__metadata__"]
    with safe_open(p, "pt") as f: up = f.get_tensor("hessian_upper").double()
    return up, int(meta["rows"]), meta
files = sorted(glob.glob(f"{NG}/*.input.safetensors")); assert files, NG
for w0 in W0:
    od = f"{OUT}/w{w0:g}"; os.makedirs(od, exist_ok=True); st = open(f"{od}/stats.jsonl", "w")
    for f in files:
        g = os.path.basename(f)[:-len(".safetensors")]
        uc, rc, meta = load(f"{CAL}/{g}.safetensors"); un, rn, mn = load(f)
        assert uc.shape == un.shape and int(meta["k"]) == int(mn["k"]), g
        up = uc - (1 - w0) * un; rows = round(rc - (1 - w0) * rn)
        k = int(meta["k"]); iu = torch.triu_indices(k, k); dg = iu[0] == iu[1]
        d, dc = up[dg], uc[dg]
        r = {"group": g, "w0": w0, "removed_tr": float(1 - d.sum() / dc.sum()), "rows": rows, "rows_calib": rc, "rows_ng": rn,
             "min_diag_rel": float(d.min() / d.mean()), "min_diag_rel_calib": float(dc.min() / dc.mean())}
        assert float(d.min()) > 0, (g, r)
        save_file({"hessian_upper": up.float().contiguous()}, f"{od}/{g}.safetensors.part",
                  metadata=dict(meta, rows=str(rows), source=f"mkgenh w0 {w0:g}: calib - (1 - w0) x {NG}"))
        os.replace(f"{od}/{g}.safetensors.part", f"{od}/{g}.safetensors")
        st.write(json.dumps(r) + "\n")
    st.close()
    rs = [json.loads(l) for l in open(f"{od}/stats.jsonl")]
    print(f"w0 {w0:g}: {len(rs)} groups, trace removed median {sorted(r['removed_tr'] for r in rs)[len(rs) // 2]:.3f} "
          f"(max {max(r['removed_tr'] for r in rs):.3f}), rows {rs[0]['rows']} of {rs[0]['rows_calib']}")
print("MKGENH DONE")
