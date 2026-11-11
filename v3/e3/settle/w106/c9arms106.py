#!/usr/bin/env python3
"""c9arms106.py QDIR SWJWQ OUTROOT TAG K --model M --hess H -- #140 (../w106.md): the C9 arms of one (K, seed) quant on
fixed codes. QDIR: the rxq quant output (t-<key>.safetensors, c9-<key>.safetensors sidecars); SWJWQ: the swj'd wq dir
(150 f16 weights). Per writer (o_proj / out_proj; down has alpha 0 in every arm) the job's codes are decoded as shipped
(rxq.deq, f16 suh / svh) and c comes from the shipped rxq.post.c9_rows with each arm's alpha / e_clip:
  off  c = 1;  cur  v3.toml's alpha, e_clip 0.5;  ak  alpha * clip(K - 1, 0, 1), e_clip 0.5;  e3  v3.toml's alpha, e_clip 0.3
Writes OUTROOT/TAG-<arm>/ (links to SWJWQ for the other tensors, the writers as Q * c rows, f16) unless the arm's c
equals an earlier arm's on every writer (then OUTROOT/TAG-<arm>.same holds that arm's name and no dir is written).
OUTROOT/TAG-c9.json: G1 (cur's c vs the job's sidecar), per arm and family the c / e statistics. Exit 1 on a G1 fail."""
import argparse, json, os, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file
from rxq import post, deq, quant, qwen35, config as C

ap = argparse.ArgumentParser()
ap.add_argument("qdir"); ap.add_argument("swjwq"); ap.add_argument("outroot"); ap.add_argument("tag"); ap.add_argument("K", type=float)
ap.add_argument("--model", required=True); ap.add_argument("--hess", required=True); ap.add_argument("--codebook", default="i8")
a = ap.parse_args()
cfg = C.load(None); AL = cfg["post"]["c9"]["alpha"]
sk = min(max(a.K - 1, 0.0), 1.0)
ARMS = {"off": ({"attn_o": 0.0, "gdn_out": 0.0}, 0.5), "cur": ({f: AL[f] for f in ("attn_o", "gdn_out")}, 0.5),
        "ak": ({f: AL[f] * sk for f in ("attn_o", "gdn_out")}, 0.5), "e3": ({f: AL[f] for f in ("attn_o", "gdn_out")}, 0.3)}
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
ck = quant.Ckpt(a.model)
keys = sorted(f[:-12] for f in os.listdir(a.swjwq) if f.endswith(".safetensors"))
assert len(keys) == 150, len(keys)
wkeys = [k for k in keys if post.c9_family(k) in ("attn_o", "gdn_out")]
lut = torch.from_numpy(deq.lut(a.codebook))
cs = {arm: {} for arm in ARMS}; stats = {arm: {} for arm in ARMS}; g1 = []; qs = {}
for key in wkeys:
    fam = post.c9_family(key)
    with safe_open(f"{a.qdir}/t-{key}.safetensors", "pt", device="cpu") as f: t = {k: f.get_tensor(k) for k in f.keys()}
    tr = t[f"{key}.trellis"]
    Q = deq.weight(tr, t[f"{key}.suh"], t[f"{key}.svh"], tr.shape[2] / 16, lut).T.double()
    W = ck.get(key + ".weight").double()
    _, _, _, hname, _ = qwen35.locate(key)
    M = post.hessian(f"{a.hess}/{hname}.safetensors", "cpu")
    for arm, (al, ecl) in ARMS.items():
        c, rec = post.c9_rows(Q, W, M, al[fam], ecl)
        cs[arm][key] = c; stats[arm][key] = dict(rec, fam=fam, alpha=al[fam], e_clip=ecl)
    side = load_file(f"{a.qdir}/c9-{key}.safetensors")["c"].double()
    g1.append(float(((cs["cur"][key] - side).abs() / side.abs()).max()))
    qs[key] = Q.half()
    del W, M
g1max = max(g1)
out = {"tag": a.tag, "K": a.K, "g1_max_rel": g1max, "g1_pass": g1max <= 1e-6, "arms": {}, "same": {}}
done = []
for arm in ARMS:
    same = next((b for b in done if all(torch.equal(cs[arm][k], cs[b][k]) for k in wkeys)), None)
    fams = {}
    for fam in ("attn_o", "gdn_out"):
        R = [stats[arm][k] for k in wkeys if stats[arm][k]["fam"] == fam]
        n = sum(r["rows"] for r in R)
        cc = torch.cat([cs[arm][k] for k in wkeys if stats[arm][k]["fam"] == fam])
        fams[fam] = {"alpha": R[0]["alpha"], "e_clip": R[0]["e_clip"], "tensors": len(R), "rows": n,
                     "c_mean": float(cc.mean()), "c_max": float(cc.max()), "c_min": float(cc.min()),
                     "c_q50": float(cc.float().quantile(0.5)), "at_clip": sum(r["clipped"] for r in R) / n,
                     "e_mean": sum(r["e_mean"] * r["rows"] for r in R) / n, "e_max": max(r["e_max"] for r in R),
                     "per_tensor": {k.split(".")[3]: {x: stats[arm][k][x] for x in ("c_mean", "c_max", "e_mean", "e_max", "clipped")}
                                    for k in wkeys if stats[arm][k]["fam"] == fam}}
    out["arms"][arm] = fams
    if same:
        out["same"][arm] = same
        with open(f"{a.outroot}/{a.tag}-{arm}.same", "w") as fh: fh.write(same + "\n")
    elif out["g1_pass"]:
        d = f"{a.outroot}/{a.tag}-{arm}"
        if not os.path.exists(f"{d}/.ok"):
            os.makedirs(d + ".part", exist_ok=True)
            for k in keys:
                dst = f"{d}.part/{k}.safetensors"
                if k in cs[arm]:
                    save_file({"weight": (qs[k].double() * cs[arm][k][:, None]).half().contiguous()}, dst)
                elif not os.path.lexists(dst):
                    os.symlink(os.path.realpath(f"{a.swjwq}/{k}.safetensors"), dst)
            open(f"{d}.part/.ok", "w").close()
            if os.path.exists(d): os.system(f"rm -rf '{d}'")
            os.replace(d + ".part", d)
    done.append(arm)
with open(f"{a.outroot}/{a.tag}-c9.json.part", "w") as fh: json.dump(out, fh, indent=1)
os.replace(f"{a.outroot}/{a.tag}-c9.json.part", f"{a.outroot}/{a.tag}-c9.json")
print(json.dumps({"tag": a.tag, "g1_max_rel": g1max, "same": out["same"],
                  "c_mean": {arm: {f: round(v["c_mean"], 4) for f, v in fa.items()} for arm, fa in out["arms"].items()}}))
sys.exit(0 if out["g1_pass"] else 1)
