#!/usr/bin/env python3
"""c9e27.py LEVEL OUT --alpha attn_o=A,gdn_out=B[,down=C] [--model CKPT] [--hess DIR] -- w75 #112-02 (../w75.md): C9's
per-row svh multipliers (#85; w64's c9w.py arithmetic) for the 27B writers of an ourq level, in swjapply.py's scales
layout, so that `swjapply.py LEVEL OUT DST` writes the level with svh *= c (codes, suh, every other tensor unchanged).
  Q   the writer as the level holds it (deq.py's decode: the plugin's exl3i arithmetic, f16 suh / svh)
  t   W, the bf16 checkpoint's weight;  M  the published calib H_in of the writer's group (raw: sum x x^T / rows)
  e_n = (q_n - t_n) M (q_n - t_n)^T / t_n M t_n^T, clipped to [0, 0.5];  c_n = (1 - e_n)^(-A/2)   (f64)
Writers: attn_o = self_attn.o_proj (16), gdn_out = linear_attn.out_proj (48), down = mlp.down_proj (64); families with
A = 0 get no scales file (swjapply leaves them byte for byte). OUT/scales/<key>.safetensors: "c" (out,) f32, "r" (in,)
ones; OUT/c9.jsonl: per writer e_n mean / max, c mean / sd / min / max / quantiles, rows c > 1.5 / > 2;
OUT/.done last. Restartable per writer. Env DEV (default cuda); C9E27_ONLY=key,key (tests only)."""
import argparse, json, os, struct, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deq

ap = argparse.ArgumentParser()
ap.add_argument("level"); ap.add_argument("out")
ap.add_argument("--alpha", required=True)
ap.add_argument("--model", default="/data/workspace/models/qwen38-27b-bf16")
ap.add_argument("--hess", default="/root/challenge/hessians/calib")
a = ap.parse_args()
AL = {k: float(v) for k, v in (x.split("=") for x in a.alpha.split(","))}
assert set(AL) <= {"attn_o", "gdn_out", "down"}, AL
dev = os.environ.get("DEV", "cuda")
if os.path.exists(f"{a.out}/.done"): print(f"{a.out} done"); sys.exit(0)
os.makedirs(f"{a.out}/scales", exist_ok=True)
Lp = "model.language_model.layers"
keys = []
for l in range(64):
    full = l % 4 == 3
    if AL.get("attn_o", 0) and full: keys.append((f"{Lp}.{l}.self_attn.o_proj", "attn_o", f"{Lp}.{l}.self_attn.o_proj.input"))
    if AL.get("gdn_out", 0) and not full: keys.append((f"{Lp}.{l}.linear_attn.out_proj", "gdn_out", f"{Lp}.{l}.linear_attn.out_proj.input"))
    if AL.get("down", 0): keys.append((f"{Lp}.{l}.mlp.down_proj", "down", f"{Lp}.{l}.mlp.down_proj.input"))
if os.environ.get("C9E27_ONLY"): keys = [k for k in keys if k[0] in os.environ["C9E27_ONLY"].split(",")]   # tests only
lv = deq.Level(a.level, dev=dev)
ix = json.load(open(f"{a.model}/model.safetensors.index.json"))["weight_map"]


def hessian(p):
    with open(p, "rb") as fh: n = struct.unpack("<Q", fh.read(8))[0]; meta = json.loads(fh.read(n))["__metadata__"]
    k, rows = int(meta["k"]), int(meta["rows"])
    with safe_open(p, "pt", device="cpu") as f: up = f.get_tensor("hessian_upper")
    H = torch.zeros(k, k, dtype=torch.float64, device=dev); iu = torch.triu_indices(k, k, device=dev)
    H[iu[0], iu[1]] = up.to(dev).double(); del up
    return (H + H.triu(1).T) / rows


done = {json.loads(l)["key"] for l in open(f"{a.out}/c9.jsonl")} if os.path.exists(f"{a.out}/c9.jsonl") else set()
with open(f"{a.out}/c9.jsonl", "a") as log:
    for key, fam, grp in keys:
        if key in done and os.path.exists(f"{a.out}/scales/{key}.safetensors"): continue
        al = AL[fam]
        Q = lv.get(key).double()                                                   # (out, in)
        with safe_open(f"{a.model}/{ix[key + '.weight']}", "pt", device="cpu") as f: W = f.get_tensor(key + ".weight").to(dev).double()
        assert Q.shape == W.shape, (key, Q.shape, W.shape)
        M = hessian(f"{a.hess}/{grp}.safetensors")
        E = Q - W
        e = ((E @ M) * E).sum(1) / ((W @ M) * W).sum(1).clamp(min=1e-300)
        ncl = int((e > 0.5).sum()); nneg = int((e < 0).sum()); e = e.clamp(min=0.0, max=0.5)
        c = (1 - e).pow(-al / 2)
        save_file({"c": c.float().cpu().contiguous(), "r": torch.ones(W.shape[1], dtype=torch.float32)},
                  f"{a.out}/scales/{key}.safetensors.part")
        os.replace(f"{a.out}/scales/{key}.safetensors.part", f"{a.out}/scales/{key}.safetensors")
        q = torch.quantile(c.float().cpu(), torch.tensor([0.01, 0.5, 0.99])).tolist()
        log.write(json.dumps({"key": key, "fam": fam, "layer": int(key.split(".")[3]), "alpha": al, "rows": int(c.numel()),
                              "e_mean": float(e.mean()), "e_max": float(e.max()), "clipped": ncl, "neg": nneg,
                              "c_mean": float(c.mean()), "c_sd": float(c.std()), "c_min": float(c.min()), "c_max": float(c.max()),
                              "c_q01_q50_q99": q, "n_gt15": int((c > 1.5).sum()), "n_gt2": int((c > 2).sum())}) + "\n")
        log.flush()
        print(f"{key}: e mean {float(e.mean()):.4f} max {float(e.max()):.3f}, c mean {float(c.mean()):.4f} max {float(c.max()):.3f}", flush=True)
        del Q, W, M, E
        if dev != "cpu": torch.cuda.empty_cache()
open(f"{a.out}/.done", "w").close()
print(f"C9E27 DONE {len(keys)} writers -> {a.out}")
