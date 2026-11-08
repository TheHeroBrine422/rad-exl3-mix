#!/usr/bin/env python3
"""pw82.py ARM LEVEL SCALES HOUT OUTDIR [LAYERS] -- #118 2(a) (../w82.md): one-sided and Σdy²-weighted proxy of the swj'd
vs the plain gate / up of an ourq level, on the CPU (no GPU: CUDA_VISIBLE_DEVICES / ROCR hidden by the caller).

Per layer and part: W (bf16 checkpoint, MODEL env), W_hat (exl3deq.Level, the level's own codebook), the swj row factor
k_r = f16(svh_r c_r) / svh_r (swjapply.py's stored form), H = calib post_attention.input (HESS env, undamped), h = the
Σdy² diagonal (HOUT/<key>.safetensors). Row sums (f32 GEMMs, f64 sums) with E = W - W_hat: ee = E H E^T, eq = E H W_hat^T, qq = W_hat H W_hat^T;
then ||w - k w_hat||_H^2 = ee + 2 (1 - k) eq + (1 - k)^2 qq and ||w||_H^2 = ee + 2 eq + qq (exact algebra).
e1 = sum(err) / sum(ref), e2 = sum(h err) / sum(h ref) for plain (k = 1), swj (k), and the per-row least-squares
refit (k* = 1 + eq / qq, the proxy's own optimum). One JSON per layer: OUTDIR/<ARM>/L<NN>.json (written whole: rerun
skips finished layers). Peak RAM ~2.5 GB; one thread (OMP_NUM_THREADS=1 by the caller)."""
import json, os, sys, time
import torch
from safetensors import safe_open

_h = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _h)
import exl3deq                                                     # copied beside this file (with exl3dec.py)
torch.set_grad_enabled(False)
MODEL = os.environ.get("MODEL", "/data/workspace/models/qwen38-27b-bf16")
HESS = os.environ.get("HESS", "/data/challenge/hessians/calib")
P = "model.language_model.layers."


def load_h(layer):
    with safe_open(f"{HESS}/{P}{layer}.post_attention.input.safetensors", "pt") as f:
        k = int(f.metadata()["k"]); up = f.get_tensor("hessian_upper").float()
    H = torch.zeros(k, k)
    o = 0
    for i in range(k):                                             # packed upper triangle, row-major (triu_indices order)
        H[i, i:] = up[o:o + k - i]; o += k - i
    return H + H.T - torch.diag(H.diagonal())


def main():
    arm, level, scales, hout, outd = sys.argv[1:6]
    layers = range(64) if len(sys.argv) < 7 else range(int(sys.argv[6].split("-")[0]), int(sys.argv[6].split("-")[1]) + 1)
    os.makedirs(f"{outd}/{arm}", exist_ok=True)
    lv = exl3deq.Level(level, dev="cpu")
    idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    for L in layers:
        out = f"{outd}/{arm}/L{L:02d}.json"
        if os.path.exists(out): continue
        t0 = time.time(); H = load_h(L); res = {"arm": arm, "layer": L, "level": level}
        for part in ("gate_proj", "up_proj"):
            key = f"{P}{L}.mlp.{part}"
            with safe_open(f"{MODEL}/{idx[key + '.weight']}", "pt") as f: W = f.get_tensor(key + ".weight").float()
            Wq = lv.get(key).float()
            svh = lv.raw(key)["svh"].float()
            with safe_open(f"{scales}/scales/{key}.safetensors", "pt") as f: c = f.get_tensor("c").float()
            k = ((svh * c).half().float() / svh).double()
            with safe_open(f"{hout}/{key}.safetensors", "pt") as f: h = f.get_tensor("hout_diag").double()
            ee = torch.empty(W.shape[0], dtype=torch.float64); eq = torch.empty_like(ee); qq = torch.empty_like(ee)
            for a in range(0, W.shape[0], 2048):
                E = W[a:a + 2048] - Wq[a:a + 2048]; Q = Wq[a:a + 2048]
                EH = (E @ H).double(); QH = (Q @ H).double(); E = E.double(); Q = Q.double()   # f32 GEMMs, f64 sums
                ee[a:a + 2048] = (EH * E).sum(1); eq[a:a + 2048] = (EH * Q).sum(1); qq[a:a + 2048] = (QH * Q).sum(1)
            ref = ee + 2 * eq + qq
            ks = 1 + eq / qq.clamp(min=1e-300)
            arms = {"plain": torch.ones_like(k), "swj": k, "ls": ks}
            r = {}
            for n, kk in arms.items():
                err = ee + 2 * (1 - kk) * eq + (1 - kk) ** 2 * qq
                r[n] = {"e1": float(err.sum() / ref.sum()), "e2": float((h * err).sum() / (h * ref).sum())}
            r["k_mean"] = float(k.mean()); r["k_sd"] = float(k.std()); r["kls_mean"] = float(ks.mean())
            r["kls_sd"] = float(ks.std()); r["corr_k_kls"] = float(torch.corrcoef(torch.stack([k, ks]))[0, 1])
            res[part] = r
            del W, Wq, ee, eq, qq
        res["sec"] = round(time.time() - t0, 1)
        with open(out + ".part", "w") as f: json.dump(res, f)
        os.replace(out + ".part", out)
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
