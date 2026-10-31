#!/usr/bin/env python3
"""ourq.py -K K -o OUTDIR [--layers A-B] [--head] [--device N] -- our own EXL3 quantization of Qwen3.8-27B's linears at a
uniform K (1..8), with exllamav3's quantize_exl3 (LDLQ over the trellis, mul1 codebook, output scales always -- the
settings of turboderp's published exports) on the challenge's calibration Hessians (hessians/calib: every linear's
input over the whole 10.26M-token calib corpus, the same distribution as the private scoring set).

OUTDIR becomes an EXL3 export radiance's exl3 import reads (src=OUTDIR): config.json (quant_method exl3) +
model.safetensors.index.json + one shard per layer (q-LNN.safetensors, q-head.safetensors). Each shard is written
whole when its layer is done, so a run can be resumed: layers whose shard exists are skipped. proxy.jsonl beside
them holds exllamav3's own proxy error per tensor.

Run in $RXM_VENV (torch ROCm + exllamav3 built for gfx1201)."""
import argparse, json, os, struct, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
import exllamav3.modules.quant.exl3_lib.quantize as _Q
# paths (README: Paths)
RXM_WORK = os.environ.get("RXM_WORK", "/root")
RXM_CKPT = os.environ.get("RXM_CKPT", "/data/workspace/models/qwen38-27b-bf16")
# block_trace only feeds a logged proxy that the scale refit overwrites (quantize.py ~1843): 55% of the time, no effect
# on the output. OURQ_TRACE=1 restores it.
if not os.environ.get("OURQ_TRACE"): _Q.block_trace = lambda A, B, block_size=1024: 1.0

CKPT = RXM_CKPT
HESS = f"{RXM_WORK}/challenge/hessians/calib"
TEMPLATE = os.environ.get("RXM_TEMPLATE", "/data/workspace/references/td-SC_4.00bpw_H5/config.json")   # a turboderp config.json
L = "model.language_model.layers"

ap = argparse.ArgumentParser()
ap.add_argument("-K", type=int, required=True)
ap.add_argument("-o", required=True)
ap.add_argument("--layers", default="0-63")
ap.add_argument("--head", action="store_true", help="also lm_head")
ap.add_argument("--only-head", action="store_true")
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--sigma-reg", type=float, default=None, help="exllamav3's diagonal regularisation (default its own)")
ap.add_argument("--hout", default=None, help="dir of output-side Hessians (<key>.safetensors, 'hout', v3/yaqa/fold.py): "
                "two-sided LDLQ for the linears that have one")
ap.add_argument("--save-wq", action="store_true", help="also write each linear's dequantized weight (out, in) f16 to "
                "OUTDIR/wq/<key>.safetensors, and per output row the error (w-q)ᵀH(w-q) and the reference wᵀHw (H: the "
                "raw input Hessian) to OUTDIR/wq/<key>.rowerr.npz (v3/yaqa/headslab.py)")
ap.add_argument("--head-perm", default=None, help="int64 .npy permutation of the head's vocab rows: lm_head is "
                "quantized in that row order (rows sorted by sensitivity: slabs = contiguous 128-row blocks); --save-wq "
                "outputs stay in the permuted order")
ap.add_argument("--sigma-reg-out", type=float, default=None, help="regularisation of hout (default: --sigma-reg's)")
a = ap.parse_args()
dev = torch.device("cuda", a.device)
os.makedirs(a.o, exist_ok=True)

# ------------------------------------------------------------------ the bf16 checkpoint
idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
_open = {}
def ck(name):
    f = idx[name]
    if f not in _open: _open[f] = safe_open(f"{CKPT}/{f}", "pt", device="cpu")
    return _open[f].get_tensor(name)

# ------------------------------------------------------------------ a Hessian (packed upper triangle, sum over rows)
def hessian(name):
    p = f"{HESS}/{name}.safetensors"
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
    meta = hdr["__metadata__"]; k = int(meta["k"]); rows = int(meta["rows"])
    with safe_open(p, "pt", device="cpu") as f:
        up = f.get_tensor("hessian_upper").to(dev)
    H = torch.zeros(k, k, dtype=torch.float32, device=dev)
    iu = torch.triu_indices(k, k, device=dev)
    H[iu[0], iu[1]] = up; del up
    H = H + H.triu(1).T
    return {"H": H, "first_key": name, "count": rows, "finalized": False, "num_total": rows,
            "inf_nan": torch.zeros(2, dtype=torch.long, device=dev), "device": dev}

# ------------------------------------------------------------------ one layer's linears, grouped by their Hessian
def layer_plan(l):
    p = f"{L}.{l}"
    full = l % 4 == 3
    plan = [(f"{p}.input", [f"{p}.self_attn.q_proj", f"{p}.self_attn.k_proj", f"{p}.self_attn.v_proj"] if full else
                           [f"{p}.linear_attn.in_proj_qkv", f"{p}.linear_attn.in_proj_z"]),
            (f"{p}.self_attn.o_proj.input" if full else f"{p}.linear_attn.out_proj.input",
             [f"{p}.self_attn.o_proj" if full else f"{p}.linear_attn.out_proj"]),
            (f"{p}.post_attention.input", [f"{p}.mlp.gate_proj", f"{p}.mlp.up_proj"]),
            (f"{p}.mlp.down_proj.input", [f"{p}.mlp.down_proj"])]
    return plan

def quantize_group(hname, keys, seed0, log):
    out = {}
    Hd = hessian(hname)
    for j, key in enumerate(keys):
        t0 = time.time()
        w = ck(key + ".weight").to(dev).float().T.contiguous()     # (in_features, out_features)
        if a.head_perm and key == "lm_head":
            import numpy as np
            w = w[:, torch.from_numpy(np.load(a.head_perm)).to(dev)].contiguous()
        qa = {"seed": seed0 + j, "K": a.K, "devices": [a.device], "device_ratios": None,
              "apply_out_scales": True, "mul1": True, "debug_dir": os.path.join(a.o, "debug")}
        if a.sigma_reg is not None: qa["sigma_reg"] = a.sigma_reg
        if a.sigma_reg_out is not None: qa["sigma_reg_out"] = a.sigma_reg_out
        hp = f"{a.hout}/{key}.safetensors" if a.hout else None
        if hp and os.path.exists(hp):
            with safe_open(hp, "pt", device="cpu") as f: qa["H_out"] = f.get_tensor("hout").to(dev)
        wq, perr, t = quantize_exl3(w, Hd, qa, a.save_wq)
        if a.save_wq:
            import numpy as np
            os.makedirs(os.path.join(a.o, "wq"), exist_ok=True)
            H = hessian(hname)["H"]                     # fresh: quantize_exl3 may have regularised Hd in place
            err, ref = [], []
            for c in range(0, w.shape[1], 8192):        # per output row, in column chunks
                E = w[:, c:c + 8192] - wq[:, c:c + 8192]
                err.append(((H @ E) * E).sum(0).double().cpu())
                W = w[:, c:c + 8192]; ref.append(((H @ W) * W).sum(0).double().cpu())
                del E, W
            np.savez(os.path.join(a.o, "wq", f"{key}.rowerr.npz"), err=torch.cat(err).numpy(), ref=torch.cat(ref).numpy())
            del H
            save_file({"weight": wq.T.half().contiguous().cpu()}, os.path.join(a.o, "wq", f"{key}.safetensors"))
        del w, wq
        for k2, v in t.items(): out[f"{key}.{k2}"] = v.contiguous().cpu()
        two = qa.pop("H_out", None) is not None
        log.write(json.dumps({"key": key, "K": a.K, "proxy_err": float(perr), "fallback": bool(qa.get("q_fallback")), "hout": two,
                              "s": round(time.time() - t0, 1)}) + "\n"); log.flush()
        print(f"  {key:60s} K{a.K} proxy {float(perr):.3e} {time.time() - t0:.1f}s", flush=True)
    del Hd; torch.cuda.empty_cache()
    return out

def write_index():
    wm = {}
    for f in sorted(os.listdir(a.o)):
        if not (f.startswith("q-") and f.endswith(".safetensors")): continue
        with open(os.path.join(a.o, f), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
        for k in hdr:
            if k != "__metadata__": wm[k] = f
    json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(a.o, "model.safetensors.index.json"), "w"), indent=1)
    c = json.load(open(TEMPLATE))
    c["quantization_config"].update({"bits": float(a.K), "head_bits": a.K, "calibration": {"source": "challenge calib Hessians"}})
    json.dump(c, open(os.path.join(a.o, "config.json"), "w"), indent=1)

log = open(os.path.join(a.o, "proxy.jsonl"), "a")
lo, hi = map(int, a.layers.split("-"))
if not a.only_head:
    for l in range(lo, hi + 1):
        shard = os.path.join(a.o, f"q-L{l:02d}.safetensors")
        if os.path.exists(shard): continue
        t0 = time.time(); out = {}
        for gi, (hn, keys) in enumerate(layer_plan(l)):
            out.update(quantize_group(hn, keys, l * 16 + gi * 4, log))
        save_file(out, shard + ".part"); os.replace(shard + ".part", shard)
        # drop the checkpoint shard handles (their mapped pages piled up to ~23 GB over a run: K8 OOM, 2026-10-30)
        del out; _open.clear(); import gc; gc.collect(); torch.cuda.empty_cache()
        rss = next((l.split()[1] for l in open("/proc/self/status") if l.startswith("RssAnon")), "?")
        print(f"layer {l}: {time.time() - t0:.0f}s  anon rss {int(rss) >> 10 if rss != chr(63) else rss} MiB", flush=True)
        write_index()
if a.head or a.only_head:
    shard = os.path.join(a.o, "q-head.safetensors")
    if not os.path.exists(shard):
        out = quantize_group("lm_head.input", ["lm_head"], 9999, log)
        save_file(out, shard + ".part"); os.replace(shard + ".part", shard)
write_index()
print("done", flush=True)
