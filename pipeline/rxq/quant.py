# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The per-tensor quant stage: one linear at one level (K, codebook), from its bf16 weight, its group's input Hessian
and optionally its H_out diagonal / the asym writers' G (hin) and F (wpre). Port of rad-exl3-mix quantize/ourq.py
quantize_group() (2026-11-06, stamp lines included), one tensor at a time:

- the same quant_args dict (seed, K, devices, apply_out_scales, mul1, sigma_reg / _in / _out, scaled_hessian, H_out,
  tile CD args), the same weight layout (in, out) fp32 on the device, the same Hessian unpacking;
- a tensor that is not the first of its Hessian group first finalizes the group Hessian under the group's first seed
  (the su draw ourq's group run makes for tensor 0), so a per-tensor job equals the group run (tests/check_quant.py);
- outputs the same tensors (<key>.suh / .svh / .trellis / codebook marker) and the same proxy-log fields; --save-wq's
  wq / rowerr files as ourq writes them (optional).
Not ported (not in the V3 settings): --sigma-in-sweep, per-linear Sketch-B --hin files, --hout-diag (output scales from
H_out), --head-perm / --head-mean-row (head jobs: a later stage), the multi-level memo (each job is one level)."""
import json, os, struct, time

import torch
from safetensors import safe_open

from . import qwen35, engine, refit


class Ckpt:
    """bf16 checkpoint reader by tensor name (index or single file); handles kept open, as ourq's ck()"""
    def __init__(self, d):
        self.d = d
        ix = f"{d}/model.safetensors.index.json"
        self.map = json.load(open(ix))["weight_map"] if os.path.exists(ix) else None
        self.h = {}

    def get(self, name):
        f = self.map[name] if self.map else "model.safetensors"
        if f not in self.h: self.h[f] = safe_open(f"{self.d}/{f}", "pt", device="cpu")
        return self.h[f].get_tensor(name)

    def close(self): self.h.clear()


def read_meta(p):
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; return json.loads(fh.read(n))["__metadata__"]


def hessian(path, name, dev, on=None):
    """ourq's hessian(): the challenge layout (packed upper triangle "hessian_upper", metadata k / rows = a sum over rows
    tokens) as exllamav3's H_data dict, H = the full symmetric fp32 sum on the device"""
    meta = read_meta(path); k = int(meta["k"]); rows = int(meta["rows"])
    with safe_open(path, "pt", device="cpu") as f: up = f.get_tensor("hessian_upper")
    d_ = on or dev
    up = up.to(d_)
    H = torch.zeros(k, k, dtype=torch.float32, device=d_)
    iu = torch.triu_indices(k, k, device=d_)
    H[iu[0], iu[1]] = up; del up
    H = H + H.triu(1).T
    return {"H": H, "first_key": name, "count": rows, "finalized": False, "num_total": rows,
            "inf_nan": torch.zeros(2, dtype=torch.long, device=dev), "device": dev}


def _devid(dev):
    """quant_args' device entry: the CUDA index (ourq's), or "cpu" (the stand-in kernels, rxq/cpukern.py)"""
    return "cpu" if dev.type == "cpu" else (dev.index if dev.index is not None else 0)


def quant_args(s, key, seed, device, outdir, hname):
    """ourq's qa for one tensor (same keys, same order of updates)"""
    qa = {"seed": seed, "K": s["K"], "devices": [device], "device_ratios": None,
          "apply_out_scales": True, "mul1": True, "debug_dir": os.path.join(outdir, "debug")}
    if s["scaled_hessian"]: qa["scaled_hessian"] = True
    if s.get("sigma_reg") is not None: qa["sigma_reg"] = s["sigma_reg"]
    if s.get("sigma_reg_in") is not None and qwen35.is_input_group(hname): qa["sigma_reg"] = s["sigma_reg_in"]
    if s.get("sigma_reg_out") is not None: qa["sigma_reg_out"] = s["sigma_reg_out"]
    t = engine.tilecd()
    if t is not None and s.get("tile_cd"): qa.update(t.args(key, s["K"], outdir))   # s: the family and the level's K (k_max)
    return qa


def quantize_tensor(Q, ck, key, s, hess_dir, device, outdir, hin_dir=None, wpre_dir=None, hout_dir=None,
                    save_wq=False, prof=None, diag_out=None):
    """-> (tensors {<key>.<name>: cpu tensor}, log record). s = config.quant_settings(...). The refit runs as
    s["refit"] says (rxq.refit); its diagnostics' scalars go to the record's "diag", their tensors into diag_out (a dict)"""
    from contextlib import nullcontext
    st = prof.stage if prof else (lambda _: nullcontext())
    t0 = time.time()
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    l, gi, j, hname, gkeys = qwen35.locate(key)
    seed, seed_first = qwen35.seed(key, s.get("seed_offset", 0))
    hp = f"{hin_dir}/{hname}.safetensors" if hin_dir and os.path.exists(f"{hin_dir}/{hname}.safetensors") else f"{hess_dir}/{hname}.safetensors"
    with st("hessian_load"):
        Hd = hessian(hp, hname, dev)
    engine.set_gain(Q, s.get("gain", 1.0))
    if j > 0:                                   # the group's su draw, as tensor 0 makes it in ourq's group run
        with st("group_finalize"):
            qa0 = quant_args(s, gkeys[0], seed_first, _devid(dev), outdir, hname)
            torch.manual_seed(qa0["seed"])
            Q.finalize_capture_H(Hd, qa0, False)
    with st("weight_load"):
        w = ck.get(key + ".weight").to(dev).float()
        if wpre_dir:                            # ourq --wpre: the target W F (asym writers), F^T from mkdenoise
            pf = f"{wpre_dir}/{hname}.safetensors"
            if not os.path.exists(pf): raise SystemExit(f"wpre {wpre_dir}: no transform for {hname} ({key})")
            with safe_open(pf, "pt", device="cpu") as f: Ft = f.get_tensor("Ft").to(dev).float()
            assert Ft.shape == (w.shape[1], w.shape[1]), (key, tuple(Ft.shape), tuple(w.shape))
            w = w @ Ft.T; del Ft
        w = w.T.contiguous()                    # (in_features, out_features)
    w_orig = w.detach().cpu().clone() if save_wq else None   # quantize_exl3 regularizes w in place (ourq-rowerr.patch, w46 / w90)
    qa = quant_args(s, key, seed, _devid(dev), outdir, hname)
    hraw = None                                 # the raw H_out diagonal (refit "hout" weights, the diagnostics' f2)
    if hout_dir:                                # ourq skips a key without a file; a job refuses (the bn dirs hold every linear)
        if not os.path.exists(f"{hout_dir}/{key}.safetensors"): raise SystemExit(f"H_out {hout_dir}: no file for {key}")
        with st("hout_load"), safe_open(f"{hout_dir}/{key}.safetensors", "pt", device="cpu") as f:
            if s.get("hout_blocks"):            # #115: the 1-D diagonal selects the fork's block-diagonal path (patch 0008)
                if "hout_diag" not in f.keys(): raise SystemExit(f"H_out {hout_dir}/{key}: ldlq = blocks needs a diagonal (hout_diag)")
                if not hasattr(Q, "ldlq_2hess_blocks"): raise SystemExit("inputs.hout.ldlq = blocks: the fork lacks patch 0008")
                qa["H_out"] = f.get_tensor("hout_diag").float(); qa["hout_schedule"] = s["hout_blocks"]
            else:
                qa["H_out"] = (f.get_tensor("hout") if "hout" in f.keys() else torch.diag(f.get_tensor("hout_diag").float()))
            hraw = f.get_tensor("hout_diag").float() if "hout_diag" in f.keys() else qa["H_out"].diagonal().float()
    refit.begin(key, s.get("refit", True), hraw, s.get("sigma_reg_out") or 0.01)
    try:
        with st("other"):
            wq, perr, t = Q.quantize_exl3(w, Hd, qa, save_wq)
    finally:
        dg = refit.end()
    rowerr = None
    if save_wq:
        with st("save_wq"):
            w = w_orig.to(wq.device); del w_orig      # the original weight, not the regularized one
            try: H = hessian(hp, hname, dev)["H"]
            except torch.OutOfMemoryError:
                if not (os.environ.get("RXQ_REFIT_CPU_FALLBACK") or os.environ.get("OURQ_REFIT_CPU_FALLBACK")): raise
                torch.cuda.empty_cache(); H = hessian(hp, hname, dev, on="cpu")["H"]; w, wq = w.cpu().float(), wq.cpu().float()
            err, ref = [], []
            for c in range(0, w.shape[1], 8192):
                E = w[:, c:c + 8192] - wq[:, c:c + 8192]
                err.append(((H @ E) * E).sum(0).double().cpu())
                W = w[:, c:c + 8192]; ref.append(((H @ W) * W).sum(0).double().cpu())
                del E, W
            rowerr = {"err": torch.cat(err).numpy(), "ref": torch.cat(ref).numpy()}
            wq = wq.T.half().contiguous().cpu(); del H
    else: wq = None
    del w
    out = {f"{key}.{k2}": v.contiguous().cpu() for k2, v in t.items()}
    two = qa.pop("H_out", None) is not None
    rec = {"key": key, "K": s["K"], "proxy_err": float(perr), "fallback": bool(qa.get("q_fallback")), "hout": two,
           "hin_key": False, "hout_diag_scales": bool(qa.get("out_scales_hessian")), "sigma_reg": qa.get("sigma_reg"),
           "sweep_rel": None, "sweep_all": None,
           "g_scale": None if qa.get("g_scale") is None else float(qa["g_scale"]), "drift": Q.ldlq_drift(s["K"]),
           "scaled_hessian": bool(s["scaled_hessian"]), "gain": s.get("gain", 1.0), "codebook": s["codebook"],
           "out_scales": qa.get("apply_out_scales"), "head_mean_row": False, "s": round(time.time() - t0, 1)}
    if s.get("hout_blocks"): rec["hout_blocks"] = s["hout_blocks"]
    if "proxy_err_2s" in qa: rec["proxy_err_2s"] = qa["proxy_err_2s"]   # two-sided, quantization domain, before the refit (fork 0008)
    if qa.get("block_order"): rec["block_order"] = True
    if dg: rec["diag"] = _diag_record(dg, s, qa, outdir, key)
    if diag_out is not None: diag_out.update(dg)
    del Hd, qa
    if dev.type == "cuda": torch.cuda.empty_cache()
    return out, rec, wq, rowerr


def _diag_record(dg, s, qa, outdir, key):
    """the log record's "diag" (#139): trajectory, the proxy check, overload counters, tile CD's quantization-domain rel"""
    r = {"refit": s.get("refit", True), "g_scale": None if qa.get("g_scale") is None else float(qa["g_scale"])}
    for k in ("traj", "check", "overload", "arms", "d_off", "damping"):
        if k in dg: r[k] = dg[k]
    lp = qa.get("tile_cd_log")
    if lp and os.path.exists(lp):                # tile CD's record of this tensor: rel after LDLQ and each sweep
        for l in open(lp):
            try: x = json.loads(l)
            except ValueError: continue
            if x.get("key") == key: r.setdefault("traj", {})["tilecd_q"] = x.get("rel")
    return r
