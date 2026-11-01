#!/usr/bin/env python3
"""ourq.py -K K -o OUTDIR [--layers A-B] [--head] [--device N] -- our own EXL3 quantization of Qwen3.8-27B's linears at a
uniform K (1..8, or fractional e.g. 2.25 / 5.5), with exllamav3's quantize_exl3 (LDLQ over the trellis, mul1 codebook, output scales always -- the
settings of turboderp's published exports) on the challenge's calibration Hessians (hessians/calib: every linear's
input over the whole 10.26M-token calib corpus, the same distribution as the private scoring set).

OUTDIR becomes an EXL3 export radiance's exl3 import reads (src=OUTDIR): config.json (quant_method exl3) +
model.safetensors.index.json + one shard per layer (q-LNN.safetensors, q-head.safetensors). Each shard is written
whole when its layer is done, so a run can be resumed: layers whose shard exists are skipped. proxy.jsonl beside
them holds exllamav3's own proxy error per tensor.
Every shard carries a code stamp in its metadata ("ourq_stamp": a hash of this file, exllamav3's quantize.py, the
--frac-ext sources and every result-affecting argument; STAMP_SRC lists the inputs in OUTDIR/stamp.json). A resume
refuses to mix stamps (OURQ_MIXED=1 overrides); shards without a stamp (written before 2026-11-01) are accepted with a
warning.

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
# refit_scales on the head runs with VRAM full: a kernel used there for the first time cannot load its code object
# (HSA_STATUS_ERROR_OUT_OF_RESOURCES) and the HIP runtime segfaults (K4.5 / K5.5 heads, 2026-11-01). Before it: hand
# the caching allocator's free blocks back and load every kernel the refit uses on a tiny problem. Output unchanged.
_orig_refit = _Q.refit_scales
def _warm_refit(dev_):
    t = torch.rand(4, 4, device=dev_) + torch.eye(4, device=dev_)
    r = torch.linalg.solve(t, torch.ones(4, device=dev_))
    r = torch.where(torch.isfinite(r) & (r > 0), r, torch.ones_like(r)); _ = (t.diagonal().mean(), (t * t).sum(0), t @ t.T)
    torch.cuda.synchronize(dev_)
def _refit(*args, **kw):
    torch.cuda.empty_cache(); _warm_refit(dev); torch.cuda.empty_cache()
    return _orig_refit(*args, **kw)
_Q.refit_scales = _refit

CKPT = RXM_CKPT
HESS = f"{RXM_WORK}/challenge/hessians/calib"
TEMPLATE = os.environ.get("RXM_TEMPLATE", "/data/workspace/references/td-SC_4.00bpw_H5/config.json")   # a turboderp config.json
L = "model.language_model.layers"

ap = argparse.ArgumentParser()
ap.add_argument("--frac-ext", metavar="SRC", help="quarter-step / K5-8 half-step rates: build+install v3/e3/exl3frac from "
                "the rxm-frac-quarter exllamav3_ext sources SRC (needs PYTORCH_ROCM_ARCH)")
ap.add_argument("--codebook", choices=("mul1", "i8", "i4"), default="mul1", help="search with the int8- or int4-"
                "rounded mul1 codebook (v3/e3/exl3cb, needs --frac-ext SRC for the sources): what V3's int kernels multiply")
ap.add_argument("-K", type=lambda v: int(v) if float(v).is_integer() else float(v), required=True,
                help="bitrate: integer 1-8, or fractional (K = KA + n/16, n even; needs the rxm-frac-quarter exllamav3 for n != 8)")
ap.add_argument("-o", required=True)
ap.add_argument("--layers", default="0-63")
ap.add_argument("--head", action="store_true", help="also lm_head")
ap.add_argument("--only-head", action="store_true")
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--sigma-reg", type=float, default=None, help="exllamav3's diagonal regularisation (default its own)")
ap.add_argument("--hout", default=None, help="dir of output-side Hessians (<key>.safetensors, 'hout', v3/yaqa/fold.py): "
                "two-sided LDLQ for the linears that have one")
ap.add_argument("--hin", default=None, help="dir of replacement input Hessians in the challenge layout "
                "(<name>.safetensors, hessian_upper; v3/yaqa/collect.py YAQA_HIN): used for the groups that have one")
ap.add_argument("--hout-diag", default=None, help="dir of output-side Hessians as for --hout, but only their diagonal is "
                "used: one-sided LDLQ with exllamav3's sensitivity output scales (t ~ (sigma^2/s)^(1/4), as convert.py "
                "--out_scales yaqa) for the linears that have one (output scales on auto instead of always)")
ap.add_argument("--save-wq", action="store_true", help="also write each linear's dequantized weight (out, in) f16 to "
                "OUTDIR/wq/<key>.safetensors, and per output row the error (w-q)ᵀH(w-q) and the reference wᵀHw (H: the "
                "raw input Hessian) to OUTDIR/wq/<key>.rowerr.npz (v3/yaqa/headslab.py)")
ap.add_argument("--head-perm", default=None, help="int64 .npy permutation of the head's vocab rows: lm_head is "
                "quantized in that row order (rows sorted by sensitivity: slabs = contiguous 128-row blocks); --save-wq "
                "outputs stay in the permuted order")
ap.add_argument("--sigma-reg-in", type=float, default=None, help="regularisation for the layer-input group only (attention "
                "q/k/v, GDN qkv/z; damptest: 0.005 beats 0.025 there, ~0 elsewhere)")
ap.add_argument("--sigma-in-sweep", default=None, help="for the layer-input group (attention q/k/v, GDN qkv/z): quantize at "
                "each of these sigma_reg values (e.g. 0.025,0.005,0.001) and keep, per tensor, the one with the lowest "
                "proxy error against the UNDAMPED Hessian (deterministic, no sampling noise)")
ap.add_argument("--gain", default=None, help="multipliers on exllamav3's chosen g_scale per linear, by key suffix, e.g. "
                "'o_proj=1.16,out_proj=1.16' (gaintest.py)")
ap.add_argument("--sigma-reg-out", type=float, default=None, help="regularisation of hout (default: --sigma-reg's)")
ap.add_argument("--out-scales-off", default=None, metavar="FILE", help="checkpoint keys (one per line, e.g. "
                "model.language_model.layers.40.mlp.up_proj) quantized WITHOUT output channel scales (default: always on); "
                "v3/e3/osr/mkoff.py writes it from upstream's test (osr.jsonl, ratio > 1.05)")
ap.add_argument("--only", default=None, metavar="FILE", help="quantize only these checkpoint keys (one per line); "
                "seeds stay those of the full run (by position in the layer); list whole Hessian groups (gate+up, q/k/v, ...) -- the first of a group prepares the shared H in place -- and a subset equals the full run")
ap.add_argument("--head-mean-row", action="store_true", help="lm_head: subtract the mean vocabulary row before "
                "quantizing (logits shift by one constant a position: softmax and KL unchanged). Every donor of a slab "
                "head must agree on this, or mixed rows stop sharing the shift")
ap.add_argument("--qtf2", action="store_true", help="fractional rates on v3/e3/exl3frac2 (qtf2: the V3 kernel pass; "
                "bit-identical on the old rates, plus 1/8 steps and sub-1 K); needs --frac-ext SRC")
a = ap.parse_args()
if a.frac_ext:
    _h = os.path.dirname(os.path.abspath(__file__))           # next to ourq.py (GPU box copy) or in the repo
    sys.path.insert(0, next(d for d in (os.path.join(_h, "exl3frac"), os.path.join(_h, "../v3/e3/exl3frac")) if os.path.isdir(d)))
    import exl3frac
    exl3frac.install(a.frac_ext)
if a.codebook != "mul1":
    assert a.frac_ext, "--codebook needs --frac-ext SRC (the fork's exllamav3_ext sources)"
    sys.path.insert(0, next(d for d in (os.path.join(_h, "exl3cb"), os.path.join(_h, "../v3/e3/exl3cb")) if os.path.isdir(d)))
    import exl3cb
    exl3cb.install(a.frac_ext, a.codebook)
if a.qtf2:
    assert a.frac_ext and a.codebook in ("mul1", "i8"), "--qtf2 needs --frac-ext SRC and the mul1 or i8 codebook"
    sys.path.insert(0, next(d for d in (os.path.join(_h, "exl3frac2"), os.path.join(_h, "../v3/e3/exl3frac2")) if os.path.isdir(d)))
    import exl3frac2
    exl3frac2.install(a.frac_ext, a.codebook)
_keys = lambda f: set(l.strip() for l in open(f) if l.strip() and not l.startswith("#"))
OFF = _keys(a.out_scales_off) if a.out_scales_off else set()
ONLY = _keys(a.only) if a.only else None
dev = torch.device("cuda", a.device)
_warm_refit(dev)                                   # load the refit kernels while VRAM is empty
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
    if a.hin and os.path.exists(f"{a.hin}/{name}.safetensors"): p = f"{a.hin}/{name}.safetensors"
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
        if ONLY is not None and key not in ONLY: continue
        t0 = time.time()
        w = ck(key + ".weight").to(dev).float().T.contiguous()     # (in_features, out_features)
        if a.head_perm and key == "lm_head":
            import numpy as np
            w = w[:, torch.from_numpy(np.load(a.head_perm)).to(dev)].contiguous()
        if a.head_mean_row and key == "lm_head":
            w -= w.mean(dim=1, keepdim=True)              # columns are vocabulary rows: drop their mean
        qa = {"seed": seed0 + j, "K": a.K, "devices": [a.device], "device_ratios": None,
              "apply_out_scales": True, "mul1": True, "debug_dir": os.path.join(a.o, "debug")}
        if key in OFF: qa["apply_out_scales"] = False
        if a.sigma_reg is not None: qa["sigma_reg"] = a.sigma_reg
        if a.sigma_reg_in is not None and hname.endswith(".layers." + hname.split(".layers.")[1].split(".")[0] + ".input"):
            qa["sigma_reg"] = a.sigma_reg_in
        GAIN[0] = next((float(m) for suf, m in GAINS.items() if key.endswith(suf)), 1.0)
        if a.sigma_reg_out is not None: qa["sigma_reg_out"] = a.sigma_reg_out
        hp = f"{a.hout}/{key}.safetensors" if a.hout else None
        if hp and os.path.exists(hp):
            with safe_open(hp, "pt", device="cpu") as f: qa["H_out"] = f.get_tensor("hout").to(dev)
        hd = f"{a.hout_diag}/{key}.safetensors" if a.hout_diag else None
        if hd and os.path.exists(hd):
            with safe_open(hd, "pt", device="cpu") as f:
                qa["out_hessian_diag"] = (f.get_tensor("hout_diag") if "hout_diag" in f.keys() else f.get_tensor("hout").diagonal()).float().to(dev)
            qa["apply_out_scales"] = None             # the diagonal path runs only with output scales on auto
        is_in = hname.endswith(".layers." + hname.split(".layers.")[1].split(".")[0] + ".input") if ".layers." in hname else False
        if a.sigma_in_sweep and is_in:
            Hu = hessian(hname)["H"]; ref = float((w * (Hu @ w)).sum()); best = None
            for sg in [float(v) for v in a.sigma_in_sweep.split(",")]:
                qs = dict(qa); qs["sigma_reg"] = sg
                wq_, perr_, t_ = quantize_exl3(w.clone(), hessian(hname), qs, True)
                E = w - wq_.float(); rel = float((E * (Hu @ E)).sum()) / ref; del E
                if best is None or rel < best[0]: best = (rel, sg, wq_, perr_, t_)
                else: del wq_, t_
            _, qa["sigma_reg"], wq, perr, t = best; qa["sweep_rel"] = best[0]; del Hu
        else:
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
                              "hout_diag_scales": bool(qa.get("out_scales_hessian")), "sigma_reg": qa.get("sigma_reg"), "sweep_rel": qa.get("sweep_rel"), "gain": GAIN[0], "codebook": a.codebook,
                              "out_scales": qa.get("apply_out_scales"), "head_mean_row": bool(a.head_mean_row and key == "lm_head"),
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

GAINS = dict(kv.split("=") for kv in a.gain.split(",")) if a.gain else {}
GAIN = [1.0]
if GAINS:
    _orig_gss = _Q.g_scale_gss
    def _gss(*x, **k):
        g_, m_ = _orig_gss(*x, **k); return g_ * GAIN[0], m_
    _Q.g_scale_gss = _gss
# ---- code stamp (mining merge B13): never resume a run with different code or settings
import hashlib, glob
def _stamp():
    h = hashlib.sha256(); src = []
    files = [os.path.abspath(__file__), _Q.__file__]
    if a.frac_ext:
        files += sorted(f for e in ("cu", "cuh", "cpp", "h", "py") for f in glob.glob(f"{a.frac_ext}/**/*.{e}", recursive=True))
        files += sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exl3frac", "*.py")))
        files += sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exl3cb", "*")))
        files += sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exl3frac2", "*.*")))
    if a.out_scales_off: files.append(os.path.abspath(a.out_scales_off))      # its contents, not just the path
    if a.only: files.append(os.path.abspath(a.only))
    for f in files:
        if os.path.isfile(f): h.update(f.encode()); h.update(open(f, "rb").read()); src.append(f)
    keep = {k: v for k, v in vars(a).items() if k not in ("o", "layers", "head", "only_head", "device", "save_wq")}
    h.update(json.dumps(keep, sort_keys=True).encode())
    return h.hexdigest()[:16], src, keep
STAMP, STAMP_SRC, STAMP_ARGS = _stamp()
def _shard_stamp(f):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; return (json.loads(fh.read(n)).get("__metadata__") or {}).get("ourq_stamp")
def resume_ok(shard):
    """True: the shard is done and was made by this code / settings. Exits on a mismatch."""
    if not os.path.exists(shard): return False
    st = _shard_stamp(shard)
    if st is None: print(f"warning: {os.path.basename(shard)} has no code stamp (pre-2026-11-01): resumed as is", flush=True)
    elif st != STAMP and not os.environ.get("OURQ_MIXED"):
        sys.exit(f"{shard}: stamp {st} != this run's {STAMP} (code or settings changed): refusing to resume; "
                 f"new OUTDIR, or OURQ_MIXED=1 to override")
    return True
_sj = os.path.join(a.o, "stamp.json")
_prev = json.load(open(_sj)) if os.path.exists(_sj) else []
if not _prev or _prev[-1]["stamp"] != STAMP:
    json.dump(_prev + [{"stamp": STAMP, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "args": STAMP_ARGS,
                        "sources": STAMP_SRC}], open(_sj, "w"), indent=1)
print(f"code stamp {STAMP}", flush=True)

log = open(os.path.join(a.o, "proxy.jsonl"), "a")
lo, hi = map(int, a.layers.split("-"))
if not a.only_head:
    for l in range(lo, hi + 1):
        shard = os.path.join(a.o, f"q-L{l:02d}.safetensors")
        if resume_ok(shard): continue
        t0 = time.time(); out = {}
        if ONLY is not None and not any(k in ONLY for _, ks in layer_plan(l) for k in ks): continue
        for gi, (hn, keys) in enumerate(layer_plan(l)):
            if ONLY is not None and not any(k in ONLY for k in keys): continue
            out.update(quantize_group(hn, keys, l * 16 + gi * 4, log))
        save_file(out, shard + ".part", metadata={"ourq_stamp": STAMP}); os.replace(shard + ".part", shard)
        # drop the checkpoint shard handles (their mapped pages piled up to ~23 GB over a run: K8 OOM, 2026-10-30)
        del out; _open.clear(); import gc; gc.collect(); torch.cuda.empty_cache()
        rss = next((l.split()[1] for l in open("/proc/self/status") if l.startswith("RssAnon")), "?")
        print(f"layer {l}: {time.time() - t0:.0f}s  anon rss {int(rss) >> 10 if rss != chr(63) else rss} MiB", flush=True)
        write_index()
if a.head or a.only_head:
    shard = os.path.join(a.o, "q-head.safetensors")
    if not resume_ok(shard):
        out = quantize_group("lm_head.input", ["lm_head"], 9999, log)
        save_file(out, shard + ".part", metadata={"ourq_stamp": STAMP}); os.replace(shard + ".part", shard)
write_index()
print("done", flush=True)
