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
# kill-safe writes (w41, 2026-11-06; the 4070 and the RDNA4 boxes are preempted at random): every output is written as                     # +io
# X.part, fsynced and renamed; a layer's wq files are staged and renamed only after its shard and the index, so a                          # +io
# wq/<key>.safetensors that exists belongs to a finished layer (the callers' "150 wq files" check can trust it). Output                    # +io
# bytes unchanged. Lines tagged +io are left out of the code stamp and #io-was: lines put back (_io_canon): same stamps.                   # +io
def _io_fsync(p):                                                                                                                          # +io
    fd = os.open(p, os.O_RDONLY); os.fsync(fd); os.close(fd)                                                                               # +io
def _io_put(write, path):          # write(path + ".part"), fsync, rename into place                                                       # +io
    write(path + ".part"); _io_fsync(path + ".part"); os.replace(path + ".part", path); _io_fsync(os.path.dirname(os.path.abspath(path)))  # +io
def _io_json(obj, path, **kw):                                                                                                             # +io
    def w(p):                                                                                                                              # +io
        with open(p, "w") as fh: json.dump(obj, fh, **kw)                                                                                  # +io
    _io_put(w, path)                                                                                                                       # +io
_IO_STAGED = []                    # wq files written as .part, renamed by _io_promote once their layer's shard is in place                # +io
def _io_stage(write, path):                                                                                                                # +io
    write(path + ".part"); _io_fsync(path + ".part"); _IO_STAGED.append(path)                                                              # +io
def _io_promote():                                                                                                                         # +io
    for p in _IO_STAGED: os.replace(p + ".part", p)                                                                                        # +io
    if _IO_STAGED: _io_fsync(os.path.dirname(_IO_STAGED[0]))                                                                               # +io
    _IO_STAGED.clear()                                                                                                                     # +io
def _io_leftover(prefix):          # a shard already done: rename the wq parts a kill left between it and _io_promote                      # +io
    d = os.path.join(a.o, "wq")                                                                                                            # +io
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:                                                                            # +io
        if f.startswith(prefix) and f.endswith(".safetensors.part"): os.replace(os.path.join(d, f), os.path.join(d, f[:-5]))               # +io
def _io_canon(f):                  # what the code stamp hashes of f: this file without the +io lines, #io-was: lines restored             # +io
    b = open(f, "rb").read()                                                                                                               # +io
    if os.path.abspath(f) != os.path.abspath(__file__): return b                                                                           # +io
    return b"".join(x[8:] if x.startswith(b"#io-was:") else x for x in b.splitlines(True) if not x.rstrip().endswith(b"# +io"))            # +io
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
    try:
        return _orig_refit(*args, **kw)
    except torch.OutOfMemoryError:
        # small cards (the 4070's ~8 GB beside a desktop: 27B gate_up's Q Q^T): the same refit on the CPU, fp32
        if not os.environ.get("OURQ_REFIT_CPU_FALLBACK"): raise
        torch.cuda.empty_cache(); print("refit_scales: OOM on the card, refit on the CPU", flush=True)
        mv = lambda x, d: x.to(d) if torch.is_tensor(x) else x
        r = _orig_refit(*[mv(x, "cpu") for x in args], **{k: mv(v, "cpu") for k, v in kw.items()})
        return tuple(mv(x, dev) for x in r) if isinstance(r, tuple) else mv(r, dev)
_Q.refit_scales = _refit
# LDLQ drift (the expected inflation of the tiles LDLQ feeds the search; the global scale search pre-inflates its samples
# by it): exllamav3 looks it up by exact K in {1, 1.5, 2, 2.5, 3} and gives every other K 1.0, so the 1/8 and quarter
# levels below K4 got none (trunk review 2026-11-02). Log-linear interpolation between the table points, 1.0 from K4,
# 1.08 below K1; the table points themselves are unchanged (identity gates hold there). OURQ_DRIFT_EXACT=1: old lookup.
if not os.environ.get("OURQ_DRIFT_EXACT"):
    import math as _m
    _DT = sorted({**_Q.LDLQ_DRIFT, 4: 1.0}.items())
    def _drift(K):
        K = float(K)
        if K in dict(_DT): return dict(_DT)[K]
        if K <= _DT[0][0]: return _DT[0][1]
        if K >= _DT[-1][0]: return 1.0
        (k0, d0), (k1, d1) = next((p, q) for p, q in zip(_DT, _DT[1:]) if p[0] < K < q[0])
        return _m.exp(_m.log(d0) + (_m.log(d1) - _m.log(d0)) * (K - k0) / (k1 - k0))
    _Q.ldlq_drift = _drift

CKPT = RXM_CKPT
HESS = f"{RXM_WORK}/challenge/hessians/calib"
TEMPLATE = os.environ.get("RXM_TEMPLATE", "/data/workspace/references/td-SC_4.00bpw_H5/config.json")   # a turboderp config.json
L = "model.language_model.layers"

ap = argparse.ArgumentParser()
ap.add_argument("--frac-ext", metavar="SRC", help="quarter-step / K5-8 half-step rates: build+install v3/e3/exl3frac from "
                "the rxm-frac-quarter exllamav3_ext sources SRC (needs PYTORCH_ROCM_ARCH)")
ap.add_argument("--codebook", choices=("mul1", "i8", "i4", "i8c"), default="mul1", help="search with the int8- or int4-"
                "rounded mul1 codebook (v3/e3/exl3cb, needs --frac-ext SRC for the sources): what V3's int kernels multiply")
ap.add_argument("-K", required=True,
                help="bitrate: integer 1-8, or fractional (K = KA + n/16, n even; needs the rxm-frac-quarter exllamav3 for n != 8); "
                "a comma list quantizes several levels in one pass, layer by layer, reusing each tensor's K-independent work "
                "(Hessian factorization, its unrotated copy); -o must then contain {K}. Output identical to one run per level")
ap.add_argument("-o", required=True, help="output dir; {K} is replaced by the level")
ap.add_argument("--layers", default="0-63")
ap.add_argument("--head", action="store_true", help="also lm_head")
ap.add_argument("--only-head", action="store_true")
ap.add_argument("--device", type=int, default=0)
ap.add_argument("--seed-offset", type=int, default=0, help="added (x 100000) to every tensor's rotation seed: replicate quants of the same setting (0 = the usual seeds)")
ap.add_argument("--sigma-reg", type=float, default=None, help="exllamav3's diagonal regularisation (default its own)")
ap.add_argument("--hout", default=None, help="dir of output-side Hessians (<key>.safetensors, 'hout' square or 'hout_diag' "
                "vector = a diagonal factor; v3/yaqa/fold.py, collect.py YAQA_DIAG): two-sided LDLQ for the linears that have one")
ap.add_argument("--hin", default=None, help="dir of replacement input Hessians in the challenge layout (a file named "
                "<checkpoint key>.safetensors replaces the input Hessian of that linear alone: YAQA Sketch B) "
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
ap.add_argument("--wpre", default=argparse.SUPPRESS, metavar="DIR", help="target transforms: DIR/<input-Hessian "
                "name>.safetensors, \"Ft\" (in x in, = F^T): every trunk linear of that group quantizes W F instead of its "
                "weight (activation denoising, proxy2b/exp/w07-denoise/mkdenoise.py; pair with its --hin dir). A trunk group "
                "without a file is an error. Unset: not in the code stamp's arguments")
ap.add_argument("--head-mean-row", action="store_true", help="lm_head: subtract the mean vocabulary row before "
                "quantizing (logits shift by one constant a position: softmax and KL unchanged). Every donor of a slab "
                "head must agree on this, or mixed rows stop sharing the shift")
ap.add_argument("--scaled-hessian", action="store_true", help="exllamav3's scaled_hessian: LDLQ factors P diag(su) H diag(su) P "
                "per tensor (the metric of the row-scaled weight it actually quantizes) instead of the shared rotated H")
ap.add_argument("--qtf2", action="store_true", help="fractional rates on v3/e3/exl3frac2 (qtf2: the V3 kernel pass; "
                "bit-identical on the old rates, plus 1/8 steps and sub-1 K); needs --frac-ext SRC")
a = ap.parse_args()
_kp = lambda v: int(v) if float(v).is_integer() else float(v)
KS = [_kp(v) for v in a.K.split(",")]
assert len(KS) == len(set(KS)), "duplicate level in -K"
O_TMPL = a.o
assert len(KS) == 1 or "{K}" in O_TMPL, "several levels: -o needs {K}"
a.K = KS[0]; a.o = O_TMPL.replace("{K}", str(KS[0]))
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
    assert a.frac_ext and a.codebook in ("mul1", "i8", "i8c"), "--qtf2 needs --frac-ext SRC and the mul1, i8 or i8c codebook"
    sys.path.insert(0, next(d for d in (os.path.join(_h, "exl3frac2"), os.path.join(_h, "../v3/e3/exl3frac2")) if os.path.isdir(d)))
    import exl3frac2
    exl3frac2.install(a.frac_ext, a.codebook)
_keys = lambda f: set(l.strip() for l in open(f) if l.strip() and not l.startswith("#"))
OFF = _keys(a.out_scales_off) if a.out_scales_off else set()
ONLY = _keys(a.only) if a.only else None
dev = torch.device("cuda", a.device)
_warm_refit(dev)                                   # load the refit kernels while VRAM is empty
for _k in KS: os.makedirs(O_TMPL.replace("{K}", str(_k)), exist_ok=True)

# ------------------------------------------------------------------ several levels in one pass: K-independent work once
# finalize_capture_H (Hessian scaling, damping, sign / Hadamard rotation, block LDL) and unrotate_H depend on the tensor,
# the Hessian, the damping and the seed, not on K. With several levels, the first level's results are kept per (tensor,
# damping) for the layer and replayed for the others: the dict fields finalize sets, and its one RNG draw (the sign
# vector su), so every later draw -- sv, the g_scale samples -- sees the RNG exactly as in a single-level run.
_CUR = [None]
_MEMO, _UMEMO = {}, {}
def _memo_clear(): _MEMO.clear(); _UMEMO.clear()
if len(KS) > 1:
    _orig_fin, _orig_unrot = _Q.finalize_capture_H, _Q.unrotate_H
    _FIELDS = ("L", "H", "su", "finalized", "diag", "q_fallback")
    def _fin(H_data, quant_args, verbose):
        if _CUR[0] is None or H_data["finalized"] or H_data["H"].is_meta or "H_swap_device" in H_data:
            return _orig_fin(H_data, quant_args, verbose)
        mk = (_CUR[0], quant_args.get("sigma_reg", 0.025), quant_args.get("seed"))
        if mk in _MEMO:
            m = _MEMO[mk]
            torch.randn(m["k"], device=H_data["H"].device)      # the su draw a fresh finalize makes
            H_data.update(m["fields"])
            return m["ret"]
        k_ = H_data["H"].shape[0]
        ret = _orig_fin(H_data, quant_args, verbose)
        _MEMO[mk] = {"k": k_, "fields": {f: H_data[f] for f in _FIELDS}, "ret": ret}
        return ret
    def _unrot(H_rot, su):                         # pure; keyed by the (kept-alive) objects finalize returned
        uk = (id(H_rot), id(su))
        if uk not in _UMEMO: _UMEMO[uk] = (H_rot, su, _orig_unrot(H_rot, su))
        return _UMEMO[uk][2]
    _Q.finalize_capture_H, _Q.unrotate_H = _fin, _unrot

# ------------------------------------------------------------------ the bf16 checkpoint
idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
_open = {}
def ck(name):
    f = idx[name]
    if f not in _open: _open[f] = safe_open(f"{CKPT}/{f}", "pt", device="cpu")
    return _open[f].get_tensor(name)
if getattr(a, "wpre", None):                       # --wpre: a trunk linear's weight is read as W F (out, in), fp32 on dev
    _ck_raw = ck
    def ck(name):
        w = _ck_raw(name)
        key = name[:-len(".weight")]
        if not key.startswith(L + "."): return w      # lm_head: unchanged
        hn = next(h for h, ks in layer_plan(int(key.split(".")[3])) if key in ks)
        pf = f"{a.wpre}/{hn}.safetensors"
        if not os.path.exists(pf): sys.exit(f"--wpre {a.wpre}: no transform for {hn} ({key})")
        with safe_open(pf, "pt", device="cpu") as f: Ft = f.get_tensor("Ft").to(dev).float()
        w = w.to(dev).float()
        assert Ft.shape == (w.shape[1], w.shape[1]), (key, tuple(Ft.shape), tuple(w.shape))
        return w @ Ft.T                                 # W F

# ------------------------------------------------------------------ a Hessian (packed upper triangle, sum over rows)
_HUP = {}
def hessian(name, path=None, on=None):
    p = path or f"{HESS}/{name}.safetensors"
    if not path and a.hin and os.path.exists(f"{a.hin}/{name}.safetensors"): p = f"{a.hin}/{name}.safetensors"
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
    meta = hdr["__metadata__"]; k = int(meta["k"]); rows = int(meta["rows"])
    if p not in _HUP:                              # the damping sweep reloads one Hessian up to 4x a tensor: keep the last
        _HUP.clear()
        with safe_open(p, "pt", device="cpu") as f: _HUP[p] = f.get_tensor("hessian_upper")
    d_ = on or dev
    up = _HUP[p].to(d_)
    H = torch.zeros(k, k, dtype=torch.float32, device=d_)
    iu = torch.triu_indices(k, k, device=d_)
    H[iu[0], iu[1]] = up; del up
    H = H + H.triu(1).T
    return {"H": H, "first_key": name if not path else os.path.basename(path), "count": rows, "finalized": False, "num_total": rows,
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
        qa = {"seed": seed0 + j + 100000 * a.seed_offset, "K": a.K, "devices": [a.device], "device_ratios": None,
              "apply_out_scales": True, "mul1": True, "debug_dir": os.path.join(a.o, "debug")}
        if key in OFF: qa["apply_out_scales"] = False
        if a.scaled_hessian: qa["scaled_hessian"] = True
        if a.sigma_reg is not None: qa["sigma_reg"] = a.sigma_reg
        if a.sigma_reg_in is not None and hname.endswith(".layers." + hname.split(".layers.")[1].split(".")[0] + ".input"):
            qa["sigma_reg"] = a.sigma_reg_in
        GAIN[0] = next((float(m) for suf, m in GAINS.items() if key.endswith(suf)), 1.0)
        if a.sigma_reg_out is not None: qa["sigma_reg_out"] = a.sigma_reg_out
        hp = f"{a.hout}/{key}.safetensors" if a.hout else None
        if hp and os.path.exists(hp):
            with safe_open(hp, "pt", device="cpu") as f:     # "hout_diag": a diagonal-only factor (collect.py YAQA_DIAG)
                # kept on the CPU: prepare_H_out copies it to the card itself (a dense 17408^2 fp32 here is 1.2 GB of VRAM
                # that nothing else reads; the 4070's hconv MLP arms ran out by 170 MB). Same values, same result.
                qa["H_out"] = (f.get_tensor("hout") if "hout" in f.keys() else torch.diag(f.get_tensor("hout_diag").float()))
        hd = f"{a.hout_diag}/{key}.safetensors" if a.hout_diag else None
        if hd and os.path.exists(hd):
            with safe_open(hd, "pt", device="cpu") as f:
                qa["out_hessian_diag"] = (f.get_tensor("hout_diag") if "hout_diag" in f.keys() else f.get_tensor("hout").diagonal()).float().to(dev)
            qa["apply_out_scales"] = None             # the diagonal path runs only with output scales on auto
        is_in = hname.endswith(".layers." + hname.split(".layers.")[1].split(".")[0] + ".input") if ".layers." in hname else False
        if a.sigma_in_sweep and is_in:
            Hu = hessian(hname)["H"]; ref = float((w * (Hu @ w)).sum()); best = None
            sweep_all = {}
            for sg in [float(v) for v in a.sigma_in_sweep.split(",")]:
                qs = dict(qa); qs["sigma_reg"] = sg
                _CUR[0] = key; wq_, perr_, t_ = quantize_exl3(w.clone(), hessian(hname), qs, True)
                E = w - wq_.float(); rel = float((E * (Hu @ E)).sum()) / ref; del E
                gs = qs.get("g_scale"); sweep_all[str(sg)] = [rel, None if gs is None else float(gs)]
                if best is None or rel < best[0]: best = (rel, sg, wq_, perr_, t_)
                else: del wq_, t_
            _, qa["sigma_reg"], wq, perr, t = best; qa["sweep_rel"] = best[0]; qa["sweep_all"] = sweep_all; del Hu
        else:
            # a per-linear input Hessian (YAQA Sketch B: every linear its own) overrides the group's
            pk = f"{a.hin}/{key}.safetensors" if a.hin else None
            Hk = hessian(hname, pk) if pk and os.path.exists(pk) else Hd
            if Hk is not Hd: qa["hin_key"] = True
            _CUR[0] = key; wq, perr, t = quantize_exl3(w, Hk, qa, a.save_wq)
        if a.save_wq:
            import numpy as np
            os.makedirs(os.path.join(a.o, "wq"), exist_ok=True)
            try: H = hessian(hname)["H"]                # fresh: quantize_exl3 may have regularised Hd in place
            except torch.OutOfMemoryError:              # small cards (ffn_down's 17408^2 beside the layer): rows on the CPU
                if not os.environ.get("OURQ_REFIT_CPU_FALLBACK"): raise
                torch.cuda.empty_cache(); H = hessian(hname, on="cpu")["H"]; w, wq = w.cpu().float(), wq.cpu().float()
            err, ref = [], []
            for c in range(0, w.shape[1], 8192):        # per output row, in column chunks
                E = w[:, c:c + 8192] - wq[:, c:c + 8192]
                err.append(((H @ E) * E).sum(0).double().cpu())
                W = w[:, c:c + 8192]; ref.append(((H @ W) * W).sum(0).double().cpu())
                del E, W
            np.savez(os.path.join(a.o, "wq", f"{key}.rowerr.npz"), err=torch.cat(err).numpy(), ref=torch.cat(ref).numpy())
            del H
#io-was:            save_file({"weight": wq.T.half().contiguous().cpu()}, os.path.join(a.o, "wq", f"{key}.safetensors"))
            _io_stage(lambda p: save_file({"weight": wq.T.half().contiguous().cpu()}, p), os.path.join(a.o, "wq", f"{key}.safetensors"))  # +io
        del w, wq
        for k2, v in t.items(): out[f"{key}.{k2}"] = v.contiguous().cpu()
        two = qa.pop("H_out", None) is not None
        log.write(json.dumps({"key": key, "K": a.K, "proxy_err": float(perr), "fallback": bool(qa.get("q_fallback")), "hout": two,
                              "hin_key": bool(qa.get("hin_key")),
                              "hout_diag_scales": bool(qa.get("out_scales_hessian")), "sigma_reg": qa.get("sigma_reg"), "sweep_rel": qa.get("sweep_rel"), "sweep_all": qa.get("sweep_all"),
                              "g_scale": None if qa.get("g_scale") is None else float(qa["g_scale"]), "drift": _Q.ldlq_drift(a.K), "scaled_hessian": bool(a.scaled_hessian), "gain": GAIN[0], "codebook": a.codebook,
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
#io-was:    json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(a.o, "model.safetensors.index.json"), "w"), indent=1)
    _io_json({"metadata": {}, "weight_map": wm}, os.path.join(a.o, "model.safetensors.index.json"), indent=1)  # +io
    c = json.load(open(TEMPLATE))
    c["quantization_config"].update({"bits": float(a.K), "head_bits": a.K, "calibration": {"source": "challenge calib Hessians"}})
#io-was:    json.dump(c, open(os.path.join(a.o, "config.json"), "w"), indent=1)
    _io_json(c, os.path.join(a.o, "config.json"), indent=1)  # +io

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
#io-was:        if os.path.isfile(f): h.update(f.encode()); h.update(open(f, "rb").read()); src.append(f)
        if os.path.isfile(f): h.update(f.encode()); h.update(_io_canon(f)); src.append(f)  # +io
    keep = {k: v for k, v in vars(a).items() if k not in ("o", "layers", "head", "only_head", "device", "save_wq")}
    h.update(json.dumps(keep, sort_keys=True).encode())
    return h.hexdigest()[:16], src, keep
_LOGS, _STAMPS = {}, {}
def set_level(K):
    """switch a.K / a.o / STAMP / log to level K (stamp and stamp.json exactly as a single-level run's)"""
    global STAMP, STAMP_SRC, STAMP_ARGS, log
    a.K = K; a.o = O_TMPL.replace("{K}", str(K))
    if K not in _STAMPS:
        _STAMPS[K] = _stamp()
        st, src, args = _STAMPS[K]
        _sj = os.path.join(a.o, "stamp.json")
        _prev = json.load(open(_sj)) if os.path.exists(_sj) else []
        if not _prev or _prev[-1]["stamp"] != st:
#io-was:            json.dump(_prev + [{"stamp": st, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "args": args,
#io-was:                                "sources": src}], open(_sj, "w"), indent=1)
            _io_json(_prev + [{"stamp": st, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "args": args,  # +io
                               "sources": src}], _sj, indent=1)  # +io
        print(f"code stamp {st} (K{K})", flush=True)
        _LOGS[K] = open(os.path.join(a.o, "proxy.jsonl"), "a")
    STAMP, STAMP_SRC, STAMP_ARGS = _STAMPS[K]; log = _LOGS[K]
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
for _k in KS: set_level(_k)
lo, hi = map(int, a.layers.split("-"))
if not a.only_head:
    for l in range(lo, hi + 1):
        _memo_clear()
        for K in KS:
            set_level(K)
            shard = os.path.join(a.o, f"q-L{l:02d}.safetensors")
            if os.path.exists(shard): _io_leftover(f"{L}.{l}.")  # +io
            if resume_ok(shard): continue
            t0 = time.time(); out = {}
            if ONLY is not None and not any(k in ONLY for _, ks in layer_plan(l) for k in ks): continue
            for gi, (hn, keys) in enumerate(layer_plan(l)):
                if ONLY is not None and not any(k in ONLY for k in keys): continue
                out.update(quantize_group(hn, keys, l * 16 + gi * 4, log))
#io-was:            save_file(out, shard + ".part", metadata={"ourq_stamp": STAMP}); os.replace(shard + ".part", shard)
            _io_put(lambda p: save_file(out, p, metadata={"ourq_stamp": STAMP}), shard)  # +io
            # drop the checkpoint shard handles (their mapped pages piled up to ~23 GB over a run: K8 OOM, 2026-10-30)
            del out; _open.clear(); import gc; gc.collect(); torch.cuda.empty_cache()
            rss = next((l.split()[1] for l in open("/proc/self/status") if l.startswith("RssAnon")), "?")
            print(f"layer {l}{f' K{K}' if len(KS) > 1 else ''}: {time.time() - t0:.0f}s  anon rss {int(rss) >> 10 if rss != chr(63) else rss} MiB", flush=True)
            write_index()
            _io_promote()  # +io
        _memo_clear()
if a.head or a.only_head:
    _memo_clear()
    for K in KS:
        set_level(K)
        shard = os.path.join(a.o, "q-head.safetensors")
        if os.path.exists(shard): _io_leftover("lm_head.")  # +io
        if not resume_ok(shard):
            out = quantize_group("lm_head.input", ["lm_head"], 9999, log)
#io-was:            save_file(out, shard + ".part", metadata={"ourq_stamp": STAMP}); os.replace(shard + ".part", shard)
            _io_put(lambda p: save_file(out, p, metadata={"ourq_stamp": STAMP}), shard)  # +io
            write_index(); _io_promote()  # +io
    _memo_clear()
for K in KS: set_level(K); write_index()
print("done", flush=True)
