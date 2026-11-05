#!/usr/bin/env python3
"""realq.py [REAL-Q options] -- [ourq.py arguments] -- REAL-Q (arXiv 2609.00049) on our quantizer: quantize/ourq.py runs
unchanged (exec'd in this process: same arguments, settings, code stamp, shards, --save-wq output), and four functions
of exllamav3's quantize module are wrapped from here (no shared code edited):

  quantize_exl3   which linear is being quantized (ourq's _CUR); before it, the calibration streams are brought to its
                  layer; after it, its dequantized weight replaces the bf16 one in the quantized-prefix model;
  regularize      captures su / sv (input / output scales incl. g_scale) to map the rotated weight to the original;
  ldlq / ldlq_2hess  exllamav3's loops, copied, with one REAL-Q step after every 128-row span (one-sided) or at every
                  1/(k/128) of the tiles (two-sided --hout sweep along anti-diagonals): one Adam step on the still
                  unquantized entries against the Fisher-weighted block-output loss (below);
  refit_scales    the post-LDLQ scale refit fits the quantized weight to the GD-updated target W + ΣΔ (--refit-target
                  gd, default) instead of the original W.

The REAL-Q step (paper §4.2-4.4, App. A / D.1): calibration windows (rqdata, qad pool) are kept as three streams on the
host: X^_l (block l's input through the already-quantized prefix), X*_{l+1}, X*_{l+2} (the bf16 model's block outputs).
For a minibatch of --batch windows (--micro at a time), block l runs with the linears already quantized at their
quantized values, the current one at [quantized rows | unquantized rows incl. LDLQ's error feedback], later ones bf16;
loss = a L_l + (1 - a) L_{l+1}, L = 1/2 mean_t dy_tᵀ F dy_t (fisher.py's F, dy clipped per token at the --clip quantile
of |dy| by a detached factor), a = 1 - (b - 1) / (B - 1) over the layer's B column blocks (sliding window; block l+1
runs bf16); the last layer uses the true KL against the head (gen-masked mean) with --lr-last. Adam (0.9, 0.999, 1e-8,
bias-corrected, state per linear) runs in the rotated basis the LDLQ works in (exllamav3's incoherence basis, the
analog of the paper's QuaRot weights), with lr scaled by rms(W_r) / rms(W) so a step is lr-sized in weight units.
lr per layer: reverse-cosine from --lr-first-frac x --lr up to --lr (D.1 eq. 29).

--gd-off: everything above except the step itself (the identity gate: shards and wq equal a plain ourq run's).
OUTDIR/realq.jsonl: one line per linear (steps, loss first / last, step size). OUTDIR/realq-stamp.json: these options
and the hashes of this code and the Fisher; a resume with different ones exits (RQ_MIXED=1 overrides)."""
import argparse, hashlib, json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
OURQ = os.path.join(HERE, "../../../quantize/ourq.py")
argv = sys.argv[1:]
assert "--" in argv, "usage: realq.py [REAL-Q options] -- [ourq.py arguments]"
own, rest = argv[:argv.index("--")], argv[argv.index("--") + 1:]
ap = argparse.ArgumentParser(allow_abbrev=False)
ap.add_argument("--fisher", required=True, help="fisher.py's OUT/fisher.pt")
ap.add_argument("--ref-model", default=os.environ.get("RXM_CKPT"), help="the bf16 reference (HF dir); default RXM_CKPT")
ap.add_argument("--ncal", type=int, default=256, help="calibration windows (the first of rqdata's order)")
ap.add_argument("--seqlen", type=int, default=2048)
ap.add_argument("--batch", type=int, default=32, help="windows per Adam step (paper: 32)")
ap.add_argument("--micro", type=int, default=4, help="windows per forward / backward")
ap.add_argument("--lr", type=float, default=3e-4, help="eta_final (paper Qwen3-0.6B W4A16: 3e-4)")
ap.add_argument("--lr-first-frac", type=float, default=0.01, help="first block's lr = this x --lr (paper 0.01)")
ap.add_argument("--lr-const", action="store_true", help="no reverse-cosine schedule (constant --lr)")
ap.add_argument("--lr-last", type=float, default=1e-5, help="last block (true KL) lr (paper 0.6B: 1e-5)")
ap.add_argument("--clip", type=float, default=0.95, help="dy clip quantile (paper a_loss_clip 0.95 for small Qwen3); 0 off")
ap.add_argument("--no-window", action="store_true", help="no loss sliding window (a = 1)")
ap.add_argument("--no-genmask", action="store_true", help="last block: KL over every position (default: scored ones)")
ap.add_argument("--refit-target", choices=("gd", "orig"), default="gd")
ap.add_argument("--base-wq", default=None, help="wq dir of a finished quant: linears not quantized by this run (--only) "
                "take these weights in the quantized prefix and the window block")
ap.add_argument("--gd-off", action="store_true")
ap.add_argument("--max-sec", type=float, default=0, help="exit 3 at a layer boundary when the next layer would end past "
                "this many seconds (a rerun resumes from the shards: a long quant in several GPU-lock holds)")
ap.add_argument("--max-steps-per-linear", type=int, default=0, help="debug: cap GD steps per linear (0 = all)")
r = ap.parse_args(own)
op = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
op.add_argument("-o"); op.add_argument("-K"); op.add_argument("--only"); op.add_argument("--layers", default="0-63")
op.add_argument("--seed-offset", type=int, default=0); op.add_argument("--save-wq", action="store_true")
op.add_argument("--sigma-in-sweep"); op.add_argument("--device", type=int, default=0)
oa, _ = op.parse_known_args(rest)
assert oa.save_wq, "REAL-Q needs --save-wq (the dequantized weight feeds the quantized prefix)"
assert oa.K and "," not in oa.K and "{K}" not in oa.o, "one level per run"
assert not oa.sigma_in_sweep, "--sigma-in-sweep is not supported"

import numpy as np, torch
import exllamav3.modules.quant.exl3_lib.quantize as _Q
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
import rqdata
from common import load_model
from safetensors.torch import load_file
dev = torch.device("cuda", oa.device)
if os.environ.get("VRAM_CAP_GB"):
    torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["VRAM_CAP_GB"]) * 2**30 / torch.cuda.get_device_properties(dev).total_memory), dev)
os.makedirs(oa.o, exist_ok=True)

# ------------------------------------------------------------------ stamp (ourq's own stamp does not see these options)
_h = hashlib.sha256()
for f in (__file__, rqdata.__file__): _h.update(open(f, "rb").read())
FI = torch.load(r.fisher)
_h.update(json.dumps({"fisher": FI["args"], "n_tok": FI["n_tok"]}, sort_keys=True).encode())
STAMP = {"code_fisher": _h.hexdigest()[:16], "args": {k: v for k, v in vars(r).items() if k != "max_sec"}, "fisher_args": FI["args"]}
_sp = os.path.join(oa.o, "realq-stamp.json")
if os.path.exists(_sp) and json.load(open(_sp)) != STAMP and not os.environ.get("RQ_MIXED"):
    sys.exit(f"{_sp}: made with other REAL-Q options or code; new OUTDIR, or RQ_MIXED=1")
json.dump(STAMP, open(_sp, "w"), indent=1)
RQLOG = open(os.path.join(oa.o, "realq.jsonl"), "a")

# ------------------------------------------------------------------ reference model and calibration streams
model, dec, head = load_model(r.ref_model, dev=dev)
model.requires_grad_(False)
NL = len(dec.layers); D = model.config.get_text_config().hidden_size
LTYPES = model.config.get_text_config().layer_types
F = FI["F"].to(dev)
assert F.shape == (NL, D, D), F.shape
toks, gpos, wins = rqdata.windows(r.seqlen, FI["args"]["seed"])
wins = wins[:r.ncal]; N, T = len(wins), r.seqlen
TOK = torch.from_numpy(np.stack([toks[w:w + T] for w in wins]).astype(np.int64))
GEN = torch.from_numpy(np.stack([gpos[w:w + T] for w in wins])) if not r.no_genmask else torch.ones(N, T)
LP = "model.language_model.layers."
def lmod(key):                       # checkpoint key -> (layer, module path inside the decoder layer)
    l, m = key[len(LP):].split(".", 1); return int(l), m

KW = {}                               # the decoder's per-layer-type call kwargs (captured from one forward, batch 1)
def _cap(l):
    def h(mod, args, kwargs): KW.setdefault(LTYPES[l], {k: v for k, v in kwargs.items() if k != "past_key_values"})
    return h
_hs = [dec.layers[LTYPES.index(t)].register_forward_pre_hook(_cap(LTYPES.index(t)), with_kwargs=True) for t in set(LTYPES)]
with torch.no_grad(): dec(input_ids=TOK[:1].to(dev))
for h_ in _hs: h_.remove()
for t, kw in KW.items():
    am = kw.get("attention_mask")
    assert am is None or am.shape[0] == 1, (t, None if am is None else am.shape)
def fwd(l, x, ov=None):
    return torch.func.functional_call(dec.layers[l], ov or {}, (x,), KW[LTYPES[l]])

ONLY = set(x.strip() for x in open(oa.only) if x.strip() and not x.startswith("#")) if oa.only else None
QW = {}                               # key -> quantized weight (out, in) bf16 on the host: this run's, as quantized
def final_w(key, final_layer):
    """the weight key has in the quantized prefix (None = bf16)"""
    if key in QW: return QW[key]
    p = os.path.join(oa.o, "wq", key + ".safetensors")
    l = lmod(key)[0]
    if (ONLY is None or key in ONLY) and final_layer and os.path.exists(p) and os.path.exists(os.path.join(oa.o, f"q-L{l:02d}.safetensors")):
        QW[key] = load_file(p)["weight"].to(torch.bfloat16); return QW[key]
    if r.base_wq and (ONLY is not None and key not in ONLY) and os.path.exists(os.path.join(r.base_wq, key + ".safetensors")):
        QW[key] = load_file(os.path.join(r.base_wq, key + ".safetensors"))["weight"].to(torch.bfloat16); return QW[key]
    return None
def layer_keys(l):
    return [k for _, ks in NS["layer_plan"](l) for k in ks]
def overrides(l, final_layer=True, skip=()):
    ov = {}
    for k in layer_keys(l):
        if k in skip: continue
        w = final_w(k, final_layer)
        if w is not None: ov[lmod(k)[1] + ".weight"] = w.to(dev, non_blocking=True)
    return ov

class Streams:
    def __init__(self):
        self.l = None
        mk = lambda: torch.empty(N, T, D, dtype=torch.bfloat16).pin_memory()
        self.xq, self.y1, self.y2 = mk(), mk(), (mk() if NL > 1 else None)
    @torch.no_grad()
    def init(self, l0):
        t0 = time.time(); ovs = [overrides(j) for j in range(l0)]
        src = {j: sum(1 for k in layer_keys(j) if final_w(k, True) is not None) for j in range(l0)}
        for c in range(0, N, r.micro):
            x = dec.embed_tokens(TOK[c:c + r.micro].to(dev)); xs = x
            for j in range(l0): x = fwd(j, x, ovs[j])
            self.xq[c:c + r.micro] = x.cpu()
            for j in range(min(l0 + 2, NL)):
                xs = fwd(j, xs)
                if j == l0: self.y1[c:c + r.micro] = xs.cpu()
                if j == l0 + 1: self.y2[c:c + r.micro] = xs.cpu()
        self.l = l0
        print(f"realq: streams at layer {l0} ({N} x {T}, {time.time() - t0:.0f}s; quantized linears below: {src})", flush=True)
    @torch.no_grad()
    def advance(self):
        l = self.l; ov = overrides(l)
        miss = [k for k in layer_keys(l) if (ONLY is None or k in ONLY) and k not in QW]
        if miss: print(f"realq: warning: layer {l} leaves with bf16 {miss}", flush=True)
        for c in range(0, N, r.micro):
            self.xq[c:c + r.micro] = fwd(l, self.xq[c:c + r.micro].to(dev), ov).cpu()
            if l + 3 <= NL:                      # X*_{l+3} = layer l+2 (X*_{l+2}), into y1's buffer
                self.y1[c:c + r.micro] = fwd(l + 2, self.y2[c:c + r.micro].to(dev)).cpu()
        if l + 3 <= NL: self.y1, self.y2 = self.y2, self.y1
        else: self.y1, self.y2 = self.y2, None
        self.l = l + 1
    def goto(self, l):
        if self.l is None: self.init(l)
        while self.l < l: self.advance()
S = Streams()

# ------------------------------------------------------------------ the REAL-Q step
class Ctx: active = False
C = Ctx()
HAD = _Q.get_hadamard_dt(128, dev, torch.float, 1 / math.sqrt(128))
def dequant(Z, su, sv):              # rotated (k, n) -> original (k, n): as quantize_exl3's reconstruction
    k, n = Z.shape
    x = (HAD @ Z.reshape(-1, 128, n)).reshape(k, n) * su.view(k, 1)
    return (x.reshape(k, -1, 128) @ HAD).reshape(k, n) * sv.view(1, n)

def lr_of(l):
    if l == NL - 1: return r.lr_last
    if r.lr_const: return r.lr
    lo = r.lr * r.lr_first_frac
    return lo + (r.lr - lo) * math.sin(math.pi / 2 * l / max(NL - 1, 1))   # eq. 29 (i - 1) / (L - 1), 0-based

def fisher_mse(dy, Fm):
    dy = dy.float()
    if r.clip > 0:
        a = dy.abs(); kk = max(1, math.ceil(r.clip * dy.shape[-1]))
        tau = a.kthvalue(kk, dim=-1, keepdim=True).values
        dy = dy * torch.where(a > tau, tau / a.clamp_min(1e-30), torch.ones_like(a)).detach()
    return 0.5 * ((dy @ Fm) * dy).sum(-1).mean()

def kl_backward(yhat, ystar, m, wt):
    """d/d yhat of wt x sum over masked positions of KL(p* || q) (last block, true KL), chunked; returns the value"""
    yd = yhat.detach().requires_grad_(True); idx = m.flatten().nonzero()[:, 0]; tot = 0.0
    if len(idx) == 0: return 0.0
    yf, sf = yd.reshape(-1, D), ystar.reshape(-1, D)
    for c in range(0, len(idx), 512):
        ii = idx[c:c + 512]
        with torch.no_grad(): lp = torch.log_softmax(head(dec.norm(sf[ii])).float(), -1)
        lq = torch.log_softmax(head(dec.norm(yf[ii])).float(), -1)
        kl = (lp.exp() * (lp - lq)).sum() * wt
        kl.backward(); tot += float(kl); del lp, lq, kl
    yhat.backward(yd.grad)
    return tot

@torch.enable_grad()
def gd_step(Zeff, pend):
    """one Adam step for C's linear: Zeff the rotated effective weight (k, n), pend the unquantized mask -> delta"""
    C.b += 1
    if r.gd_off or (r.max_steps_per_linear and C.steps >= r.max_steps_per_linear): return None
    t0 = time.time()
    Zl = Zeff.detach().requires_grad_(True)
    W = dequant(Zl, C.su, C.sv)
    if C.lr_r is None:
        C.lr_r = C.lr * float(Zeff.square().mean().sqrt() / W.detach().square().mean().sqrt().clamp_min(1e-30))
        C.w_rms = float(W.detach().square().mean().sqrt())
    Wl = W.detach().requires_grad_(True)
    alpha = 1.0 if (r.no_window or C.btot <= 1) else 1 - (C.b - 1) / (C.btot - 1)
    l = C.layer; last = l == NL - 1
    ov = dict(C.ov)
    pick = torch.randperm(N, generator=C.gen)[:r.batch]
    wsum = float(GEN[pick].sum()) if last else 0.0
    lsum = 0.0
    for c in range(0, len(pick), r.micro):
        ii = pick[c:c + r.micro]; frac = len(ii) / len(pick)
        x = S.xq[ii].to(dev, non_blocking=True)
        ov[C.mod + ".weight"] = Wl.T.to(torch.bfloat16)
        yhat = fwd(l, x, ov)
        if last:
            lsum += kl_backward(yhat, S.y1[ii].to(dev), GEN[ii].to(dev), 1.0 / max(wsum, 1.0))
        else:
            loss = alpha * fisher_mse(yhat.float() - S.y1[ii].to(dev).float(), F[l])
            if alpha < 1:
                y2 = fwd(l + 1, yhat, C.ov1)
                loss = loss + (1 - alpha) * fisher_mse(y2.float() - S.y2[ii].to(dev).float(), F[l + 1])
            (loss * frac).backward(); lsum += float(loss) * frac
        del yhat
    if Wl.grad is None: return None                   # (last block: no scored position drawn)
    g, = torch.autograd.grad(W, Zl, Wl.grad)
    g = g * pend
    C.t += 1; b1, b2 = 0.9, 0.999
    C.m.mul_(b1).add_(g, alpha=1 - b1); C.v.mul_(b2).addcmul_(g, g, value=1 - b2)
    step = (C.m / (1 - b1 ** C.t)) / ((C.v / (1 - b2 ** C.t)).sqrt() + 1e-8)
    delta = -C.lr_r * step * pend
    C.steps += 1; C.losses.append(lsum); C.sec += time.time() - t0
    if C.steps == 1: C.g_rms = float(g[pend.expand_as(g) > 0].square().mean().sqrt())
    return delta

def _apply(weight, delta, rows=None):
    if delta is None: return
    if rows is None: weight.add_(delta); C.Dsum.add_(delta)
    else: weight[:rows].add_(delta[:rows]); C.Dsum[:rows].add_(delta[:rows])

# ---- exllamav3's ldlq (one-sided), copied; REAL-Q step after every span
def ldlq_rq(weight, L, quant_args, pb=None):
    devices = quant_args["devices"]
    for device in devices: torch.cuda.synchronize(device)
    main_stream = _Q.get_quant_stream(devices[0])
    with torch.cuda.stream(main_stream):
        device = L.device
        assert device == torch.device(devices[0])
        assert weight.device == device, "REAL-Q: weight must be on the card"
        size_k, size_n = weight.shape
        assert size_k % 16 == 0 and size_n % 128 == 0
        tiles_k, tiles_n = size_k // 16, size_n // 16
        buf_size_k = max(quant_args.get("buf_size_k", 128), 16)
        assert buf_size_k % 16 == 0 and size_k % buf_size_k == 0
        p_row = 0
        prod_cache = torch.zeros((size_k, size_n), dtype=torch.float, device=device)
        weight_q = torch.zeros((size_k, size_n), dtype=torch.float, device=device)
        encoded = torch.zeros((tiles_k, tiles_n, 256), dtype=torch.short, device=device)
        for j in range(size_k, 0, -buf_size_k):
            i = j - buf_size_k
            b_weight = weight[i:j]
            b_weight_q = weight_q[i:j]
            b_encoded = encoded[i // 16: j // 16]
            b_prod_cache = prod_cache[i:j]
            b_L = L[i:j]
            for bj in range(buf_size_k, 0, -16):
                bi = bj - 16
                bb_err = b_weight[bj:] - b_weight_q[bj:]
                bb_L = b_L[bj:, i + bi:i + bj]
                compensation_term = b_prod_cache[bi:bj]
                compensation_term.addmm_(bb_L.T, bb_err, alpha=1.0, beta=1.0)
                rows = b_weight[bi:bj] + compensation_term
                tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)
                tiles = tiles[:, _Q.tensor_core_perm(device)]
                quant_w, quant_i = _Q.quantize_tiles_multigpu(tiles, quant_args)
                quant_w = quant_w[:, _Q.tensor_core_perm_i(device)]
                quant_w = quant_w.reshape(tiles_n, 16, 16).permute(1, 0, 2).reshape(16, size_n)
                b_weight_q[bi:bj] = quant_w
                b_encoded[bi // 16: bj // 16] = quant_i.unsqueeze(0)
                if pb:
                    p_row += 1; pb.update(p_row)
            b_err = b_weight - b_weight_q
            prod_cache.addmm_(b_L.T, b_err, alpha=1.0, beta=1.0)
            if C.active:                                    # REAL-Q: rows 0:i are still unquantized
                if i > 0:
                    pend = torch.zeros(size_k, 1, device=device); pend[:i] = 1
                    Zeff = torch.cat([weight[:i] + prod_cache[:i], weight_q[i:]])
                    _apply(weight, gd_step(Zeff, pend), rows=i)
                else:
                    C.b += 1
        for device in devices: torch.cuda.synchronize(device)
    return weight_q, encoded

# ---- exllamav3's ldlq_2hess (two-sided, anti-diagonals), copied; REAL-Q step at every 1/nblk of the tiles
def ldlq_2hess_rq(weight, L_in, L_out, quant_args, pb=None):
    devices = quant_args["devices"]
    for device in devices: torch.cuda.synchronize(device)
    main_stream = _Q.get_quant_stream(devices[0])
    with torch.cuda.stream(main_stream):
        device = L_in.device
        weight = weight.to(device)
        size_k, size_n = weight.shape
        assert size_k % 16 == 0 and size_n % 16 == 0
        tiles_k, tiles_n = size_k // 16, size_n // 16
        Lk = L_in.clone(); Lk.diagonal().fill_(1.0)
        Ln = L_out.clone(); Ln.diagonal().fill_(1.0)
        F_ = torch.zeros((size_k, size_n), dtype=torch.float, device=device)
        weight_q = torch.zeros_like(weight)
        encoded = torch.zeros((tiles_k, tiles_n, 256), dtype=torch.short, device=device)
        W4 = weight.view(tiles_k, 16, tiles_n, 16)
        Q4 = weight_q.view(tiles_k, 16, tiles_n, 16)
        F4 = F_.view(tiles_k, 16, tiles_n, 16)
        perm, perm_i = _Q.tensor_core_perm(device), _Q.tensor_core_perm_i(device)
        ar16 = torch.arange(16, device=device)
        nblk = size_k // 128; ntile = tiles_k * tiles_n; done_t = 0; nb = 0
        ta, tc = torch.arange(tiles_k, device=device).view(-1, 1), torch.arange(tiles_n, device=device).view(1, -1)
        steps = tiles_k + tiles_n - 1
        for step, s in enumerate(range(steps - 1, -1, -1)):
            a = torch.arange(max(0, s - (tiles_n - 1)), min(tiles_k - 1, s) + 1, device=device)
            c = s - a
            tiles = (W4[a, :, c, :] + F4[a, :, c, :]).reshape(-1, 256)
            quant_w, quant_i = _Q.quantize_tiles_multigpu(tiles[:, perm].contiguous(), quant_args)
            quant_w = quant_w[:, perm_i].view(-1, 16, 16)
            Q4[a, :, c, :] = quant_w
            encoded[a, c] = quant_i
            if s > 0:
                dE = W4[a, :, c, :] - quant_w
                rows = (a.unsqueeze(1) * 16 + ar16).flatten()
                cols = (c.unsqueeze(1) * 16 + ar16)
                k_hi = int(a.max().item()) * 16 + 16
                n_hi = int(c.max().item()) * 16 + 16
                right = torch.bmm(dE, Ln[cols][:, :, :n_hi])
                F_[:k_hi, :n_hi].addmm_(Lk[rows, :k_hi].T, right.reshape(-1, n_hi))
            if C.active:                                    # REAL-Q: tiles (a, c) with a + c < s are still pending
                done_t += len(a)
                while nb < nblk and done_t >= (nb + 1) * ntile / nblk - 1e-9:
                    nb += 1
                    if s > 0 and nb < nblk:
                        pend = ((ta + tc) < s).float().repeat_interleave(16, 0).repeat_interleave(16, 1)
                        Zeff = torch.where(pend > 0, weight + F_, weight_q)
                        _apply(weight, gd_step(Zeff, pend))
                    else:
                        C.b += 1
            if pb and step % 4 == 0:
                pb.update(min(tiles_k, step * tiles_k // steps))
        for device in devices: torch.cuda.synchronize(device)
    return weight_q, encoded

_orig_reg = _Q.regularize
def regularize_rq(*args, **kw):
    w0 = args[0].clone() if C.active else None       # regularize works in place
    out = _orig_reg(*args, **kw)
    if C.active:
        C.su = out[3].flatten().float().clone(); C.sv = out[4].flatten().float().clone()
        C.deq_err = float((dequant(out[1], C.su, C.sv) - w0).norm() / w0.norm().clamp_min(1e-30))   # self-check of the map
    return out

_orig_refit = _Q.refit_scales
def refit_rq(weight, weight_q, H, su, sv, *args, **kw):
    if C.active and r.refit_target == "gd" and C.steps > 0:
        weight = weight + dequant(C.Dsum, C.su, C.sv).to(weight.device)
    return _orig_refit(weight, weight_q, H, su, sv, *args, **kw)

_orig_qe = _Q.quantize_exl3
LT, LT0 = [], [0.0]                  # per-layer wall times (for --max-sec)
def quantize_exl3_rq(weight, H_data, quant_args, return_weight_q, *args, **kw):
    key = NS["_CUR"][0]
    if key is None or not key.startswith(LP):
        return _orig_qe(weight, H_data, quant_args, return_weight_q, *args, **kw)
    l, mod = lmod(key)
    if S.l is not None and l > S.l:
        LT.append(time.time() - LT0[0]); LT0[0] = time.time()
        if r.max_sec and time.time() - t_all + 1.25 * max(LT) > r.max_sec:
            print(f"REALQ STOP before layer {l} (--max-sec {r.max_sec:.0f}, {time.time() - t_all:.0f}s)", flush=True)
            sys.exit(3)
    elif S.l is None: LT0[0] = time.time()
    S.goto(l)
    keys = [k for k in layer_keys(l) if ONLY is None or k in ONLY]
    k_of = lambda kk: dict(dec.layers[l].named_modules())[lmod(kk)[1]].weight.shape[1]
    btot = sum(k_of(kk) // 128 for kk in keys)
    b0 = sum(k_of(kk) // 128 for kk in keys[:keys.index(key)])
    k, n = weight.shape
    C.__dict__.update(active=True, key=key, layer=l, mod=mod, btot=btot, b=b0, lr=lr_of(l), lr_r=None, t=0, steps=0,
                      losses=[], sec=0.0, g_rms=None, w_rms=None, deq_err=None,
                      m=torch.zeros(k, n, device=dev), v=torch.zeros(k, n, device=dev), Dsum=torch.zeros(k, n, device=dev),
                      ov=overrides(l, final_layer=False, skip=(key,)),
                      ov1=overrides(l + 1, final_layer=False) if l + 1 < NL else {},
                      gen=torch.Generator().manual_seed(int(hashlib.sha256(f"{key}|{oa.seed_offset}".encode()).hexdigest()[:12], 16)))
    t0 = time.time()
    try:
        wq, perr, t = _orig_qe(weight, H_data, quant_args, return_weight_q, *args, **kw)
    finally:
        C.active = False
    QW[key] = wq.T.half().to(torch.bfloat16).cpu()
    dn = float(dequant(C.Dsum, C.su, C.sv).square().mean().sqrt()) if C.steps else 0.0
    rec = {"key": key, "layer": l, "deq_err": C.deq_err, "steps": C.steps, "blocks": C.b - b0, "btot": btot, "lr": C.lr, "lr_r": C.lr_r,
           "loss_first": C.losses[0] if C.losses else None, "loss_last": C.losses[-1] if C.losses else None,
           "losses": [round(x, 8) for x in C.losses], "g_rms0": C.g_rms, "w_rms": C.w_rms, "delta_rms": dn,
           "gd_sec": round(C.sec, 1), "sec": round(time.time() - t0, 1), "max_mem_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)}
    RQLOG.write(json.dumps(rec) + "\n"); RQLOG.flush()
    if C.steps:
        print(f"  realq {key[len(LP):]}: {C.steps} steps, loss {C.losses[0]:.4e} -> {C.losses[-1]:.4e}, "
              f"|ΣΔ|/|W| {dn / max(C.w_rms, 1e-30):.4f}, gd {C.sec:.0f}s", flush=True)
    del C.m, C.v, C.Dsum
    return wq, perr, t

# the in-layer overrides: for the current layer only, a linear quantized earlier in this run counts; final_w with
# final_layer=False must not pick up stale wq files of the layer being redone
_final_w = final_w
def final_w(key, final_layer):
    if not final_layer and key not in QW:
        if r.base_wq and ONLY is not None and key not in ONLY: return _final_w(key, True)
        return None
    return _final_w(key, final_layer)

_Q.quantize_exl3, _Q.regularize, _Q.refit_scales = quantize_exl3_rq, regularize_rq, refit_rq
_Q.ldlq, _Q.ldlq_2hess = ldlq_rq, ldlq_2hess_rq

# ------------------------------------------------------------------ run ourq.py in this process
NS = {"__name__": "__main__", "__file__": os.path.abspath(OURQ)}
sys.argv = [os.path.abspath(OURQ)] + rest
t_all = time.time()
exec(compile(open(OURQ).read(), os.path.abspath(OURQ), "exec"), NS)
print(f"REALQ DONE {time.time() - t_all:.0f}s", flush=True)
