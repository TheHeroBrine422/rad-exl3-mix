#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""qad220.py (w220) = qad200.py (w217's trainer, copied) + the w220 switches; every one is off by default and then the trainer is
qad200's (v3/qad220/README.md "w220 flags"): --train-layers / --train-tensors / --mask (trainable set; the layers before the first
trainable one run once under no_grad: automatic), --tie-groups / --tie-layers, --clip-gnorm / --beta2, --lr-steps / --decay-shape /
--fork-from / --decay-data / --save-at, --prompt-weight, --local-mse / --local-taps (+ --make-taps --tap-layers), --no-frozen-prefix.
w223: --a8-int8 (= --a8 with the GEMM as a real int8 x int8 -> int32 product, v3/qad223/a8int8.py; backward dX bf16 as --a8's STE).
qad200.py's header follows.

qad200.py OUT --model BF16 --level LEVEL --chunks DIR,... [--calib GOLDEN --pools POOLS] [options] -- #200 (RULES200):
#196's trainer (qad196.py, copied) with the #200 switches. All off = #196 arm A (checks in data/qad200/checks200.md):
  --wd W                     AdamW (decoupled weight decay) on the trunk log multipliers: each step multiplies them by
                             (1 - lr W) toward 0 (= the stored scales) before the Adam update; the norm gains are not decayed.
                             W = 0 keeps torch.optim.Adam (#196's code path, unchanged).
  --accum N                  N consecutive plan steps (micro-batches of the same size: 24 x 1024 positions) per optimizer
                             step. Each micro-batch's KL sum is divided by the optimizer step's total rows, so the summed
                             gradient is the gradient of the mean KL over N x the micro-batch rows (loss scaled 1/N per
                             micro-batch). Streams carry their states across micro-batches as in #196. N = 1 is #196.
  --rows T                   the budget in gen rows (overrides --budgets); the data stream is #181's seeded order past 1M rows
                             (the plan is built on the whole order; the 1M run is the prefix of the 4M run).
  --order-seed S             the data-order seed: an alias of --seed (w186's A-s1 is --seed 1 = the same switch).
  --warmup-frac F            warmup = max(1, round(F x optimizer steps)); F = 8/49 is #191's 8-of-49 warmup stretched to the step count.
#196's header (qad196.py, copied):
qad196.py OUT --model BF16 --level LEVEL --chunks DIR,... [--calib GOLDEN --pools POOLS] [options] -- #196 (RULES196):
#191's trainer (qad191.py, copied) with three switches on the S-only recipe. All off = #191's arm A, exactly (unit check):
  --factors all|attn_ssmout  R: trainable row scales only on attn_qg (self_attn.q_proj), attn_k, attn_v, attn_output (o_proj)
                             and ssm_out (linear_attn.out_proj); every other linear's r / c frozen at their start values
  --norms                    N: RMSNorm gains (exp(g) on each norm's output, hook form as qmodel's arm C) on every
                             input_layernorm, post_attention_layernorm, q_norm / k_norm, GDN linear_attn.norm and the final
                             norm; trained at lr / 10. Exported to OUT/ckpt-*/norms.safetensors (HF names, bf16; the 1 + w
                             form for Qwen3_5RMSNorm, plain w for the gated GDN norm)
  --unshrink FILE            U: per-row un-shrink c_r (unshrink196.py, a safetensors of key -> [n] f32) set as the start
                             value of each listed linear's output-side log multiplier (a frozen parameter where the output
                             side is not trained: attn_v under the gauge, ffn_down / ssm_qkv under R)
  --export-only              write OUT/ckpt-<budget>/ (scales, norms, params) with no training, then exit (the U arms)

#191's header (qad191.py, copied):
trainer (qad181.py, copied; arm A = trunk suh / svh log multipliers on frozen codes, the mix's quantized head frozen)
with the combination screen's factors as composable flags (RULES191; every combination valid, all off = qad181):
  L  --lora R [--lr-lora]       rank-R adapters on every trunk linear, trained with the scales (qmodel's arm L)
  B  --block-frac F             block-wise stage first: per-block output MSE vs the bf16 checkpoint's block (inputs: the
                                student's own trajectory; targets: the bf16 trajectory) over the documents of the plan's
                                first F x budget rows, one block at a time; then the end-to-end stage on the rest
  T  --taps DIR --tap-weight W  + W x the DFlash2-tap hidden-state MSE (layers 5/19/33/47/61, the teacher's states read
                                from DIR, written by --make-taps) on the end-to-end loss, over the same gen rows
  Q  --a8                       a8g128 fake-quant activations on every trunk linear (#181 arm Q)
  D  --mix50 SEED               data order: half calib qad pool, half xcorp by gen rows (else #181's seeded shuffle)
The served form's other pieces: --emb FILE (the served token table, raw bf16 [V, E]: emb191.py), --ab FILE (the
served GDN a / b weights, bf16, blk.N.ssm_a / ssm_b). --make-taps DIR: the bf16 teacher's tap states (2 cards), no
training. --check: the flag unit checks on one segment, no training. --tapcal N: KL and tap MSE of the untrained student
over the first N steps (the T weight's calibration). The text below is qad181's.

Data / windows: data181 (every document from its start to --cap tokens, V virtual streams x Q positions a step,
segments carry the student's KV / GDN / conv state detached; a missing state (resume, a branch fork) is rebuilt by a
no-grad forward over the prefix). Loss: the scorer's KL (top-K + tail cell, qad.chunk_kl) summed over a step's rows /
the step's rows. The running training loss (each batch unseen before its update) is logged per corpus.

Schedule (WSD): warmup, constant lr, linear decay to 0 over the last --decay-frac of the budget; for every budget T in
--budgets below the last, a branch forks the main run at the step where it has seen (1 - decay-frac) T positions and
decays to 0 at T on the same data: OUT/ckpt-<T>/ is then a one-epoch run of budget T. The main run's end is the last
budget. Ranks hold their streams; grads are summed with gloo; every rank takes the same Adam step.
OUT/state.pt resumes (main state + step; branches restart from their fork); OUT/train.jsonl per step;
OUT/ckpt-<T>/params.pt + scales/<key>.safetensors (rxq.swj apply_level's format) + mon.json.
--gate: decode / segment-carry exactness and timing on one card, then exit."""
import argparse, collections, contextlib, hashlib, json, math, os, re, sys, time
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qad"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "qad181"))
sys.path.insert(0, os.path.join(HERE, "..", "qad223"))                       # w223 a8int8 (imported only with --a8-int8)
import qmodel  # noqa: E402
from qad import chunk_kl, save_atomic  # noqa: E402
import data181 as D  # noqa: E402

BF = torch.bfloat16


# ------------------------------------------------------------------ student extensions
def mad_int_form():
    """mad values are q * 4 k_inv rounded to f16 with |q| <= 126 (deq.mad_lut): the i8 step, so mad tensors hold as int8"""
    old = qmodel.int_form
    qmodel.int_form = lambda cb: old("i8") if cb == "mad" else old(cb)


class _A8(torch.autograd.Function):
    """a8g128 fake quant (arm Q): per row and 128-group absmax / 127 int8 rounding of the GEMM's (rotated) input,
    straight-through gradient"""
    @staticmethod
    def forward(ctx, x):
        g = x.float().view(*x.shape[:-1], -1, 128)
        s = g.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127.0
        return (torch.round(g / s).clamp(-127, 127) * s).view_as(x).to(x.dtype)
    @staticmethod
    def backward(ctx, g): return g


def _a8_int8_gemm(q, x):
    """w223 --a8-int8: the a8g128 GEMM of the rotated input x (bf16) with q's int8 codebook integers as an int8 x int8 -> int32 product
    (per row and 128-group scales in f32; quantization bitwise = _A8's); backward dX = dy @ bf16(wq)^T (= --a8's STE). A linear held as
    bf16 W_hat (not integer-exact) takes the fake-quant path."""
    if q.wq.dtype != torch.int8: return _A8.apply(x) @ q.wq.to(BF)
    import a8int8
    return a8int8.a8_matmul(x, q.wq, qmodel.i8_to_bf16)


def set_dynamo_limit(n):
    """w222: raise torch._dynamo's per-function recompile limit (default 8: the two compiled scale functions hit it with the 400 linears' guard variants and the rest fell back to eager); 0 / None = untouched"""
    if not n: return
    import torch._dynamo
    for k in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit", "accumulated_cache_size_limit"):
        if hasattr(torch._dynamo.config, k): setattr(torch._dynamo.config, k, max(int(n) * (8 if k.startswith("acc") else 1), getattr(torch._dynamo.config, k)))


def use_compiled_scales(stu, a8=False, a8_int8=False, dynamo_limit=0):
    """QLinear's elementwise chain (x . su -> bf16, y . (sv s) -> bf16; arm Q's fake quant) as torch.compile'd functions
    (dynamic shapes): the same arithmetic in fused kernels (rounding-level differences from eager). Inductor's deterministic
    mode (no on-device autotuning: w215 saw two runs differ without it) keeps runs bitwise reproducible; numerics "strict"
    keeps eager's bf16 downcasts between fused ops and eager's reduction order"""
    import types
    import torch._inductor.config
    torch._inductor.config.deterministic = True; torch._inductor.config.numerics = "strict"
    set_dynamo_limit(dynamo_limit)
    @torch.compile(dynamic=True)
    def pre(x0, suh, r): return (x0 * (suh * r.exp() if r is not None else suh)).to(BF)
    @torch.compile(dynamic=True)
    def post(y, svh, c, s): return (y * ((svh * c.exp() if c is not None else svh) * s)).to(BF)
    def fwd(self, x):
        sh = x.shape
        x = pre(x.reshape(-1, self.k), self.suh, self.r)
        x = (x.view(-1, 128) @ self.Hb).view(-1, self.k)
        if a8_int8: y = _a8_int8_gemm(self, x)
        else:
            if a8: x = _A8.apply(x)
            y = x @ qmodel.i8_to_bf16(self.wq)
        y = (y.view(-1, 128) @ self.Hb).view(-1, self.n)
        return post(y, self.svh, self.c, self.s).view(*sh[:-1], self.n)
    for q in stu.qlin.values():
        assert q.la is None, "compiled scales: no adapters"
        q.forward = types.MethodType(fwd, q)


def use_a8(stu, a8_int8=False):
    """the student's trunk QLinears run the GEMM on a8g128 fake-quantized activations (the hadamard-rotated, suh-scaled
    input, which is what the exl3i A8 path quantizes); the head stays f16 as served. a8_int8 (w223): the GEMM as int8 x int8"""
    def fwd(self, x):
        sh = x.shape
        x0 = x.reshape(-1, self.k)
        x = (x0 * self.su()).to(BF)
        x = (x.view(-1, 128) @ self.Hb).view(-1, self.k)
        if a8_int8: y = _a8_int8_gemm(self, x)
        else: y = _A8.apply(x) @ self.wq.to(BF)
        y = (y.view(-1, 128) @ self.Hb).view(-1, self.n)
        y = y * (self.sv() * self.s)
        if self.la is not None: y = y + (x0.float() @ self.la.T) @ self.lb.T            # adapters (arm L) compose, f32 (use_lora32)
        return y.to(BF).view(*sh[:-1], self.n)
    import types
    for q in stu.qlin.values(): q.forward = types.MethodType(fwd, q)


def use_lora32(stu):
    """L without Q: qmodel.QLinear's forward with the adapter GEMMs in f32 (x0 f32, A / B as held) instead of bf16.
    11-15: the bf16 adapter weight-gradient GEMM (dA = dT^T x0, [16, k], K = the segment's rows) hit a Tensile memory
    fault (Cijk_Ailk_Bjlk_BBS_BH_..._MT64x64x32, 80 workgroups) at fixed steps, on hipBLASLt and on rocBLAS alike"""
    def fwd(self, x):
        sh = x.shape
        x0 = x.reshape(-1, self.k)
        x = (x0 * self.su()).to(BF)
        x = (x.view(-1, 128) @ self.Hb).view(-1, self.k)
        y = x @ self.wq.to(BF)
        y = (y.view(-1, 128) @ self.Hb).view(-1, self.n)
        y = y * (self.sv() * self.s)
        if self.la is not None: y = y + (x0.float() @ self.la.T) @ self.lb.T
        return y.to(BF).view(*sh[:-1], self.n)
    import types
    for q in stu.qlin.values(): q.forward = types.MethodType(fwd, q)


class SegCache:
    """the transformers Cache surface Qwen3.5's layers call, for one segment: reads the previous segment's state
    (detached, immutable) and records this segment's final state in .next (detached; idempotent under the checkpoint's
    recompute)"""
    def __init__(self, prev=None): self.prev, self.next = prev or {}, {}
    def get_conv_state(self, l, state_idx=0): return self.prev.get(("conv", l))
    def get_recurrent_state(self, l, state_idx=0): return self.prev.get(("rec", l))
    def is_recording_past(self, l): return False
    def update_conv_state(self, x, l, conv_kernel_size):
        p = self.prev.get(("conv", l))
        if p is not None: x = torch.cat([p.to(x.dtype), x], -1)
        self.next[("conv", l)] = x[:, :, -(conv_kernel_size - 1):].detach().clone()     # a copy: a view keeps all of x alive
        return x
    def update_recurrent_state(self, s, l): self.next[("rec", l)] = s.detach()
    def update(self, k, v, l, *a, **kw):
        p = self.prev.get(("kv", l))
        if p is not None: k, v = torch.cat([p[0], k], -2), torch.cat([p[1], v], -2)
        self.next[("kv", l)] = (k.detach(), v.detach())
        return k, v


class KVView:
    """w225: an attention layer's carried K / V in a preallocated buffer [1, kv heads, capacity, head dim] shared by a document's segments;
    the first n positions are valid"""
    __slots__ = ("k", "v", "n")
    def __init__(self, k, v, n): self.k, self.v, self.n = k, v, n
    @property
    def device(self): return self.k.device
    def nbytes(self): return 2 * self.k.numel() * self.k.element_size()
    def to(self, dev, cap=0):
        """a copy on dev in a buffer of max(cap, n) positions (cap 0: just the valid part, the CPU park)"""
        sh = list(self.k.shape); sh[2] = max(cap, self.n)
        k = torch.empty(sh, dtype=self.k.dtype, device=dev); v = torch.empty(sh, dtype=self.v.dtype, device=dev)
        k[:, :, :self.n].copy_(self.k[:, :, :self.n]); v[:, :, :self.n].copy_(self.v[:, :, :self.n])
        return KVView(k, v, self.n)


def kv_room(need, cap):
    """w226: a KV buffer's capacity for `need` positions: 1/8 headroom in 4,096-position steps, at most the document's length (cap); buffers grow
    with the document (two 245k documents in flight at once would otherwise hold 2 x 8 GiB a card from their first segment)"""
    return max(need, min(cap, -(-(need + need // 8) // 4096) * 4096))


class SegCacheBuf(SegCache):
    """w225 (--split-layers / --state-card-gb / --depth-test): SegCache with each attention layer's carried K / V in one buffer per document
    (KVView, capacity cap = the document's length): a segment writes its own k / v after the carried prefix in place and attends to
    cat(buffer[:P], k) (a one-layer transient), so the prefix is held once (SegCache's cat kept it twice: the previous state and the new one).
    The attention's inputs and the carried values are bitwise SegCache's"""
    def __init__(self, prev=None, cap=0):
        super().__init__(dict(prev) if prev else None); self.cap = cap
    def room(self, need): return kv_room(need, self.cap)
    def update(self, k, v, l, *a, **kw):
        p = self.prev.get(("kv", l)); n = k.shape[2]
        P = p.n if p is not None else 0
        if p is None:
            sh = list(k.shape); sh[2] = self.room(n)
            p = KVView(torch.empty(sh, dtype=k.dtype, device=k.device), torch.empty(sh, dtype=v.dtype, device=v.device), 0)
            self.prev[("kv", l)] = p
        elif p.k.shape[2] < P + n or p.device != k.device:
            p = p.to(k.device, self.room(P + n)); self.prev[("kv", l)] = p
        with torch.no_grad():                    # idempotent under the checkpoint's recompute (same positions, same values)
            p.k[:, :, P:P + n].copy_(k.detach()); p.v[:, :, P:P + n].copy_(v.detach())
        self.next[("kv", l)] = KVView(p.k, p.v, P + n)
        if P == 0: return k, v
        return torch.cat([p.k[:, :, :P], k], -2), torch.cat([p.v[:, :, :P], v], -2)


def devctx(dev):
    """w225: make dev the current device (raw Triton kernels launch on the current device, not their tensors')"""
    return torch.cuda.device(dev) if dev.type == "cuda" else contextlib.nullcontext()


ATTN_GQA = [False]              # w225 --attn-gqa
PIPE_INFLIGHT = 4 * 2**30       # w225 --pipeline: bytes of carried states brought onto the cards (parked / rebuilt) that the segments in flight may hold


def triton_autotune_lock():
    """w225 --pipeline: triton's Autotuner keeps per-call state on the shared kernel object (self.nargs, read by its reset / restore hooks while
    it benchmarks a new key), so the two cards' threads must not interleave inside it: one lock around Autotuner.run (a launch holds it for
    microseconds; a first-time benchmark for its duration)"""
    import threading
    import triton.runtime.autotuner as at
    if getattr(at.Autotuner, "_w225_lock", None) is not None: return
    lk = threading.RLock(); run0 = at.Autotuner.run
    def run(self, *args, **kwargs):
        with lk: return run0(self, *args, **kwargs)
    at.Autotuner.run = run; at.Autotuner._w225_lock = lk


def seg_attention(module, q, k, v, attention_mask, dropout=0.0, scaling=None, **kw):
    """sdpa for a segment after a prefix: query i (absolute P + i) sees keys 0 .. P + i (lower-right causal)"""
    from torch.nn.attention.bias import causal_lower_right
    L, S = q.shape[2], k.shape[2]
    if ATTN_GQA[0]:             # w225: the KV heads shared in the kernel (no 6x repeated K / V: 2 GiB instead of 12 GiB a layer at 256k; forward bitwise, backward rounding-level)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=None if S == L else causal_lower_right(L, S), is_causal=S == L, scale=scaling, enable_gqa=True)
        return o.transpose(1, 2).contiguous(), None
    rep = q.shape[1] // k.shape[1]
    if rep > 1: k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    if S == L: o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scaling)
    else: o = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_lower_right(L, S), scale=scaling)
    return o.transpose(1, 2).contiguous(), None


def use_seg_attention(stu):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["qad181"] = seg_attention
    stu.cfg._attn_implementation = "qad181"


TAPS = (5, 19, 33, 47, 61)      # DFlash2 target_layer_ids: radiance copies layer li's output (residual stream) into the taps


def run_segment(stu, ids, a, prev=None, ckpt=True, taps=None, loc=None):
    """final-norm hidden states [n, H] of tokens a .. a + n - 1 of a document given the state after 0 .. a - 1 (prev:
    a SegCache.next of the segment before, or None at a = 0); returns (h, next state); with taps (layer indices) also
    {layer: its output [1, n, H]}; w220 loc (a LocalMSE): the same third value for loc's layers, and in mode "local" those layers
    are cut out of the KL's graph (input and output detached: they train on their own MSE only)"""
    if getattr(stu, "split", 0): return run_segment_split(stu, ids, a, prev, ckpt, taps, loc)
    q35 = stu.q35
    dev = stu.devs[0]
    assert len(stu.devs) == 1, "qad181: one card a student"
    x = stu.embed(ids)
    n = x.shape[1]
    o = a + getattr(stu, "pos_off", 0)                  # w225 --depth-test: positions after the synthetic prefix
    pid = torch.arange(o, o + n, device=dev).view(1, 1, -1).expand(4, 1, -1)
    pe = stu.rotary[dev](x, pid[1:])
    cache = (SegCacheBuf(prev, getattr(stu, "kv_cap", 0)) if getattr(stu, "kvbuf", False) else SegCache(prev)); tp = {}; lo = {}
    ft = getattr(stu, "first_train", 0)                  # w220: layers before the first trainable one run once, no autograd
    for i, ly in enumerate(stu.layers):
        kw = dict(position_embeddings=pe, attention_mask=None, position_ids=pid[0], past_key_values=cache)
        if loc is not None and i in loc.set:
            xi = x.detach() if loc.mode == "local" else x          # block (w222): a listed layer ends a block; x is already detached when the layer before it was listed, so a block of layers trains jointly on its end layer's MSE
            o = checkpoint(lambda h, _l=ly, _k=kw: _l(h, **_k), xi, use_reentrant=False) if ckpt and torch.is_grad_enabled() else ly(xi, **kw)
            lo[i] = o; x = o.detach() if loc.mode in ("local", "block") else o
        elif i < ft and torch.is_grad_enabled():
            with torch.no_grad(): x = ly(x, **kw)
        elif ckpt and torch.is_grad_enabled(): x = checkpoint(lambda h, _l=ly, _k=kw: _l(h, **_k), x, use_reentrant=False)
        else: x = ly(x, **kw)
        if taps is not None and i in taps: tp[i] = x
    if taps is not None: return stu.fnorm(x)[0], cache.next, tp
    if loc is not None: return stu.fnorm(x)[0], cache.next, lo
    return stu.fnorm(x)[0], cache.next


def run_half(stu, x, a, cache, lo, hi, ckpt=True):
    """w225 (--split-layers): decoder layers lo .. hi - 1 on x [1, n, H] (on those layers' card) for tokens a .. a + n - 1 (+ the --depth-test
    offset), as run_segment's loop (layers before the first trainable one under no_grad, the rest checkpointed); returns the last output"""
    dev = x.device; n = x.shape[1]; o = a + getattr(stu, "pos_off", 0)
    pid = torch.arange(o, o + n, device=dev).view(1, 1, -1).expand(4, 1, -1)
    pe = stu.rotary[dev](x, pid[1:])
    ft = getattr(stu, "first_train", 0)
    for i in range(lo, hi):
        ly = stu.layers[i]
        kw = dict(position_embeddings=pe, attention_mask=None, position_ids=pid[0], past_key_values=cache)
        if i < ft and torch.is_grad_enabled():
            with torch.no_grad(): x = ly(x, **kw)
        elif ckpt and torch.is_grad_enabled(): x = checkpoint(lambda h, _l=ly, _k=kw: _l(h, **_k), x, use_reentrant=False)
        else: x = ly(x, **kw)
    return x


def run_segment_split(stu, ids, a, prev=None, ckpt=True, taps=None, loc=None):
    """w225 --split-layers K: run_segment over two cards as one connected graph (the naive split): the token table and layers 0 .. K - 1 on
    devs[0], layers K .. end and the final norm on devs[1]; the hidden state crosses once (its gradient once back), each layer's carried
    state lives on its card. Returns (final-norm hidden states [n, H] on devs[1], next state)"""
    assert taps is None and loc is None, "--split-layers: no taps / local MSE"
    A, B, K = stu.devs[0], stu.devs[-1], stu.split
    cache = SegCacheBuf(prev, getattr(stu, "kv_cap", 0))
    with devctx(A): x = run_half(stu, stu.embed(ids), a, cache, 0, K, ckpt)
    with devctx(B):
        x = run_half(stu, x.to(B), a, cache, K, len(stu.layers), ckpt); h = stu.fnorm(x)[0]
    return h, cache.next


def seg_kl(stu, gold, i, a, b, h, ch=256):
    """w225 (--pipeline): seg_loss's KL sum and rows from the final-norm hidden states h [n, H] of tokens a .. b - 1 of document i"""
    j, r = gold.scored(i, a, b)
    if len(j) == 0: return torch.zeros((), device=h.device), 0
    dev = h.device
    h = h[torch.from_numpy(j - a).to(dev)]
    tid, tlp, ptail = gold.targets(r, dev)
    if stu.v2row is not None: tid = stu.v2row[tid]
    tot = 0.0
    for c in range(0, len(j), ch):
        args = (h[c:c + ch], stu.head, tid[c:c + ch], tlp[c:c + ch], ptail[c:c + ch])
        tot = tot + (checkpoint(chunk_kl, *args, use_reentrant=False) if torch.is_grad_enabled() else chunk_kl(*args))
    return tot, len(j)


def tap_mse(tp, tt, jj):
    """the T term of a segment: sum over its rows of the mean over TAPS of |s - t|^2 / |t|^2 (student tap rows tp[l][0, jj]
    vs the teacher's tt [m, taps, H])"""
    ts = 0.0
    for k, l in enumerate(TAPS):
        s = tp[l][0, jj].float(); t = tt[:, k].float()
        ts = ts + ((s - t).pow(2).sum(-1) / t.pow(2).sum(-1).clamp_min(1e-12)).sum()
    return ts / len(TAPS)


def seg_loss(stu, g, it, a, b, prev, ch=256, per_row=False, tap=None):
    """(KL sum over the segment's rows, rows, next state); per_row: the KL vector (no grad); tap (a TapStore): also
    the T term's sum over the same rows (a 4th value)"""
    gi, i = it
    gold = g[gi]
    if tap is not None: h, nxt, tp = run_segment(stu, gold.ids(i, a, b), a, prev, taps=TAPS)
    else: h, nxt = run_segment(stu, gold.ids(i, a, b), a, prev)
    j, r = gold.scored(i, a, b)
    if len(j) == 0:
        z = torch.zeros((), device=h.device) if not per_row else torch.zeros(0, device=h.device)
        return (z, 0, nxt, torch.zeros((), device=h.device)) if tap is not None else (z, 0, nxt)
    dev = h.device
    ts = tap_mse(tp, tap.rows(gold, i, j, dev), torch.from_numpy(j - a).to(dev)) if tap is not None else None
    h = h[torch.from_numpy(j - a).to(dev)]
    tid, tlp, ptail = gold.targets(r, dev)
    if stu.v2row is not None: tid = stu.v2row[tid]
    if per_row:
        return torch.cat([chunk_kl(h[c:c + ch], stu.head, tid[c:c + ch], tlp[c:c + ch], ptail[c:c + ch], True)
                          for c in range(0, len(j), ch)]), len(j), nxt
    tot = 0.0
    for c in range(0, len(j), ch):
        args = (h[c:c + ch], stu.head, tid[c:c + ch], tlp[c:c + ch], ptail[c:c + ch])
        tot = tot + (checkpoint(chunk_kl, *args, use_reentrant=False) if torch.is_grad_enabled() else chunk_kl(*args))
    if tap is not None: return tot, len(j), nxt, ts
    return tot, len(j), nxt


class TapStore:
    """the bf16 teacher's TAPS states at the gen rows of the documents of a data stream (--make-taps): DIR/taps.bin
    bf16 [rows, taps, H], DIR/index.json {"golden:doc": [row offset, window]}; a document's stored rows are its
    scored positions in [0, window) in order"""
    def __init__(self, d, any_layers=False):
        z = json.load(open(f"{d}/index.json")); assert os.path.exists(f"{d}/.done"), f"{d}: taps not finished"
        if not any_layers: assert tuple(z["taps"]) == TAPS, z["taps"]          # the T arm: the DFlash2 taps
        self.layers = tuple(z["taps"])                                          # w220 (--local-taps): any layers (--make-taps --tap-layers)
        self.off, self.H = z["docs"], z["hidden"]
        self.mm = np.memmap(f"{d}/taps.bin", np.uint16, "r", shape=(z["rows"], len(self.layers), self.H))
        self.J = {}

    def rows(self, gold, i, j, dev):
        key = f"{gold.name}:{i}"; o, n = self.off[key]
        if key not in self.J:
            if len(self.J) > 256: self.J.clear()
            self.J[key] = gold.scored(i, 0, n)[0]
        J = self.J[key]; lo = int(np.searchsorted(J, j[0]))
        assert np.array_equal(J[lo:lo + len(j)], j), (key, j[0], j[-1])
        x = torch.from_numpy(np.array(self.mm[o + lo:o + lo + len(j)])).view(BF)             # a copy (the memmap is read-only)
        return x.to(dev, non_blocking=True)


# ------------------------------------------------------------------ loss options (w220)
def parse_layers(s):
    """'0-31', '5+19+33', '0-15+20' -> sorted layer list"""
    out = set()
    for t in s.split("+"):
        lo, hi = parse_range(t); out.update(range(lo, hi + 1))
    return sorted(out)


class LocalMSE:
    """--local-mse LAYERS,lambda[,local|joint] with --local-taps DIR: the bf16 teacher's hidden states (a TapStore of any layers) at
    the gen rows as targets of the student's outputs of the listed layers; the term is lambda x (the sum over a segment's gen rows of
    the mean over the listed layers of |s - t|^2 / |t|^2) and joins the KL sum before the division by the step's rows.
    mode local (default): a listed layer's input is the previous output detached and its own output continues detached, so it trains
    on its MSE alone (the KL never reaches it) and the MSE of layer i reaches only layer i; each layer is fed the student's own
    state, the target is the bf16 trajectory's (arm B's block-wise loss). joint: gradients flow through the whole chain.
    block (w222): the listed layers are block ENDS (a few taps, e.g. 3+7+15+23+31): each block (the layers after the previous listed layer up to
    the listed one) is fed the student's own state detached and trains jointly on the MSE at its end layer; the KL reaches only the layers after the last listed layer"""
    def __init__(self, spec, store):
        f = spec.split(",")
        self.layers = parse_layers(f[0]); self.lam = float(f[1]); self.mode = f[2] if len(f) > 2 else "local"
        assert self.mode in ("local", "joint", "block") and len(f) in (2, 3), f"--local-mse LAYERS,lambda[,local|joint|block]: {spec}"
        self.store = store
        miss = [L for L in self.layers if L not in store.layers]
        assert not miss, f"--local-mse: layers {miss} are not in the --local-taps store {store.layers}"
        self.set = set(self.layers)

    def restrict(self, first_train, log):
        """layers before the first trainable one have nothing to train (and run under no_grad): drop them"""
        drop = [L for L in self.layers if L < first_train]
        if drop: log(f"--local-mse: layers {drop} are before the first trainable layer {first_train}: dropped")
        self.layers = [L for L in self.layers if L >= first_train]; self.set = set(self.layers)
        assert self.layers, "--local-mse: no listed layer is trainable"

    def term(self, outs, tt, jj):
        """sum over the rows jj of the segment of the mean over the layers of |s - t|^2 / |t|^2 (tt: the store's rows [m, layers, H])"""
        ts = 0.0
        for L in self.layers:
            s = outs[L][0, jj].float(); t = tt[:, self.store.layers.index(L)].float()
            ts = ts + ((s - t).pow(2).sum(-1) / t.pow(2).sum(-1).clamp_min(1e-12)).sum()
        return ts / len(self.layers)


def scored_all(gold, i, a, b):
    """every row of the golden in [a, b) with its gen flag: (positions, rows, is_gen bool array); a golden without a gen mask has gen rows only"""
    d = gold.docs[i]
    j = np.arange(max(a, d["score_from"]), min(b, d["tokens"] - 1))
    r = d["row0"] + j - d["score_from"]
    if gold.gen is None or not len(r): return j, r, np.ones(len(j), bool)
    return j, r, np.asarray(gold.gen[r[0]:r[-1] + 1]).astype(bool)[r - r[0]]


def chunk_kl_rows(h, head, tid, tlp, ptail): return chunk_kl(h, head, tid, tlp, ptail, True)


def seg_loss_ext(stu, g, it, a, b, prev, pw=0.0, loc=None, ch=256):
    """seg_loss with the w220 loss options: (loss for backward, gen KL sum, gen rows, all rows, next state, local MSE term sum).
    loss = sum over the segment's gen rows of KL + pw x sum over its prompt rows of KL (--prompt-weight; the golden's rows outside
    its gen mask) + loc.lam x the local MSE term; the trainer divides it by the step's gen rows. gen KL / local term: detached"""
    gi, i = it
    gold = g[gi]
    jg, rg = gold.scored(i, a, b)
    if loc is not None: h, nxt, outs = run_segment(stu, gold.ids(i, a, b), a, prev, loc=loc)
    else: h, nxt = run_segment(stu, gold.ids(i, a, b), a, prev)
    dev = h.device; z = torch.zeros((), device=dev)
    if pw > 0: j, r, isg = scored_all(gold, i, a, b)
    else: j, r, isg = jg, rg, np.ones(len(jg), bool)
    if len(j) == 0: return z, z, 0, 0, nxt, z
    ls = z
    if loc is not None and len(jg): ls = loc.term(outs, loc.store.rows(gold, i, jg, dev), torch.from_numpy(jg - a).to(dev))
    h = h[torch.from_numpy(j - a).to(dev)]
    tid, tlp, ptail = gold.targets(r, dev)
    if stu.v2row is not None: tid = stu.v2row[tid]
    wg = torch.from_numpy(isg).to(dev).float(); wt = torch.where(wg > 0, torch.ones_like(wg), torch.full_like(wg, pw))
    tot = 0.0; gen = z
    for c in range(0, len(j), ch):
        args = (h[c:c + ch], stu.head, tid[c:c + ch], tlp[c:c + ch], ptail[c:c + ch])
        v = checkpoint(chunk_kl_rows, *args, use_reentrant=False) if torch.is_grad_enabled() else chunk_kl_rows(*args)
        tot = tot + (v * wt[c:c + ch]).sum(); gen = gen + (v.detach() * wg[c:c + ch]).sum()
    if loc is not None: tot = tot + loc.lam * ls
    return tot, gen, int(isg.sum()), len(j), nxt, ls.detach()


def lr_groups(a, names, params):
    """lead 11-22 --lr-layers "0-31:1e-4[,40-47:2e-4]": the peak lr of each scale vector by its layer (others: --lr); returns
    [(base lr, [params])] as consecutive runs in parameter order (the optimizer's index order stays = opt_order). Without the flag:
    one group at --lr (unchanged)."""
    rules = []
    for x in (a.lr_layers.split(",") if a.lr_layers else []):
        rg, lr_ = x.split(":"); lo, _, hi = rg.partition("-"); rules.append((int(lo), int(hi or lo), float(lr_)))
    def lr_of(n):
        m = re.search(r"\.layers\.(\d+)\.", n)
        if m:
            for lo, hi, v in rules:
                if lo <= int(m.group(1)) <= hi: return v
        return a.lr
    out = []
    for n, p in zip(names, params):
        b = lr_of(n)
        if out and out[-1][0] == b: out[-1][1].append(p)
        else: out.append((b, [p]))
    return out if out else [(a.lr, [])]


def to_cpu(st): return {k: (tuple(t.cpu() for t in v) if isinstance(v, tuple) else v.cpu()) for k, v in st.items()}
def to_dev(st, dev): return {k: (tuple(t.to(dev, non_blocking=True) for t in v) if isinstance(v, tuple) else v.to(dev, non_blocking=True)) for k, v in st.items()}


@torch.no_grad()
def prefix_state(stu, g, it, a, L, st0=None):
    """the state after tokens 0 .. a - 1 of a document, by no-grad segments of <= L tokens (w225 st0: the state before token 0, --depth-test)"""
    gi, i = it; st = st0; p = 0
    while p < a:
        q = min(a, p + L); _, st = run_segment(stu, g[gi].ids(i, p, q), p, st, ckpt=False); p = q
    return st


# ------------------------------------------------------------------ distributed
class Dist:
    def __init__(self):
        self.rank = int(os.environ.get("RANK", 0)); self.world = int(os.environ.get("WORLD_SIZE", 1))
        self.local = int(os.environ.get("LOCAL_RANK", 0))
        if self.world > 1:
            import torch.distributed as dist
            dist.init_process_group("gloo", timeout=__import__("datetime").timedelta(hours=2))
            self.dist = dist
    def allsum(self, t):
        """sum of a CPU f64 / f32 tensor over ranks (in place)"""
        if self.world > 1: self.dist.all_reduce(t)
        return t
    def barrier(self):
        if self.world > 1: self.dist.barrier()


# ------------------------------------------------------------------ schedule
def schedule(cum, budgets, frac):
    """main run's last step (cum rows >= the last budget) and, per smaller budget T, (fork step, end step):
    fork = first step index whose start has seen >= (1 - frac) T rows, end = first step count with >= T rows"""
    def first(x): return int(np.searchsorted(cum, x, side="left")) + 1     # steps needed to reach x rows
    n = len(cum); Tm = budgets[-1]; last = min(n, first(Tm))
    br = {T: (min(n, first((1 - frac) * T)), min(n, first(T))) for T in budgets[:-1]}
    return last, br


EXACT_NAMES = [False]          # lead 11-22: --exact-ckpt-names


def _bfmt(x, spec):
    """budget / checkpoint label: python's spec as before; with --exact-ckpt-names a '.0e' label that would round the row count
    (1.5e6 -> '2e+06', 2.5e5 -> '2e+05') gets one more digit ('1.5e+06', '2.5e+05') so labels never collide or mislead"""
    t = float.__format__(float(x), spec)
    if EXACT_NAMES[0] and spec == ".0e": t = _exact(x, t)
    return t


def _exact(x, t):
    """the shortest e-format label that is exactly x (1.25e6 -> '1.25e+06'); t = the '.0e' label"""
    p = 0
    while float(t) != float(x) and p < 6: p += 1; t = float.__format__(float(x), f".{p}e")
    return t


class BK(float):
    """a budget (a checkpoint key): formats via _bfmt"""
    def __format__(self, spec): return _bfmt(self, spec)


class RW(float):
    """lead 11-22: the key of an undecayed snapshot (--snap-at): the main run's state as it is (no decay), checkpoint dir ckpt-raw-<rows>"""
    def __format__(self, spec):
        t = float.__format__(float(self), spec)
        if spec == ".0e": t = _exact(self, t)                         # always exact (new key, no old names to keep)
        return "raw-" + t


SNAP_GRID = (2.5e5, 5e5, 7.5e5, 1e6)       # then every 5e5 (keep in step with v3/qadq/qq.py snap_grid)


def snap_grid(spec, final):
    """the --snap-at row counts: 'none' -> [], an explicit comma list -> it, default ('auto' / None) -> 0.25 / 0.5 / 0.75 / 1 M, then every
    0.5 M; always only the points before the final budget (the final checkpoint is the run's end)"""
    if spec == "none": return []
    if spec not in (None, "auto"): pts = [float(x) for x in spec.split(",")]
    else:
        pts = list(SNAP_GRID); x = 1.5e6
        while x < final: pts.append(x); x += 5e5
    return [p for p in pts if p < float(final)]


class EMK(float):
    """lead 11-22: the key of a parameter-EMA checkpoint (--ema-params): ckpt-ema<decay>-<rows> (exact rows)"""
    def __new__(cls, x, tag): o = float.__new__(cls, x); o.tag = tag; return o
    def __format__(self, spec): return f"{self.tag}-" + (_exact(self, float.__format__(float(self), spec)) if spec == ".0e" else float.__format__(float(self), spec))


class SD(float):
    """w222: the key of a short-decay branch (--short-decay): formats as `sd-<rows>` so its checkpoint dir is ckpt-sd-2e+05"""
    def __format__(self, spec): return "sd-" + _bfmt(self, spec)


def branch_lr(bs, s, end, last, fork_last, a_lr, warm, lr_at, scheduled):
    """lr of plan step bs of a branch that forked at step s and ends at step end: linear to ~0 from the main run's lr. Unscheduled runs (WSD): a_lr x the ramp, as always; scheduled runs (w222:
    --lr-steps / --decay-shape / --short-decay): the schedule's own lr at the fork, so a branch decays from where the run actually is"""
    if scheduled: return min(lr_at(bs), lr_at(s) * (end - bs) / (end - s))
    return min(lr_main(bs, last, fork_last, a_lr, warm), a_lr * (end - bs) / (end - s))


def lr_main(step, last, fork_last, lr, warm):
    if step < warm: return lr * (step + 1) / warm
    if step < fork_last: return lr
    return lr * max(0.0, (last - step) / max(1, last - fork_last))


def branch_plan(fk, end, N):
    """w222: a WSD branch of budget T (decay from plan step fk to its end step `end`, k0 = 0) with N plan steps per optimizer step, on optimizer-step boundaries: the groups the standalone run of budget T
    executes are 0 .. GT - 1 (GT = ceil(end / N), the last one partial); its lr is constant before group g_f = ceil(fk / N), so the main run (same lr there) is forked at group g0 = min(g_f, GT - 1)
    = plan step g0 * N and the branch runs groups g0 .. GT - 1 with the standalone lr. Returns (g0, GT, g0 * N)"""
    cd = lambda x: -(-x // N)
    GT = cd(end); g0 = min(cd(fk), GT - 1)
    return g0, GT, g0 * N


def group_lr(s, k0, last, fork_last, lr, warm, N):
    """#200: lr of the optimizer step whose first plan step is s, N plan steps per optimizer step (N = 1: lr_main of plan
    step s, as #196). Groups index the schedule: warmup, constant, linear decay over the groups from ceil(fork_last / N)"""
    def cd(x): return -(-x // N)
    return lr_main((s - k0) // N, cd(last - k0), cd(fork_last - k0), lr, warm)


def parse_lr_steps(s):
    """--lr-steps "lr1:rows1,lr2:rows2,..." -> [(lr, rows)] (rows = cumulative gen rows at which the phase ends, increasing)"""
    ph = [(float(x.split(":")[0]), float(x.split(":")[1])) for x in s.split(",")]
    assert all(b[1] > a_[1] for a_, b in zip(ph, ph[1:])) and all(r > 0 and l > 0 for l, r in ph), s
    return ph


def decay_mult(shape, f):
    """the lr multiplier of the decay phase at remaining fraction f = (last - g) / (last - decay start) in (0, 1]: linear: f;
    sqrt (Hagele et al.: 1 - sqrt(progress)): 1 - sqrt(1 - f); cosine-all: (1 + cos(pi progress)) / 2; linear-all: f"""
    if shape in ("linear", "linear-all"): return f
    if shape == "sqrt": return 1.0 - math.sqrt(1.0 - f)
    if shape == "cosine-all": return 0.5 * (1.0 + math.cos(math.pi * (1.0 - f)))
    raise ValueError(shape)


def sched_lr(g, U, FU, warm, lr, shape="linear", phases=None, rows_before=0.0):
    """w220: lr of optimizer step g of U (decay from step FU, warmup warm). Warmup ramps to the first phase's lr; piecewise constant
    over the phases (phase i holds while the rows seen before the step are < its end; the last phase's lr after that); then the
    decay of --decay-shape from the last phase's lr (or lr) to ~0 at step U. Linear without phases is lr_main's arithmetic"""
    peak = phases[0][0] if phases else lr
    if g < warm: return peak * (g + 1) / warm
    if g < FU:
        if not phases: return lr
        for plr, prows in phases:
            if rows_before < prows: return plr
        return phases[-1][0]
    base = phases[-1][0] if phases else lr
    return base * decay_mult(shape, max(0.0, (U - g) / max(1, U - FU)))


def make_lr(a, cum, last, fork_last, k0, N):
    """the lr of the optimizer step starting at plan step s: group_lr (#200) unless a w220 schedule flag is set"""
    if not (a.lr_steps or a.decay_shape != "linear"):
        return lambda s: group_lr(s, k0, last, fork_last, a.lr, a.warmup, N)
    cd = lambda x: -(-x // N)
    phases = parse_lr_steps(a.lr_steps) if a.lr_steps else None
    U = cd(last - k0); FU = a.warmup if a.decay_shape.endswith("-all") else cd(fork_last - k0)
    return lambda s: sched_lr((s - k0) // N, U, FU, a.warmup, a.lr, a.decay_shape, phases, float(cum[s - 1]) if s > 0 else 0.0)


def decay_start_step(a, cum, budget):
    """plan step where the decay starts: the steps needed to see (1 - decay-frac) x budget rows, or --lr-steps' last phase end"""
    x = parse_lr_steps(a.lr_steps)[-1][1] if a.lr_steps else (1 - a.decay_frac) * budget
    return int(np.searchsorted(cum, x, side="left")) + 1


def rows_cum(C, steps):
    """(rows_of {(item, a, b): gen rows}, cumulative gen rows after each step)"""
    rows_of, cum, tot = {}, [], 0
    for st in steps:
        for segs in st:
            for (it, sa, sb) in segs:
                gi, i = C.items[it][0], C.items[it][1]
                rows_of[(it, sa, sb)] = len(C.g[gi].scored(i, sa, sb)[0]); tot += rows_of[(it, sa, sb)]
        cum.append(tot)
    return rows_of, np.array(cum)


def calib_cooldown(C, steps, fork_last, V, Q, seed):
    """--decay-data calib: plan steps 0 .. fork_last - 1 as planned; from fork_last on, a fresh plan over the calib-pool documents
    the first fork_last steps did not touch (seeded shuffle), same V x Q streams"""
    used = {it for st in steps[:fork_last] for segs in st for (it, _, _) in segs}
    cal = [k for k, x in enumerate(C.items) if C.tags[x[0]] == "calib" and k not in used]
    assert cal, "--decay-data calib: no unused calib documents"
    return steps[:fork_last] + D.plan(C.items, np.random.default_rng(seed).permutation(np.array(cal)), V, Q)


def plan_sha(steps, upto):
    """sha1 of the first `upto` plan steps (--fork-from checks the fork state's data prefix is this run's)"""
    return hashlib.sha1(str([[[(int(i), int(x), int(y)) for (i, x, y) in segs] for segs in st] for st in steps[:upto]]).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ tied scales (w220)
class TiedQLinear(qmodel.QLinear):
    """a QLinear whose log multipliers r (input side) / c (output side) are held as one shared value per --tie-groups G consecutive
    channels (parameters r_t / c_t, [k / G] and [n / G]). r / c read as the per-channel vectors (a group's value repeated: expand +
    reshape, whose backward is a deterministic sum over the group), so the forward, the export and the checkpoint format are those
    of the untied student: export writes per-channel scales"""
    tie_g = 1

    @staticmethod
    def expand(t, g): return t.view(-1, 1).expand(-1, g).reshape(-1)

    def _side(self, nm):
        t = self._parameters.get(nm + "_t")
        return self.expand(t, self.tie_g) if t is not None else self._parameters.get(nm, self.__dict__.get(nm))

    @property
    def r(self): return self._side("r")

    @property
    def c(self): return self._side("c")


def tie_linear(ql, g, sides=("r", "c")):
    """convert a QLinear in place into a TiedQLinear with groups of g channels on its trainable sides; returns {side: tied Parameter}"""
    out = {}
    ql.__class__ = TiedQLinear; ql.tie_g = g
    for nm in sides:
        p = ql._parameters.get(nm)
        if p is None or not p.requires_grad: continue
        assert p.numel() % g == 0, f"--tie-groups {g}: {ql.key} {nm} has {p.numel()} channels"
        ql._parameters.pop(nm)
        t = torch.nn.Parameter(p.detach().view(-1, g).mean(1).clone())          # start value: the group mean (0 at a fresh start)
        ql._parameters[nm + "_t"] = t; out[nm] = t
    return out


def apply_ties(a, stu, log):
    """--tie-groups G [--tie-layers A-B]: every trainable r / c of the trunk linears in layers A-B (all by default) becomes one
    parameter per G channels. stu.P[<key>:r|c] then holds the tied parameter"""
    if a.tie_groups <= 1: return
    lo, hi = parse_range(a.tie_layers) if a.tie_layers else (0, len(stu.layers) - 1)
    nt = nv = 0
    for k, ql in stu.qlin.items():
        if not (lo <= layer_of(stu, k + ":r") <= hi): continue
        for nm, t in tie_linear(ql, a.tie_groups).items():
            stu.P[f"{k}:{nm}"] = t; nt += 1; nv += t.numel() * a.tie_groups
    log(f"tied scales: {nt} vectors in layers {lo}-{hi} share one multiplier per {a.tie_groups} channels ({nv} scales -> {nv // a.tie_groups} parameters)")


# ------------------------------------------------------------------ trainable set (w220)
def layer_of(stu, name):
    """decoder layer index of a student parameter name (<linear or norm key>:<r|c|g|la|lb>); the final norm and the head: after the last layer"""
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else len(stu.layers)


def parse_range(s):
    """'A-B' (inclusive) or 'A' -> (A, B)"""
    a, _, b = s.partition("-")
    lo, hi = int(a), int(b if b else a)
    assert 0 <= lo <= hi, s
    return lo, hi


def apply_trainable(a, stu, log):
    """--train-layers A-B (inclusive), --train-tensors REGEX (re.search on the parameter name <linear key>:<r|c>), --mask FILE.npz
    (per scale vector name -> 0 / 1 per scale, mkmask220.py): the trainable set is their intersection. Frozen parameters stay in
    stu.P (state, export) with requires_grad False; a mask that zeroes a whole vector freezes it; a vector the mask file does not list is not masked. Returns {name: bool tensor on the
    parameter's device} of the partial masks (None when no entry is partly masked)"""
    mk = None
    if a.train_layers is not None:
        lo, hi = parse_range(a.train_layers)
    rx = re.compile(a.train_tensors) if a.train_tensors else None
    if a.mask:
        z = np.load(a.mask)
        extra = sorted(set(z.files) - set(stu.P))
        assert not extra, f"--mask: {len(extra)} names not in the student, e.g. {extra[:3]}"
        mk = {}
    for n, p in stu.P.items():
        keep = True
        if a.train_layers is not None: keep = lo <= layer_of(stu, n) <= hi
        if keep and rx is not None: keep = rx.search(n) is not None
        if keep and a.mask and n in z.files:
            m = z[n]; assert m.shape == tuple(p.shape), f"--mask {n}: shape {m.shape} vs the parameter's {tuple(p.shape)} (tied scales need a mask of the tied shape)"
            m = m.astype(bool)
            if not m.any(): keep = False
            elif not m.all(): mk[n] = torch.from_numpy(m).to(p.device)
        if not keep: p.requires_grad_(False)
    if a.train_layers is not None or rx is not None or a.mask:
        log(f"trainable set: layers {a.train_layers}, tensors {a.train_tensors}, mask {a.mask}"
            + (f" ({len(mk)} partly masked vectors, {sum(int(m.sum()) for m in mk.values())} of {sum(m.numel() for m in mk.values())} scales kept)" if mk else ""))
    return mk or None


def set_first_train(stu, log, skip=False):
    """stu.first_trainable = the first decoder layer holding a trainable parameter; stu.first_train = the same: run_segment runs the
    layers before it once under no_grad (no activations kept, no checkpoint recompute, no backward). skip (--no-frozen-prefix, for
    the equivalence check and the speed table): every layer takes the full path, as before w220"""
    tl = [layer_of(stu, n) for n, p in stu.P.items() if p.requires_grad]
    assert tl, "no trainable parameter"
    stu.first_trainable = min(tl); stu.first_train = 0 if skip else min(tl)
    nt = sum(p.numel() for p in stu.P.values() if p.requires_grad); nf = sum(p.numel() for p in stu.P.values() if not p.requires_grad)
    log(f"trainable: {len(tl)} vectors / {nt} values (frozen: {len(stu.P) - len(tl)} vectors / {nf} values); first trainable layer {stu.first_train}"
        + (f": layers 0 .. {stu.first_train - 1} run once under no_grad" if stu.first_train else (" (--no-frozen-prefix: full backward path)" if skip and stu.first_trainable else "")))


# ------------------------------------------------------------------ optimizer hygiene (w220)
CLIP_MIN, CLIP_WIN = 5, 256        # --clip-gnorm: clipping starts after CLIP_MIN steps; the median is over the last CLIP_WIN steps' raw norms


def clip_factor(hist, g, x):
    """(factor, threshold) of --clip-gnorm X for a step whose global gradient norm is g, given the raw norms of the earlier steps:
    threshold = X x the median of the last CLIP_WIN norms; factor = threshold / g when g exceeds it, else 1 (no clipping before
    CLIP_MIN steps of history, threshold None)"""
    if x <= 0 or len(hist) < CLIP_MIN: return 1.0, None
    thr = x * float(np.median(hist[-CLIP_WIN:]))
    return (thr / g if g > thr else 1.0), thr


# ------------------------------------------------------------------ trainer
def map_opt_state(old, old_order, new_order, new_groups):
    """w222: Adam state_dict of an optimizer over the parameters old_order -> a state_dict for the optimizer over new_order (param_groups = the new optimizer's): per-parameter state copied
    by name, absent (fresh) for the new ones"""
    pos = {n: i for i, n in enumerate(old_order)}
    return {"state": {j: old["state"][pos[n]] for j, n in enumerate(new_order) if n in pos and pos[n] in old["state"]}, "param_groups": new_groups}


class Trainer:
    def __init__(self, a, stu, C, steps, rows_of, dist, log, tap=None, loc=None):
        self.a, self.stu, self.C, self.steps, self.rows_of, self.d, self.log = a, stu, C, steps, rows_of, dist, log
        self.tap = tap                                       # T: a TapStore (weight a.tap_weight)
        self.loc = loc                                       # w220: a LocalMSE (--local-mse); with a.prompt_weight: seg_loss_ext
        self.dev = stu.devs[0]
        tr = [(n, p) for n, p in stu.P.items() if p.requires_grad]       # w220: frozen parameters (--train-*, --mask) stay in stu.P (state, export)
        self.P = [p for _, p in tr]; self.names = [n for n, _ in tr]
        nrm = [p for n, p in zip(self.names, self.P) if n.endswith(":g")]          # N: RMSNorm gains at lr / 10
        sc = [p for n, p in zip(self.names, self.P) if not n.endswith((":la", ":lb", ":g"))]
        lo = [p for n, p in zip(self.names, self.P) if n.endswith((":la", ":lb"))]
        self.scal = sc + nrm
        scg = lr_groups(a, [n for n in self.names if not n.endswith((":la", ":lb", ":g"))], sc)   # lead 11-22 --lr-layers: consecutive runs of scales with one lr (parameter order kept)
        if a.lr_layers: log(f"--lr-layers {a.lr_layers}: " + ", ".join(f"{len(g_)} vectors at lr x {b_ / a.lr:g}" for b_, g_ in scg))
        if a.wd > 0:       # #200: decoupled weight decay on the log multipliers (toward 0 = the stored scales); not on the norm gains
            groups = [{"params": g_, "base": b_, "weight_decay": a.wd} for b_, g_ in scg] + ([{"params": nrm, "base": a.lr / 10, "weight_decay": 0.0}] if nrm else [])
            self.opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, a.beta2), eps=a.eps, weight_decay=0.0)
        else:              # W = 0: #196's Adam, unchanged
            groups = [{"params": g_, "base": b_} for b_, g_ in scg] + ([{"params": nrm, "base": a.lr / 10}] if nrm else [])
            self.opt = torch.optim.Adam(groups, lr=a.lr, betas=(0.9, a.beta2), eps=a.eps)
        # L: the adapters' Adam runs on the CPU on f32 master copies (its state, 2 x 0.46 GB at r = 16, does not fit
        # beside a card's training peak); the card's adapters are overwritten with the masters after every step
        self.lo = lo; self.is_lo = [n.endswith((":la", ":lb")) for n in self.names]
        self.lo_m = [p.detach().cpu().clone().requires_grad_(True) for p in lo]
        self.opt_lo = torch.optim.Adam([{"params": self.lo_m, "base": a.lr_lora}], lr=a.lr_lora, betas=(0.9, a.beta2)) if lo else None
        self.gn_hist = []                                    # w220 --clip-gnorm: the raw global gradient norms of the earlier steps
        self.seg_hist = []; self.nclip_doc = 0               # lead 11-22 --clip-doc: recent per-row segment gradient norms
        self.pin = []                                        # w220 --mask: partly masked vectors (mask, start value on the card)
        self.fmask = None
        if getattr(stu, "mask", None):
            parts = []
            for n, p in zip(self.names, self.P):
                m = stu.mask.get(n)
                parts.append(torch.ones(p.numel(), dtype=torch.float64) if m is None else m.reshape(-1).cpu().double())
                if m is not None: self.pin.append((p, m, p.detach().clone()))
            self.fmask = torch.cat(parts)
        self.mine = [s for s in range(a.V) if s % dist.world == dist.rank]
        self.states = {}                                     # stream -> (item index, offset, state on CPU)
        self.w225 = bool(getattr(stu, "split", 0) or getattr(stu, "kvbuf", False))     # w225: split / buffered-KV segment loop (run_segs_w225)
        tags = sorted(set(C.tags)); self.tags = tags; self.tix = {t: k for k, t in enumerate(tags)}

    def step_rows(self, s): return sum(self.rows_of[(it, a_, b_)] for st in self.steps[s] for (it, a_, b_) in st)

    def clip_seg(self, snap, m, N):
        """lead 11-22 (--clip-doc C): per-document (segment) gradient clipping. The segment's own gradient d (grad after its backward minus
        before) is measured per row (|d| x N / m: the segment loss is its KL sum / N); when that exceeds C x the running median of the
        recent segments' per-row norms, d is scaled down to the threshold. Bounds the few documents that dominate the step's gradient
        (w218: chunk norms vary ~35x) without touching the rest. History: the last 2,000 segments; no clipping before 32 are seen."""
        with torch.no_grad():
            dl = [p.grad - (s0 if s0 is not None else 0.0) if p.grad is not None else None for p, s0 in zip(self.P, snap)]
            nrm = math.sqrt(sum(float((x.float() ** 2).sum()) for x in dl if x is not None)); u = nrm * N / max(m, 1)
            h = self.seg_hist
            if len(h) >= 32:
                thr = self.a.clip_doc * float(np.median(h))
                if u > thr:
                    f = thr / u; self.nclip_doc += 1
                    for p, s0, x in zip(self.P, snap, dl):
                        if x is not None: p.grad = (s0 if s0 is not None else 0.0) + f * x
            h.append(u)
            if len(h) > 2000: del h[0]

    def run_group(self, ss, lr):
        """one optimizer step over the consecutive plan steps ss (#200: --accum micro-batches) at lr; returns per-tag (KL sum,
        rows) over all ranks and the wall. Each micro-batch's KL is divided by the step's total rows (see the header)"""
        a, stu = self.a, self.stu
        for o_ in (self.opt, self.opt_lo):
            if o_ is None: continue
            for g in o_.param_groups: g["lr"] = lr * g.get("base", self.a.lr) / self.a.lr     # adapters keep their ratio
            o_.zero_grad(set_to_none=True)
        for p in self.lo: p.grad = None
        N = sum(self.step_rows(s_) for s_ in ss)                # the optimizer step's rows (all its micro-batches)
        acc = torch.zeros(2 * len(self.tags) + 2, dtype=torch.float64)      # per tag (KL, rows), T term sum, prefix tokens
        t0 = time.time()
        self.nclip_doc = 0
        for s, sid in ([] if self.w225 else [(s_, q_) for s_ in ss for q_ in self.mine]):
            for (it, sa, sb) in self.steps[s][sid]:
                gi, i = self.C.items[it][0], self.C.items[it][1]
                prev = None
                snap = [None if p.grad is None else p.grad.detach().clone() for p in self.P] if a.clip_doc > 0 else None
                if sa > 0:
                    k = self.states.get(sid)
                    if k is not None and k[0] == it and k[1] == sa: prev = to_dev(k[2], self.dev)
                    else: prev = prefix_state(stu, self.C.g, (gi, i), sa, a.len); acc[-1] += sa
                if self.a.prompt_weight > 0 or self.loc is not None:
                    loss, kl, m, n_all, nxt, ls = seg_loss_ext(stu, self.C.g, (gi, i), sa, sb, prev, self.a.prompt_weight, self.loc)
                    if n_all: (loss / N).backward()
                    acc[-2] += float(ls)
                elif self.tap is not None:
                    kl, m, nxt, ts = seg_loss(stu, self.C.g, (gi, i), sa, sb, prev, tap=self.tap)
                    if m: ((kl + a.tap_weight * ts) / N).backward()
                    acc[-2] += float(ts)
                else:
                    kl, m, nxt = seg_loss(stu, self.C.g, (gi, i), sa, sb, prev)
                    if m: (kl / N).backward()
                if snap is not None and m: self.clip_seg(snap, m, N)
                t = self.tix[self.C.tags[gi]]; acc[2 * t] += float(kl); acc[2 * t + 1] += m
                n_doc = self.C.items[it][2]
                if sb < n_doc: self.states[sid] = (it, sb, to_cpu(nxt))
                else: self.states.pop(sid, None)
                del prev, nxt
        if self.w225: self.run_segs_w225(ss, N, acc)
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).double().cpu() for p in self.P] + [acc])
        self.d.allsum(flat)
        if self.fmask is not None: flat[:len(self.fmask)] *= self.fmask         # w220 --mask: masked scales get gradient 0
        nflat = sum(p.numel() for p in self.P)
        self.gnorm = float(flat[:nflat].norm())                 # the optimizer step's gradient norm (logged)
        self.clip = None
        if a.clip_gnorm > 0:                                    # w220: clip the global norm at X x the running median norm
            f, thr = clip_factor(self.gn_hist, self.gnorm, a.clip_gnorm)
            self.gn_hist.append(self.gnorm)
            if f < 1.0: flat[:nflat] *= f; self.clip = (f, thr)
        o = 0; k = 0
        for p, il in zip(self.P, self.is_lo):
            n = p.numel()
            if il: p.grad = None; self.lo_m[k].grad = flat[o:o + n].view_as(p).float(); k += 1
            else: p.grad = flat[o:o + n].view_as(p).to(p.device, torch.float32)
            o += n
        accs = flat[o:]
        self.opt.step()
        if self.opt_lo is not None:
            self.opt_lo.step()
            with torch.no_grad():
                for p, m in zip(self.lo, self.lo_m): p.copy_(m.to(p.device, non_blocking=True)); m.grad = None
        with torch.no_grad():
            for p, m, p0 in self.pin: p.copy_(torch.where(m, p, p0))        # masked scales stay at their start value (also under --wd)
            for p in self.scal: p.clamp_(-a.trust, a.trust)                 # the log multipliers only (not adapters)
        return accs, time.time() - t0

    def run_step(self, s, lr): return self.run_group([s], lr)

    # ---------------------------------------------------------------- w225: layer split, buffered KV, depth test
    def layer_dev(self, key):
        return self.stu.dev_of[key[1]] if hasattr(self.stu, "dev_of") else self.stu.devs[0]

    def place(self, st):
        """a stored state on its layers' cards: a KV buffer kept on its card is used in place (SegCacheBuf.update grows it when the segment needs
        room), a parked one is copied into a buffer with room for the next segment (kv_room, at most the document's capacity stu.kv_cap)"""
        cap = self.stu.kv_cap; out = {}
        for key, v in st.items():
            dev = self.layer_dev(key)
            if isinstance(v, KVView): out[key] = v if v.device == dev else v.to(dev, kv_room(v.n + self.a.Q, cap))
            elif isinstance(v, tuple): out[key] = tuple(t.to(dev, non_blocking=True) for t in v)
            else: out[key] = v.to(dev, non_blocking=True)
        return out

    @staticmethod
    def st_bytes(st):
        out = {}
        for v in st.values():
            for t in ((v.k, v.v) if isinstance(v, KVView) else v if isinstance(v, tuple) else (v,)):
                out[t.device] = out.get(t.device, 0) + t.numel() * t.element_size()
        return out

    def resident(self, skip=None):
        """bytes of the carried states kept on each card (KV buffers at capacity)"""
        used = {}
        for s2, (_, _, st2) in self.states.items():
            if s2 == skip: continue
            for d_, b_ in self.st_bytes(st2).items():
                if d_.type != "cpu": used[d_] = used.get(d_, 0) + b_
        return used

    def park(self, sid, nxt):
        """--state-card-gb G: keep a stream's next state on its cards when every card's kept states stay within G GiB, else park it on the CPU
        (KV compacted to its valid part). G = 0: always park (today's CPU park)"""
        G = getattr(self.a, "state_card_gb", 0.0) * 2**30
        if G > 0:
            used = self.resident(skip=sid)
            if all(d_.type == "cpu" or used.get(d_, 0) + b_ <= G for d_, b_ in self.st_bytes(nxt).items()): return nxt
        return {k: (v.to(torch.device("cpu")) if isinstance(v, KVView) else tuple(t.cpu() for t in v) if isinstance(v, tuple) else v.cpu())
                for k, v in nxt.items()}

    def synth(self):
        """--depth-test D: the state before a document's token 0 = D synthetic positions: zero K / V in every attention layer (on its card, in a
        buffer of the document's capacity), the GDN layers empty"""
        stu = self.stu; cfg = stu.cfg; D_ = stu.pos_off; out = {}
        for l, t in enumerate(cfg.layer_types[:len(stu.layers)]):
            if t != "full_attention": continue
            k = torch.zeros(1, cfg.num_key_value_heads, max(stu.kv_cap, D_), cfg.head_dim, dtype=BF, device=self.layer_dev(("kv", l)))
            out[("kv", l)] = KVView(k, torch.zeros_like(k), D_)
        return out

    def state_in(self, sid, it, sa, acc):
        """the carried state of stream sid's segment starting at sa, on its layers' cards: the stored one, else a no-grad rebuild over the prefix
        (counted in acc[-1]); at a document's start None (--depth-test: the synthetic prefix)"""
        stu = self.stu
        st0 = lambda: self.synth() if getattr(stu, "pos_off", 0) else None       # built only when used (a 256k synthetic prefix is 8 GiB a card)
        if sa == 0: return st0()
        k = self.states.get(sid)
        if k is not None and k[0] == it and k[1] == sa: return self.place(k[2])
        acc[-1] += sa
        return prefix_state(stu, self.C.g, tuple(self.C.items[it][:2]), sa, self.a.len, st0())

    def state_out(self, sid, it, sb, n_doc, nxt):
        if sb < n_doc: self.states[sid] = (it, sb, self.park(sid, nxt))
        else: self.states.pop(sid, None)

    def run_segs_w225(self, ss, N, acc):
        """the segment loop of run_group for --split-layers (naive split: one connected graph over both cards), --state-card-gb and --depth-test
        (buffered KV, states kept on the cards); --pipeline: run_pipe. Same segments, order, losses and state carry as the default loop"""
        a, stu = self.a, self.stu
        assert not (a.prompt_weight > 0 or self.loc is not None or self.tap is not None), "w225 loop: no --prompt-weight / --local-mse / taps"
        work = [(sid, it, sa, sb) for s_ in ss for sid in self.mine for (it, sa, sb) in self.steps[s_][sid]]
        if getattr(a, "pipeline", False): self.run_pipe(work, N, acc); stu.kv_cap = 0; return
        for sid, it, sa, sb in work:
            gi, i, n_doc = self.C.items[it][:3]
            snap = [None if p.grad is None else p.grad.detach().clone() for p in self.P] if a.clip_doc > 0 else None
            stu.kv_cap = n_doc + getattr(stu, "pos_off", 0)
            prev = self.state_in(sid, it, sa, acc)
            with devctx(stu.devs[-1]):
                kl, m, nxt = seg_loss(stu, self.C.g, (gi, i), sa, sb, prev)
                if m: (kl / N).backward()
            if snap is not None and m: self.clip_seg(snap, m, N)
            t = self.tix[self.C.tags[gi]]; acc[2 * t] += float(kl); acc[2 * t + 1] += m
            self.state_out(sid, it, sb, n_doc, nxt)
            del prev, nxt
        stu.kv_cap = 0

    def run_pipe(self, work, N, acc):
        """--pipeline (with --split-layers): the filled two-card pipeline over a step's segments. Card A (this thread) runs segment j's first half
        forward while card B (a worker thread) runs segment j - 2's second half forward + loss + backward; then A runs j - 2's first-half backward
        seeded with the hidden-state gradient B sent back. The graph is the naive split's cut at the card boundary (B's input = A's output
        detached), each card accumulates its parameters' gradients in segment order, and states / losses are booked in segment order, so the
        step equals the naive split's bit for bit"""
        import queue, threading
        stu, a = self.stu, self.a
        A, B, K, NL = stu.devs[0], stu.devs[-1], stu.split, len(stu.layers)
        assert a.clip_doc == 0, "--pipeline: no --clip-doc"
        qb, qg = queue.Queue(), queue.Queue()
        tm = self.pipe_s = dict.fromkeys(("a_fwd", "a_wait", "a_bwd", "a_out", "b_wait", "b_fwd", "b_bwd"), 0.0)     # host-side seconds per phase (logged as pipe_s)
        sync = (lambda d_: torch.cuda.synchronize(d_)) if os.environ.get("W225_SYNC") == "1" and A.type == "cuda" else (lambda d_: None)   # profiling: each phase waits for its card (GPU time per phase)
        def side_b():
            try:
                if B.type == "cuda": torch.cuda.set_device(B)
                while True:
                    t_ = time.time(); job = qb.get(); tm["b_wait"] += time.time() - t_
                    if job is None: return
                    t_ = time.time()
                    j, hA, prevB, gi, i, sa, sb, cap = job
                    xB = hA.detach().to(B)
                    if hA.requires_grad: xB.requires_grad_(True)
                    cache = SegCacheBuf(prevB, cap)
                    h = stu.fnorm(run_half(stu, xB, sa, cache, K, NL))[0]
                    kl, m = seg_kl(stu, self.C.g[gi], i, sa, sb, h)
                    sync(B); t1 = time.time(); tm["b_fwd"] += t1 - t_
                    if m: (kl / N).backward()
                    sync(B); tm["b_bwd"] += time.time() - t1
                    qg.put((j, xB.grad if (m and hA.requires_grad) else None, float(kl.detach()), m, cache.next))
                    del job, hA, xB, h, kl, cache, prevB
            except BaseException as e:
                qg.put((None, e))
        import contextvars                      # torch's dynamo / inductor configs (recompile limit, deterministic, strict numerics) are context variables: the worker runs in a copy of this thread's context
        th = threading.Thread(target=contextvars.copy_context().run, args=(side_b,), name="w225-card-B", daemon=True); th.start()
        pend = collections.deque()
        def finish():
            j, sid, it, sb, n_doc, gi, hA, cacheA, _ = pend.popleft()
            t_ = time.time(); r = qg.get(); tm["a_wait"] += time.time() - t_
            if r[0] is None: raise r[1]
            assert r[0] == j, (r[0], j)
            _, gx, kl, m, nxtB = r
            t_ = time.time()
            if gx is not None: hA.backward(gx.to(A))
            sync(A); t1 = time.time(); tm["a_bwd"] += t1 - t_
            t = self.tix[self.C.tags[gi]]; acc[2 * t] += kl; acc[2 * t + 1] += m
            self.state_out(sid, it, sb, n_doc, {**cacheA.next, **nxtB})
            tm["a_out"] += time.time() - t1
        try:
            with devctx(A):
                for j, (sid, it, sa, sb) in enumerate(work):
                    while any(p_[1] == sid for p_ in pend): finish()     # the stream's previous segment (its state) first (--accum, V = 1)
                    gi, i, n_doc = self.C.items[it][:3]
                    cap = n_doc + getattr(stu, "pos_off", 0); stu.kv_cap = cap
                    t_ = time.time()
                    k_ = self.states.get(sid)
                    kept = sa > 0 and k_ is not None and k_[0] == it and k_[1] == sa and all(d_.type != "cpu" for d_ in self.st_bytes(k_[2]))
                    prev = self.state_in(sid, it, sa, acc) or {}
                    nb = 0 if kept else sum(self.st_bytes(prev).values())         # a state brought to the cards for this segment (parked / rebuilt / synthetic)
                    while pend and sum(p_[-1] for p_ in pend) + nb > PIPE_INFLIGHT: finish()   # deep states: few in flight (a 256k stream's KV is 16 GiB)
                    pa = {k: v for k, v in prev.items() if k[1] < K}; pb = {k: v for k, v in prev.items() if k[1] >= K}
                    cacheA = SegCacheBuf(pa, cap)
                    hA = run_half(stu, stu.embed(self.C.g[gi].ids(i, sa, sb)), sa, cacheA, 0, K)
                    qb.put((j, hA, pb, gi, i, sa, sb, cap)); pend.append((j, sid, it, sb, n_doc, gi, hA, cacheA, nb)); sync(A); tm["a_fwd"] += time.time() - t_
                    del prev, pa, pb, hA, cacheA
                    if len(pend) > 2: finish()       # A runs two segments ahead: B always has the next one queued (B's work per segment is the larger half)
                while pend: finish()
        finally:
            qb.put(None); th.join()

    def snapshot(self):
        import copy
        def cp(o_):
            sd = o_.state_dict()
            return {"state": {k: {n: (v.detach().cpu().clone() if torch.is_tensor(v) else v) for n, v in s.items()} for k, s in sd["state"].items()},
                    "param_groups": copy.deepcopy(sd["param_groups"])}
        z = {"cur": self.stu.state(), "opt": cp(self.opt), "gn_hist": list(self.gn_hist), "seg_hist": list(self.seg_hist), "opt_order": self.opt_order()}
        if self.opt_lo is not None: z["opt_lo"] = cp(self.opt_lo)
        return z

    def opt_order(self):
        """the trainable parameter names in the main optimizer's index order (the scale vectors, then the norm gains)"""
        return [n for n in self.names if not n.endswith((":la", ":lb", ":g"))] + [n for n in self.names if n.endswith(":g")]

    def restore(self, z):
        self.stu.load_state(z["cur"]); self.states = {}; self.gn_hist = list(z.get("gn_hist", [])); self.seg_hist = list(z.get("seg_hist", [])); self.partial_restore = False
        if "opt_order" in z and z["opt_order"] != self.opt_order():     # w222: a fork into a different trainable set (staged MSE -> KL): Adam state by name, fresh for the rest
            new = map_opt_state(z["opt"], z["opt_order"], self.opt_order(), self.opt.state_dict()["param_groups"])
            self.opt.load_state_dict(new); self.partial_restore = True
            self.log(f"partial restore: Adam state kept for {len(new['state'])} of {len(self.opt_order())} parameters (the state's {len(z['opt_order'])}), fresh for the rest")
        else: self.opt.load_state_dict(z["opt"])
        if self.opt_lo is not None:
            self.opt_lo.load_state_dict(z["opt_lo"])
            with torch.no_grad():
                for p, m in zip(self.lo, self.lo_m): m.copy_(p.detach().cpu())

    @torch.no_grad()
    def monitor(self, mon):
        """KL / row over the monitor docs (first --len tokens), split over ranks"""
        s = torch.zeros(2, dtype=torch.float64)
        for k, (gi, i) in enumerate(mon):
            if k % self.d.world != self.d.rank: continue
            n = min(self.a.len, self.C.g[gi].ntok(i))
            kl, m, _ = seg_loss(self.stu, self.C.g, (gi, i), 0, n, None)
            s[0] += float(kl); s[1] += m
        self.d.allsum(s)
        return float(s[0] / s[1])

    @torch.no_grad()
    def fixed(self, fe):
        """per-doc (KL sum, rows) of the fixed eval docs (first --len tokens), split over ranks"""
        out = torch.zeros(2 * len(fe), dtype=torch.float64)
        for k, (gi, i) in enumerate(fe):
            if k % self.d.world != self.d.rank: continue
            n = min(self.a.len, self.C.g[gi].ntok(i))
            kl, m, _ = seg_loss(self.stu, self.C.g, (gi, i), 0, n, None)
            out[2 * k] = float(kl); out[2 * k + 1] = m
        self.d.allsum(out)
        return out.view(-1, 2).numpy()

    def checksum(self):
        v = torch.tensor([float(sum(p.detach().double().sum().to(self.dev) for p in self.P))], dtype=torch.float64)    # w225: parameters on two cards
        mine = float(v); self.d.allsum(v)
        assert abs(float(v) / self.d.world - mine) <= 1e-9 * max(1.0, abs(mine)), f"ranks out of sync: {mine} vs mean {float(v) / self.d.world}"
        return mine


R196_KEEP = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "linear_attn.out_proj")


def apply196(a, stu, log):
    """#196 switches on the built student: R (factor freeze), U (un-shrink start values), N (norm gains). Each is a no-op
    when its flag is off, so the default path is #191's arm A."""
    if a.factors == "attn_ssmout":
        off = 0; kept = 0
        for k, ql in stu.qlin.items():
            if k.endswith(tuple("." + s for s in R196_KEEP)): kept += 1; continue
            for nm in ("r", "c"):
                if getattr(ql, nm) is not None:
                    stu.P.pop(f"{k}:{nm}", None); setattr(ql, nm, None); off += 1
        log(f"arm R (#196 factors attn_ssmout): {kept} linears keep their row scales, {off} row-scale vectors frozen")
    if a.unshrink:
        from safetensors.torch import load_file
        C = load_file(a.unshrink)
        assert set(C) <= set(stu.qlin), sorted(set(C) - set(stu.qlin))[:3]
        cs = []
        for k, c in C.items():
            ql = stu.qlin[k]; c = c.float()
            assert tuple(c.shape) == (ql.n,), (k, tuple(c.shape), ql.n)
            assert bool((c > 0).all()), k
            lc = torch.log(c).to(ql.svh.device)
            with torch.no_grad():
                if ql.c is not None: ql.c.copy_(lc)
                else: ql.c = torch.nn.Parameter(lc, requires_grad=False)      # output side not trained: a fixed start value
            cs.append(c)
        allc = torch.cat(cs)
        log(f"U (#196 un-shrink, {a.unshrink}): {len(C)} linears; c mean {float(allc.mean()):.5f}, min {float(allc.min()):.4f}, "
            f"max {float(allc.max()):.4f}, at the clamp {float(((allc <= 0.9 + 1e-6) | (allc >= 1.2 - 1e-6)).float().mean()):.4f}")
    if a.norms:
        n0 = len(stu.norm)
        for i, ly in enumerate(stu.layers):
            pre = f"{stu.L}.{i}."
            mods = [("input_layernorm", ly.input_layernorm), ("post_attention_layernorm", ly.post_attention_layernorm)]
            sa = getattr(ly, "self_attn", None)
            if sa is not None: mods += [("self_attn.q_norm", sa.q_norm), ("self_attn.k_norm", sa.k_norm)]
            la = getattr(ly, "linear_attn", None)
            if la is not None: mods += [("linear_attn.norm", la.norm)]
            for nm, m in mods: stu._add_norm(f"{pre}{nm}.weight", m)
        stu._add_norm(f"{stu.LP}norm.weight", stu.fnorm)
        log(f"N (#196 norm gains): {len(stu.norm) - n0} RMSNorm modules, gains at lr / 10")


def write_ckpt(stu, out, meta):
    from safetensors.torch import save_file
    os.makedirs(out, exist_ok=True)
    sd = os.path.join(out, "scales"); os.makedirs(sd, exist_ok=True)
    for k, q in stu.qlin.items():
        with torch.no_grad():
            r = q.r.exp().cpu() if q.r is not None else torch.ones(q.k)
            c = q.c.exp().cpu() if q.c is not None else torch.ones(q.n)
        save_file({"r": r.float().contiguous(), "c": c.float().contiguous()}, f"{sd}/{k}.safetensors")
    hq = getattr(stu, "hq", None)
    if hq is not None and hq.r is not None and hq.r.requires_grad:             # w222 (H): the trained head scales (r always; c = 1 when only the input side was trained), stored row order
        with torch.no_grad():
            save_file({"r": hq.r.exp().float().cpu().contiguous(), "c": (hq.c.exp().float().cpu() if hq.c is not None else torch.ones(hq.n)).contiguous()}, f"{sd}/lm_head.safetensors")
    lo = {k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in stu.P.items() if k.endswith((":la", ":lb"))}
    if lo: save_file(lo, os.path.join(out, "lora.safetensors"))
    if getattr(stu, "norm", None):                                          # N (#196): the trained norm weights, as the checkpoint stores them
        nw = {}
        with torch.no_grad():
            for k, m in stu.norm.items():
                g = m._qg.detach().float().exp().cpu(); w = m.weight.detach().float().cpu()
                new = (1.0 + w) * g - 1.0 if type(m).__name__ == "Qwen3_5RMSNorm" else w * g      # Qwen3_5RMSNorm: norm * (1 + w)
                nw[k] = new.to(torch.bfloat16).contiguous()
        save_file(nw, os.path.join(out, "norms.safetensors"))
    save_atomic({"state": stu.state(), "meta": meta}, os.path.join(out, "params.pt"))


# ------------------------------------------------------------------ gate
def profile(a, stu, C, log):
    """torch.profiler over two 1,024-token segment fwd + bwd: the top ops by device time -> OUT/profile.txt"""
    from torch.profiler import profile as tp, ProfilerActivity
    dev = stu.devs[0]
    gi, i = next((gi, i) for gi, i, n, nr in C.items if n >= 2048)
    for rep in range(3):
        if rep == 2: P = tp(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]); P.__enter__()
        torch.cuda.synchronize(dev); t0 = time.time()
        kl, m, _ = seg_loss(stu, C.g, (gi, i), 0, 1024, None); (kl / m).backward(); torch.cuda.synchronize(dev)
        for p in stu.P.values(): p.grad = None
        log(f"profile rep {rep}: {time.time() - t0:.2f} s")
    P.__exit__(None, None, None)
    t = P.key_averages().table(sort_by="self_cuda_time_total" if hasattr(P.key_averages()[0], "self_cuda_time_total") else "self_device_time_total", row_limit=45)
    open(os.path.join(a.out, "profile.txt"), "w").write(t); log(t)


def bench(a, stu, C, log):
    """wall of 1,024-token segment fwd + bwd (4 reps after 2 warm-ups) and its loss / grad norm (for flag comparisons)"""
    dev = stu.devs[0]; gi, i = next((gi, i) for gi, i, n, nr in C.items if n >= 2048)
    ts = []
    for rep in range(6):
        torch.cuda.synchronize(dev); t0 = time.time()
        kl, m, _ = seg_loss(stu, C.g, (gi, i), 0, 1024, None); (kl / m).backward(); torch.cuda.synchronize(dev)
        g = sum(float(p.grad.double().norm()) ** 2 for p in stu.P.values() if p.grad is not None) ** 0.5
        for p in stu.P.values(): p.grad = None
        if rep >= 2: ts.append(time.time() - t0)
    log(f"bench: {np.mean(ts):.3f} s a 1,024-token segment (fwd + bwd, {len(ts)} reps), KL/row {float(kl) / m:.6f}, grad norm {g:.6e}")


def gate(a, stu, C, log):
    """(1) carry exactness: one 2,048-token window vs 1,024 + 1,024 with the carried state, and vs 3 uneven segments;
    (2) fwd + bwd wall and peak memory of a 1,024 / 2,048-token segment and of a 1,024 segment after a 6k prefix"""
    dev = stu.devs[0]
    its = [x for t in sorted(set(C.tags)) for x in [(gi, i) for gi, i, n, nr in C.items if n >= 2048 and nr >= 1500 and C.tags[gi] == t][:1]]
    for (gi, i) in its:
        with torch.no_grad():
            full, m, _ = seg_loss(stu, C.g, (gi, i), 0, 2048, None, per_row=True)
            k1, m1, s1 = seg_loss(stu, C.g, (gi, i), 0, 1024, None, per_row=True)
            k2, m2, _ = seg_loss(stu, C.g, (gi, i), 1024, 2048, s1, per_row=True)
            u = [0, 300, 1500, 2048]; ks, st = [], None
            for p, q in zip(u[:-1], u[1:]):
                k, _, st = seg_loss(stu, C.g, (gi, i), p, q, st, per_row=True); ks.append(k)
            again, _, _ = seg_loss(stu, C.g, (gi, i), 0, 2048, None, per_row=True)
        two = torch.cat([k1, k2]); three = torch.cat(ks); j = torch.from_numpy(C.g[gi].scored(i, 0, 2048)[0]).to(two.device)
        dd = (two - full).abs(); top = torch.topk(dd, min(5, len(dd)))
        log(f"gate carry {C.g[gi].name}:{i}: KL/row whole {float(full.mean()):.6f} (rerun {float(again.mean()):.6f}, max |diff| "
            f"{float((again - full).abs().max()):.2e}), 2 segments {float(two.mean()):.6f}, 3 uneven {float(three.mean()):.6f}; "
            f"rel mean diff {float((two - full).mean() / full.mean()):+.2e} / {float((three - full).mean() / full.mean()):+.2e}; "
            f"largest |diff| at positions {[int(j[x]) for x in top.indices]}: {[round(float(x), 3) for x in top.values]} "
            f"(whole {[round(float(full[x]), 3) for x in top.indices]}); median |diff| {float(dd.median()):.2e}; first segment "
            f"(no carry: shape noise) rel {float((two - full)[j < 1024].mean() / full[j < 1024].mean()):+.2e} median "
            f"{float(dd[j < 1024].median()):.2e}, second (carried) rel {float((two - full)[j >= 1024].mean() / full[j >= 1024].mean()):+.2e} "
            f"median {float(dd[j >= 1024].median()):.2e}; alloc "
            f"{torch.cuda.memory_allocated(dev) / 2**30:.2f} peak {torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GiB")
    big = max(C.items, key=lambda x: x[2])
    for (L, pre) in ((1024, 0), (1536, 0), (1024, 7168)):
        gi, i = big[0], big[1]
        torch.cuda.reset_peak_memory_stats(dev)
        for rep in range(2):
            prev = prefix_state(stu, C.g, (gi, i), pre, a.len) if pre else None
            torch.cuda.synchronize(dev); t0 = time.time()
            kl, m, _ = seg_loss(stu, C.g, (gi, i), pre, pre + L, prev)
            torch.cuda.synchronize(dev); tf = time.time() - t0
            (kl / m).backward(); torch.cuda.synchronize(dev); tb = time.time() - t0 - tf
            for p in stu.P.values(): p.grad = None
        log(f"gate segment {L} after {pre}: fwd {tf:.2f} s, bwd {tb:.2f} s -> {L / (tf + tb):.0f} positions/s, peak "
            f"{torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GiB")


# ------------------------------------------------------------------ main
# ------------------------------------------------------------------ #191: the served form's pieces, data order, B, T
def build_student(a, dev, log, lora=None, layers=None, quant_layers=None, served=True):
    """qmodel.Student on dev(s) (one build at a time on the host when --build-lock: a build holds the decoded trunk in
    host RAM), then the served token table (--emb) and GDN a / b weights (--ab) when served"""
    import fcntl
    lk = open(a.build_lock, "w") if a.build_lock else None
    if lk: fcntl.flock(lk, fcntl.LOCK_EX)
    try:
        t1 = time.time(); devs = dev if isinstance(dev, list) else [dev]
        stu = qmodel.Student(a.model, a.level, devs, log=log, head_dir=a.head if served else None,
                             head_perm=a.head_perm if served else None, lora=a.lora if lora is None else lora,
                             layers=layers, quant_layers=quant_layers, head_train={"none": False, "in": "in", "both": True}[getattr(a, "head_train", "none")] if served else False,
                             split=getattr(a, "split_layers", 0))
        log(f"student built in {time.time() - t1:.0f} s: {len(stu.qlin)} linears ({stu.cb}, {sum(q.exact for q in stu.qlin.values())} "
            f"int8-exact), {sum(p.numel() for p in stu.P.values())} values, alloc {torch.cuda.memory_allocated(devs[0]) / 2**30:.2f} GiB")
    finally:
        if lk: fcntl.flock(lk, fcntl.LOCK_UN); lk.close()
    stu.emb_bf16 = stu.emb                                    # the checkpoint's table (the B stage's teacher input)
    if served and a.emb:
        stu.emb = np.memmap(a.emb, np.int16, "r", shape=stu.emb_bf16.shape)
        r = [0, 1000, 248070, 248319]
        d = np.abs(stu.emb[r].view(np.uint16).astype(np.int64) - stu.emb_bf16[r].view(np.uint16).astype(np.int64))
        log(f"token table: {a.emb} (served form; rows {r}: {int((d > 0).sum())} of {d.size} bf16 values differ from the checkpoint's)")
    if served and a.ab:
        from safetensors import safe_open
        rel = []
        with safe_open(a.ab, "pt") as f:
            for i, ly in enumerate(stu.layers):
                if stu.cfg.layer_types[i] != "linear_attention": continue
                for nm, key in (("in_proj_a", f"blk.{i}.ssm_a.weight"), ("in_proj_b", f"blk.{i}.ssm_b.weight")):
                    p = getattr(ly.linear_attn, nm).weight; w = f.get_tensor(key).to(p.device, p.dtype)
                    assert w.shape == p.shape, (key, w.shape, p.shape)
                    rel.append(float((w.float() - p.float()).norm() / p.float().norm())); p.data.copy_(w)
        assert max(rel) < 0.2, f"a / b weights far from the checkpoint's (max rel {max(rel):.3f}): wrong mapping?"
        log(f"GDN a / b: {a.ab} ({len(rel)} tensors, rel diff vs the checkpoint median {np.median(rel):.4f}, max {max(rel):.4f})")
    return stu


def mix50_order(C, seed):
    """D+: one seeded shuffle of the calib qad-pool documents and one of the rest (xcorp, traces), merged so that the
    cumulative gen rows of the two halves stay equal (the next document comes from the half with fewer rows so far)"""
    rng = np.random.default_rng(seed)
    cal = [k for k, x in enumerate(C.items) if C.tags[x[0]] == "calib"]; xs = [k for k, x in enumerate(C.items) if C.tags[x[0]] != "calib"]
    rng.shuffle(cal); rng.shuffle(xs)
    out, ca, xa, ic, ix = [], 0, 0, 0, 0
    while ic < len(cal) or ix < len(xs):
        if ix >= len(xs) or (ic < len(cal) and ca <= xa): k = cal[ic]; ic += 1; ca += C.items[k][3]
        else: k = xs[ix]; ix += 1; xa += C.items[k][3]
        out.append(k)
    return np.array(out)


def build_data(a, d, log):
    """corpus, fixed eval hold-out, document order (D), stream plan, rows, schedule (qad181's, factored out)"""
    chunks = [x.strip() for x in (open(a.chunks[1:]).read().split() if a.chunks.startswith("@") else a.chunks.split(",")) if x.strip()]
    t0 = time.time()
    C = D.Corpus(chunks, a.cap, calib=a.calib, pools=a.pools, pool=a.pool, genmask=a.genmask,
                 cache=os.path.join(a.out, "items.json")) if d.rank == 0 else None
    d.barrier()
    if d.rank: C = D.Corpus(chunks, a.cap, calib=a.calib, pools=a.pools, pool=a.pool, genmask=a.genmask, cache=os.path.join(a.out, "items.json"))
    keep8 = set(json.load(open(a.cap_keep))) if a.cap_keep else set()    # lead 11-23: calib docs that stay at --cap-keep-len (the long-context judge's docs)
    calib_gi = len(C.g) - 1 if a.calib else None
    def capfix(gi, i, n, nr):
        if gi == calib_gi and i in keep8 and n > a.cap_keep_len:
            g = C.g[gi]; return (gi, i, a.cap_keep_len, g.nrows(i, a.cap_keep_len))
        return (gi, i, n, nr)
    if keep8:
        nb = sum(1 for x in C.items if x[0] == calib_gi and x[1] in keep8 and x[2] > a.cap_keep_len)
        C.items = [capfix(*x) for x in C.items]; log(f"--cap-keep: {nb} base items cut back to {a.cap_keep_len} tokens ({len(keep8)} listed calib docs)")
    log(f"corpus {len(C.items)} documents in {time.time() - t0:.0f} s: {json.dumps(C.summary())}")
    log(f"chunk list: {len(chunks)} golden dirs, sha1 {hashlib.sha1(chr(10).join(chunks).encode()).hexdigest()[:12]} ({chunks[0]} .. {chunks[-1]})")
    fe = []
    if a.fixed_eval:
        assert a.calib, "--fixed-eval needs --calib"
        gc = len(C.g) - 1
        pl = json.load(open(a.pools))["pool"]
        cands = [i for i, p in enumerate(pl) if p == a.pool]
        rng = np.random.default_rng(a.fe_seed); rng.shuffle(cands); tot = 0
        for i in cands:
            nr = len(C.g[gc].scored(i, 0, min(a.len, C.g[gc].ntok(i)))[0])
            if nr >= 200: fe.append((gc, i)); tot += nr
            if tot >= a.fixed_eval: break
        fes = {i for _, i in fe}
        before = len(C.items); C.items = [x for x in C.items if not (x[0] == gc and x[1] in fes)]
        log(f"fixed eval: {len(fe)} qad-pool docs, {tot} rows (first {a.len} tokens), held out of training ({before - len(C.items)} items removed)")
        json.dump([i for _, i in fe], open(os.path.join(a.out, "fixed-eval-docs.json"), "w"))
    if a.plan_ext:                                    # w222 (ledger use qad214-ext): w214's plan + new documents appended after it, w214's order as the prefix of the new order
        px = json.load(open(a.plan_ext))
        if a.cap > px["cap"]:                         # w226: a document with < min_rows gen rows in its first cap tokens is not in the plan's base, whatever this run's cap
            nb = len(C.items); C.items = [x for x in C.items if C.g[x[0]].nrows(x[1], px["cap"]) >= px["min_rows"]]
            log(f"plan-ext at cap {a.cap} > the plan's {px['cap']}: {nb - len(C.items)} base items under {px['min_rows']} rows at cap {px['cap']} dropped (not in w214's base)")
        n_old = len(C.items)
        assert px["n_base_items"] == n_old and (px["cap"] == a.cap or a.cap > px["cap"]), f"--plan-ext: its base has {px['n_base_items']} items (cap {px['cap']}), this run's {n_old} (cap {a.cap}): not w214's data"
        relen = a.cap != px["cap"]                    # lead 11-23: a larger cap than the plan was built with: same documents and order, rows recounted at this cap
        for nm in px["new_goldens"]:
            C.g.append(D.Gold(os.path.join(a.ext_root, "gold", nm), genfile=os.path.join(a.ext_root, "genmask", nm + ".u8"))); C.tags.append(D.corpus_tag(nm))
        gidx = {g.name: k for k, g in enumerate(C.g)}
        for nm, i, nr in px["items"] + px.get("front_items", []):
            g = C.g[gidx[nm]]; r_ = g.nrows(i, a.cap); assert relen or r_ == nr, f"--plan-ext: {nm}:{i} has {r_} rows here, {nr} in the plan"
            C.items.append(capfix(gidx[nm], i, min(a.cap, g.ntok(i)), r_))
        n_front = len(px.get("front_items", [])); n_ext = len(C.items) - n_old - n_front
        log(f"plan-ext {a.plan_ext}: {n_ext} documents appended to w214's {n_old}, {n_front} front ({', '.join(px['new_goldens'])} + calib docs): {json.dumps(C.summary())}")
        # w226: front_items (stored after the extension in C.items) take the first places of the order, one per stream (deep documents need
        # the whole run: a stream trains Q tokens a step), then w214's order, then the extension; without front_items the order is w222's
        order = np.concatenate([n_old + n_ext + np.arange(n_front), np.random.default_rng(a.seed).permutation(n_old), n_old + np.arange(n_ext)])
    else:
        order = mix50_order(C, a.mix50) if a.mix50 is not None else np.random.default_rng(a.seed).permutation(len(C.items))
    EXACT_NAMES[0] = bool(getattr(a, "exact_ckpt_names", False))
    budgets = [BK(a.rows)] if a.rows else [BK(x) for x in a.budgets.split(",")]
    steps = D.plan(C.items, order, a.V, a.Q)
    rows_of, cum = rows_cum(C, steps)
    if a.lr_steps or a.decay_shape != "linear" or a.decay_data != "mix":
        assert len(budgets) == 1, "w220 schedule flags: one budget (no WSD branches)"
    fork_last = decay_start_step(a, cum, budgets[-1])
    if a.decay_data == "calib":                                  # w220: the cooldown (steps from fork_last) reads unused calib documents only
        assert not a.decay_shape.endswith("-all"), "--decay-data calib needs a constant phase (not a *-all shape)"
        steps = calib_cooldown(C, steps, fork_last, a.V, a.Q, a.seed)
        rows_of, cum = rows_cum(C, steps)
    last, br = schedule(cum, budgets, a.decay_frac)
    if a.short_decay:                                           # w222: a cheap decayed point at each listed row count: fork there, decay linearly from the run's lr over --short-decay-rows rows
        for Pk in [float(x) for x in a.short_decay.split(",")]:
            if Pk < budgets[-1]:
                br[SD(Pk)] = (int(np.searchsorted(cum, Pk, side="left")) + 1, min(len(cum), int(np.searchsorted(cum, Pk + a.short_decay_rows, side="left")) + 1))
    assert cum[-1] >= budgets[-1] or a.gate or a.profile, f"the data hold {cum[-1]} rows < budget {budgets[-1]}"
    share = {}
    for s in range(last):
        for segs in steps[s]:
            for (it, sa, sb) in segs: t = C.tags[C.items[it][0]]; share[t] = share.get(t, 0) + rows_of[(it, sa, sb)]
    if a.decay_data == "calib": log(f"decay data: calib only from plan step {fork_last} ({int(cum[fork_last - 1])} rows before)")
    log(f"plan: {len(steps)} steps of {a.V} x {a.Q} (data {cum[-1]} rows); main to step {last} ({cum[last - 1]} rows), decay from "
        f"step {fork_last}; branches " + ", ".join(f"{T:.0e}: fork {f} end {e} ({cum[e - 1]} rows)" for T, (f, e) in br.items())
        + f"; order {'mix50 seed ' + str(a.mix50) if a.mix50 is not None else 'shuffle seed ' + str(a.seed)} (sha1 "
        f"{__import__('hashlib').sha1(np.asarray(order, np.int64).tobytes()).hexdigest()[:12]}); gen rows to the budget by corpus "
        + json.dumps({t: [v, round(v / cum[last - 1], 4)] for t, v in sorted(share.items())}))
    return dict(C=C, fe=fe, order=order, budgets=budgets, steps=steps, rows_of=rows_of, cum=cum, last=last, br=br, fork_last=fork_last, share=share)


def layer_run(stu, ly, x, L, y=None, scale=1.0):
    """one decoder layer over a whole window (x [1, n, H] on the card, from position 0) in <= L-token segments that
    carry the layer's state (KV / GDN / conv, detached), as the end-to-end stage carries a document: no y -> the output
    [1, n, H]; y (the target [1, n, H], any device) -> the sum over positions of |s - y|^2 / |y|^2, each segment's part
    backpropagated (x scale) before the next segment runs (truncated backprop: a whole 8k window at once does not fit
    beside the student, 11-15)"""
    dev = x.device; n = x.shape[1]; st = None; outs = []; tot = 0.0
    k = -(-n // L); cut = [round(i * n / k) for i in range(k + 1)]          # near-equal segments (a 1-token one after a
    for p, q in zip(cut[:-1], cut[1:]):                                      # state would take GDN's no-backward path)
        pid = torch.arange(p, q, device=dev).view(1, 1, -1).expand(4, 1, -1)
        xs = x[:, p:q]; cache = SegCache(st)
        o = ly(xs, position_embeddings=stu.rotary[dev](xs, pid[1:]), attention_mask=None, position_ids=pid[0], past_key_values=cache)
        st = cache.next
        if y is None: outs.append(o); continue
        s = o[0].float(); tg = y[0, p:q].to(dev).float()
        l = ((s - tg).pow(2).sum(-1) / tg.pow(2).sum(-1).clamp_min(1e-12)).sum()
        (l * scale).backward(); tot += float(l); del o, s, tg, l
    return torch.cat(outs, 1) if y is None else tot


def teacher_layer(stu, i, dev):
    """the bf16 checkpoint's decoder layer i on dev (frozen)"""
    pre = f"{stu.L}.{i}."
    with torch.device("meta"): tl = stu.q35.Qwen3_5DecoderLayer(stu.cfg, i)
    tl.load_state_dict(stu._load([k for k in stu.wmap if k.startswith(pre)], pre), strict=True, assign=True)
    for f in stu._fh.values(): f.__exit__(None, None, None)
    stu._fh = {}
    return tl.to(dev).eval().requires_grad_(False)


def block_stage(a, stu, P, k0, log):
    """B: the documents of plan steps 0 .. k0 - 1 (each from its start to the furthest position those steps reach),
    one decoder layer at a time (each window in <= --len-token segments with the layer's state carried: layer_run): the
    student's layer i (its scales, and adapters under L) trained for one pass over
    the documents in batches of >= --block-batch tokens (fresh Adam, lr --block-lr, warmup 2 batches, linear decay to
    0 over the last --decay-frac of the batches; |log| <= --trust) on the per-token relative MSE |s - t|^2 / |t|^2 of
    its output s on the student's own trajectory (the trained layers before it) vs the bf16 layer's output t on the
    bf16 trajectory (checkpoint table in, checkpoint layers); then both trajectories advance one layer. OUT/block.pt
    (params after each layer) resumes; the trajectories are rebuilt through the finished layers."""
    C, steps = P["C"], P["steps"]; dev = stu.devs[0]
    reach, first = {}, []
    for s in range(k0):
        for segs in steps[s]:
            for (it, sa, sb) in segs:
                if it not in reach: assert sa == 0, (it, sa); first.append(it)
                reach[it] = max(reach.get(it, 0), sb)
    docs = [(C.items[it][0], C.items[it][1], reach[it]) for it in first]
    ntok = sum(n for _, _, n in docs)
    batches, cur, ct = [], [], 0
    for k, (_, _, n) in enumerate(docs):
        cur.append(k); ct += n
        if ct >= a.block_batch: batches.append(cur); cur, ct = [], 0
    if cur: batches.append(cur)
    log(f"B stage: {len(docs)} documents, {ntok} tokens (plan steps 0 .. {k0 - 1}, {int(P['cum'][k0 - 1])} gen rows), "
        f"{len(batches)} batches a layer")
    bp = os.path.join(a.out, "block.pt"); done = 0
    if os.path.exists(bp):
        z = torch.load(bp, weights_only=False); stu.load_state(z["state"]); done = z["layer"]; log(f"B stage: resumed after layer {done - 1}")
    if done >= len(stu.layers): return
    def table(emb, gi, i, n): return torch.from_numpy(np.ascontiguousarray(emb[C.g[gi].ids(i, 0, n)])).view(BF)[None]
    XS = [table(stu.emb, gi, i, n) for gi, i, n in docs]                   # the student's trajectory (served table)
    XT = [table(stu.emb_bf16, gi, i, n) for gi, i, n in docs]              # the bf16 trajectory
    def advance(i, ly_s, ly_t):
        with torch.no_grad():
            for k in range(len(docs)):
                XS[k] = layer_run(stu, ly_s, XS[k].to(dev), a.len).cpu()
                XT[k] = layer_run(stu, ly_t, XT[k].to(dev), a.len).cpu()
    for i in range(done):                                                  # resume: replay the finished layers
        tl = teacher_layer(stu, i, dev); advance(i, stu.layers[i], tl); del tl
    for i in range(done, len(stu.layers)):
        t0 = time.time()
        tl = teacher_layer(stu, i, dev); ly = stu.layers[i]
        with torch.no_grad(): YT = [layer_run(stu, tl, x.to(dev), a.len).cpu() for x in XT]
        names = [n for n in stu.P if n.startswith(f"{stu.L}.{i}.")]
        sc = [stu.P[n] for n in names if not n.endswith((":la", ":lb"))]; lo = [stu.P[n] for n in names if n.endswith((":la", ":lb"))]
        opt = torch.optim.Adam([{"params": sc, "base": a.block_lr}] + ([{"params": lo, "base": a.block_lr * a.lr_lora / a.lr}] if lo else []),
                               lr=a.block_lr, betas=(0.9, 0.99))
        nb = len(batches); dk = max(1, int(round(a.decay_frac * nb))); losses = []
        for t, bt in enumerate(batches):
            f = min(1.0, (t + 1) / 2) * min(1.0, (nb - t) / dk)
            for g in opt.param_groups: g["lr"] = g["base"] * f
            opt.zero_grad(set_to_none=True)
            n = sum(XS[k].shape[1] for k in bt); tot = 0.0
            for k in bt: tot += layer_run(stu, ly, XS[k].to(dev), a.len, y=YT[k], scale=1.0 / n)
            opt.step()
            with torch.no_grad():
                for p in sc: p.clamp_(-a.trust, a.trust)
            losses.append(tot / n)
        XT = YT
        with torch.no_grad():
            for k in range(len(docs)): XS[k] = layer_run(stu, ly, XS[k].to(dev), a.len).cpu()
            after = sum(float(((XS[k].float() - XT[k].float()).pow(2).sum(-1) / XT[k].float().pow(2).sum(-1).clamp_min(1e-12)).sum())
                        for k in range(len(docs))) / ntok
        del tl, opt
        torch.cuda.empty_cache()
        rec = {"layer": i, "first": losses[0], "last": losses[-1], "mean": float(np.mean(losses)), "traj_after": after,
               "wall": round(time.time() - t0, 1), "peak_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)}
        with open(os.path.join(a.out, "block.jsonl"), "a") as fh: fh.write(json.dumps(rec) + "\n")
        log("B " + json.dumps(rec))
        save_atomic({"layer": i + 1, "state": stu.state()}, bp)
    open(os.path.join(a.out, "block.done"), "w").close()


@torch.no_grad()
def make_taps(a, P, log):
    """the bf16 teacher's TAPS states at the gen rows of every document the plan's steps 0 .. last - 1 touch (each
    from its start to its window): OUT/taps.bin + index.json; restartable (progress.json, flushed every 64 docs)"""
    C, steps, last = P["C"], P["steps"], P["last"]
    out = a.make_taps; os.makedirs(out, exist_ok=True)
    tl = [int(x) for x in a.tap_layers.split(",")] if a.tap_layers else list(TAPS)
    seen, docs = set(), []
    for s in range(last):
        for segs in steps[s]:
            for (it, sa, sb) in segs:
                if it not in seen: seen.add(it); docs.append(it)
    idx, o = {}, 0
    J = []
    for it in docs:
        gi, i, n, _ = C.items[it]; j = C.g[gi].scored(i, 0, n)[0]
        idx[f"{C.g[gi].name}:{i}"] = [o, n]; J.append(j); o += len(j)
    H = 5120
    ip = os.path.join(out, "index.json")
    z = {"taps": tl, "hidden": H, "rows": o, "docs": idx, "plan_last": last, "order": "mix50" if a.mix50 is not None else "shuffle"}
    if os.path.exists(ip): assert json.load(open(ip)) == z, "index differs from the existing one"
    else: json.dump(z, open(ip + ".part", "w")); os.replace(ip + ".part", ip)
    bf = os.path.join(out, "taps.bin")
    mm = np.memmap(bf, np.uint16, "r+" if os.path.exists(bf) else "w+", shape=(o, len(tl), H))
    pp = os.path.join(out, "progress.json"); done = json.load(open(pp))["done"] if os.path.exists(pp) else 0
    log(f"taps: {len(docs)} documents, {o} rows ({o * len(tl) * H * 2 / 2**30:.1f} GiB), {done} done")
    devs = [f"cuda:{k}" for k in range(torch.cuda.device_count())]
    tea = build_student(a, devs, log, lora=0, layers=max(tl) + 1, quant_layers=[], served=False)
    use_seg_attention(tea)
    t0 = time.time(); ntok = 0
    for k in range(done, len(docs)):
        it = docs[k]; gi, i, n, _ = C.items[it]
        x = torch.from_numpy(np.ascontiguousarray(tea.emb_bf16[C.g[gi].ids(i, 0, n)])).view(BF)[None]
        cache = SegCache(None); pes = {}; rows = np.empty((len(J[k]), len(tl), H), np.uint16)
        jj = None
        for l, ly in enumerate(tea.layers):
            dev = tea.dev_of[l]
            if x.device != dev: x = x.to(dev)
            if dev not in pes:
                pid = torch.arange(n, device=dev).view(1, 1, -1).expand(4, 1, -1); pes[dev] = (tea.rotary[dev](x, pid[1:]), pid[0])
            x = ly(x, position_embeddings=pes[dev][0], attention_mask=None, position_ids=pes[dev][1], past_key_values=cache)
            if l in tl:
                jj = torch.from_numpy(J[k]).to(dev)
                rows[:, tl.index(l)] = x[0, jj].view(torch.int16).cpu().numpy().view(np.uint16)
        o0 = idx[f"{C.g[gi].name}:{i}"][0]; mm[o0:o0 + len(J[k])] = rows; ntok += n
        if (k + 1) % 64 == 0 or k + 1 == len(docs):
            mm.flush(); json.dump({"done": k + 1}, open(pp + ".part", "w")); os.replace(pp + ".part", pp)
            if (k + 1) % 512 == 0: log(f"taps: {k + 1} / {len(docs)} documents, {ntok / (time.time() - t0):.0f} tokens/s")
    open(os.path.join(out, ".done"), "w").close()
    log(f"TAPS DONE {len(docs)} documents, {o} rows")


@torch.no_grad()
def tapcal(a, stu, P, tap, log):
    """the T weight's calibration: KL and T-term sums of the untrained student over plan steps 0 .. --tapcal - 1 (every
    stream in order, states carried as in training)"""
    C, steps = P["C"], P["steps"]; dev = stu.devs[0]
    kl = ts = 0.0; m = 0
    for sid in range(a.V):
        prev, cur = None, None
        for s in range(a.tapcal):
            for (it, sa, sb) in steps[s][sid]:
                gi, i = C.items[it][0], C.items[it][1]
                p = prev if (cur == it and sa > 0) else (prefix_state(stu, C.g, (gi, i), sa, a.len) if sa > 0 else None)
                k_, m_, nxt, t_ = seg_loss(stu, C.g, (gi, i), sa, sb, p, tap=tap)
                kl += float(k_); ts += float(t_); m += m_; prev, cur = nxt, it
    r = {"steps": a.tapcal, "rows": m, "kl_row": kl / m, "tap_row": ts / m, "ratio": kl / ts}
    json.dump(r, open(os.path.join(a.out, "tapcal.json"), "w")); log("tapcal " + json.dumps(r))


def check(a, stu, P, tap, log):
    """flag unit checks on one 1,024-token segment of the plan's first document (a student built with --lora R):
    base (adapters detached, A8 off) vs L (adapters attached, B = 0: same loss and scale grads, adapter grads), Q (A8 on:
    loss changes), T (KL unchanged, a T term), B (one layer's block step moves only that layer's params)"""
    C = P["C"]; it = P["steps"][0][0][0][0]; gi, i = C.items[it][0], C.items[it][1]; n = min(1024, C.items[it][2])
    las = {k: (q.la, q.lb) for k, q in stu.qlin.items()}
    def detach_lora(on):
        for k, q in stu.qlin.items(): q.la, q.lb = (las[k] if on else (None, None))
    def run(t=None):
        for p in stu.P.values(): p.grad = None
        r = seg_loss(stu, C.g, (gi, i), 0, n, None, tap=t)
        l = r[0] + (a.tap_weight * r[3] if t is not None else 0.0); (l / r[1]).backward()
        g = {k: (p.grad.detach().clone() if p.grad is not None else None) for k, p in stu.P.items()}
        return float(r[0]) / r[1], (float(r[3]) / r[1] if t is not None else None), g
    out = {}
    detach_lora(False); kb, _, gb = run()
    sk = [k for k in gb if not k.endswith((":la", ":lb"))]
    out["base"] = {"kl": kb, "scale_grad_norm": float(sum(gb[k].double().norm() ** 2 for k in sk if gb[k] is not None) ** 0.5)}
    detach_lora(True); kl_, _, gl = run()
    out["L"] = {"kl": kl_, "same_kl": kl_ == kb, "max_scale_grad_diff": max(float((gl[k] - gb[k]).abs().max()) for k in sk if gb[k] is not None),
                "lb_grad_norm": float(sum(gl[k].double().norm() ** 2 for k in gl if k.endswith(":lb")) ** 0.5),
                "la_grad_norm": float(sum(gl[k].double().norm() ** 2 for k in gl if k.endswith(":la") and gl[k] is not None) ** 0.5)}
    detach_lora(False)
    if tap is not None:
        kt, tt, gt = run(tap)
        out["T"] = {"kl": kt, "same_kl": kt == kb, "tap_row": tt, "weighted": a.tap_weight * tt,
                    "grad_changed": max(float((gt[k] - gb[k]).abs().max()) for k in sk if gb[k] is not None)}
    use_a8(stu); kq, _, gq = run()
    out["Q"] = {"kl": kq, "rel_change": kq / kb - 1, "grad_changed": max(float((gq[k] - gb[k]).abs().max()) for k in sk if gb[k] is not None)}
    import types
    if a.lora: use_lora32(stu)                                                             # A8 off again
    else:
        for q in stu.qlin.values(): q.forward = types.MethodType(qmodel.QLinear.forward, q)
    k2, _, _ = run(); out["Q"]["restored"] = k2 == kb
    before = stu.state()
    a2 = argparse.Namespace(**{**vars(a), "out": os.path.join(a.out, "check-block")}); os.makedirs(a2.out, exist_ok=True)
    k0 = 1
    stu.layers_all = stu.layers
    stu.layers = stu.layers_all[:1]
    try: block_stage(a2, stu, P, k0, log)
    finally: stu.layers = stu.layers_all
    after = stu.state()
    moved = sorted({k.split(".")[3] for k in before if not torch.equal(before[k], after[k])})
    out["B"] = {"layers_moved": moved, "block_log": [json.loads(l) for l in open(os.path.join(a2.out, "block.jsonl"))]}
    stu.load_state(before)
    log("CHECK " + json.dumps(out)); json.dump(out, open(os.path.join(a.out, "check.json"), "w"), indent=1)


# ------------------------------------------------------------------ main
def parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("--model", required=True); ap.add_argument("--level", required=True)
    ap.add_argument("--chunks", required=True, help="comma list of golden dirs, or @FILE (one per line)")
    ap.add_argument("--calib", default=None, help="the whole-calib golden: its --pool documents join the training data")
    ap.add_argument("--pools", default=os.path.join(HERE, "..", "..", "eval", "pools", "pools-v3.json"))
    ap.add_argument("--pool", default="qad")
    ap.add_argument("--genmask", default=None, help="dir of gen masks <chunk>.u8 (genmask181.py) for the --chunks goldens: train on the board's rows only")
    ap.add_argument("--head", default=None); ap.add_argument("--head-perm", default=None)
    ap.add_argument("--dynamo-limit", type=int, default=64, help="w222: torch._dynamo recompile limit for --compile-scales / --compile-ew (default 64; 0 = torch default 8, which the compiled scale functions exceed: the later linear shapes ran eager; 64: 6 to 11 percent (full model, layers 32-63) faster, same peak memory, 2-step KL / gnorm identical)")
    ap.add_argument("--short-decay", default=None, help="w222: comma list of row counts: at each, a branch forks off the main run and decays linearly from the run's lr to ~0 over --short-decay-rows rows (ckpt-sd-<rows>); for runs without a constant phase (cosine-all, linear-all) and --lr-steps runs")
    ap.add_argument("--short-decay-rows", type=float, default=5e4, help="w222: length of the --short-decay branches in rows (default 50,000: the WSD 0.25 M branch's window)")
    ap.add_argument("--plan-ext", default=None, help="w222: the extended data plan (plan222-build.py): w214's documents in w214's order, then these new ones (xcorp / trace goldens under --ext-root/gold, calib qad documents), one epoch; budgets up to ~1e7 rows")
    ap.add_argument("--ext-root", default=None, help="w222: with --plan-ext: dir holding gold/<name>/ (the new goldens) and genmask/<name>.u8")
    ap.add_argument("--head-train", choices=("none", "in", "both"), default="none", help="w222 (H): train the served head's log multipliers: in = only r (the 5,120 hidden-side scales, suh of output.weight; the per-vocabulary scales stay frozen), both = r and c; none = frozen (default)")
    ap.add_argument("--cap", type=int, default=8192); ap.add_argument("--len", type=int, default=2048, help="max segment for prefix rebuilds / monitor")
    ap.add_argument("--V", type=int, default=24); ap.add_argument("--Q", type=int, default=1024)
    ap.add_argument("--budgets", default="1e6,4e6,16e6,48e6")
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--trust", type=float, default=0.3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--monitor", type=int, default=32); ap.add_argument("--mon-every", type=float, default=2e6)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--build-group", type=int, default=2, help="ranks building their student at once (host RAM)")
    ap.add_argument("--gate", action="store_true"); ap.add_argument("--profile", action="store_true"); ap.add_argument("--bench", action="store_true")
    ap.add_argument("--a8", action="store_true", help="arm Q: a8g128 fake-quant activations in the student (STE)")
    ap.add_argument("--a8-int8", action="store_true", help="w223: --a8 with the GEMM as a real int8 x int8 -> int32 product (implies --a8)")
    ap.add_argument("--compile-scales", action="store_true", help="torch.compile QLinear's scale / cast chain (speed)")
    ap.add_argument("--lora", type=int, default=0, help="rank of adapters on every trunk linear (qmodel's arm L), composable with the rest")
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--fixed-eval", type=int, default=0, help="rows target of a fixed eval set of calib qad-pool docs (first --len tokens, held out of training)")
    ap.add_argument("--fe-every", type=int, default=10); ap.add_argument("--fe-seed", type=int, default=2181)
    ap.add_argument("--init", default=None, help="a ckpt's params.pt: start from its state (fresh optimizer)")
    ap.add_argument("--conv", choices=("torch", "shift"), default="shift", help="GDN causal conv: F.conv1d (MIOpen) or qmodel.shift_conv1d")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many main steps (tests)")
    # #191
    ap.add_argument("--emb", default=None, help="the served token table, raw bf16 [V, E] (emb191.py)")
    ap.add_argument("--ab", default=None, help="the served GDN a / b weights (bf16 safetensors, blk.N.ssm_a / ssm_b.weight)")
    ap.add_argument("--mix50", type=int, default=None, help="D+: half calib qad pool, half the rest by gen rows (merge seed)")
    ap.add_argument("--block-frac", type=float, default=0.0, help="B: the block-wise stage's share of the budget (0 = off)")
    ap.add_argument("--block-lr", type=float, default=1e-3); ap.add_argument("--block-batch", type=int, default=4096)
    ap.add_argument("--taps", default=None, help="T: the teacher's tap states of this data order (--make-taps)")
    ap.add_argument("--tap-weight", type=float, default=0.0)
    ap.add_argument("--make-taps", default=None, help="write the bf16 teacher's tap states of the plan to DIR (all visible cards), then exit")
    ap.add_argument("--tapcal", type=int, default=0, help="KL / T-term ratio of the untrained student over the first N steps, then exit")
    ap.add_argument("--check", action="store_true", help="flag unit checks on one segment (build with --lora R), then exit")
    ap.add_argument("--build-lock", default=None, help="a host-wide lock file held while the student is built")
    ap.add_argument("--factors", choices=("all", "attn_ssmout"), default="all", help="R (#196): trainable row scales on attn_qg/k/v/output + ssm_out only")
    ap.add_argument("--norms", action="store_true", help="N (#196): RMSNorm gains, trained at lr / 10")
    ap.add_argument("--unshrink", default=None, help="U (#196): safetensors of per-row un-shrink factors c (key = linear, value = [n] f32)")
    ap.add_argument("--export-only", action="store_true", help="#196: write the checkpoint of the start state (U arms), no training")
    ap.add_argument("--eps", type=float, default=1e-8, help="#217: Adam's eps on the log multipliers (torch's default 1e-8)")
    ap.add_argument("--compile-ew", action="store_true", help="w215: torch.compile the shift conv and the MLP's silu * up (speed; forward bitwise, backward rounding-level)")
    ap.add_argument("--cap-keep", default=None, help="lead 11-23: JSON list of calib doc ids kept at --cap-keep-len tokens whatever --cap is (the long-context judge golden-jlong's docs, so its positions >= 8192 stay unseen)")
    ap.add_argument("--cap-keep-len", type=int, default=8192)
    ap.add_argument("--expandable-segments", action="store_true", help="w226: torch's CUDA allocator with expandable segments (growing KV buffers of deep documents fragment the default allocator)")
    ap.add_argument("--lr-layers", default=None, help="lead 11-22: per-layer-range peak lr for the scales, e.g. 0-31:1e-4 (the schedule scales it like --lr; other layers --lr); a fresh run only (a fork's optimizer groups must match)")
    ap.add_argument("--clip-doc", type=float, default=0.0, help="lead 11-22: per-document (segment) gradient clipping at C x the running median per-row segment gradient norm (0 = off)")
    ap.add_argument("--ema-params", default=None, help="lead 11-22: comma list of decays (e.g. 0.9,0.97): keep a bias-corrected EMA of the trainable parameters over the main run's optimizer steps and write it as ckpt-ema<decay>-<rows> at every snapshot point and at the end (training itself unchanged); off by default until its ablation")
    ap.add_argument("--snap-at", default="auto", help="lead 11-22: undecayed snapshots, the norm: the main run's state as it is (no decay, no branch) written as ckpt-raw-<rows> when the run first passes each row count. auto (default) = 0.25 / 0.5 / 0.75 / 1 M then every 0.5 M before the final budget; a comma list; none = off. A fork skips the points at / before its start")
    ap.add_argument("--exact-ckpt-names", action="store_true", help="lead 11-22: checkpoint labels keep the exact row count when python's %%.0e would round it (1.5e6 -> ckpt-1.5e+06, not ckpt-2e+06); default off = the old names")
    ap.add_argument("--gradnoise-from", type=int, default=0, help="lead 11-22: the gradnoise batches are plan steps FROM..FROM+N-1 (default 0 = #217's first N); use steps a checkpoint never trained on")
    ap.add_argument("--gradnoise", type=int, default=0, help="#217: gradients of the first N plan steps at the start state (no optimizer step): per-scale mean / std across batches to OUT/gradnoise.npz, then exit")
    ap.add_argument("--wd", type=float, default=0.0, help="#200: AdamW decoupled weight decay on the log multipliers (0 = Adam)")
    ap.add_argument("--accum", type=int, default=1, help="#200: plan steps (micro-batches) per optimizer step")
    ap.add_argument("--rows", type=float, default=None, help="#200: the budget in gen rows (overrides --budgets)")
    ap.add_argument("--order-seed", type=int, default=None, help="#200: data-order seed (alias of --seed)")
    ap.add_argument("--warmup-frac", type=str, default=None, help="#200: warmup as a fraction of the optimizer steps (8/49), overrides --warmup")
    # w220
    ap.add_argument("--train-layers", default=None, help="w220: train only the parameters of decoder layers A-B (inclusive); the layers before A run once under no_grad")
    ap.add_argument("--train-tensors", default=None, help="w220: train only the parameters whose name (<linear key>:<r|c|g>) matches this regex (re.search)")
    ap.add_argument("--tie-groups", type=int, default=1, help="w220: one shared log multiplier per G consecutive channels (exports per channel; 1 = off)")
    ap.add_argument("--tie-layers", default=None, help="w220: tie only in layers A-B (inclusive; default all layers)")
    ap.add_argument("--lr-steps", default=None, help='w220: piecewise-constant lr "lr1:rows1,lr2:rows2,..." (rows = cumulative gen rows at each phase end), then the final decay from the last phase\'s lr (starts at the last end)')
    ap.add_argument("--decay-shape", choices=("linear", "sqrt", "cosine-all", "linear-all"), default="linear", help="w220: the final decay's shape (sqrt = 1 - sqrt; *-all = decay over the whole run after warmup, no constant phase)")
    ap.add_argument("--fork-from", default=None, help="w220: start from a saved state (DIR/state.pt or a state-<rows>.pt file: parameters, optimizer, step) in a fresh OUT and run on from its step with this run's schedule (a cooldown)")
    ap.add_argument("--decay-data", choices=("mix", "calib"), default="mix", help="w220: data of the decay phase: the plan's own mix, or unused calib documents only")
    ap.add_argument("--save-at", default=None, help="w220: comma list of gen-row counts: OUT/state-<rows>.pt is written when the run first passes each (fork points for --fork-from)")
    ap.add_argument("--prompt-weight", type=float, default=0.0, help="w220: add the golden's prompt-side rows (outside the gen mask) to the loss at this weight per row (0 = gen rows only, as before); the loss is still divided by the step's gen rows")
    ap.add_argument("--local-mse", default=None, help="w220: LAYERS,lambda[,local|joint|block]: lambda x the hidden-state MSE vs the bf16 model at those layers (LAYERS: 0-31, 5+19+33; needs --local-taps), added to the KL")
    ap.add_argument("--local-taps", default=None, help="w220: the bf16 teacher's hidden states of the --local-mse layers (--make-taps DIR --tap-layers L,L,...)")
    ap.add_argument("--tap-layers", default=None, help="w220: with --make-taps: comma list of layers to store (default the DFlash2 taps 5,19,33,47,61)")
    ap.add_argument("--clip-gnorm", type=float, default=0.0, help="w220: clip the global gradient norm at X x the running median batch norm (logs the clipped steps; 0 = off)")
    ap.add_argument("--beta2", type=float, default=0.99, help="w220: Adam beta2 (the trainer's 0.99)")
    ap.add_argument("--no-frozen-prefix", action="store_true", help="w220 (test / benchmark switch): do not skip the backward of the layers before the first trainable one (the old full path, same trainable set)")
    ap.add_argument("--mask", default=None, help="w220: npz of per-scale 0 / 1 masks (mkmask220.py); a vector it does not list is not masked")
    ap.add_argument("--split-layers", type=int, default=0, help="w225: two cards, one process: the token table and layers 0..K-1 on the first visible card, layers K..63 + final norm + head on the second; carried KV in per-document buffers, each layer's state on its card (0 = off: one card)")
    ap.add_argument("--pipeline", action="store_true", help="w225 (with --split-layers): the filled pipeline (card A runs segment j's first half while card B runs segment j-1's second half); bit for bit the naive split")
    ap.add_argument("--state-card-gb", type=float, default=0.0, help="w225: keep carried stream states on their cards while each card's kept states stay within this many GiB, park the rest on the CPU (0 = park all, as before); implies the buffered KV")
    ap.add_argument("--attn-gqa", action="store_true", help="w225: segment attention with sdpa's enable_gqa (no 6x repeated K / V: the long-context memory); forward bitwise, backward rounding-level vs the default")
    ap.add_argument("--depth-test", type=int, default=0, help="w225 test switch: every document starts after D synthetic positions (zero K / V in the attention layers, positions offset by D): memory / speed at depth D on any data; not for training")
    return ap


def gradnoise(a, T, d, log):
    """#217: the log-multiplier gradient of each of the first a.gradnoise plan steps (one normal batch: V x Q positions) at the current
    state, no optimizer step. Writes OUT/gradnoise.npz: per trainable scale vector (name) its mean and std (ddof 1) across the batches."""
    P = [(n, p) for n, p in zip(T.names, T.P) if p.requires_grad and not n.startswith("lm_head")]
    T.opt.step = lambda *x, **k: None
    s1 = [torch.zeros(p.numel(), dtype=torch.float64) for _, p in P]; s2 = [torch.zeros_like(x) for x in s1]
    rec = []
    for s in range(a.gradnoise):
        accs, w = T.run_group([a.gradnoise_from + s], 0.0)
        for (_, p), x1, x2 in zip(P, s1, s2):
            g = p.grad.reshape(-1).double().cpu(); x1 += g; x2 += g * g
        kl = sum(float(accs[2 * k]) for k in range(len(T.tags))); rows = sum(float(accs[2 * k + 1]) for k in range(len(T.tags)))
        rec.append({"step": a.gradnoise_from + s, "rows": rows, "kl_row": kl / rows, "gnorm": T.gnorm, "wall": round(w, 1)})
        log(json.dumps(rec[-1]))
    n = a.gradnoise
    out = {}
    for (nm, _), x1, x2 in zip(P, s1, s2):
        m = x1 / n; var = ((x2 - n * m * m) / (n - 1)).clamp_min(0)
        out[nm + "|mean"] = m.float().numpy(); out[nm + "|std"] = var.sqrt().float().numpy()
    if d.rank == 0:
        np.savez(os.path.join(a.out, "gradnoise.npz"), **out)
        json.dump(rec, open(os.path.join(a.out, "gradnoise.json"), "w"))
    log(f"gradnoise: {n} batches, {len(P)} scale vectors, {sum(x.numel() for x in s1)} scales")


def main():
    a = parser().parse_args()
    if a.a8_int8: a.a8 = True
    if a.pipeline: assert a.split_layers, "--pipeline needs --split-layers"
    if a.split_layers or a.state_card_gb > 0 or a.depth_test:     # w225: the w225 segment loop covers the plain KL path only
        assert not (a.prompt_weight > 0 or a.local_mse or a.taps or a.lora or a.block_frac or a.make_taps or a.gate or a.profile or a.bench or a.check or a.tapcal), \
            "w225 switches: plain KL training only (no --prompt-weight / --local-mse / --taps / --lora / --block-frac / --make-taps / --gate / --profile / --bench / --check / --tapcal)"
    if a.expandable_segments: torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    if a.order_seed is not None: a.seed = a.order_seed          # #200: the data order (--seed and --order-seed are one switch)
    d = Dist()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, f"run.log" if d.rank == 0 else f"run-r{d.rank}.log"), "a")
    def log(s, all_ranks=False):
        if d.rank != 0 and not all_ranks: return
        line = time.strftime("%H:%M:%S ") + s; print(line, flush=True); logf.write(line + "\n"); logf.flush()
    log("qad191.py " + json.dumps(vars(a)) + f" world {d.world}")
    mad_int_form()
    if a.conv == "shift": qmodel.use_shift_conv()
    if a.compile_ew: set_dynamo_limit(a.dynamo_limit); qmodel.use_compiled_elementwise(); log("shift conv / MLP silu * up compiled")
    P = build_data(a, d, log)
    C, fe, budgets, steps, rows_of, cum, last, br, fork_last = (P[k] for k in ("C", "fe", "budgets", "steps", "rows_of", "cum", "last", "br", "fork_last"))
    if a.make_taps: return make_taps(a, P, log)
    mon = []
    if a.calib and a.monitor:                                   # monitor: search-pool calib docs (the pilot's rule)
        pl = json.load(open(a.pools))["pool"]; gc = len(C.g) - 1
        cands = [i for i, p in enumerate(pl) if p == "search"]
        rng = np.random.default_rng(1000 + a.seed); rng.shuffle(cands)
        mon = [(gc, i) for i in cands if len(C.g[gc].scored(i, 0, min(a.len, C.g[gc].ntok(i)))[0]) >= 16][:a.monitor]
    k0 = 0                                                      # B: the end-to-end stage starts at plan step k0
    if a.block_frac:
        assert len(budgets) == 1, "B: one budget (no WSD branches)"
        k0 = int(np.searchsorted(cum, a.block_frac * budgets[-1], side="left")) + 1
        fork_last = int(np.searchsorted(cum, cum[k0 - 1] + (1 - a.decay_frac) * (budgets[-1] - cum[k0 - 1]), side="left")) + 1
        log(f"B: block stage on plan steps 0 .. {k0 - 1} ({cum[k0 - 1]} rows); end-to-end steps {k0} .. {last - 1}, decay from step {fork_last}")
    N = max(1, a.accum)                                      # #200: plan steps per optimizer step
    assert N == 1 or k0 == 0, "--accum > 1: no block stage"
    if N > 1 and br:                                         # w222: WSD branches with --accum > 1: fork points on optimizer-step boundaries, each branch = a standalone run of its budget (branch_plan)
        assert not (a.lr_steps or a.decay_shape != "linear" or a.short_decay or a.warmup_frac is not None), "--accum > 1 with branches: WSD only (no --lr-steps / --decay-shape / --short-decay / --warmup-frac: a branch must equal a standalone run)"
        for Tb, (fk, end) in br.items():
            g0, GT, sf = branch_plan(fk, end, N)
            assert g0 <= -(-fork_last // N), f"--accum {N}: the branch at {Tb:.0e} forks at optimizer step {g0}, after the main run's decay has started (step {-(-fork_last // N)}): give the main run a longer constant phase (--decay-frac) or fewer / larger branches"
    NU = -(-(last - k0) // N)                                # optimizer steps of the main run
    if a.warmup_frac is not None:                            # #200: warmup stretched to the optimizer steps (8 of 49 = #191)
        a.warmup = max(1, round(float(Fraction(a.warmup_frac)) * NU))
    assert not (a.block_frac and (a.lr_steps or a.decay_shape != "linear" or a.decay_data != "mix")), "B stage: no w220 schedule flags"
    log(f"#200: {N} plan step(s) per optimizer step, {NU} optimizer steps, warmup {a.warmup}, wd {a.wd}, lr {a.lr}, budget {budgets[-1]:.0f} rows")
    json.dump({"monitor": mon, "budgets": budgets, "last": last, "branches": {str(k): v for k, v in br.items()},
               "fork_last": fork_last, "steps": len(steps), "e2e_from": k0, "share": P["share"], "accum": N, "wd": a.wd,
               "warmup": a.warmup, "opt_steps": NU, "order_seed": a.seed}, open(os.path.join(a.out, "plan.json"), "w"))
    lr_at = make_lr(a, cum, last, fork_last, k0, N)             # w220: the lr of the optimizer step starting at plan step s
    dev = f"cuda:{d.local}"; torch.cuda.set_device(dev)
    if a.split_layers:                                          # w225: one process, two cards (the main thread's current card is the second: loss, head)
        assert d.world == 1 and torch.cuda.device_count() >= 2, "--split-layers: one process with two visible cards"
        dev = ["cuda:0", "cuda:1"]; torch.cuda.set_device("cuda:1")
    for grp in range(0, d.world, a.build_group):                # ranks build in groups (each build holds the decoded trunk in host RAM once)
        if grp <= d.rank < grp + a.build_group: stu = build_student(a, dev, lambda s: log(s, True))
        d.barrier()
    apply196(a, stu, log)
    ATTN_GQA[0] = a.attn_gqa                                     # w225
    if a.pipeline: triton_autotune_lock()
    if a.split_layers or a.state_card_gb > 0 or a.depth_test:
        stu.split = a.split_layers; stu.kvbuf = True; stu.pos_off = a.depth_test; stu.kv_cap = 0
        log(f"w225: split {a.split_layers}" + (f" (layers 0-{a.split_layers - 1} on {stu.devs[0]}: {torch.cuda.memory_allocated(stu.devs[0]) / 2**30:.2f} GiB, "
            f"{a.split_layers}-{len(stu.layers) - 1} + head on {stu.devs[1]}: {torch.cuda.memory_allocated(stu.devs[1]) / 2**30:.2f} GiB)" if a.split_layers else "")
            + f", pipeline {a.pipeline}, states kept on the cards up to {a.state_card_gb} GiB each, attention gqa {a.attn_gqa}, depth test {a.depth_test}")
    if a.export_only:
        o = os.path.join(a.out, f"ckpt-{budgets[-1]:.0e}")
        if d.rank == 0:
            write_ckpt(stu, o, {"budget": budgets[-1], "steps": 0, "rows": 0, "args": vars(a), "export_only": True})
            open(o + "/.done", "w").close()
        log(f"export-only (#196): {o}")
        return
    apply_ties(a, stu, log)
    stu.mask = apply_trainable(a, stu, log)
    set_first_train(stu, log, a.no_frozen_prefix)
    use_seg_attention(stu)
    if a.compile_scales: use_compiled_scales(stu, a.a8, a.a8_int8, a.dynamo_limit); log(f"QLinear scales compiled (a8 {a.a8}, int8 {a.a8_int8})")
    elif a.a8 and not a.check: use_a8(stu, a.a8_int8)
    elif a.lora: use_lora32(stu); log("arm L: adapters in f32 (use_lora32)")
    if a.a8: log("arm Q: a8g128 fake-quant activations on every trunk linear (STE)" + ("; GEMM int8 x int8 (w223 --a8-int8)" if a.a8_int8 else ""))
    assert not (a.prompt_weight > 0 and a.taps), "--prompt-weight with the T arm (--taps) is not implemented"
    loc = None
    if a.local_mse:
        assert a.local_taps and not a.taps, "--local-mse needs --local-taps DIR (and not the T arm's --taps)"
        loc = LocalMSE(a.local_mse, TapStore(a.local_taps, any_layers=True)); loc.restrict(stu.first_trainable, log)
        log(f"local MSE: layers {loc.layers[0]}..{loc.layers[-1]} ({len(loc.layers)}), lambda {loc.lam}, mode {loc.mode}, targets {a.local_taps}")
    tap = TapStore(a.taps) if a.taps else None
    if tap is not None: log(f"arm T: tap MSE at layers {TAPS} from {a.taps}, weight {a.tap_weight}")
    if a.check: return check(a, stu, P, tap, log)
    if a.tapcal: return tapcal(a, stu, P, tap, log)
    if a.gate: return gate(a, stu, C, log)
    if a.profile: return profile(a, stu, C, log)
    if a.bench: return bench(a, stu, C, log)
    if a.init:
        stu.load_state(torch.load(a.init, weights_only=False)["state"]); log(f"state from {a.init}")
    sp = os.path.join(a.out, "state.pt")
    if a.block_frac and not os.path.exists(sp):
        assert d.world == 1, "B: one card"
        block_stage(a, stu, P, k0, log)
    T = Trainer(a, stu, C, steps, rows_of, d, log, tap=tap, loc=loc)
    if a.gradnoise: return gradnoise(a, T, d, log)
    st = {"step": k0, "done_br": [], "wall": 0.0, "ema": None, "next_mon": a.mon_every}
    def fixed_log(step, rows, what):
        if not fe: return
        z = T.fixed(fe)
        if d.rank == 0:
            with open(os.path.join(a.out, "fixed.jsonl"), "a") as f:
                f.write(json.dumps({"step": step, "rows": rows, "what": what, "kl": float(z[:, 0].sum() / z[:, 1].sum()),
                                    "sums": z[:, 0].round(6).tolist(), "n": z[:, 1].astype(int).tolist()}) + "\n")
    forked = False
    if os.path.exists(sp):
        z = torch.load(sp, weights_only=False); T.restore(z["snap"]); st = z["st"]; log(f"resumed at main step {st['step']}")
    elif a.fork_from:                                        # w220: a cooldown from a saved state (parameters, Adam state, step)
        fp = a.fork_from if os.path.isfile(a.fork_from) else os.path.join(a.fork_from, "state.pt")
        z = torch.load(fp, weights_only=False); T.restore(z["snap"]); st = z["st"]; st["done_br"] = []; forked = True
        if "plan_sha" in st:
            assert st["plan_sha"] == plan_sha(steps, st["step"]), f"--fork-from {fp}: its data plan up to step {st['step']} is not this run's"
        else: log(f"WARNING: {fp} carries no plan checksum (written without --save-at): the data prefix is not verified")
        assert st["step"] < last, f"--fork-from: the state's step {st['step']} is not before the run's end {last}"
        if getattr(T, "partial_restore", False):            # w222: parameters that start without Adam state: the lr ramps over --warmup optimizer steps from the fork (as at a run's start)
            fk, base_lr = st["step"], lr_at
            lr_at = lambda q, _f=fk, _b=base_lr: _b(q) * min(1.0, ((q - _f) // N + 1) / a.warmup)
            log(f"fork into a different trainable set: lr ramps up over {a.warmup} optimizer steps")
        log(f"fork from {fp} at main step {st['step']} ({int(cum[st['step'] - 1])} rows); decay starts at plan step {fork_last}, lr here {lr_at(st['step']):.3e}")
        fixed_log(st["step"], int(cum[st["step"] - 1]), "fork")
        st["fork_rows"] = int(cum[st["step"] - 1])                  # kept in st: a resumed fork still skips the points before its start
    else:
        v = T.monitor(mon) if mon else None
        log(json.dumps({"step": k0, "rows": 0, "mon_kl": v}))
        if d.rank == 0:
            with open(os.path.join(a.out, "mon.jsonl"), "a") as f: f.write(json.dumps({"step": k0, "rows": 0, "mon_kl": v, "what": "main"}) + "\n")
    if not os.path.exists(sp) and not forked: fixed_log(k0, int(cum[k0 - 1]) if k0 else 0, "main")
    save_at = [float(x) for x in a.save_at.split(",")] if a.save_at else []
    snap_at = [RW(x) for x in snap_grid(a.snap_at, budgets[-1])]   # lead 11-22: undecayed snapshots are the norm (grid by default; --snap-at none to turn off)
    PE = {}                                                           # lead 11-22: parameter EMAs (--ema-params): decay -> [count, [fp32 tensors]]
    def save():
        if PE: st["pema"] = {k: [n_, [t.cpu() for t in v]] for k, (n_, v) in PE.items()}
        if d.rank == 0: save_atomic({"snap": T.snapshot(), "st": st}, sp)
        d.barrier()
    def logstep(s, lr, accs, wall, what):
        kl = {t: (float(accs[2 * k]), int(accs[2 * k + 1])) for t, k in T.tix.items()}
        rows = sum(v[1] for v in kl.values()); tkl = sum(v[0] for v in kl.values()) / max(1, rows)
        if what == "main":
            st["ema"] = tkl if st["ema"] is None else 0.95 * st["ema"] + 0.05 * tkl; st["wall"] += wall
        rec = {"what": what, "step": s + 1, "rows": int(cum[s]), "lr": lr, "train_kl": tkl, "ema": st["ema"],
               "by": {t: v[0] / v[1] for t, v in kl.items() if v[1]}, "n": {t: v[1] for t, v in kl.items() if v[1]},
               "prefix_tok": int(accs[-1]), "wall": round(wall, 2), "pos_s": round(rows / wall, 1)}
        if tap is not None: rec["tap_row"] = float(accs[-2]) / max(1, rows)
        if loc is not None: rec["loc_row"] = float(accs[-2]) / max(1, rows)
        rec["peak_gib"] = round(torch.cuda.max_memory_allocated(T.dev) / 2**30, 2)
        if a.split_layers: rec["peak_gib_b"] = round(torch.cuda.max_memory_allocated(stu.devs[1]) / 2**30, 2)    # w225: the second card
        if getattr(T, "pipe_s", None): rec["pipe_s"] = {k_: round(v_, 1) for k_, v_ in T.pipe_s.items()}
        if T.w225:
            rec["states_gib"] = {str(k_): round(v_ / 2**30, 2) for k_, v_ in T.resident().items()}
            rec["rss_gib"] = round(int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30, 2)
        rec["accum"] = N; rec["gnorm"] = round(T.gnorm, 6) if hasattr(T, "gnorm") else None
        if a.clip_doc > 0: rec["clip_doc"] = T.nclip_doc
        if a.clip_gnorm > 0:
            rec["clip"] = None if T.clip is None else round(T.clip[0], 6)
            if T.clip is not None: log(f"CLIP step {s + 1}: gradient norm {T.gnorm:.4e} > {a.clip_gnorm} x median {T.clip[1] / a.clip_gnorm:.4e}: scaled by {T.clip[0]:.4f}")
        if d.rank == 0:
            with open(os.path.join(a.out, "train.jsonl"), "a") as f: f.write(json.dumps(rec) + "\n")
        return rec
    def ckpt(Tb, s):
        o = os.path.join(a.out, f"ckpt-{Tb:.0e}")
        v = T.monitor(mon) if mon else None
        cs = T.checksum()
        if d.rank == 0:
            write_ckpt(stu, o, {"budget": Tb, "steps": s, "rows": int(cum[s - 1]), "mon_kl": v, "args": vars(a), "checksum": cs})
            json.dump({"budget": Tb, "steps": s, "rows": int(cum[s - 1]), "mon_kl": v}, open(o + "/mon.json", "w"))
            open(o + "/.done", "w").close()
            with open(os.path.join(a.out, "mon.jsonl"), "a") as f: f.write(json.dumps({"step": s, "rows": int(cum[s - 1]), "mon_kl": v, "what": f"ckpt {Tb:.0e}"}) + "\n")
        fixed_log(s, int(cum[s - 1]), f"ckpt {Tb:.0e}")
        log(f"CKPT {Tb:.0e} at step {s} ({cum[s - 1]} rows): monitor KL {v}")
        d.barrier()
    ema_ds = [float(x) for x in a.ema_params.split(",")] if a.ema_params else []
    if forked: st.pop("pema", None)                                   # a fork starts its own averages
    for e_d in ema_ds:
        k = f"{e_d:g}"
        if st.get("pema") and k in st["pema"]: PE[k] = [st["pema"][k][0], [t.to(p.device) for t, p in zip(st["pema"][k][1], T.P)]]
        else: PE[k] = [0, [torch.zeros(p.shape, dtype=torch.float32, device=p.device) for p in T.P]]
    def ema_ckpt(rows, s_):
        """write each parameter EMA (bias-corrected: the weighted average of the main-run states since the start / fork) as a checkpoint"""
        for k, (n_, v) in PE.items():
            if n_ == 0: continue
            key = EMK(rows, f"ema{k}")
            if os.path.exists(os.path.join(a.out, f"ckpt-{key:.0e}", ".done")): continue
            bak = [p.detach().clone() for p in T.P]; c = 1.0 - float(k) ** n_
            with torch.no_grad():
                for p, e_ in zip(T.P, v): p.copy_((e_ / c).to(p.dtype))
            ckpt(key, s_)
            with torch.no_grad():
                for p, b_ in zip(T.P, bak): p.copy_(b_)
    if d.rank == 0:                                                   # lead 11-22: the snapshot list for the queue's reads (qq.py auto_snaps)
        sn = [r_ for r_ in snap_at if r_ > st.get("fork_rows", 0)]
        json.dump([f"ckpt-{r_:.0e}" for r_ in sn] + [f"ckpt-{EMK(r_, 'ema' + k):.0e}" for k in PE for r_ in sn + [budgets[-1]]], open(os.path.join(a.out, "snaps.json"), "w"))
    while st["step"] < last:
        s = st["step"]
        for Tb, (fk, end) in br.items():                 # a branch forks at this step
            g0, GT, sf = branch_plan(fk, end, N) if N > 1 else (None, None, fk - 1)
            if s == sf and Tb not in st["done_br"] and not os.path.exists(os.path.join(a.out, f"ckpt-{Tb:.0e}", ".done")):
                save(); snap = T.snapshot()
                log(f"branch {Tb:.0e}: steps {fk} .. {end}" + (f" (optimizer steps {g0} .. {GT - 1}, forked at plan step {sf})" if N > 1 else ""))
                if N > 1:                                 # w222: the standalone run of budget Tb, step for step: its lr (group_lr with last = end, fork_last = fk) and its (partial) last group
                    for g in range(g0, GT):
                        s0 = g * N; e_ = min(s0 + N, end); lr = group_lr(s0, 0, end, fk, a.lr, a.warmup, N)
                        accs, w = T.run_group(list(range(s0, e_)), lr); r = logstep(e_ - 1, lr, accs, w, f"branch {Tb:.0e}")
                        log(json.dumps(r))
                    ckpt(Tb, end)
                    T.restore(snap); st["done_br"].append(Tb); save()
                    continue
                for bs in range(s, end):
                    lr = branch_lr(bs, s, end, last, fork_last, a.lr, a.warmup, lr_at, bool(a.lr_steps or a.decay_shape != "linear" or a.short_decay))   # linear to ~0 at end
                    accs, w = T.run_step(bs, lr); r = logstep(bs, lr, accs, w, f"branch {Tb:.0e}")
                    log(json.dumps(r))
                ckpt(Tb, end)
                T.restore(snap); st["done_br"].append(Tb); save()
        lr = lr_at(s)                                                 # per optimizer step (N plan steps)
        e = min(s + N, last)
        accs, w = T.run_group(list(range(s, e)), lr); st["step"] = e; s = e - 1
        if PE:
            with torch.no_grad():
                for k, ent in PE.items():
                    ent[0] += 1
                    for p, e_ in zip(T.P, ent[1]): e_.mul_(float(k)).add_(p.detach().float(), alpha=1.0 - float(k))
        r = logstep(s, lr, accs, w, "main")
        log(json.dumps(r))
        if cum[s] >= st["next_mon"] and mon:
            v = T.monitor(mon); st["next_mon"] += a.mon_every
            if d.rank == 0:
                with open(os.path.join(a.out, "mon.jsonl"), "a") as f: f.write(json.dumps({"step": s + 1, "rows": int(cum[s]), "mon_kl": v, "what": "main"}) + "\n")
            log(f"monitor at step {s + 1} ({cum[s]} rows, lr {lr:.2e}): {v}")
        if fe and (st["step"] // N) % a.fe_every == 0: fixed_log(st["step"], int(cum[s]), "main")
        if (st["step"] // N) % max(1, a.save_every // N) == 0: save()
        for r_ in save_at:                                            # w220: fork points
            fn = os.path.join(a.out, f"state-{int(r_)}.pt")
            if cum[s] >= r_ and not os.path.exists(fn):
                if d.rank == 0: save_atomic({"snap": T.snapshot(), "st": {**st, "plan_sha": plan_sha(steps, st["step"])}}, fn)
                d.barrier(); log(f"fork point state: {fn} (step {st['step']}, {int(cum[s])} rows)")
        for r_ in snap_at:                                            # lead 11-22: undecayed snapshots (curves for nearly free); a fork skips the points at / before its start
            if cum[s] >= r_ and not r_ <= st.get("fork_rows", 0) and not os.path.exists(os.path.join(a.out, f"ckpt-{r_:.0e}", ".done")): ckpt(r_, st["step"])
            if PE and cum[s] >= r_ and not r_ <= st.get("fork_rows", 0): ema_ckpt(float(r_), st["step"])
        if a.max_steps and st["step"] >= a.max_steps: save(); log("max steps"); return
        stop = torch.tensor([float(os.path.exists(os.path.join(a.out, "STOP")))]); d.allsum(stop)
        if float(stop) > 0: save(); log(f"STOP file: saved at main step {st['step']}, exiting"); return
    ckpt(budgets[-1], last)
    if PE: ema_ckpt(float(budgets[-1]), last)
    save()
    log(f"QAD200 DONE {last} plan steps ({NU} optimizer steps), {cum[last - 1]} rows, train wall {st['wall']:.0f} s")


if __name__ == "__main__":
    main()
