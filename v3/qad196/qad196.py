#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""qad196.py OUT --model BF16 --level LEVEL --chunks DIR,... [--calib GOLDEN --pools POOLS] [options] -- #196 (RULES196):
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
import argparse, json, math, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qad"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "qad181"))
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


def use_compiled_scales(stu, a8=False):
    """QLinear's elementwise chain (x . su -> bf16, y . (sv s) -> bf16; arm Q's fake quant) as torch.compile'd functions
    (dynamic shapes): the same arithmetic in fused kernels (rounding-level differences from eager)"""
    import types
    @torch.compile(dynamic=True)
    def pre(x0, suh, r): return (x0 * (suh * r.exp() if r is not None else suh)).to(BF)
    @torch.compile(dynamic=True)
    def post(y, svh, c, s): return (y * ((svh * c.exp() if c is not None else svh) * s)).to(BF)
    def fwd(self, x):
        sh = x.shape
        x = pre(x.reshape(-1, self.k), self.suh, self.r)
        x = (x.view(-1, 128) @ self.Hb).view(-1, self.k)
        if a8: x = _A8.apply(x)
        y = x @ self.wq.to(BF)
        y = (y.view(-1, 128) @ self.Hb).view(-1, self.n)
        return post(y, self.svh, self.c, self.s).view(*sh[:-1], self.n)
    for q in stu.qlin.values():
        assert q.la is None, "compiled scales: no adapters"
        q.forward = types.MethodType(fwd, q)


def use_a8(stu):
    """the student's trunk QLinears run the GEMM on a8g128 fake-quantized activations (the hadamard-rotated, suh-scaled
    input, which is what the exl3i A8 path quantizes); the head stays f16 as served"""
    def fwd(self, x):
        sh = x.shape
        x0 = x.reshape(-1, self.k)
        x = (x0 * self.su()).to(BF)
        x = (x.view(-1, 128) @ self.Hb).view(-1, self.k)
        x = _A8.apply(x)
        y = x @ self.wq.to(BF)
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


def seg_attention(module, q, k, v, attention_mask, dropout=0.0, scaling=None, **kw):
    """sdpa for a segment after a prefix: query i (absolute P + i) sees keys 0 .. P + i (lower-right causal)"""
    from torch.nn.attention.bias import causal_lower_right
    L, S = q.shape[2], k.shape[2]
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


def run_segment(stu, ids, a, prev=None, ckpt=True, taps=None):
    """final-norm hidden states [n, H] of tokens a .. a + n - 1 of a document given the state after 0 .. a - 1 (prev:
    a SegCache.next of the segment before, or None at a = 0); returns (h, next state); with taps (layer indices) also
    {layer: its output [1, n, H]}"""
    q35 = stu.q35
    dev = stu.devs[0]
    assert len(stu.devs) == 1, "qad181: one card a student"
    x = stu.embed(ids)
    n = x.shape[1]
    pid = torch.arange(a, a + n, device=dev).view(1, 1, -1).expand(4, 1, -1)
    pe = stu.rotary[dev](x, pid[1:])
    cache = SegCache(prev); tp = {}
    for i, ly in enumerate(stu.layers):
        kw = dict(position_embeddings=pe, attention_mask=None, position_ids=pid[0], past_key_values=cache)
        if ckpt and torch.is_grad_enabled(): x = checkpoint(lambda h, _l=ly, _k=kw: _l(h, **_k), x, use_reentrant=False)
        else: x = ly(x, **kw)
        if taps is not None and i in taps: tp[i] = x
    if taps is not None: return stu.fnorm(x)[0], cache.next, tp
    return stu.fnorm(x)[0], cache.next


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
    def __init__(self, d):
        z = json.load(open(f"{d}/index.json")); assert os.path.exists(f"{d}/.done"), f"{d}: taps not finished"
        assert tuple(z["taps"]) == TAPS, z["taps"]
        self.off, self.H = z["docs"], z["hidden"]
        self.mm = np.memmap(f"{d}/taps.bin", np.uint16, "r", shape=(z["rows"], len(TAPS), self.H))
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


def to_cpu(st): return {k: (tuple(t.cpu() for t in v) if isinstance(v, tuple) else v.cpu()) for k, v in st.items()}
def to_dev(st, dev): return {k: (tuple(t.to(dev, non_blocking=True) for t in v) if isinstance(v, tuple) else v.to(dev, non_blocking=True)) for k, v in st.items()}


@torch.no_grad()
def prefix_state(stu, g, it, a, L):
    """the state after tokens 0 .. a - 1 of a document, by no-grad segments of <= L tokens"""
    gi, i = it; st = None; p = 0
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


def lr_main(step, last, fork_last, lr, warm):
    if step < warm: return lr * (step + 1) / warm
    if step < fork_last: return lr
    return lr * max(0.0, (last - step) / max(1, last - fork_last))


# ------------------------------------------------------------------ trainer
class Trainer:
    def __init__(self, a, stu, C, steps, rows_of, dist, log, tap=None):
        self.a, self.stu, self.C, self.steps, self.rows_of, self.d, self.log = a, stu, C, steps, rows_of, dist, log
        self.tap = tap                                       # T: a TapStore (weight a.tap_weight)
        self.dev = stu.devs[0]
        self.P = list(stu.P.values()); self.names = list(stu.P.keys())
        nrm = [p for n, p in zip(self.names, self.P) if n.endswith(":g")]          # N: RMSNorm gains at lr / 10
        sc = [p for n, p in zip(self.names, self.P) if not n.endswith((":la", ":lb", ":g"))]
        lo = [p for n, p in zip(self.names, self.P) if n.endswith((":la", ":lb"))]
        self.scal = sc + nrm
        groups = [{"params": sc, "base": a.lr}] + ([{"params": nrm, "base": a.lr / 10}] if nrm else [])
        self.opt = torch.optim.Adam(groups, lr=a.lr, betas=(0.9, 0.99))
        # L: the adapters' Adam runs on the CPU on f32 master copies (its state, 2 x 0.46 GB at r = 16, does not fit
        # beside a card's training peak); the card's adapters are overwritten with the masters after every step
        self.lo = lo; self.is_lo = [n.endswith((":la", ":lb")) for n in self.names]
        self.lo_m = [p.detach().cpu().clone().requires_grad_(True) for p in lo]
        self.opt_lo = torch.optim.Adam([{"params": self.lo_m, "base": a.lr_lora}], lr=a.lr_lora, betas=(0.9, 0.99)) if lo else None
        self.mine = [s for s in range(a.V) if s % dist.world == dist.rank]
        self.states = {}                                     # stream -> (item index, offset, state on CPU)
        tags = sorted(set(C.tags)); self.tags = tags; self.tix = {t: k for k, t in enumerate(tags)}

    def step_rows(self, s): return sum(self.rows_of[(it, a_, b_)] for st in self.steps[s] for (it, a_, b_) in st)

    def run_step(self, s, lr):
        """one optimizer step on plan step s at lr; returns per-tag (KL sum, rows) over all ranks and the wall"""
        a, stu = self.a, self.stu
        for o_ in (self.opt, self.opt_lo):
            if o_ is None: continue
            for g in o_.param_groups: g["lr"] = lr * g.get("base", self.a.lr) / self.a.lr     # adapters keep their ratio
            o_.zero_grad(set_to_none=True)
        for p in self.lo: p.grad = None
        N = self.step_rows(s)
        acc = torch.zeros(2 * len(self.tags) + 2, dtype=torch.float64)      # per tag (KL, rows), T term sum, prefix tokens
        t0 = time.time()
        for sid in self.mine:
            for (it, sa, sb) in self.steps[s][sid]:
                gi, i = self.C.items[it][0], self.C.items[it][1]
                prev = None
                if sa > 0:
                    k = self.states.get(sid)
                    if k is not None and k[0] == it and k[1] == sa: prev = to_dev(k[2], self.dev)
                    else: prev = prefix_state(stu, self.C.g, (gi, i), sa, a.len); acc[-1] += sa
                if self.tap is not None:
                    kl, m, nxt, ts = seg_loss(stu, self.C.g, (gi, i), sa, sb, prev, tap=self.tap)
                    if m: ((kl + a.tap_weight * ts) / N).backward()
                    acc[-2] += float(ts)
                else:
                    kl, m, nxt = seg_loss(stu, self.C.g, (gi, i), sa, sb, prev)
                    if m: (kl / N).backward()
                t = self.tix[self.C.tags[gi]]; acc[2 * t] += float(kl); acc[2 * t + 1] += m
                n_doc = self.C.items[it][2]
                if sb < n_doc: self.states[sid] = (it, sb, to_cpu(nxt))
                else: self.states.pop(sid, None)
                del prev, nxt
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).double().cpu() for p in self.P] + [acc])
        self.d.allsum(flat)
        o = 0; k = 0
        for p, il in zip(self.P, self.is_lo):
            n = p.numel()
            if il: p.grad = None; self.lo_m[k].grad = flat[o:o + n].view_as(p).float(); k += 1
            else: p.grad = flat[o:o + n].view_as(p).to(self.dev, torch.float32)
            o += n
        accs = flat[o:]
        self.opt.step()
        if self.opt_lo is not None:
            self.opt_lo.step()
            with torch.no_grad():
                for p, m in zip(self.lo, self.lo_m): p.copy_(m.to(p.device, non_blocking=True)); m.grad = None
        with torch.no_grad():
            for p in self.scal: p.clamp_(-a.trust, a.trust)                 # the log multipliers only (not adapters)
        return accs, time.time() - t0

    def snapshot(self):
        import copy
        def cp(o_):
            sd = o_.state_dict()
            return {"state": {k: {n: (v.detach().cpu().clone() if torch.is_tensor(v) else v) for n, v in s.items()} for k, s in sd["state"].items()},
                    "param_groups": copy.deepcopy(sd["param_groups"])}
        z = {"cur": self.stu.state(), "opt": cp(self.opt)}
        if self.opt_lo is not None: z["opt_lo"] = cp(self.opt_lo)
        return z

    def restore(self, z):
        self.stu.load_state(z["cur"]); self.opt.load_state_dict(z["opt"]); self.states = {}
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
        v = torch.tensor([float(sum(p.detach().double().sum() for p in self.P))], dtype=torch.float64)
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
                             layers=layers, quant_layers=quant_layers)
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
    log(f"corpus {len(C.items)} documents in {time.time() - t0:.0f} s: {json.dumps(C.summary())}")
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
    order = mix50_order(C, a.mix50) if a.mix50 is not None else np.random.default_rng(a.seed).permutation(len(C.items))
    budgets = [float(x) for x in a.budgets.split(",")]
    steps = D.plan(C.items, order, a.V, a.Q)
    rows_of, cum = {}, []
    tot = 0
    for st in steps:
        for segs in st:
            for (it, sa, sb) in segs:
                gi, i = C.items[it][0], C.items[it][1]
                rows_of[(it, sa, sb)] = len(C.g[gi].scored(i, sa, sb)[0]); tot += rows_of[(it, sa, sb)]
        cum.append(tot)
    cum = np.array(cum)
    last, br = schedule(cum, budgets, a.decay_frac)
    assert cum[-1] >= budgets[-1] or a.gate or a.profile, f"the data hold {cum[-1]} rows < budget {budgets[-1]}"
    fork_last = int(np.searchsorted(cum, (1 - a.decay_frac) * budgets[-1], side="left")) + 1
    share = {}
    for s in range(last):
        for segs in steps[s]:
            for (it, sa, sb) in segs: t = C.tags[C.items[it][0]]; share[t] = share.get(t, 0) + rows_of[(it, sa, sb)]
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
    z = {"taps": list(TAPS), "hidden": H, "rows": o, "docs": idx, "plan_last": last, "order": "mix50" if a.mix50 is not None else "shuffle"}
    if os.path.exists(ip): assert json.load(open(ip)) == z, "index differs from the existing one"
    else: json.dump(z, open(ip + ".part", "w")); os.replace(ip + ".part", ip)
    bf = os.path.join(out, "taps.bin")
    mm = np.memmap(bf, np.uint16, "r+" if os.path.exists(bf) else "w+", shape=(o, len(TAPS), H))
    pp = os.path.join(out, "progress.json"); done = json.load(open(pp))["done"] if os.path.exists(pp) else 0
    log(f"taps: {len(docs)} documents, {o} rows ({o * len(TAPS) * H * 2 / 2**30:.1f} GiB), {done} done")
    devs = [f"cuda:{k}" for k in range(torch.cuda.device_count())]
    tea = build_student(a, devs, log, lora=0, layers=TAPS[-1] + 1, quant_layers=[], served=False)
    use_seg_attention(tea)
    t0 = time.time(); ntok = 0
    for k in range(done, len(docs)):
        it = docs[k]; gi, i, n, _ = C.items[it]
        x = torch.from_numpy(np.ascontiguousarray(tea.emb_bf16[C.g[gi].ids(i, 0, n)])).view(BF)[None]
        cache = SegCache(None); pes = {}; rows = np.empty((len(J[k]), len(TAPS), H), np.uint16)
        jj = None
        for l, ly in enumerate(tea.layers):
            dev = tea.dev_of[l]
            if x.device != dev: x = x.to(dev)
            if dev not in pes:
                pid = torch.arange(n, device=dev).view(1, 1, -1).expand(4, 1, -1); pes[dev] = (tea.rotary[dev](x, pid[1:]), pid[0])
            x = ly(x, position_embeddings=pes[dev][0], attention_mask=None, position_ids=pes[dev][1], past_key_values=cache)
            if l in TAPS:
                jj = torch.from_numpy(J[k]).to(dev)
                rows[:, TAPS.index(l)] = x[0, jj].view(torch.int16).cpu().numpy().view(np.uint16)
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
    return ap


def main():
    a = parser().parse_args()
    d = Dist()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, f"run.log" if d.rank == 0 else f"run-r{d.rank}.log"), "a")
    def log(s, all_ranks=False):
        if d.rank != 0 and not all_ranks: return
        line = time.strftime("%H:%M:%S ") + s; print(line, flush=True); logf.write(line + "\n"); logf.flush()
    log("qad191.py " + json.dumps(vars(a)) + f" world {d.world}")
    mad_int_form()
    if a.conv == "shift": qmodel.use_shift_conv()
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
    json.dump({"monitor": mon, "budgets": budgets, "last": last, "branches": {str(k): v for k, v in br.items()},
               "fork_last": fork_last, "steps": len(steps), "e2e_from": k0, "share": P["share"]}, open(os.path.join(a.out, "plan.json"), "w"))
    dev = f"cuda:{d.local}"; torch.cuda.set_device(dev)
    for grp in range(0, d.world, a.build_group):                # ranks build in groups (each build holds the decoded trunk in host RAM once)
        if grp <= d.rank < grp + a.build_group: stu = build_student(a, dev, lambda s: log(s, True))
        d.barrier()
    apply196(a, stu, log)
    if a.export_only:
        o = os.path.join(a.out, f"ckpt-{budgets[-1]:.0e}")
        if d.rank == 0:
            write_ckpt(stu, o, {"budget": budgets[-1], "steps": 0, "rows": 0, "args": vars(a), "export_only": True})
            open(o + "/.done", "w").close()
        log(f"export-only (#196): {o}")
        return
    use_seg_attention(stu)
    if a.compile_scales: use_compiled_scales(stu, a.a8); log(f"QLinear scales compiled (a8 {a.a8})")
    elif a.a8 and not a.check: use_a8(stu)
    elif a.lora: use_lora32(stu); log("arm L: adapters in f32 (use_lora32)")
    if a.a8: log("arm Q: a8g128 fake-quant activations on every trunk linear (STE)")
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
    T = Trainer(a, stu, C, steps, rows_of, d, log, tap=tap)
    st = {"step": k0, "done_br": [], "wall": 0.0, "ema": None, "next_mon": a.mon_every}
    if os.path.exists(sp):
        z = torch.load(sp, weights_only=False); T.restore(z["snap"]); st = z["st"]; log(f"resumed at main step {st['step']}")
    else:
        v = T.monitor(mon) if mon else None
        log(json.dumps({"step": k0, "rows": 0, "mon_kl": v}))
        if d.rank == 0:
            with open(os.path.join(a.out, "mon.jsonl"), "a") as f: f.write(json.dumps({"step": k0, "rows": 0, "mon_kl": v, "what": "main"}) + "\n")
    def fixed_log(step, rows, what):
        if not fe: return
        z = T.fixed(fe)
        if d.rank == 0:
            with open(os.path.join(a.out, "fixed.jsonl"), "a") as f:
                f.write(json.dumps({"step": step, "rows": rows, "what": what, "kl": float(z[:, 0].sum() / z[:, 1].sum()),
                                    "sums": z[:, 0].round(6).tolist(), "n": z[:, 1].astype(int).tolist()}) + "\n")
    if not os.path.exists(sp): fixed_log(k0, int(cum[k0 - 1]) if k0 else 0, "main")
    def save():
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
        rec["peak_gib"] = round(torch.cuda.max_memory_allocated(T.dev) / 2**30, 2)
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
    while st["step"] < last:
        s = st["step"]
        for Tb, (fk, end) in br.items():                 # a branch forks at this step
            if s == fk - 1 and Tb not in st["done_br"] and not os.path.exists(os.path.join(a.out, f"ckpt-{Tb:.0e}", ".done")):
                save(); snap = T.snapshot()
                log(f"branch {Tb:.0e}: steps {fk} .. {end}")
                for bs in range(s, end):
                    lr = min(lr_main(bs, last, fork_last, a.lr, a.warmup), a.lr * (end - bs) / (end - s))   # linear to ~0 at end
                    accs, w = T.run_step(bs, lr); r = logstep(bs, lr, accs, w, f"branch {Tb:.0e}")
                    log(json.dumps(r))
                ckpt(Tb, end)
                T.restore(snap); st["done_br"].append(Tb); save()
        lr = lr_main(s - k0, last - k0, fork_last - k0, a.lr, a.warmup)
        accs, w = T.run_step(s, lr); st["step"] = s + 1
        r = logstep(s, lr, accs, w, "main")
        log(json.dumps(r))
        if cum[s] >= st["next_mon"] and mon:
            v = T.monitor(mon); st["next_mon"] += a.mon_every
            if d.rank == 0:
                with open(os.path.join(a.out, "mon.jsonl"), "a") as f: f.write(json.dumps({"step": s + 1, "rows": int(cum[s]), "mon_kl": v, "what": "main"}) + "\n")
            log(f"monitor at step {s + 1} ({cum[s]} rows, lr {lr:.2e}): {v}")
        if fe and st["step"] % a.fe_every == 0: fixed_log(st["step"], int(cum[s]), "main")
        if st["step"] % a.save_every == 0: save()
        if a.max_steps and st["step"] >= a.max_steps: save(); log("max steps"); return
        stop = torch.tensor([float(os.path.exists(os.path.join(a.out, "STOP")))]); d.allsum(stop)
        if float(stop) > 0: save(); log(f"STOP file: saved at main step {st['step']}, exiting"); return
    ckpt(budgets[-1], last); save()
    log(f"QAD191 DONE {last} steps, {cum[last - 1]} rows, train wall {st['wall']:.0f} s")


if __name__ == "__main__":
    main()
