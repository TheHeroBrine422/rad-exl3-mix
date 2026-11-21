#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""qad181.py OUT --model BF16 --level LEVEL --chunks DIR,... [--calib GOLDEN --pools POOLS] [options] -- #181: arm A of
#165 (trunk suh / svh log multipliers, frozen codes, the mix's quantized head frozen) at scale: one epoch over tens of
millions of stored bf16 golden positions, data-parallel over cards (torchrun; one whole int8 student a card).

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
        y = x @ qmodel.i8_to_bf16(self.wq)
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
        y = x @ qmodel.i8_to_bf16(self.wq)
        y = (y.view(-1, 128) @ self.Hb).view(-1, self.n)
        y = y * (self.sv() * self.s)
        if self.la is not None: y = y + (x0.to(BF) @ self.la.to(BF).T) @ self.lb.to(BF).T     # adapters (arm L) compose
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


def run_segment(stu, ids, a, prev=None, ckpt=True):
    """final-norm hidden states [n, H] of tokens a .. a + n - 1 of a document given the state after 0 .. a - 1 (prev:
    a SegCache.next of the segment before, or None at a = 0); returns (h, next state)"""
    q35 = stu.q35
    dev = stu.devs[0]
    assert len(stu.devs) == 1, "qad181: one card a student"
    x = stu.embed(ids)
    n = x.shape[1]
    pid = torch.arange(a, a + n, device=dev).view(1, 1, -1).expand(4, 1, -1)
    pe = stu.rotary[dev](x, pid[1:])
    cache = SegCache(prev)
    for i, ly in enumerate(stu.layers):
        kw = dict(position_embeddings=pe, attention_mask=None, position_ids=pid[0], past_key_values=cache)
        if ckpt and torch.is_grad_enabled(): x = checkpoint(lambda h, _l=ly, _k=kw: _l(h, **_k), x, use_reentrant=False)
        else: x = ly(x, **kw)
    return stu.fnorm(x)[0], cache.next


def seg_loss(stu, g, it, a, b, prev, ch=256, per_row=False):
    """(KL sum over the segment's rows, rows, next state); per_row: the KL vector (no grad)"""
    gi, i = it
    gold = g[gi]
    h, nxt = run_segment(stu, gold.ids(i, a, b), a, prev)
    j, r = gold.scored(i, a, b)
    if len(j) == 0: return (torch.zeros((), device=h.device) if not per_row else torch.zeros(0)), 0, nxt
    dev = h.device
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
    return tot, len(j), nxt


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
    def __init__(self, a, stu, C, steps, rows_of, dist, log):
        self.a, self.stu, self.C, self.steps, self.rows_of, self.d, self.log = a, stu, C, steps, rows_of, dist, log
        self.dev = stu.devs[0]
        self.P = list(stu.P.values()); self.names = list(stu.P.keys())
        sc = [p for n, p in zip(self.names, self.P) if not n.endswith((":la", ":lb"))]
        lo = [p for n, p in zip(self.names, self.P) if n.endswith((":la", ":lb"))]
        self.scal = sc
        self.opt = torch.optim.Adam([{"params": sc, "base": a.lr}] + ([{"params": lo, "base": a.lr_lora}] if lo else []), lr=a.lr, betas=(0.9, 0.99))
        self.mine = [s for s in range(a.V) if s % dist.world == dist.rank]
        self.states = {}                                     # stream -> (item index, offset, state on CPU)
        tags = sorted(set(C.tags)); self.tags = tags; self.tix = {t: k for k, t in enumerate(tags)}

    def step_rows(self, s): return sum(self.rows_of[(it, a_, b_)] for st in self.steps[s] for (it, a_, b_) in st)

    def run_step(self, s, lr):
        """one optimizer step on plan step s at lr; returns per-tag (KL sum, rows) over all ranks and the wall"""
        a, stu = self.a, self.stu
        for g in self.opt.param_groups: g["lr"] = lr * g.get("base", self.a.lr) / self.a.lr     # adapters keep their ratio
        self.opt.zero_grad(set_to_none=True)
        N = self.step_rows(s)
        acc = torch.zeros(2 * len(self.tags) + 1, dtype=torch.float64)
        t0 = time.time()
        for sid in self.mine:
            for (it, sa, sb) in self.steps[s][sid]:
                gi, i = self.C.items[it][0], self.C.items[it][1]
                prev = None
                if sa > 0:
                    k = self.states.get(sid)
                    if k is not None and k[0] == it and k[1] == sa: prev = to_dev(k[2], self.dev)
                    else: prev = prefix_state(stu, self.C.g, (gi, i), sa, a.len); acc[-1] += sa
                kl, m, nxt = seg_loss(stu, self.C.g, (gi, i), sa, sb, prev)
                if m: (kl / N).backward()
                t = self.tix[self.C.tags[gi]]; acc[2 * t] += float(kl); acc[2 * t + 1] += m
                n_doc = self.C.items[it][2]
                if sb < n_doc: self.states[sid] = (it, sb, to_cpu(nxt))
                else: self.states.pop(sid, None)
                del prev, nxt
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).double().cpu() for p in self.P] + [acc])
        self.d.allsum(flat)
        o = 0
        for p in self.P:
            n = p.numel(); p.grad = flat[o:o + n].view_as(p).to(self.dev, torch.float32); o += n
        accs = flat[o:]
        self.opt.step()
        with torch.no_grad():
            for p in self.scal: p.clamp_(-a.trust, a.trust)                 # the log multipliers only (not adapters)
        return accs, time.time() - t0

    def snapshot(self):
        import copy
        sd = self.opt.state_dict()
        return {"cur": self.stu.state(), "opt": {"state": {k: {n: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
                                                              for n, v in s.items()} for k, s in sd["state"].items()},
                                                    "param_groups": copy.deepcopy(sd["param_groups"])}}

    def restore(self, z):
        self.stu.load_state(z["cur"]); self.opt.load_state_dict(z["opt"]); self.states = {}

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
def main():
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
    a = ap.parse_args()
    d = Dist()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, f"run.log" if d.rank == 0 else f"run-r{d.rank}.log"), "a")
    def log(s, all_ranks=False):
        if d.rank != 0 and not all_ranks: return
        line = time.strftime("%H:%M:%S ") + s; print(line, flush=True); logf.write(line + "\n"); logf.flush()
    log("qad181.py " + json.dumps(vars(a)) + f" world {d.world}")
    mad_int_form()
    if a.conv == "shift": qmodel.use_shift_conv()
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
    order = np.random.default_rng(a.seed).permutation(len(C.items))
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
    log(f"plan: {len(steps)} steps of {a.V} x {a.Q} (data {cum[-1]} rows); main to step {last} ({cum[last - 1]} rows), decay from "
        f"step {fork_last}; branches " + ", ".join(f"{T:.0e}: fork {f} end {e} ({cum[e - 1]} rows)" for T, (f, e) in br.items()))
    mon = []
    if a.calib:                                                 # monitor: search-pool calib docs (the pilot's rule)
        pl = json.load(open(a.pools))["pool"]; gc = len(C.g) - 1
        cands = [i for i, p in enumerate(pl) if p == "search"]
        rng = np.random.default_rng(1000 + a.seed); rng.shuffle(cands)
        mon = [(gc, i) for i in cands if len(C.g[gc].scored(i, 0, min(a.len, C.g[gc].ntok(i)))[0]) >= 16][:a.monitor]
    json.dump({"monitor": mon, "budgets": budgets, "last": last, "branches": {str(k): v for k, v in br.items()},
               "fork_last": fork_last, "steps": len(steps)}, open(os.path.join(a.out, "plan.json"), "w"))
    # students: ranks build in groups (each build holds the decoded trunk in host RAM once)
    dev = f"cuda:{d.local}"; torch.cuda.set_device(dev)
    for grp in range(0, d.world, a.build_group):
        if grp <= d.rank < grp + a.build_group:
            t1 = time.time()
            stu = qmodel.Student(a.model, a.level, [dev], log=lambda s: log(s, True), head_dir=a.head, head_perm=a.head_perm, lora=a.lora)
            log(f"rank {d.rank}: student built in {time.time() - t1:.0f} s: {len(stu.qlin)} linears ({stu.cb}, "
                f"{sum(q.exact for q in stu.qlin.values())} int8-exact), {sum(p.numel() for p in stu.P.values())} values, "
                f"alloc {torch.cuda.memory_allocated(dev) / 2**30:.2f} GiB", True)
        d.barrier()
    use_seg_attention(stu)
    if a.compile_scales: use_compiled_scales(stu, a.a8); log(f"QLinear scales compiled (a8 {a.a8})")
    elif a.a8: use_a8(stu)
    if a.a8: log("arm Q: a8g128 fake-quant activations on every trunk linear (STE)")
    if a.gate: return gate(a, stu, C, log)
    if a.profile: return profile(a, stu, C, log)
    if a.bench: return bench(a, stu, C, log)
    if a.init:
        stu.load_state(torch.load(a.init, weights_only=False)["state"]); log(f"state from {a.init}")
    T = Trainer(a, stu, C, steps, rows_of, d, log)
    sp = os.path.join(a.out, "state.pt")
    st = {"step": 0, "done_br": [], "wall": 0.0, "ema": None, "next_mon": a.mon_every}
    if os.path.exists(sp):
        z = torch.load(sp, weights_only=False); T.restore(z["snap"]); st = z["st"]; log(f"resumed at main step {st['step']}")
    else:
        v = T.monitor(mon) if mon else None
        log(json.dumps({"step": 0, "rows": 0, "mon_kl": v}))
        if d.rank == 0:
            with open(os.path.join(a.out, "mon.jsonl"), "a") as f: f.write(json.dumps({"step": 0, "rows": 0, "mon_kl": v, "what": "main"}) + "\n")
    def fixed_log(step, rows, what):
        if not fe: return
        z = T.fixed(fe)
        if d.rank == 0:
            with open(os.path.join(a.out, "fixed.jsonl"), "a") as f:
                f.write(json.dumps({"step": step, "rows": rows, "what": what, "kl": float(z[:, 0].sum() / z[:, 1].sum()),
                                    "sums": z[:, 0].round(6).tolist(), "n": z[:, 1].astype(int).tolist()}) + "\n")
    if not os.path.exists(sp): fixed_log(0, 0, "main")
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
        lr = lr_main(s, last, fork_last, a.lr, a.warmup)
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
    log(f"QAD181 DONE {last} steps, {cum[last - 1]} rows, train wall {st['wall']:.0f} s")


if __name__ == "__main__":
    main()
