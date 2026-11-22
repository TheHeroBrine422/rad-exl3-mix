#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""probe218.py MODE OUT [options] -- #218 code-QAD feasibility probe: the released 2.50 student (w214's rad2level214 output, no
A8 in the loop = w214's arm 25A), the trunk linears of a few probe layers studied, everything else frozen as released.

  grad   OUT --chunk C: the KL gradient G = dL/dW_hat (L = the mean KL over the chunk's gen rows; W_hat the probe tensors'
         codebook values in the stored codes' basis, [k, n]) accumulated over chunk C's training documents (data218.json train,
         consecutive runs of --rows gen rows a chunk; a round merges several chunks: encode218.py), at the codes in --codes
         (else the released ones). Writes OUT/<key>.g
         (f32 [k, n]), OUT/samp.npz (per plan step: G's sum at a fixed random sample of entries a tensor, the step's rows: the
         SNR readout), OUT/grad.json. Plan V x Q positions a step (default 6 x 1024), segments carry the student's state.
  eval   OUT --set judge|tune --variants NAME=DIR,...: per-document KL sums of each variant (DIR: <key>.q files, f32 [k, n],
         the probe tensors' new codebook values; "rel" or an empty DIR = the released codes) on the set's documents (each from
         its start to the data's cap, 2,048-token segments carrying the state). Writes OUT/<NAME>.json. Restartable a variant.
  scale  OUT --chunks 0-8: scale-only QAD (w214's arm A recipe: suh / svh log multipliers on frozen codes, Adam lr 1e-3, trust
         0.3) on the probe layers' tensors only, over the same documents as the code-QAD rounds' chunks (plan 24 x 1024), warmup 2
         steps, constant, linear decay over the last 20 %; then eval on --set judge (OUT/scale.json) and its trained
         multipliers (OUT/params.pt).

Runtime cost of the probed change: none (codes or scales of the same format)."""
import argparse, concurrent.futures, json, os, sys, time, types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qad200"))
import qad200 as Q  # noqa: E402

qmodel, D = Q.qmodel, Q.D
BF = torch.bfloat16
LP = "model.language_model.layers"


def log_to(path):
    f = open(path, "a")
    def log(s):
        line = time.strftime("%H:%M:%S ") + s; print(line, flush=True); f.write(line + "\n"); f.flush()
    return log


def build(a, log):
    Q.mad_int_form(); qmodel.use_shift_conv()
    ns = argparse.Namespace(model=a.model, level=f"{a.lv}/trunk", head=f"{a.lv}/head", head_perm=f"{a.lv}/R16.npy", lora=0,
                            build_lock=None, emb=None, ab=f"{a.lv}/ab.safetensors")
    stu = Q.build_student(ns, "cuda:0", log)
    Q.use_seg_attention(stu)
    for p in stu.P.values(): p.requires_grad_(False)
    layers = {int(x) for x in a.layers.split(",")}
    stu.probe = [k for k in stu.qlin if int(k.split(".")[3]) in layers]
    stu.rel_wq = {k: stu.qlin[k].wq.detach().cpu().clone() for k in stu.probe}
    log(f"probe: {len(stu.probe)} tensors in layers {sorted(layers)}, {sum(stu.qlin[k].k * stu.qlin[k].n for k in stu.probe) / 1e9:.3f} G values")
    return stu


@torch.no_grad()
def set_codes(stu, d):
    """the probe tensors' codes from DIR/<key>.q (missing file or no DIR: the released codes)"""
    nq = 0
    for k in stu.probe:
        ql = stu.qlin[k]; f = os.path.join(d, k + ".q") if d else None
        if f and os.path.exists(f):
            w = torch.from_numpy(np.fromfile(f, np.float32).reshape(ql.k, ql.n))
            ii = qmodel.to_int8(w, stu.cb)
            assert ii is not None and ii[1] == ql.s, f"{f}: not on the int8 grid of step {ql.s}"
            ql.wq.copy_(ii[0].to(ql.wq.device)); nq += 1
        else:
            ql.wq.copy_(stu.rel_wq[k].to(ql.wq.device))
    return nq


def load_data(a):
    z = json.load(open(a.data))
    gold = D.Gold(a.calib, "calib")
    return z, gold


def chunk_docs(z, r, rows):
    """the documents of chunk r: consecutive runs of the train order with >= rows gen rows each"""
    out, acc, cur = [[]], 0, 0
    for i, n in zip(z["train"], z["train_rows"]):
        out[-1].append(i); acc += n
        if acc >= rows:
            out.append([]); acc = 0
            if len(out) > r + 1: break
    return out[r]


def plan_for(gold, docs, cap, V, Qn):
    items = [(0, i, min(cap, gold.ntok(i)), gold.nrows(i, cap)) for i in docs]
    steps = D.plan(items, np.arange(len(items)), V, Qn)
    return items, steps


def seg_rows(gold, items, it, sa, sb):
    return len(gold.scored(items[it][1], sa, sb)[0])


# ------------------------------------------------------------------ grad
class _GW(torch.autograd.Function):
    """y = x @ m (the probe tensor's int8 integers as bf16); the backward also hands x^T dy (bf16 [k, n]) to the tensor's sink"""
    @staticmethod
    def forward(ctx, x, dummy, ql):
        ctx.ql = ql; ctx.save_for_backward(x)
        return x @ qmodel.i8_to_bf16(ql.wq)

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors; ql = ctx.ql
        dy = dy.contiguous()
        dx = dy @ qmodel.i8_to_bf16(ql.wq).t()
        ql.sink.add(x.t() @ dy)
        return dx, None, None


class Sink:
    """the host f32 sum of a tensor's x^T dy (bf16 copies through a pinned buffer, summed on a worker thread) and, on the card,
    the per-step sum at nsamp fixed random entries"""
    POOL = concurrent.futures.ThreadPoolExecutor(2)

    def __init__(self, ql, nsamp, rng):
        self.acc = torch.zeros(ql.k, ql.n, dtype=torch.float32)
        self.pin = torch.empty(ql.k, ql.n, dtype=BF).pin_memory()
        dev = ql.wq.device
        self.idx = torch.from_numpy(rng.choice(ql.k * ql.n, nsamp, replace=False)).to(dev)
        self.samp = torch.zeros(nsamp, dtype=torch.float32, device=dev)
        self.fut = None; self.calls = 0

    def add(self, g):
        self.wait()
        self.samp += g.reshape(-1)[self.idx].float()
        self.pin.copy_(g, non_blocking=True)
        ev = torch.cuda.Event(); ev.record()
        self.fut = Sink.POOL.submit(self._acc, ev); self.calls += 1

    def _acc(self, ev):
        ev.synchronize(); self.acc.add_(self.pin)

    def wait(self):
        if self.fut is not None: self.fut.result(); self.fut = None


def use_grad_probe(stu, nsamp, seed):
    dummy = torch.zeros((), device=stu.devs[0], requires_grad=True)
    rng = np.random.default_rng(seed)
    def fwd(self, x):
        sh = x.shape
        x = (x.reshape(-1, self.k) * self.su()).to(BF)
        x = (x.view(-1, 128) @ self.Hb).view(-1, self.k)
        y = _GW.apply(x, dummy, self)
        y = (y.view(-1, 128) @ self.Hb).view(-1, self.n)
        y = y * (self.sv() * self.s)
        return y.to(BF).view(*sh[:-1], self.n)
    for k in stu.probe:
        ql = stu.qlin[k]; assert ql.la is None
        ql.sink = Sink(ql, nsamp, rng); ql.forward = types.MethodType(fwd, ql)


def run_grad(a, log):
    z, gold = load_data(a)
    docs = chunk_docs(z, a.chunk, a.rows) if not a.grad_set else z[a.grad_set][:a.grad_docs or None]
    items, steps = plan_for(gold, docs, z["cap"], a.V, a.Q)
    N = sum(it[3] for it in items)
    log(f"chunk {a.chunk}: {len(docs)} documents, {N} gen rows, {len(steps)} steps of {a.V} x {a.Q}")
    stu = build(a, log)
    nq = set_codes(stu, a.codes)
    log(f"codes: {nq} probe tensors from {a.codes or '(released)'}")
    use_grad_probe(stu, a.nsamp, 218)
    G = [gold]; states = {}; samp = {k: [] for k in stu.probe}; srows = []; kls = []
    t0 = time.time()
    for s, st in enumerate(steps):
        ts = time.time(); kl_s = 0.0; rows_s = 0
        for sid, segs in enumerate(st):
            for (it, sa, sb) in segs:
                i = items[it][1]; prev = None
                if sa > 0:
                    k = states.get(sid)
                    if k is not None and k[0] == it and k[1] == sa: prev = Q.to_dev(k[2], stu.devs[0])
                    else: prev = Q.prefix_state(stu, G, (0, i), sa, 2048)
                kl, m, nxt = Q.seg_loss(stu, G, (0, i), sa, sb, prev)
                if m: (kl / N).backward()
                kl_s += float(kl); rows_s += m
                if sb < items[it][2]: states[sid] = (it, sb, Q.to_cpu(nxt))
                else: states.pop(sid, None)
                del prev, nxt
        for k in stu.probe:
            ql = stu.qlin[k]; ql.sink.wait(); samp[k].append(ql.sink.samp.cpu().numpy() / N); ql.sink.samp.zero_()
        srows.append(rows_s); kls.append(kl_s)
        log(f"step {s + 1}/{len(steps)}: {rows_s} rows, KL {kl_s / max(1, rows_s):.6f}, {time.time() - ts:.1f} s, "
            f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    os.makedirs(a.out, exist_ok=True)
    meta = {}
    for k in stu.probe:
        ql = stu.qlin[k]; ql.sink.wait()
        g = (ql.sink.acc / ql.s).numpy()                                 # dL / dW_hat (W_hat = s m)
        g.tofile(os.path.join(a.out, k + ".g.part")); os.replace(os.path.join(a.out, k + ".g.part"), os.path.join(a.out, k + ".g"))
        meta[k] = {"k": ql.k, "n": ql.n, "K": ql.K, "s": ql.s, "calls": ql.sink.calls, "g_rms": float(np.sqrt((g.astype(np.float64) ** 2).mean()))}
    np.savez(os.path.join(a.out, "samp.npz"), rows=np.array(srows), **{k: np.stack(v) for k, v in samp.items()},
             **{k + ":idx": stu.qlin[k].sink.idx.cpu().numpy() for k in stu.probe})
    res = {"chunk": a.chunk, "docs": docs, "rows": N, "steps": len(steps), "V": a.V, "Q": a.Q, "codes": a.codes,
           "train_kl": float(sum(kls) / N), "sec": time.time() - t0, "tensors": meta}
    json.dump(res, open(os.path.join(a.out, "grad.json.part"), "w")); os.replace(os.path.join(a.out, "grad.json.part"), os.path.join(a.out, "grad.json"))
    log(f"GRAD DONE chunk {a.chunk}: {N} rows in {time.time() - t0:.0f} s, train KL {res['train_kl']:.6f}")


# ------------------------------------------------------------------ eval
@torch.no_grad()
def eval_docs(stu, gold, docs, cap, L=2048):
    out = []
    for i in docs:
        n = min(cap, gold.ntok(i)); st = None; p = 0; s = 0.0; m = 0
        while p < n:
            q = min(n, p + L)
            if n - q < 128: q = n
            kl, mm, st = Q.seg_loss(stu, [gold], (0, i), p, q, st); s += float(kl); m += mm; p = q
        out.append([s, m])
    return out


def run_eval(a, log, stu=None):
    z, gold = load_data(a)
    docs = z[a.set]
    stu = stu or build(a, log)
    os.makedirs(a.out, exist_ok=True)
    for v in a.variants.split(","):
        name, d = v.split("=", 1) if "=" in v else (v, "")
        f = os.path.join(a.out, f"{name}.json")
        if os.path.exists(f): log(f"{name}: done before"); continue
        nq = set_codes(stu, "" if d in ("", "rel") else d)
        t0 = time.time()
        r = eval_docs(stu, gold, docs, z["cap"])
        s = sum(x[0] for x in r); m = sum(x[1] for x in r)
        json.dump({"name": name, "codes": d, "probe_from_dir": nq, "set": a.set, "docs": docs, "sums": [x[0] for x in r],
                   "rows": [x[1] for x in r], "kl": s / m, "sec": time.time() - t0}, open(f + ".part", "w"))
        os.replace(f + ".part", f)
        log(f"EVAL {a.set} {name}: KL {s / m:.6f} over {m} rows ({len(docs)} docs, {nq} tensors from {d or 'rel'}), {time.time() - t0:.0f} s")
    return stu


# ------------------------------------------------------------------ scale
def run_scale(a, log):
    z, gold = load_data(a)
    lo, hi = (int(x) for x in a.chunks.split("-"))
    docs = [i for r in range(lo, hi + 1) for i in chunk_docs(z, r, a.rows)]
    items, steps = plan_for(gold, docs, z["cap"], 24, 1024)
    N = sum(it[3] for it in items)
    stu = build(a, log)
    if a.compile_scales: Q.use_compiled_scales(stu)
    keep = {n: p for n, p in stu.P.items() if n.split(":")[0] in set(stu.probe)}
    for p in keep.values(): p.requires_grad_(True)
    stu.P = keep
    log(f"scale-only: {len(keep)} multiplier vectors ({sum(p.numel() for p in keep.values())} values) on the probe tensors; "
        f"{len(docs)} documents, {N} rows, {len(steps)} steps of 24 x 1024")
    ta = argparse.Namespace(lr=a.lr, lr_lora=1e-4, wd=0.0, V=24, len=2048, trust=0.3, tap_weight=0.0)
    C = types.SimpleNamespace(items=items, g=[gold], tags=["calib"])
    rows_of = {(it, sa, sb): seg_rows(gold, items, it, sa, sb) for st in steps for segs in st for (it, sa, sb) in segs}
    T = Q.Trainer(ta, stu, C, steps, rows_of, Q.Dist(), log)
    nst = len(steps); warm = 2; fork = int(round(0.8 * nst))
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    for s in range(nst):
        lr = Q.lr_main(s, nst, fork, a.lr, warm)
        accs, w = T.run_step(s, lr)
        log(json.dumps({"step": s + 1, "lr": lr, "train_kl": float(accs[0] / max(1, accs[1])), "rows": int(accs[1]), "wall": round(w, 1),
                        "gnorm": round(T.gnorm, 6)}))
    torch.save({"state": stu.state(), "steps": nst, "rows": N}, os.path.join(a.out, "params.pt"))
    log(f"scale-only trained in {time.time() - t0:.0f} s")
    a.variants = "scale=rel"
    run_eval(a, log, stu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("grad", "eval", "scale")); ap.add_argument("out")
    ap.add_argument("--model", default="/data/workspace/models/qwen38-27b-bf16")
    ap.add_argument("--lv", default="/data/w218/lv/2.50"); ap.add_argument("--calib", default="/data/challenge/golden/calib")
    ap.add_argument("--data", default="/data/w218/data218.json"); ap.add_argument("--layers", default="8,23,40,55")
    ap.add_argument("--codes", default=""); ap.add_argument("--chunk", type=int, default=0); ap.add_argument("--rows", type=int, default=100000)
    ap.add_argument("--V", type=int, default=6); ap.add_argument("--Q", type=int, default=1024); ap.add_argument("--nsamp", type=int, default=20000)
    ap.add_argument("--set", default="judge"); ap.add_argument("--variants", default="rel=rel")
    ap.add_argument("--chunks", default="0-8"); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--compile-scales", action="store_true")
    ap.add_argument("--grad-set", default="", help="grad: a data218 set (e.g. tune) instead of a train chunk (a check gradient, never a step)")
    ap.add_argument("--grad-docs", type=int, default=0, help="grad: the first N documents of --grad-set")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = log_to(os.path.join(a.out, f"{a.mode}.log"))
    log("probe218.py " + json.dumps(vars(a)))
    torch.manual_seed(0)
    {"grad": run_grad, "eval": run_eval, "scale": run_scale}[a.mode](a, log)


if __name__ == "__main__":
    main()
