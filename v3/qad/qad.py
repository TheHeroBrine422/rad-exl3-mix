#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""qad.py OUT --model BF16 --level LEVEL --golden GOLDEN [options] -- scale-only end-to-end QAD of a 27B mix (#165).

Student = qmodel.Student (frozen trellis codes, trainable log multipliers on suh / svh, arm C also q_norm / k_norm /
final norm gains). Loss = KL(p_bf16 || q_student) on the golden's top-512 + tail cell, the challenge scorer's formula
(proxy2b/common.py score_rows: head terms + tail cell, renormalised by S), summed over the scored (gen) rows of each
window and averaged over the rows of a step: no teacher forward for the KL term. Arm B adds lambda x the mean over
layers of the per-block hidden-state MSE vs the bf16 teacher run alongside (normalised by the teacher's mean square;
lambda fixed at step 1 as KL / MSE of the first step, so both terms start equal).

Windows: the first --len tokens of a document (the golden's rows were computed with the whole document before them,
so only document-start windows match the stored distributions); training docs from --pools' qad pool, monitor docs
from its search pool (--monitor-docs: a fixed list, or N = a seeded draw), never verify / fit. Adam on the log
parameters, warmup + cosine, |log| <= --trust, eval on the monitor every --eval-every steps, early stop after
--patience evals without a new best; OUT/state.pt resumes bit-exactly (data order from the seed); OUT/params.pt = the
best monitor state; OUT/train.jsonl the curve; OUT/scales/<key>.safetensors (r, c multipliers: rxq.swj apply_level's
format) and OUT/norms.safetensors (arm C) written at the end.

--head DIR [--head-perm R16.npy] [--head-train]: the student's head is that quantized donor (stored row order, targets
mapped by the perm), frozen (control A16) or with its r / c trained (arm H; OUT/head/ = its healed suh / svh).
--lora R: rank-R adapters on every trunk linear (arm L; lr --lr-lora, no trust clamp; OUT/lora.safetensors).
--score NAME=GOLDEN ... --init PARAMS: no training, the student (with PARAMS' state, or none = unhealed) scored on every
document of each golden whole (torch path): OUT/tscore-NAME.npz (per-doc KL sums / rows, per-row KL and position).
--gate: identity + timing checks (decode exactness, one window forward / backward per arm), then exit."""
import argparse, json, math, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qmodel  # noqa: E402

TOP = 512
REC = np.dtype([("id", "<i4"), ("lp", "<f4")])


# ------------------------------------------------------------------ data
class Golden:
    """radiance-kld-top-1 dir: row of (doc i, position j) = row0 + j - score_from, for score_from <= j <= tokens - 2"""
    def __init__(self, d):
        self.d = d
        self.m = json.load(open(f"{d}/kld.json")); self.docs = self.m["docs"]
        R = self.m["rows"]
        self.tok = np.memmap(f"{d}/tokens.i32", np.int32, "r")
        self.gen = np.memmap(f"{d}/gen.u8", np.uint8, "r") if os.path.exists(f"{d}/gen.u8") else None
        self.top = np.memmap(f"{d}/top.bin", REC, "r", shape=(R, TOP))
        self.rows = np.memmap(f"{d}/rows.f32", np.float32, "r", shape=(R, 3))
        self.off = np.concatenate([[0], np.cumsum([x["tokens"] for x in self.docs])])

    def window(self, i, L):
        """(ids [n], scored positions [m], golden rows [m]) of doc i's first L tokens"""
        d = self.docs[i]
        n = min(L, d["tokens"])
        ids = np.asarray(self.tok[self.off[i]:self.off[i] + n], np.int64)
        j = np.arange(d["score_from"], min(n, d["tokens"] - 1))
        r = d["row0"] + j - d["score_from"]
        if self.gen is not None and len(r):
            g = np.asarray(self.gen[r[0]:r[-1] + 1]).astype(bool); j, r = j[g], r[g]
        return ids, j, r

    def targets(self, r, dev):
        t = np.asarray(self.top[r[0]:r[-1] + 1])[r - r[0]]
        tid = torch.from_numpy(np.ascontiguousarray(t["id"])).long().to(dev)
        tlp = torch.from_numpy(np.ascontiguousarray(t["lp"])).float().to(dev)
        ptail = torch.from_numpy(np.ascontiguousarray(self.rows[r[0]:r[-1] + 1, 2][r - r[0]])).float().to(dev)
        return tid, tlp, ptail


def pick_docs(golden, pools, pool, n, seed, L, min_rows=16):
    """docs of a pool (golden doc i = calib doc i: the whole-calib golden) with >= min_rows scored rows in their first
    L tokens, seeded order, first n (0 = all)"""
    c = [i for i, p in enumerate(pools) if p == pool]
    rng = np.random.default_rng(seed); rng.shuffle(c)
    out = []
    for i in c:
        if len(golden.window(i, L)[1]) >= min_rows: out.append(int(i))
        if n and len(out) == n: break
    return out


# ------------------------------------------------------------------ loss
def chunk_kl(h, head, tid, tlp, ptail, per_row=False):
    """sum over rows of the scorer's KL (score_rows) from student hidden rows h [m, H], in f32; head = a weight or a
    module (a QLinear head)"""
    z = (head(h) if isinstance(head, torch.nn.Module) else F.linear(h, head)).float()
    lse = torch.logsumexp(z, -1)
    lq_head = z.gather(1, tid) - lse[:, None]
    lq_tail = torch.logsumexp(z.scatter(1, tid, float("-inf")), -1) - lse
    p = tlp.exp()
    pt = ptail.clamp_min(0)
    A = (p * (tlp - lq_head)).sum(1) + torch.where(pt > 0, pt * (torch.log(pt.clamp_min(1e-30)) - lq_tail), torch.zeros_like(pt))
    S = p.sum(1) + pt
    kl = A / S - torch.log(S)
    return kl if per_row else kl.sum()


def window_loss(stu, golden, i, L, ch=256, mse=False, per_row=False):
    """(KL sum over the scored rows, rows, [per-layer MSE terms]) of doc i's first window; per_row: the rows' KL
    vector instead of the sum (no grad)"""
    ids, j, r = golden.window(i, L)
    dev = stu.devs[-1]
    out = stu.run(ids, teacher_mse=mse)
    h, mses = out if mse else (out, None)
    h = h[torch.from_numpy(j).to(dev)]
    tid, tlp, ptail = golden.targets(r, dev)
    if stu.v2row is not None: tid = stu.v2row[tid]              # vocab ids -> the quantized head's stored rows
    if per_row:
        return torch.cat([chunk_kl(h[a:a + ch], stu.head, tid[a:a + ch], tlp[a:a + ch], ptail[a:a + ch], True)
                          for a in range(0, len(j), ch)]), j, None
    tot = 0.0
    for a in range(0, len(j), ch):
        args = (h[a:a + ch], stu.head, tid[a:a + ch], tlp[a:a + ch], ptail[a:a + ch])
        tot = tot + (checkpoint(chunk_kl, *args, use_reentrant=False) if torch.is_grad_enabled() else chunk_kl(*args))
    return tot, len(j), mses


@torch.no_grad()
def eval_kl(stu, golden, docs, L):
    s = n = 0.0
    per = []
    for i in docs:
        k, m, _ = window_loss(stu, golden, i, L)
        s += float(k); n += m; per.append((i, float(k), m))
    return s / n, per


# ------------------------------------------------------------------ training
def lr_at(step, total, lr, warm):
    if step < warm: return lr * (step + 1) / warm
    return lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))


def save_atomic(obj, path):
    torch.save(obj, path + ".part"); os.replace(path + ".part", path)


def train(stu, golden, items, mon, a, log):
    params = list(stu.P.values())
    lora = [p for n, p in stu.P.items() if n.endswith((":la", ":lb"))]
    scal = [p for n, p in stu.P.items() if not n.endswith((":la", ":lb"))]
    groups = [{"params": scal, "base_lr": a.lr}] + ([{"params": lora, "base_lr": a.lr_lora}] if lora else [])
    opt = torch.optim.Adam(groups, lr=a.lr, betas=(0.9, 0.99))
    order = np.concatenate([np.random.default_rng(a.seed + e).permutation(len(items)) for e in range(a.epochs)])
    total = len(order) // a.accum
    if a.max_steps: total = min(total, a.max_steps)
    sp = os.path.join(a.out, "state.pt")
    st = {"step": 0, "best": None, "best_state": None, "best_step": 0, "bad": 0, "base_val": None, "lam": None,
          "tokens": 0, "rows": 0, "wall": 0.0}
    if os.path.exists(sp):
        z = torch.load(sp, weights_only=False); stu.load_state(z["cur"]); opt.load_state_dict(z["opt"]); st = z["st"]
        log(f"resumed at step {st['step']}")
    if st["base_val"] is None:
        t0 = time.time(); v, per = eval_kl(stu, golden, mon, a.len)
        st["base_val"] = st["best"] = v; st["best_state"] = stu.state()
        json.dump({"step": 0, "val": v, "per_doc": per}, open(os.path.join(a.out, "mon-step0.json"), "w"))
        log(json.dumps({"step": 0, "val_kl": v, "eval_s": round(time.time() - t0, 1)}))
    tl, tm, t0 = [], [], time.time()
    while st["step"] < total and st["bad"] < a.patience:
        s = st["step"]
        for g in opt.param_groups: g["lr"] = lr_at(s, total, g["base_lr"], a.warmup)
        opt.zero_grad(set_to_none=True)
        batch = [items[k] for k in order[s * a.accum:(s + 1) * a.accum]]
        nrows = sum(len(golden.window(i, a.len)[1]) for i in batch)
        ls = lm = 0.0
        for i in batch:
            kl, m, mses = window_loss(stu, golden, i, a.len, mse=a.arm == "B")
            loss = kl / nrows
            if mses:
                mterm = torch.stack([x.to(stu.devs[-1]) for x in mses]).mean() / len(batch)
                if st["lam"] is None: st["lam"] = -1.0          # set below from this step's terms
                lam = st["lam"] if st["lam"] > 0 else 0.0
                loss = loss + lam * mterm; lm += float(mterm)
            loss.backward()
            ls += float(kl); st["tokens"] += min(a.len, golden.docs[i]["tokens"]); st["rows"] += m
        if st["lam"] == -1.0:                                   # arm B, first step: lambda = KL / MSE, then this step's
            st["lam"] = (ls / nrows) / max(lm, 1e-12)          # gradient had only the KL term (logged)
            log(f"arm B lambda = {st['lam']:.4g} (step-1 KL {ls / nrows:.4g}, MSE {lm:.4g})")
        opt.step()
        with torch.no_grad():
            for p in scal: p.clamp_(-a.trust, a.trust)
        st["step"] = s + 1; tl.append(ls / nrows); tm.append(lm)
        if st["step"] % a.eval_every == 0 or st["step"] == total:
            st["wall"] += time.time() - t0
            te = time.time(); v, per = eval_kl(stu, golden, mon, a.len); te = time.time() - te
            if v < st["best"]: st["best"], st["best_state"], st["best_step"], st["bad"] = v, stu.state(), st["step"], 0
            else: st["bad"] += 1
            mx = max(float(p.detach().abs().max()) for p in scal)
            rec = {"step": st["step"], "train_kl": float(np.mean(tl)), "train_mse": float(np.mean(tm)) if a.arm == "B" else None,
                   "val_kl": v, "val_rel": v / st["base_val"], "best": st["best"], "best_step": st["best_step"],
                   "max_abs_log": mx, "lr": opt.param_groups[0]["lr"], "tokens": st["tokens"], "rows": st["rows"],
                   "train_wall_s": round(st["wall"], 1), "eval_s": round(te, 1),
                   "s_per_step": round(st["wall"] / st["step"], 2)}
            with open(os.path.join(a.out, "train.jsonl"), "a") as fh: fh.write(json.dumps(rec) + "\n")
            log(json.dumps(rec)); tl, tm = [], []
            save_atomic({"cur": stu.state(), "opt": opt.state_dict(), "st": st}, sp)
            t0 = time.time()
    return st


def write_outputs(stu, st, a):
    """params.pt (best state), scales/<key>.safetensors (r, c multipliers as rxq.swj.apply_level reads them), norms"""
    from safetensors.torch import save_file
    stu.load_state(st["best_state"])
    save_atomic({"state": st["best_state"], "meta": {"args": vars(a), "base_val": st["base_val"], "best_val": st["best"],
                                                     "best_step": st["best_step"], "steps": st["step"], "lam": st["lam"],
                                                     "tokens": st["tokens"], "rows": st["rows"]}}, os.path.join(a.out, "params.pt"))
    sd = os.path.join(a.out, "scales"); os.makedirs(sd, exist_ok=True)
    for k, q in stu.qlin.items():
        with torch.no_grad():
            r = q.r.exp().cpu() if q.r is not None else torch.ones(q.k)
            c = q.c.exp().cpu() if q.c is not None else torch.ones(q.n)
        save_file({"r": r.float().contiguous(), "c": c.float().contiguous()}, f"{sd}/{k}.safetensors")
    if stu.norm: save_file({k: v.contiguous() for k, v in stu.export_norms().items()}, os.path.join(a.out, "norms.safetensors"))
    if stu.hq is not None and stu.hq.r is not None:
        suh, svh = stu.hq.export()
        os.makedirs(os.path.join(a.out, "head"), exist_ok=True)
        save_file({"suh": suh.contiguous(), "svh": svh.contiguous(), "r": stu.hq.r.detach().exp().float().cpu().contiguous(),
                   "c": stu.hq.c.detach().exp().float().cpu().contiguous()}, os.path.join(a.out, "head", "lm_head.safetensors"))
    lo = {k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in stu.P.items() if k.endswith((":la", ":lb"))}
    if lo: save_file(lo, os.path.join(a.out, "lora.safetensors"))


# ------------------------------------------------------------------ gate
def gate(stu, golden, items, a, log):
    """(1) every QLinear W equals rxq.deq's W (f32, same decode) on a probe; (2) one window: forward / backward wall,
    peak memory per device, gradient norm"""
    worst = 0.0
    for k, q in list(stu.qlin.items())[:a.gate_tensors]:
        dev = q.wq.device
        x = torch.randn(64, q.k, device=dev).to(torch.bfloat16)
        Wref = stu.lev.get(k).T.to(dev)                                   # [k, n] f32 = deq's W
        yref = x.float() @ Wref
        y = q(x).float()
        rel = float((y - yref).norm() / yref.norm()); worst = max(worst, rel)
        del Wref, yref, y; torch.cuda.empty_cache()
        if not q.exact: log(f"  {k}: not an integer codebook form (bf16 W_hat)")
    log(f"gate decode: max rel |y - x W_deq| over {min(a.gate_tensors, len(stu.qlin))} tensors {worst:.2e} (bf16 activations)")
    if stu.hq is not None:                              # the head vs ourq --save-wq's dequant (f16, stored rows), 4,096 rows
        from safetensors import safe_open
        with safe_open(os.path.join(a.head, "wq", "lm_head.safetensors"), "pt") as f: Wq = f.get_tensor("weight")
        dev = stu.hq.wq.device; rows = torch.randperm(Wq.shape[0], generator=torch.Generator().manual_seed(0))[:4096]
        x = torch.randn(64, stu.hq.k, device=dev).to(torch.bfloat16)
        yref = x.float() @ Wq[rows].float().to(dev).T; y = stu.hq(x).float()[:, rows.to(dev)]
        log(f"gate head: rel |y - x W_wq| {float((y - yref).norm() / yref.norm()):.2e} (4,096 stored rows; bf16 activations)")
        del Wq, yref, y
    for d in stu.devs: torch.cuda.reset_peak_memory_stats(d)
    log("gate alloc GiB after the decode check: " + ", ".join(f"{torch.cuda.memory_allocated(d) / 2**30:.2f} / reserved {torch.cuda.memory_reserved(d) / 2**30:.2f}" for d in stu.devs))
    i = max(items[:64], key=lambda d: (len(golden.window(d, a.len)[0]), len(golden.window(d, a.len)[1])))   # the longest window
    for rep in range(2):
        for d in stu.devs: torch.cuda.synchronize(d)
        t0 = time.time()
        kl, m, mses = window_loss(stu, golden, i, a.len, mse=a.arm == "B")
        loss = kl / m + (torch.stack([x.to(stu.devs[-1]) for x in mses]).mean() if mses else 0.0)
        for d in stu.devs: torch.cuda.synchronize(d)
        tf = time.time() - t0
        loss.backward()
        for d in stu.devs: torch.cuda.synchronize(d)
        tb = time.time() - t0 - tf
        g = sum(float(p.grad.norm()) ** 2 for p in stu.P.values() if p.grad is not None) ** 0.5
        ng = sum(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in stu.P.values())
        log(f"gate window {rep}: doc {i} {len(golden.window(i, a.len)[0])} tokens {m} rows, KL/row {float(kl) / m:.5f}, "
            f"fwd {tf:.2f} s, bwd {tb:.2f} s, grad norm {g:.3e}, {ng}/{len(stu.P)} params with grad")
        for p in stu.P.values(): p.grad = None
    log("gate peak GiB: " + ", ".join(f"{d}: {torch.cuda.max_memory_allocated(d) / 2**30:.2f}" for d in stu.devs))


@torch.no_grad()
def score(stu, a, log):
    """every document of each NAME=GOLDEN whole: OUT/tscore-NAME.npz (doc KL sums, rows, per-row KL, position, doc)"""
    for spec in a.score:
        name, gd = spec.split("=", 1)
        o = os.path.join(a.out, f"tscore-{name}.npz")
        if os.path.exists(o): log(f"score {name}: done"); continue
        g = Golden(gd); t0 = time.time(); S, N, KL, POS, DOC = [], [], [], [], []
        for i in range(len(g.docs)):
            if len(g.window(i, 1 << 30)[1]) == 0:      # a document without gen rows: kept (0, 0) so docs stay paired
                S.append(0.0); N.append(0); continue
            kl, j, _ = window_loss(stu, g, i, 1 << 30, per_row=True)
            kl = kl.double().cpu().numpy(); S.append(kl.sum()); N.append(len(kl)); KL.append(kl); POS.append(j); DOC.append(np.full(len(j), i))
        np.savez(o + ".part.npz", sums=np.array(S), rows=np.array(N), kl=np.concatenate(KL), pos=np.concatenate(POS), doc=np.concatenate(DOC))
        os.replace(o + ".part.npz", o)
        log(f"score {name}: KL/row {sum(S) / sum(N):.6f} over {sum(N)} rows, {len(N)} docs, {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("--model", required=True); ap.add_argument("--level", required=True); ap.add_argument("--golden", required=True)
    ap.add_argument("--pools", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "eval", "pools", "pools-v3.json"))
    ap.add_argument("--pool-override", default=None, help="a json list of pool names per golden doc (tests on a non-calib golden)")
    ap.add_argument("--devs", default="cuda:0,cuda:1")
    ap.add_argument("--arm", choices=("A", "B", "C"), default="A")
    ap.add_argument("--layers", type=int, default=None, help="truncated model: the first N layers (tests)")
    ap.add_argument("--len", type=int, default=2048); ap.add_argument("--ntrain", type=int, default=0)
    ap.add_argument("--accum", type=int, default=8); ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--trust", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=16); ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--monitor", type=int, default=32, help="monitor docs drawn from the search pool")
    ap.add_argument("--monitor-docs", default=None, help="explicit monitor doc indices (comma list)")
    ap.add_argument("--no-gauge", action="store_true")
    ap.add_argument("--gate", action="store_true"); ap.add_argument("--gate-tensors", type=int, default=8)
    ap.add_argument("--head", default=None, help="quantized head dir (ourq --only-head) for the student's lm_head")
    ap.add_argument("--head-perm", default=None, help="the head's row order (stored row s = vocab PERM[s])")
    ap.add_argument("--head-train", action="store_true", help="train the head's r / c (arm H)")
    ap.add_argument("--lora", type=int, default=0); ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--score", action="append", default=[], help="NAME=GOLDEN: score only (torch path), every doc whole")
    ap.add_argument("--init", default=None, help="a run's params.pt: start (or score) from its state")
    ap.add_argument("--conv", choices=("torch", "shift"), default="torch", help="GDN causal conv: transformers' F.conv1d "
                    "(MIOpen) or qmodel.shift_conv1d (f32 shifted adds; arm W from step 33, amendment 3)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "run.log"), "a")
    def log(s):
        line = time.strftime("%H:%M:%S ") + s; print(line, flush=True); logf.write(line + "\n"); logf.flush()
    log("qad.py " + json.dumps(vars(a)))
    if a.conv == "shift": qmodel.use_shift_conv(); log("GDN causal conv: qmodel.shift_conv1d")
    log(f"blas: {torch.backends.cuda.preferred_blas_library()} (TORCH_BLAS_PREFER_HIPBLASLT={os.environ.get('TORCH_BLAS_PREFER_HIPBLASLT')})")
    golden = Golden(a.golden)
    pools = json.load(open(a.pool_override)) if a.pool_override else json.load(open(a.pools))["pool"]
    assert len(pools) == len(golden.docs), (len(pools), len(golden.docs))
    if not a.score:                                     # score mode: no training data (and docs.json stays the run's)
        items = pick_docs(golden, pools, "qad", a.ntrain, a.seed, a.len)
        mon = [int(x) for x in a.monitor_docs.split(",")] if a.monitor_docs else pick_docs(golden, pools, "search", a.monitor, 1000 + a.seed, a.len)
        assert all(pools[i] == "qad" for i in items) and all(pools[i] == "search" for i in mon), "pool discipline"
        ntok = sum(min(a.len, golden.docs[i]["tokens"]) for i in items)
        json.dump({"train_docs": items, "monitor_docs": mon}, open(os.path.join(a.out, "docs.json"), "w"))
        log(f"{len(items)} training docs ({ntok} tokens / epoch), {len(mon)} monitor docs")
    t0 = time.time()
    stu = qmodel.Student(a.model, a.level, a.devs.split(","), layers=a.layers, norms=a.arm == "C", gauge=not a.no_gauge,
                         teacher=a.arm == "B", log=log, head_dir=a.head, head_perm=a.head_perm, head_train=a.head_train,
                         lora=a.lora)
    log(f"student built in {time.time() - t0:.0f} s: {len(stu.qlin)} quantized linears ({stu.cb}, "
        f"{sum(q.exact for q in stu.qlin.values())} int8-exact), {len(stu.P)} parameter vectors, "
        f"{sum(p.numel() for p in stu.P.values())} values; layers per device "
        + ", ".join(f"{d}: {sum(x == d for x in stu.dev_of)}" for d in stu.devs) + "; alloc GiB "
        + ", ".join(f"{torch.cuda.memory_allocated(d) / 2**30:.2f}" for d in stu.devs))
    if a.init:
        z = torch.load(a.init, weights_only=False)["state"]
        miss = [n for n in stu.P if n not in z]; assert not miss, f"--init lacks {miss[:4]}"
        stu.load_state(z); log(f"state from {a.init} ({len(z)} tensors)")
    if a.gate: return gate(stu, golden, items, a, log)
    if a.score: return score(stu, a, log)
    st = train(stu, golden, items, mon, a, log)
    write_outputs(stu, st, a)
    log(f"QAD DONE arm {a.arm} steps {st['step']} (best {st['best_step']}) monitor KL {st['base_val']:.6f} -> "
        f"{st['best']:.6f} ({st['best'] / st['base_val'] - 1:+.2%}), {st['tokens']} tokens, {st['wall']:.0f} s train wall")


if __name__ == "__main__":
    main()
