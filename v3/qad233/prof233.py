#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""prof233.py OUT <qad220 flags> -- w233: where a QAD segment's time goes, on ONE card (the student whole: screen mode's form).

Builds the student as qad220's main does (same flags: --a8-int8 --compile-scales --compile-ew ...), then on one document with >= P + 2 Q tokens:
  base     a Q-token segment after a P-token prefix: forward (layers checkpointed) + KL + backward  [trainer default]
  nockpt   the same with no checkpointing (activations kept): time, peak memory, and the gradient vs base (max abs / rel norm diff)
  fwd      forward only (no grad): one pass
  headkl   the head + top-512 KL forward + backward on fixed final hidden states (checkpointed chunks, as seg_kl)
  prof     torch.profiler over one base rep: device time by kernel family (int8 GEMM, quant, bf16 GEMMs, i8->bf16, attention, GDN, elementwise)
Writes OUT/prof233.json + OUT/prof233-kernels.txt. Env W233_Q (default 1024), W233_P (default 0 and 7168), W233_REPS (default 4)."""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qad220"))
import qad220 as T  # noqa: E402

FAM = [("int8_gemm", ("_mm_a8_k",)), ("a8_quant", ("_quant_a8_k",)), ("i8_to_bf16", ("i8_to_bf16",)),
       ("gdn_fla", ("chunk_", "fused_recurrent", "solve_tril", "wy_fast", "l2norm", "gated_delta", "kkt", "cumsum")),
       ("attention", ("attn", "flash", "fmha", "sdpa", "_efficient_attention", "ck_tile")),
       ("bf16_gemm", ("Cijk", "gemm", "Gemm", "GEMM", "matmul", "MT64", "MT128", "MT256", "addmm")),
       ("triton_inductor", ("triton_",)), ("copy", ("copy", "Copy", "memcpy", "Memcpy", "fill", "Fill")),
       ("reduce", ("reduce", "Reduce", "softmax", "logsumexp", "norm")), ("elementwise", ("elementwise", "Elementwise", "vectorized", "unrolled"))]


def family(name):
    if name == "_k": return "i8_to_bf16"                 # qmodel.i8_to_bf16's Triton copy
    for f, keys in FAM:
        if any(k in name for k in keys): return f
    return "other"


def ev():
    e = torch.cuda.Event(enable_timing=True); e.record(); return e


def main():
    a = T.parser().parse_args(sys.argv[1:])
    if a.a8_int8: a.a8 = True
    os.makedirs(a.out, exist_ok=True)
    def log(s, *_):
        line = time.strftime("%H:%M:%S ") + s; print(line, flush=True)
        with open(os.path.join(a.out, "prof233.log"), "a") as f: f.write(line + "\n")
    T.mad_int_form()
    if a.conv == "shift": T.qmodel.use_shift_conv()
    if a.compile_ew: T.set_dynamo_limit(a.dynamo_limit); T.qmodel.use_compiled_elementwise()
    d = T.Dist()
    P = T.build_data(a, d, log); C = P["C"]
    dev = "cuda:0"; torch.cuda.set_device(dev)
    stu = T.build_student(a, dev, log)
    T.apply196(a, stu, log); T.ATTN_GQA[0] = a.attn_gqa
    stu.mask = T.apply_trainable(a, stu, log); T.set_first_train(stu, log, a.no_frozen_prefix); T.use_seg_attention(stu)
    if a.compile_scales: T.use_compiled_scales(stu, a.a8, a.a8_int8, a.dynamo_limit)
    elif a.a8: T.use_a8(stu, a.a8_int8)
    Q = int(os.environ.get("W233_Q", 1024)); reps = int(os.environ.get("W233_REPS", 4))
    Ps = [int(x) for x in os.environ.get("W233_P", "0,7168").split(",")]
    gi, i, n, _ = max(C.items, key=lambda x: x[2])
    log(f"document {C.g[gi].name}:{i} ({n} tokens); Q {Q}, prefixes {Ps}, reps {reps}")
    params = [p for p in stu.P.values() if p.requires_grad]
    out = {"doc": [C.g[gi].name, int(i), int(n)], "Q": Q, "runs": {}}

    def seg(pre, prev, ckpt, grad=True):
        ids = C.g[gi].ids(i, pre, pre + Q)
        with torch.set_grad_enabled(grad):
            h, _ = T.run_segment(stu, ids, pre, prev, ckpt=ckpt)
            kl, m = T.seg_kl(stu, C.g[gi], i, pre, pre + Q, h)
        return kl, m

    def grads():
        g = torch.cat([p.grad.reshape(-1).float() for p in params]); [setattr(p, "grad", None) for p in params]; return g

    for pre in Ps:
        prev = T.prefix_state(stu, C.g, (gi, i), pre, a.len) if pre else None
        res = {}
        for mode in ("base", "nockpt", "fwd"):
            ts, tf, tb = [], [], []
            torch.cuda.reset_peak_memory_stats(dev); m0 = torch.cuda.memory_allocated(dev)
            try:
                for r in range(reps + 1):
                    torch.cuda.synchronize(); e0 = ev()
                    kl, m = seg(pre, prev, ckpt=(mode == "base"), grad=(mode != "fwd"))
                    e1 = ev()
                    if mode != "fwd": (kl / m).backward()
                    e2 = ev(); torch.cuda.synchronize()
                    g = grads() if mode != "fwd" else None
                    if r: ts.append(e0.elapsed_time(e2) / 1e3); tf.append(e0.elapsed_time(e1) / 1e3); tb.append(e1.elapsed_time(e2) / 1e3)
                res[mode] = {"s": float(np.mean(ts)), "fwd_s": float(np.mean(tf)), "bwd_s": float(np.mean(tb)), "kl_row": float(kl) / m,
                             "peak_extra_gib": (torch.cuda.max_memory_allocated(dev) - m0) / 2**30}
                if mode == "base": gb = g
                if mode == "nockpt":
                    res[mode]["grad_maxabs_diff"] = float((g - gb).abs().max()); res[mode]["grad_rel_norm_diff"] = float((g - gb).norm() / gb.norm())
            except torch.OutOfMemoryError as e:
                res[mode] = {"oom": str(e)[:200]}; [setattr(p, "grad", None) for p in params]; torch.cuda.empty_cache()
            log(f"P {pre} {mode}: {json.dumps(res[mode])}")
        # head + KL alone on detached final hidden states
        with torch.no_grad(): h, _ = T.run_segment(stu, C.g[gi].ids(i, pre, pre + Q), pre, prev, ckpt=False)
        h = h.detach().requires_grad_(True); ts = []
        for r in range(reps + 1):
            torch.cuda.synchronize(); e0 = ev()
            kl, m = T.seg_kl(stu, C.g[gi], i, pre, pre + Q, h); (kl / m).backward(); e1 = ev(); torch.cuda.synchronize(); h.grad = None
            [setattr(p, "grad", None) for p in params]
            if r: ts.append(e0.elapsed_time(e1) / 1e3)
        res["headkl"] = {"s": float(np.mean(ts))}; log(f"P {pre} headkl: {res['headkl']}")
        # profiler over one base rep
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
            kl, m = seg(pre, prev, ckpt=True); (kl / m).backward(); torch.cuda.synchronize()
        grads()
        fam = {}; rows = []
        for e in pr.key_averages():
            t = getattr(e, "self_device_time_total", None)
            if t is None: t = getattr(e, "self_cuda_time_total", 0)
            if t <= 0 or e.device_type.name != "CUDA": continue
            f = family(e.key); fam[f] = fam.get(f, 0) + t; rows.append((t, e.count, f, e.key[:150]))
        tot = sum(fam.values())
        res["families_ms"] = {k: round(v / 1e3, 1) for k, v in sorted(fam.items(), key=lambda x: -x[1])}; res["device_total_ms"] = round(tot / 1e3, 1)
        log(f"P {pre} device time by family (ms, total {tot / 1e3:.0f}): {json.dumps(res['families_ms'])}")
        with open(os.path.join(a.out, f"prof233-kernels-P{pre}.txt"), "w") as f:
            for t, c, fm, k in sorted(rows, reverse=True): f.write(f"{t / 1e3:10.2f} ms {c:7d} {fm:16s} {k}\n")
        out["runs"][str(pre)] = res
        del prev
    json.dump(out, open(os.path.join(a.out, "prof233.json"), "w"), indent=1)
    log("prof233 done")


if __name__ == "__main__":
    main()
