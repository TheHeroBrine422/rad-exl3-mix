# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""S1 level pass: input Hessians of a Qwen3.5-family model in the published hessians/calib layout, on one card in staged
layer passes. Port of rad-exl3-mix eval/gold27/mkhin27.py (same CLI, plan, checkpoints, outputs and arithmetic:
tests/check_levelpass.py "hin" is bit-identical against it on a few 27B layers on the CPU) on rxq.forward's staging.
V3 quantizes against the published calib Hessians; this pass is for other corpora (the xcorp blend, #101, dropped) and
other models.

OUT/<name>.safetensors: "hessian_upper" f32, the packed upper triangle (row major) of sum x x^T over every token of
every META document; metadata k / rows / layout / linears (rows = tokens summed). 4 files a layer (<L>.<l>.input,
o_proj / out_proj input, post_attention.input, mlp.down_proj.input) plus lm_head.input (the final norm's output).
OUT/collect.json last (the completion marker). Documents longer than --chunk are cut into independent windows.
Sums: fp32 addmm a 4096-row block at a time, flushed into fp64 files every --flush documents (the checkpoints:
SCRATCH/acc/s<s>g<g>/<name>.f64, SCRATCH/progress.json) and at each stage's end (straight to OUT, rounded to fp32).

python -m rxq.hin MODEL META OUT --scratch DIR [--stages N|auto|--split ..] [--chunk 8192] [--flush 200] [--docs N]
Env MKH27_DEV (cuda | cpu), VRAM_CAP_GB, P2B_BLOCK."""
import argparse, hashlib, json, os, resource, shutil, struct, sys, time

import numpy as np

from . import forward as F  # noqa: E402  (P2B_BLOCK before transformers)
import torch  # noqa: E402

TOOL = "rxq.hin"


def log(*x): print(time.strftime("%H:%M:%S"), *x, file=sys.stderr, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rxq.hin", description=__doc__.split("\n\n")[0])
    ap.add_argument("model"); ap.add_argument("meta"); ap.add_argument("out")
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--stages", default="auto"); ap.add_argument("--split", default=None)
    ap.add_argument("--reserve-gb", type=float, default=2.5)
    ap.add_argument("--docs", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--flush", type=int, default=200)
    ap.add_argument("--only-stages", default=None)
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args(argv)
    dev = os.environ.get("MKH27_DEV", "cuda")
    GPU = dev == "cuda"
    torch.set_grad_enabled(False)
    GB = F.GB
    write_atomic = F.write_atomic

    m = json.load(open(f"{a.meta}/kld.json"))
    alld = m["docs"]; docs = alld[:a.docs] if a.docs else alld
    toff = np.concatenate([[0], np.cumsum([d["tokens"] for d in alld])])
    toks = np.memmap(f"{a.meta}/tokens.i32", np.int32, "r")
    assert len(toks) >= toff[-1], f"{a.meta}: tokens.i32 has {len(toks)} tokens, kld.json's docs {toff[-1]}"
    units = []
    for d, D in enumerate(docs):
        for s in range(0, D["tokens"], a.chunk): units.append((d, s, min(a.chunk, D["tokens"] - s)))
    ROWS = int(sum(D["tokens"] for D in docs))
    ckpt_after = {j for j, (d, s, n) in enumerate(units) if s + n == docs[d]["tokens"] and (d + 1) % a.flush == 0}
    meta_hash = hashlib.sha256(open(f"{a.meta}/kld.json", "rb").read()).hexdigest()[:16]

    M = F.Model(a.model, dev)
    cfg, NL, H, L, LP = M.cfg, M.NL, M.H, M.L, M.LP

    def groups(i):
        p = f"{L}.{i}"
        if cfg.layer_types[i] == "full_attention":
            g = [(f"{p}.input", "self_attn.q_proj", [f"{p}.self_attn.{x}_proj" for x in "qkv"]),
                 (f"{p}.self_attn.o_proj.input", "self_attn.o_proj", [f"{p}.self_attn.o_proj"])]
        else:
            g = [(f"{p}.input", "linear_attn.in_proj_qkv", [f"{p}.linear_attn.in_proj_{x}" for x in ("qkv", "z", "a", "b")]),
                 (f"{p}.linear_attn.out_proj.input", "linear_attn.out_proj", [f"{p}.linear_attn.out_proj"])]
        g += [(f"{p}.post_attention.input", "mlp.gate_proj", [f"{p}.mlp.gate_proj", f"{p}.mlp.up_proj"]),
              (f"{p}.mlp.down_proj.input", "mlp.down_proj", [f"{p}.mlp.down_proj"])]
        return [(n, mp, int(M.shape(f"{p}.{mp}.weight")[1]), lin) for n, mp, lin in g]

    G = [groups(i) for i in range(NL)]
    HEAD = ("lm_head.input", None, H, ["lm_head"])
    wbytes = [sum(int(np.prod(M.shape(k))) * 2 for k in M.layer_keys(i)) for i in range(NL)]
    abytes = [sum(4 * k * k for _, _, k, _ in G[i]) for i in range(NL)]
    cost = [w + c for w, c in zip(wbytes, abytes)]
    head_cost = 4 * H * H + 2 * H
    cap = os.environ.get("VRAM_CAP_GB")
    total_vram = torch.cuda.get_device_properties(0).total_memory if GPU else 0
    budget = (float(cap) * GB if cap else total_vram) - a.reserve_gb * GB
    split, sb = F.split_plan(cost, budget, a.stages, a.split, head_cost)
    N = len(split) - 1
    sgb = [x / GB for x in sb]
    log(f"{len(docs)} documents, {len(units)} windows, {ROWS} tokens; {N} stages {split}, weights + accumulators "
        f"{min(sgb):.2f} .. {max(sgb):.2f} GiB a stage (budget {budget / GB:.2f} GiB)")
    if a.plan_only: return 0
    if max(sgb) * GB > budget > 0: log(f"WARNING: a stage needs {max(sgb):.2f} GiB, over the {budget / GB:.2f} GiB budget")

    plan = {"model": os.path.abspath(a.model), "meta": os.path.abspath(a.meta), "meta_sha256": meta_hash, "docs": len(docs),
            "rows": ROWS, "chunk": a.chunk, "flush": a.flush, "split": split, "out": os.path.abspath(a.out)}
    if os.path.exists(f"{a.out}/collect.json"):
        if json.load(open(f"{a.out}/collect.json")).get("plan") == plan:
            log(f"{a.out} is finished (collect.json, same plan): nothing to do"); return 0
        raise SystemExit(f"{a.out} holds another collection (collect.json); write into a new directory")
    os.makedirs(a.out, exist_ok=True); os.makedirs(f"{a.scratch}/acc", exist_ok=True)
    pf = f"{a.scratch}/plan.json"
    if os.path.exists(pf):
        old = json.load(open(pf))
        if old != plan: raise SystemExit(f"{pf} holds another plan {old}; this run is {plan}: use a fresh --scratch")
    else:
        write_atomic(pf, json.dumps(plan).encode())
    prf = f"{a.scratch}/progress.json"
    P = {"prog": json.load(open(prf)) if os.path.exists(prf) else {"stage": 0, "units": 0, "gen": 0}}
    def bdir(s): return f"{a.scratch}/b{s}"
    def bpath(s, u): return f"{bdir(s)}/{u:07d}.bf16"
    def gdir(s, g): return f"{a.scratch}/acc/s{s}g{g}"

    def cleanup():
        s, j, g = P["prog"]["stage"], P["prog"]["units"], P["prog"]["gen"]
        for x in os.listdir(f"{a.scratch}/acc"):
            if x != f"s{s}g{g}": shutil.rmtree(f"{a.scratch}/acc/{x}", ignore_errors=True)
        for x in os.listdir(a.scratch):
            if x.startswith("b") and x[1:].isdigit() and int(x[1:]) not in (s, s + 1): shutil.rmtree(f"{a.scratch}/{x}")
        for t in (s, s + 1):
            if os.path.isdir(bdir(t)):
                for x in os.listdir(bdir(t)):
                    if x.endswith(".part") or (t == s and int(x.split(".")[0]) < j): os.remove(f"{bdir(t)}/{x}")
        for x in os.listdir(a.out):
            if x.endswith(".part"): os.remove(f"{a.out}/{x}")

    def commit(s, j, g):
        P["prog"] = {"stage": s, "units": j, "gen": g}; write_atomic(prf, json.dumps(P["prog"]).encode())
        cleanup()
    cleanup()
    if GPU and cap:
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(cap) * GB / total_vram))

    def load_stage(s):
        t0 = time.time()
        mods = M.load_layers(split[s], split[s + 1])
        norm = M.load_norm() if s == N - 1 else None
        return mods, norm, time.time() - t0

    M.make_rotary()

    def accumulate(acc, x):
        x = x.reshape(-1, x.shape[-1])
        for r in range(0, x.shape[0], 4096):
            xb = x[r:r + 4096].float(); acc.addmm_(xb.T, xb)

    def combine(acc, old, path, final=None):
        k = acc.shape[0]; rb = max(1, (1 << 23) // k); cols = torch.arange(k, device=acc.device)
        fi = open(old, "rb") if old else None
        with open(path + ".part", "wb") as fo:
            if final:
                n = k * (k + 1) // 2
                hd = json.dumps({"__metadata__": {"k": str(k), "rows": str(ROWS), "layout": "upper triangle, row-major, packed",
                                                  "linears": ",".join(final[2])},
                                 "hessian_upper": {"dtype": "F32", "shape": [n], "data_offsets": [0, 4 * n]}},
                                separators=(",", ":")).encode()
                hd += b" " * (-len(hd) % 8)
                fo.write(struct.pack("<Q", len(hd))); fo.write(hd)
            for r0 in range(0, k, rb):
                r1 = min(k, r0 + rb)
                u = acc[r0:r1][cols[None, :] >= cols[r0:r1, None]].double().cpu().numpy()
                if fi: u += np.fromfile(fi, np.float64, count=u.size)
                fo.write((u.astype(np.float32) if final else u).tobytes())
            fo.flush(); os.fsync(fo.fileno())
            os.posix_fadvise(fo.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        if fi: fi.close()
        os.replace(path + ".part", path)
        acc.zero_()

    s0, s1 = (map(int, a.only_stages.split(":")) if a.only_stages else (0, N))
    embed = None
    timing = open(f"{a.scratch}/timing.jsonl", "a")
    for s in range(s0, s1):
        prog = P["prog"]
        if prog["stage"] > s:
            log(f"stage {s}: done"); continue
        if prog["stage"] < s:
            log(f"stage {s}: stage {prog['stage']} is not finished; stopping"); break
        j0, g = prog["units"], prog["gen"]
        if s > 0:
            bad = [u for u in range(j0, len(units)) if not os.path.exists(bpath(s, u))
                   or os.path.getsize(bpath(s, u)) != units[u][2] * H * 2]
            if bad: raise SystemExit(f"stage {s}: {len(bad)} input boundaries missing or torn (first {bad[0]}): scratch damaged")
        if s < N - 1: os.makedirs(bdir(s + 1), exist_ok=True)
        if GPU: torch.cuda.reset_peak_memory_stats()
        mods, norm, tl = load_stage(s)
        gs = [x for i in range(split[s], split[s + 1]) for x in G[i]] + ([HEAD] if norm is not None else [])
        acc = {name: torch.zeros(k, k, device=dev) for name, _, k, _ in gs}
        hooks = []
        for i, ly in mods:
            for name, mp, _, _ in G[i]:
                hooks.append(ly.get_submodule(mp).register_forward_pre_hook(
                    lambda mod, args, name=name: accumulate(acc[name], args[0])))
        if s == 0 and embed is None: embed = M.embed_memmap()
        wgb = torch.cuda.memory_allocated() / GB if GPU else 0.0
        log(f"stage {s}: layers {split[s]}..{split[s + 1] - 1}{' + norm (lm_head.input)' if norm is not None else ''}, "
            f"weights + accumulators {wgb:.2f} GiB, loaded in {tl:.0f} s; windows {j0}..{len(units) - 1}")
        t0, ntok, tflush = time.time(), 0, 0.0
        for j in range(j0, len(units)):
            d, st, n = units[j]
            if s == 0:
                e = embed[np.asarray(toks[toff[d] + st:toff[d] + st + n], np.int64)]
                x = torch.from_numpy(e).view(torch.bfloat16)[None].to(dev)
            else:
                x = torch.from_numpy(np.fromfile(bpath(s, j), np.int16).reshape(1, n, H)).view(torch.bfloat16).to(dev)
            x = M.run_layers(mods, x, cache=False)
            if norm is not None:
                accumulate(acc[HEAD[0]], norm(x))
            else:
                write_atomic(bpath(s + 1, j), x[0].contiguous().view(torch.int16).cpu().numpy().tobytes())
            del x; ntok += n
            if j in ckpt_after and j < len(units) - 1:
                tf = time.time(); os.makedirs(gdir(s, g + 1), exist_ok=True)
                for name in acc:
                    old = f"{gdir(s, g)}/{name}.f64"
                    combine(acc[name], old if os.path.exists(old) else None, f"{gdir(s, g + 1)}/{name}.f64")
                g += 1; commit(s, j + 1, g); tflush += time.time() - tf
            if (j + 1) % 50 == 0 or j == len(units) - 1:
                el = time.time() - t0
                print(f"\rstage {s}: {j + 1}/{len(units)} windows, {ntok} tokens, {ntok / el:.0f} tok/s, flushes {tflush:.0f} s",
                      end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
        if GPU: torch.cuda.synchronize()
        el = time.time() - t0
        for h in hooks: h.remove()
        del mods, norm
        tw = time.time()
        for name, _, k, lin in gs:
            old = f"{gdir(s, g)}/{name}.f64"
            combine(acc[name], old if os.path.exists(old) else None, f"{a.out}/{name}.safetensors", (name, k, lin))
        tw = time.time() - tw
        del acc
        if GPU: torch.cuda.empty_cache()
        commit(s + 1, 0, 0)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        peak = torch.cuda.max_memory_allocated() / GB if GPU else 0.0
        timing.write(json.dumps({"stage": s, "layers": [split[s], split[s + 1]], "head": s == N - 1, "windows": len(units) - j0,
                                 "tokens": ntok, "load_s": round(tl, 2), "run_s": round(el, 2), "flush_s": round(tflush, 2),
                                 "write_s": round(tw, 2), "resident_gib": round(wgb, 3), "peak_gib": round(peak, 3), "maxrss_gib": round(rss, 2),
                                 "gpu": torch.cuda.get_device_name(0) if GPU else "cpu",
                                 "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
        timing.flush()
        log(f"stage {s}: {ntok} tokens in {el:.0f} s ({ntok / max(el, 1e-9):.0f} tok/s; flushes {tflush:.0f} s), "
            f"OUT files {tw:.0f} s, peak {peak:.2f} GiB, max RSS {rss:.2f} GiB")

    if P["prog"]["stage"] < N:
        log(f"stage {P['prog']['stage']} of {N} next; collect.json not written (rerun the same command)"); return 0
    names = [x[0] for i in range(NL) for x in G[i]] + [HEAD[0]]
    miss = [x for x in names if not os.path.exists(f"{a.out}/{x}.safetensors")]
    if miss: raise SystemExit(f"{len(miss)} OUT files missing (first {miss[0]}) although every stage is done")
    write_atomic(f"{a.out}/collect.json", json.dumps(
        {"tool": TOOL, "plan": plan, "files": len(names), "windows": len(units),
         "kernels": F.kernel_paths(), "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1).encode())
    shutil.rmtree(f"{a.scratch}/acc", ignore_errors=True)
    log(f"HIN DONE {len(names)} Hessians, {ROWS} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
