# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""S4 level pass: swj's statistics and fit (C10: gate / up output scales fitted jointly through the SwiGLU, codes
fixed) for any number of quantized sources (levels) in one staged bf16 forward. Port of rad-exl3-mix
v3/yaqa/swj27/swjfit27.py (same CLI, plan, checkpoints and outputs; the same ops in the same order: tests/check_levelpass.py
"swj" is bit-identical against it on a few 27B layers on the CPU) on rxq.forward's staging, rxq.swj's sums / fit
(= fsutil.swiglu_acc / _fit / _resid) and rxq.deq's decoder (= exl3deq.py's).

Per source and layer: x = the MLP input of the bf16 model, a = silu(W_g x) * (W_u x) in f32, g^ / u^ = the source's
gate / up; per channel j the sums over positions of a p and p^2, p = silu(c g^) u^, at every grid point c, and of a^2,
in f64 per document half (even / odd documents: the held-out check); then c_g = the grid point minimising the
residual, c_u its closed form, clamped to [0.5, 2].
- Source DIR:wq    DIR/wq/<key>.safetensors (ourq --save-wq, (out, in) f16)
- Source DIR:level an ourq-layout level (q-L*.safetensors): gate / up decoded in f32 (what the kernels serve)
META: a golden-style dir (kld.json "docs" with "tokens", + tokens.i32), documents cut to their first --cap tokens.
OUT/<NAME>/: scales/<key>.safetensors {"r": ones (in), "c": the multiplier (out)} for every gate / up (rxq.swj
apply_level folds them into the level), swiglu.jsonl, swjfit.json, .done. --write-wq: also OUT/<NAME>/wq/.
Restartable (SCRATCH/plan.json fixes the plan; progress.json after each checkpoint). VRAM_CAP_GB caps the allocator.

python -m rxq.swjfit MODEL META OUT --src NAME=DIR[:wq|:level] ... --scratch DIR [--stages N|auto|--split ..] [--fused]
       [--no-cache] [--grid 0.6:1.4:0.01] [--cap 4096] (python -m rxq.swjfit --help)"""
import argparse, hashlib, json, os, resource, shutil, sys, time

import numpy as np

from . import forward as F  # noqa: E402  (P2B_BLOCK before transformers)
import torch  # noqa: E402

from . import swj  # noqa: E402

TOOL = "rxq.swjfit"


def log(*x): print(time.strftime("%H:%M:%S"), *x, file=sys.stderr, flush=True)


def load_wq(qdir, key, dev):
    """(k, n) fp32 from the saved (out, in) f16 (fsutil.load_wq)"""
    from safetensors import safe_open
    with safe_open(f"{qdir}/wq/{key}.safetensors", "pt", device="cpu") as f: return f.get_tensor("weight").to(dev).float().T.contiguous()


def write_scales(dst, key, r, c):
    from safetensors.torch import save_file
    os.makedirs(f"{dst}/scales", exist_ok=True)
    save_file({"r": r.float().contiguous().cpu(), "c": c.float().contiguous().cpu()}, f"{dst}/scales/{key}.safetensors")


def write_wq(dst, key, Q):
    from safetensors.torch import save_file
    save_file({"weight": Q.T.half().contiguous().cpu()}, f"{dst}/wq/{key}.safetensors.part")
    os.replace(f"{dst}/wq/{key}.safetensors.part", f"{dst}/wq/{key}.safetensors")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rxq.swjfit", description=__doc__.split("\n\n")[0])
    ap.add_argument("model"); ap.add_argument("meta"); ap.add_argument("out")
    ap.add_argument("--src", action="append", required=True, help="NAME=DIR[:wq|:level] (default :level)")
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--stages", default="auto"); ap.add_argument("--split", default=None)
    ap.add_argument("--reserve-gb", type=float, default=5.0)
    ap.add_argument("--grid", default="0.6:1.4:0.01"); ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--maxdocs", type=int, default=0)
    ap.add_argument("--ckpt-min", type=float, default=10.0)
    ap.add_argument("--write-wq", action="store_true")
    ap.add_argument("--fused", action="store_true", help="the grid sums in one Triton kernel (rxq/swjtri.py; GPU)")
    ap.add_argument("--fused-tol", type=float, default=1e-5)
    ap.add_argument("--no-cache", action="store_true", help="use_cache=False (mkhin27's forward) instead of a fresh DynamicCache")
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args(argv)
    dev = torch.device(os.environ.get("SWJ_DEV", "cuda")); GPU = dev.type == "cuda"
    torch.set_grad_enabled(False)
    GB = F.GB
    if GPU and os.environ.get("VRAM_CAP_GB"):
        tot = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["VRAM_CAP_GB"]) / tot))
    write_atomic = F.write_atomic

    SRC = []
    for s in a.src:
        name, _, rest = s.partition("=")
        d, mode = (rest.rsplit(":", 1) if rest.rsplit(":", 1)[-1] in ("wq", "level") else (rest, "level"))
        assert name and os.path.isdir(d), f"--src {s}: NAME=DIR[:wq|:level]"
        SRC.append((name, os.path.abspath(d), mode))
    assert len({n for n, _, _ in SRC}) == len(SRC), "source names must differ"
    if all(os.path.exists(f"{a.out}/{x[0]}/.done") for x in SRC): log(f"{a.out}: every source is done"); return 0
    todo = SRC

    m, toks, toff, nd, dlen = F.corpus(a.meta, a.cap, a.maxdocs)
    NTOK = int(sum(dlen))
    M = F.Model(a.model, dev)
    cfg, NL, H, NJ, L = M.cfg, M.NL, M.H, M.NJ, M.L
    def gu(l): return f"{L}.{l}.mlp.gate_proj", f"{L}.{l}.mlp.up_proj"

    from . import deq
    LV = {}
    for name, d, mode in todo:
        if mode == "level":
            LV[name] = deq.Level(d, dev=dev)
            miss = [k for l in range(NL) for k in gu(l) if f"{k}.trellis" not in LV[name].ix]
        else:
            miss = [k for l in range(NL) for k in gu(l) if not os.path.exists(f"{d}/wq/{k}.safetensors")]
        assert not miss, f"source {name} ({d}, {mode}): {len(miss)} gate / up missing (first {miss[0]})"

    GRID = torch.arange(*(lambda g0, g1, gs: (g0, g1 + gs / 2, gs))(*map(float, a.grid.split(":"))), device=dev)
    NG = len(GRID); i1 = int(torch.argmin((GRID - 1).abs()))
    assert abs(float(GRID[i1]) - 1) < 1e-6, "grid must contain 1.0"
    wbytes = [sum(int(np.prod(M.shape(k))) * 2 for k in M.layer_keys(i)) for i in range(NL)]
    cost = [w + len(todo) * (2 * NJ * H * 4 + 2 * (2 * NG + 1) * NJ * 8) for w in wbytes]
    cap = os.environ.get("VRAM_CAP_GB")
    total_vram = torch.cuda.get_device_properties(0).total_memory if GPU else 0
    budget = (float(cap) * GB if cap else total_vram) - a.reserve_gb * GB
    split, sb = F.split_plan(cost, budget, a.stages, a.split)
    N = len(split) - 1
    sgb = [x / GB for x in sb]
    log(f"{nd} documents, {NTOK} tokens (cap {a.cap}); {len(todo)} sources {[x[0] for x in todo]}; {N} stages {split}, "
        f"weights + sources + sums {min(sgb):.2f} .. {max(sgb):.2f} GiB a stage (budget {budget / GB:.2f} GiB)")
    if a.plan_only: return 0

    meta_hash = hashlib.sha256(open(f"{a.meta}/kld.json", "rb").read()).hexdigest()[:16]
    plan = {"model": os.path.abspath(a.model), "meta": os.path.abspath(a.meta), "meta_sha256": meta_hash, "docs": nd,
            "tokens": NTOK, "cap": a.cap, "grid": a.grid, "split": split, "sources": [list(x) for x in todo],
            "fused": a.fused, "no_cache": a.no_cache}
    os.makedirs(f"{a.scratch}/sums", exist_ok=True)
    pf = f"{a.scratch}/plan.json"
    if os.path.exists(pf):
        old = json.load(open(pf))
        if old != plan: raise SystemExit(f"{pf} holds another plan {old}; this run is {plan}: use a fresh --scratch")
    else:
        write_atomic(pf, json.dumps(plan).encode())
    prf = f"{a.scratch}/progress.json"
    P = {"prog": json.load(open(prf)) if os.path.exists(prf) else {"stage": 0, "docs": 0, "gen": 0}}
    def bpath(s, i): return f"{a.scratch}/b{s}/{i:06d}.bf16"
    def ackpt(s, g): return f"{a.scratch}/acc-s{s}g{g}"
    def sums_path(name, l): return f"{a.scratch}/sums/{name}.L{l:02d}.pt"

    def cleanup():
        prog = P["prog"]
        for x in os.listdir(a.scratch):
            if x.startswith("acc-") and x != os.path.basename(ackpt(prog["stage"], prog["gen"])):
                shutil.rmtree(f"{a.scratch}/{x}", ignore_errors=True)
            if x.startswith("b") and x[1:].isdigit() and int(x[1:]) not in (prog["stage"], prog["stage"] + 1):
                shutil.rmtree(f"{a.scratch}/{x}", ignore_errors=True)

    def commit(s, j, g=0):
        P["prog"] = {"stage": s, "docs": j, "gen": g}; write_atomic(prf, json.dumps(P["prog"]).encode())
        cleanup()
    cleanup()

    M.make_rotary()

    def source_wq(name, d, mode, key):
        if mode == "wq": return load_wq(d, key, dev).T.contiguous()
        return LV[name].get(key).contiguous()

    def new_sums(): return [torch.zeros(NG, NJ, dtype=torch.float64, device=dev), torch.zeros(NG, NJ, dtype=torch.float64, device=dev),
                            torch.zeros(NJ, dtype=torch.float64, device=dev)]

    def save_sums(S, name, l, path):
        torch.save({str(hf): [t.cpu() for t in S[name][(hf, l)]] for hf in (0, 1)}, path + ".part"); os.replace(path + ".part", path)

    def load_sums(S, name, l, path):
        for hf, v in torch.load(path).items():
            for t, x in zip(S[name][(int(hf), l)], v): t.copy_(x)

    silu = torch.nn.functional.silu
    embed = None
    timing = open(f"{a.scratch}/timing.jsonl", "a")
    for s in range(N):
        prog = P["prog"]
        if prog["stage"] > s: continue
        j0 = prog["docs"]
        if s > 0:
            bad = [i for i in range(j0, nd) if not os.path.exists(bpath(s, i)) or os.path.getsize(bpath(s, i)) != dlen[i] * H * 2]
            if bad: raise SystemExit(f"stage {s}: {len(bad)} input boundaries missing or torn (first {bad[0]}): scratch damaged")
        if s < N - 1: os.makedirs(f"{a.scratch}/b{s + 1}", exist_ok=True)
        if GPU: torch.cuda.reset_peak_memory_stats()
        tl = time.time(); mods = M.load_layers(split[s], split[s + 1]); layers = [i for i, _ in mods]
        WQ = {name: {l: tuple(source_wq(name, d, mode, k) for k in gu(l)) for l in layers} for name, d, mode in todo}
        S = {name: {(hf, l): new_sums() for hf in (0, 1) for l in layers} for name, _, _ in todo}
        g0 = prog["gen"]
        if j0 > 0:
            for name, _, _ in todo:
                for l in layers: load_sums(S, name, l, f"{ackpt(s, g0)}/{name}.L{l:02d}.pt")
        tl = time.time() - tl
        cur = {"half": 0}
        if a.fused:
            from . import swjtri
            ACC = swjtri.swiglu_acc
        else:
            ACC = swj.acc
        chk = {"done": not a.fused}

        def hook(l, mlp):
            def h(mod, inp):
                x = inp[0][0].float()                              # [T, hidden]
                g = x @ mlp.gate_proj.weight.float().T; u = x @ mlp.up_proj.weight.float().T
                aa = silu(g) * u; del g, u
                for name, _, _ in todo:
                    gq = x @ WQ[name][l][0].T; uq = x @ WQ[name][l][1].T
                    if not chk["done"]:
                        err = swjtri.check(swj.acc, aa, gq, uq, GRID); chk["done"] = True
                        log(f"fused check (layer {l}, source {name}, {aa.shape[0]} positions): max rel difference {err:.2e}")
                        if not err <= a.fused_tol: raise SystemExit(f"fused grid sums differ by {err:.2e} > --fused-tol {a.fused_tol}")
                    ACC(S[name][(cur["half"], l)], aa, gq, uq, GRID)
                    del gq, uq
            return h
        hooks = [ly.mlp.register_forward_pre_hook(hook(i, ly.mlp)) for i, ly in mods]
        if s == 0 and embed is None: embed = M.embed_memmap()
        rgb = torch.cuda.memory_allocated() / GB if GPU else 0.0
        log(f"stage {s}: layers {layers[0]}..{layers[-1]}, resident {rgb:.2f} GiB, loaded in {tl:.0f} s; documents {j0}..{nd - 1}")
        t0 = tck = time.time(); ntok = 0
        for i in range(j0, nd):
            n = dlen[i]; cur["half"] = i % 2
            if s == 0:
                e = embed[np.asarray(toks[toff[i]:toff[i] + n], np.int64)]
                x = torch.from_numpy(e).view(torch.bfloat16)[None].to(dev)
            else:
                x = torch.from_numpy(np.fromfile(bpath(s, i), np.int16).reshape(1, n, H)).view(torch.bfloat16).to(dev)
            x = M.run_layers(mods, x, cache=not a.no_cache)
            if s < N - 1: write_atomic(bpath(s + 1, i), x[0].contiguous().view(torch.int16).cpu().numpy().tobytes())
            del x; ntok += n
            if time.time() - tck > a.ckpt_min * 60 and i < nd - 1:
                g0 += 1; os.makedirs(ackpt(s, g0), exist_ok=True)
                for name, _, _ in todo:
                    for l in layers: save_sums(S, name, l, f"{ackpt(s, g0)}/{name}.L{l:02d}.pt")
                commit(s, i + 1, g0); tck = time.time()
            if (i + 1) % 20 == 0 or i == nd - 1:
                el = time.time() - t0
                print(f"\rstage {s}: {i + 1}/{nd} docs, {ntok} tokens, {ntok / max(el, 1e-9):.0f} tok/s", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
        if GPU: torch.cuda.synchronize()
        el = time.time() - t0
        for h in hooks: h.remove()
        for name, _, _ in todo:
            for l in layers: save_sums(S, name, l, sums_path(name, l))
        del mods, WQ, S
        if GPU: torch.cuda.empty_cache()
        commit(s + 1, 0)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        peak = torch.cuda.max_memory_allocated() / GB if GPU else 0.0
        timing.write(json.dumps({"stage": s, "layers": [split[s], split[s + 1]], "sources": len(todo), "docs": nd - j0,
                                 "tokens": ntok, "load_s": round(tl, 2), "run_s": round(el, 2), "resident_gib": round(rgb, 3),
                                 "peak_gib": round(peak, 3), "maxrss_gib": round(rss, 2),
                                 "gpu": torch.cuda.get_device_name(0) if GPU else "cpu",
                                 "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n"); timing.flush()
        log(f"stage {s}: {ntok} tokens in {el:.0f} s ({ntok / max(el, 1e-9):.0f} tok/s), peak {peak:.2f} GiB, max RSS {rss:.2f} GiB")

    if P["prog"]["stage"] < N:
        log(f"stage {P['prog']['stage']} of {N} next (rerun the same command)"); return 0

    CLAMP = (0.5, 2.0)
    for name, d, mode in todo:
        o = f"{a.out}/{name}"
        os.makedirs(f"{o}/scales", exist_ok=True)
        if a.write_wq: os.makedirs(f"{o}/wq", exist_ok=True)
        logj = open(f"{o}/swiglu.jsonl", "w")
        done = set()
        for l in range(NL):
            blob = torch.load(sums_path(name, l))
            Sl = {hf: [t.to(dev) for t in blob[str(hf)]] for hf in (0, 1)}
            tot = [x + y for x, y in zip(Sl[0], Sl[1])]
            gi, cu, cu1 = swj.fit(tot, i1)
            one = torch.full_like(gi, i1)
            cu = cu.clamp(*CLAMP); cu1 = cu1.clamp(*CLAMP)
            rb, aa = swj.resid(tot, one, torch.ones_like(cu)); ru, _ = swj.resid(tot, one, cu1); rj, _ = swj.resid(tot, gi, cu)
            ge, cue, cu1e = swj.fit(Sl[0], i1)
            rbo, aao = swj.resid(Sl[1], one, torch.ones_like(cu)); ruo, _ = swj.resid(Sl[1], one, cu1e.clamp(*CLAMP))
            rjo, _ = swj.resid(Sl[1], ge, cue.clamp(*CLAMP))
            cg = GRID[gi]
            rec = {"layer": l, "rel_base": rb / aa, "rel_up": ru / aa, "rel_joint": rj / aa,
                   "heldout_base": rbo / aao, "heldout_up": ruo / aao, "heldout_joint": rjo / aao,
                   "cg_mean": float(cg.mean()), "cg_sd": float(cg.std()), "cu_mean": float(cu.mean()), "cu_sd": float(cu.std()),
                   "cg_at_edge": int(((gi == 0) | (gi == NG - 1)).sum())}
            logj.write(json.dumps(rec) + "\n"); logj.flush()
            print(f"{name} L{l:02d} rel resid base {rb / aa:.4e} up {ru / aa:.4e} joint {rj / aa:.4e} | held-out {rbo / aao:.4e} "
                  f"{ruo / aao:.4e} {rjo / aao:.4e} | c_g {float(cg.mean()):.3f}±{float(cg.std()):.3f} edge {rec['cg_at_edge']}", flush=True)
            for key, mult in zip(gu(l), (cg.float(), cu.float())):
                write_scales(o, key, torch.ones(H), mult)
                if a.write_wq:
                    wq = source_wq(name, d, mode, key)
                    write_wq(o, key, (wq * mult.view(-1, 1)).T)
                    del wq
                done.add(key)
        if a.write_wq and mode == "wq":
            for key in sorted(f[:-len(".safetensors")] for f in os.listdir(f"{d}/wq") if f.endswith(".safetensors")):
                if key not in done:
                    dd = f"{o}/wq/{key}.safetensors"
                    if not os.path.exists(dd): os.link(f"{d}/wq/{key}.safetensors", dd)
        json.dump({"tool": TOOL, "src": d, "mode": mode, "meta": os.path.abspath(a.meta), "grid": a.grid,
                   "docs": nd, "tokens": NTOK, "cap": a.cap, "split": split, "fused": a.fused, "no_cache": a.no_cache,
                   "kernels": F.kernel_paths(), "codebook": LV[name].codebook if mode == "level" else None,
                   "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(f"{o}/swjfit.json", "w"), indent=1)
        open(f"{o}/.done", "w").close()
        log(f"{name}: scales -> {o}")
    log(f"SWJFIT DONE {len(todo)} sources, {nd} docs, {NTOK} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
