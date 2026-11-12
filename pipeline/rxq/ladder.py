# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The trunk ladder as rad-quant-at-home jobs (w86, 2026-11-08): levels and build order from the settings file's [ladder]
and [quant.codebook], one rqah `rxq_quant` job per Hessian group (or tensor) of each level, every input pinned by its
sha256 (the datapack manifest), resource requirements per job from the measured table below, card-hour estimates.

  python -m rxq.ladder levels [--pass P]                    the levels in build order (name, K, codebook, pass, priority)
  python -m rxq.ladder jobs MANIFEST --model M --rev R [--pass P] [--levels a,b] [--split group|tensor] [--layers A-B]
                                                           [-o JOBS.jsonl]
                                                           rqah jobs (admin.py enqueue-jobs reads them); MANIFEST = the
                                                           datapack's manifest.json (sha256 of every input)
  python -m rxq.ladder estimate [--ldlq dense|blocks]       R9700 card-hours per pass (+ swj level passes, CPU packs)
  python -m rxq.ladder req                                  the per-tensor requirement table at a few K
  python -m rxq.ladder post [--pass P] [--levels a,b] [--export D --model M --meta G --scratch S --out O]
                                                           per level, the steps after its jobs: export + assemble, swj
                                                           level pass, swj + C9 apply, pack recipe

Job params (rqah jobs/rxq_quant.py checks them): model, revision, arch, level, K, codebook, layer, group, keys,
settings {rxq: the settings stamp, seed_offset}, stamp (rqah's settings stamp of the params), config (the settings
file's datapack relpath, models/M@R/rxq/<stamp>.toml), hout (the H_out snapshot id or null), c9 (the writers whose job
makes C9 scales), req {vram_gb, ram_gb, est_s}, pins {relpath: sha256} (= the job's inputs), verify "score".
The inputs of a job: the settings file, the group's weight file (weights/LNN-gG.safetensors, admin.py datapack
--groups), the group's input Hessian, and with H_out (every level since 11-08) each key's H_out diagonal."""
import argparse, hashlib, json, math, os, sys

from . import config as C, qwen35

ARCH = "qwen3_5"


def _rng(a, b, step):
    n = int(round((b - a) / step))
    return [round(a + i * step, 6) for i in range(n + 1)]


def _kn(K):
    K = float(K)
    return int(K) if K.is_integer() else K


def fmt_k(K):
    K = float(K)
    return str(int(K)) if K.is_integer() else repr(K)


# ------------------------------------------------------------------------------------------------------ levels
def levels(cfg, which=None):
    """[{name, K, codebook, pass, priority, conditional}] in build order. [ladder] levels = K ranges [first, last, step];
    pass "1" = on a [ladder] pass1 grid, "3" = sub-1 (K < 1), "2" = the rest. The codebook of every level comes from
    [quant.codebook] (i8 everywhere since the int8-only decision, 11-09; i4 twins / mul1 levels retired)."""
    L = cfg["ladder"]
    st = C.settings_stamp(cfg)
    Ks = sorted({K for r in L["levels"] for K in _rng(*r)})

    def on_grid(K):
        return any(a - 1e-9 <= K <= b + 1e-9 and abs(K / s - round(K / s)) < 1e-9 for a, b, s in L["pass1"])
    out = [{"K": _kn(K), "codebook": C.codebook_for(cfg, K), "pass": "3" if K < 1 else "1" if on_grid(K) else "2",
            "conditional": False} for K in Ks]
    fu = L["first_use"]
    rank = {"1": 0, "2": 1, "2b": 2, "3": 3}
    base = {"1": 60, "2": 40, "2b": 20, "3": 10}
    for lv in out:
        early = fu[0] <= float(lv["K"]) <= fu[1]
        lv["priority"] = base[lv["pass"]] + (5 if early else 0)
        lv["name"] = f"{L['prefix']}-{st[:8]}-k{fmt_k(lv['K'])}-{lv['codebook']}"
    out.sort(key=lambda v: (rank[v["pass"]], -v["priority"], abs(float(v["K"]) - 3.5), v["codebook"]))
    if which: out = [v for v in out if v["pass"] in which.split(",")]
    return out


# ------------------------------------------------------------------------- resources (R9700 seconds, VRAM, RAM)
# Measured 2026-11-08 on gpubox's R9700 (card 0) by w81's #115 gate (local/workers/w81/gate-report.txt): rxq per-tensor
# jobs, 27B layer 10 (GDN + MLP) and layer 39 (attention + MLP), ladder settings, bn H_out; per tensor the wall seconds
# and the peak allocated VRAM (GiB), at K4 i8 without H_out (N), with the dense two-sided LDLQ (D) and with w81's
# block-diagonal path (G, group schedule); the i4 series K0.125 / 0.5 / 1 / 2 / 3 on layer 10 (N and G).
T_K4 = {"ssm_qkv": 5.0, "ssm_z": 2.3, "ssm_out": 2.4, "attn_qg": 5.6, "attn_k": 0.8, "attn_v": 0.7, "attn_output": 2.4,
        "ffn_gate": 5.85, "ffn_up": 5.85, "ffn_down": 10.4}                    # N, K4 i8 (L10 / L39 means)
HOUT_S = {"dense": {"ssm_qkv": 14.7, "ssm_z": 6.0, "ssm_out": 5.9, "attn_qg": 21.9, "attn_k": 0.6, "attn_v": 0.4,
                    "attn_output": 5.8, "ffn_gate": 40.6, "ffn_up": 40.6, "ffn_down": 37.4},      # D - N
          "blocks": {"ssm_qkv": 0.4, "ssm_z": 0.2, "ssm_out": 0.2, "attn_qg": 0.5, "attn_k": 0.1, "attn_v": 0.1,
                     "attn_output": 0.2, "ffn_gate": 0.4, "ffn_up": 0.4, "ffn_down": 1.1}}        # G - N
# rate multiplier of the no-H_out time vs K4 (layer 10's six tensors, N): i4 K0.125 158.8 s, K0.5 144.7, K1 47.2,
# K2 34.9, K3 34.0 vs i8 K4 32.1 (the sub-1 rates run the ka = 0 trellis: ~4.5-5x); above K4 the 4070 profile's
# K6 / K4 = 2.13 (PROFILE.md §1) extrapolated as 2^(0.545 (K - 4)); fractional K = 1.17 x the integer neighbours'
# geometric interpolation (R9700: K2.5 34.7 s a layer vs K4 29.7, PROFILE.md §2)
K_MULT = {0.125: 4.95, 0.5: 4.51, 1: 1.47, 2: 1.09, 3: 1.06, 4: 1.0}
CD_S = {"dense": 38.7 - 27.5, "blocks": 17.5 - 6.1}  # tile CD on q_proj with H_out (w81 gate cdD / cdG - D / G, L39 K4)
C9_S = 1.1                          # in-job C9 per writer on the R9700 (gpucheck27 c9, 11-08: 1.06-1.08 s)
JOB_S = 5.0                         # per job: process start, imports, kernel load (cached), H_in load


def k_mult(K):
    K = float(K)
    if K < 1:                                            # ka = 0: K0.125 4.95, K0.5 4.51, linear in K between / beyond
        return max(1.47, 4.95 + (K - 0.125) * (4.51 - 4.95) / 0.375)

    def mi(k):
        return K_MULT[k] if k in K_MULT else 2 ** (0.545 * (k - 4))
    if K.is_integer(): return mi(int(K))
    lo, hi = math.floor(K), math.ceil(K)
    return 1.17 * math.exp(math.log(mi(lo)) * (hi - K) + math.log(mi(hi)) * (K - lo))


def _mode(cfg):
    return "blocks" if C.hout_ldlq(cfg) else "dense"


def tensor_seconds(cfg, key, K):
    """R9700 seconds of one tensor's job body at level K under cfg (no-H_out time x rate multiplier + H_out's F update
    for the configured path + tile CD + in-job C9)"""
    fam = qwen35.family(key)
    s = T_K4[fam] * k_mult(K)
    ho = C.hout_applies(cfg, K)
    if ho: s += HOUT_S[_mode(cfg)][fam]
    q = C.quant_settings(cfg, K, C.codebook_for(cfg, K), key)
    if q.get("tile_cd"): s += CD_S[_mode(cfg)] if ho else 9.0      # one-sided CD (11-06, R9700 attn_qg ~x3.7 of 5.1 s - 5.1)
    if C.c9_injob(cfg, key): s += C9_S
    return s


# Peak allocated VRAM (GiB) at K4 i8 without H_out (w81 gate, R9700): the qtf2 temp_edges buffer of rate KA is
# 128 x 256 x 2^(16 - KA) int16 = 2^(2 - KA) GiB (0.25 at K4); other rates move the peak by the difference (i4 K2
# gate 3.92 = 3.17 + 0.75, K1 4.93 = 3.17 + 1.75, K0.125 6.94 = 3.17 + 3.75: measured, w81 gate). ffn_down's peak
# (7.72) is its scale refit's second 17408^2 copy; the jobs set RXQ_REFIT_CPU_FALLBACK=1, so a smaller card runs that
# refit on the CPU (~10 GB host RAM, +50-90 s): its declared VRAM is the fallback peak, ~1.1 GiB lower (4070, 11-06).
VRAM_K4 = {"ssm_qkv": 2.21, "ssm_z": 1.54, "ssm_out": 1.67, "attn_qg": 2.54, "attn_k": 0.85, "attn_v": 0.85,
           "attn_output": 1.67, "ffn_gate": 3.17, "ffn_up": 3.17, "ffn_down": 7.72}
HOUT_VRAM = {"dense": {"ssm_qkv": 0.95, "ssm_z": 0.29, "ssm_out": 0.06, "attn_qg": 1.43, "attn_k": 0.0, "attn_v": 0.0,
                       "attn_output": 0.06, "ffn_gate": 3.23, "ffn_up": 3.23, "ffn_down": 0.0},   # D - N (w81 gate)
             "blocks": {}}                                                                         # G = N
CD_VRAM = {"dense": 0.10, "blocks": 0.37}             # cdD - D, cdG - G (q_proj, w81 gate)
DOWN_FALLBACK = 1.1
GIB = 2 ** 30


def edges_gib(K):
    return 128 * 256 * (65536 >> min(int(float(K)), 16)) * 2 / GIB


def tensor_vram(cfg, key, K):
    """modeled peak allocated GiB of one tensor at level K (the K4 measurement moved by the rate's trellis buffer, plus
    the configured H_out path's and tile CD's measured extras)"""
    fam = qwen35.family(key)
    v = VRAM_K4[fam] - edges_gib(4) + edges_gib(K)
    ho = C.hout_applies(cfg, K)
    if ho: v += HOUT_VRAM[_mode(cfg)].get(fam, 0.0)
    q = C.quant_settings(cfg, K, C.codebook_for(cfg, K), key)
    if q.get("tile_cd"): v += CD_VRAM[_mode(cfg)] if ho else 0.37
    return v


def job_req(cfg, keys, K):
    """{vram_gb: declared card VRAM (peak x 1.1 + 0.6 GiB context / fragmentation, rounded up), ram_gb, est_s}.
    Host RAM: 4 GB (measured RSS 2.5-2.6 GiB a job, R9700, w80 11-08); ffn_down 12 GB (its CPU refit fallback, ~10 GB)"""
    v, ram = 0.0, 4.0
    for key in keys:
        x = tensor_vram(cfg, key, K)
        if qwen35.family(key) == "ffn_down": x -= DOWN_FALLBACK; ram = 12.0    # CPU refit fallback peak
        v = max(v, x)
    return {"vram_gb": int(math.ceil(v * 1.1 + 0.6)), "ram_gb": ram,
            "est_s": round(JOB_S + sum(tensor_seconds(cfg, key, K) for key in keys), 1)}


# --------------------------------------------------------------------------------------------------- the jobs
def units(cfg, l, split):
    plan = qwen35.layer_plan(l)
    if split == "tensor": return [(gi, [k]) for gi, (_, ks) in enumerate(plan) for k in ks]
    return [(gi, list(ks)) for gi, (_, ks) in enumerate(plan)]


def canon(o): return json.dumps(o, sort_keys=True, separators=(",", ":"))


def rqah_stamp(p):
    """rad-quant-at-home's settings stamp (jobs/quant_common.settings_stamp): model, revision, arch, K, codebook, settings"""
    return hashlib.sha256(canon({k: p[k] for k in ("model", "revision", "arch", "K", "codebook", "settings")}).encode()).hexdigest()[:16]


def hout_snap(cfg):
    return os.path.basename(cfg["inputs"]["hout"]["dir"].rstrip("/"))


def job_inputs(cfg, root, l, g, keys, K, st):
    hname = qwen35.layer_plan(l)[g][0]
    names = [f"{root}/rxq/{st}.toml", f"{root}/weights/L{l:02d}-g{g}.safetensors", f"{root}/hessians/{hname}.safetensors"]
    if C.hout_applies(cfg, K): names += [f"{root}/hout/{hout_snap(cfg)}/{k}.safetensors" for k in keys]
    return names


def layer_list(spec, n_layers):
    """None -> every layer; "A-B" / "a,b,c" / an int N (the first N)"""
    if spec is None: return list(range(n_layers))
    if isinstance(spec, int): return list(range(spec))
    out = []
    for part in str(spec).split(","):
        if "-" in part: a, b = map(int, part.split("-")); out += list(range(a, b + 1))
        elif part: out.append(int(part))
    return out


def level_jobs(cfg, lv, model, rev, man=None, split=None, seed_offset=0, n_layers=None, layers=None):
    """the rqah jobs of one level: [{type, params, level, ukey, priority}]; man = manifest files {relpath: {sha256}}
    (None: pins without hashes, for planning only); layers: a subset ("2,3", "0-15"; dry runs)"""
    st = C.settings_stamp(cfg); split = split or cfg["ladder"]["split"]; n_layers = n_layers or cfg["ladder"]["n_layers"]
    root = f"models/{model}@{rev}"
    K = lv["K"]; cb = lv["codebook"]
    out = []
    for l in layer_list(layers, n_layers):
        for j, (g, keys) in enumerate(units(cfg, l, split)):
            names = job_inputs(cfg, root, l, g, keys, K, st)
            if man is not None:
                miss = [n for n in names if n not in man]
                if miss: raise SystemExit(f"{lv['name']} L{l} g{g}: not in the manifest: {miss[:3]}")
            p = {"model": model, "revision": rev, "arch": ARCH, "level": lv["name"], "K": K, "codebook": cb, "layer": l,
                 "group": g, "keys": keys, "settings": {"rxq": st, "seed_offset": seed_offset},
                 "config": f"{root}/rxq/{st}.toml", "hout": hout_snap(cfg) if C.hout_applies(cfg, K) else None,
                 "c9": [k for k in keys if C.c9_injob(cfg, k)], "req": job_req(cfg, keys, K),
                 "pins": {n: (man[n]["sha256"] if man is not None else None) for n in names}, "verify": "score"}
            p["stamp"] = rqah_stamp(p)
            u = f"L{l:02d}:g{g}" + (f":{keys[0].rsplit('.', 1)[-1]}" if split == "tensor" else "")
            out.append({"type": "rxq_quant", "params": p, "level": lv["name"], "ukey": f"rxq_quant/v1:{p['stamp']}:{u}",
                        "priority": lv["priority"]})
    return out


# ------------------------------------------------------------------------------------------------- estimates
def estimate(cfg, n_layers=64):
    """per pass: levels, R9700 card-hours of the quant jobs (sum of est_s), the swj level passes (~0.75 card-h a level
    on a 32 GB card, ~1.5 card-h for two levels in one pass, PROFILE / DESIGN S4) and the CPU packs (~1,600 s a level)"""
    rows = {}
    for lv in levels(cfg):
        r = rows.setdefault(lv["pass"], {"levels": 0, "quant_h": 0.0, "jobs": 0, "vram": {}})
        r["levels"] += 1
        for jb in level_jobs(cfg, lv, "m", "r", None, n_layers=n_layers):
            q = jb["params"]["req"]; r["quant_h"] += q["est_s"] / 3600; r["jobs"] += 1
            r["vram"][q["vram_gb"]] = r["vram"].get(q["vram_gb"], 0) + 1
    for r in rows.values():
        r["swj_h"] = 0.75 * r["levels"] if cfg["post"]["swj"]["enable"] else 0.0
        r["pack_cpu_h"] = 1600 / 3600 * r["levels"]
        r["quant_h"] = round(r["quant_h"], 1)
    return rows


def post_script(cfg, lv, export, model, meta, scratch, out):
    """the per-level steps after its jobs are accepted (bash text): rqah export (rxq assemble of the accepted job dirs:
    the level + LEVEL/c9 from the writers' in-job C9), the swj level pass (S4, our 32 GB card; post.swj), the one
    rxq.post apply that folds swj's and C9's scales into svh (swj is fitted on the level without C9, as #112-02
    measured it), and the pack recipe (S8; rxq.mix pack_script holds the lock rule)"""
    from . import post as P
    steps = P.plan_level(cfg, lv["K"])
    name = lv["name"]; L = f"{export}/{name}/level"; S = f"{out}/swj/{name}"; F = f"{out}/levels/{name}"
    sh = [f"# {name}: K{fmt_k(lv['K'])} {lv['codebook']}, post steps {[x[0] for x in steps]}",
          f"admin.py export --level {name} {export}/{name} --checked --rxq-python python3 --rxq $RXQ"]
    scales = []
    if any(x[0] == "swj" for x in steps):
        sh.append(f"python -m rxq.swjfit {model} {meta} {out}/swj --src {name}={L}:level --scratch {scratch}/{name} --fused --no-cache")
        scales.append(f"{S}/scales")
    if any(x == ("c9", "job") or x[0] == "c9" and x[1] == "job" for x in steps): scales.append(f"{L}/c9")
    if any(x[0] == "asym" for x in steps): sh.append("# asym is per mix (#111): rxq.post plan_asym after the knapsack")
    sh.append(f"python -m rxq.post apply {L} {','.join(scales)} {F}" if scales else f"cp -a {L} {F}")
    sh.append(f"python -m rxq.mix recipe {F} > {F}.recipe   # then rad-convert exl3i (conv.lock + ram.lock exclusive)")
    return "\n".join(sh) + "\n"


def main(argv=None):
    global JOB_S
    ap = argparse.ArgumentParser(prog="rxq.ladder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("levels"); a1.add_argument("--pass", dest="which")
    a2 = sub.add_parser("jobs"); a2.add_argument("manifest"); a2.add_argument("--model", required=True); a2.add_argument("--rev", required=True)
    a2.add_argument("--pass", dest="which"); a2.add_argument("--levels"); a2.add_argument("--split", choices=("group", "tensor"))
    a2.add_argument("--layers", help="a subset of layers: A-B or a,b (dry runs)"); a2.add_argument("--seed-offset", type=int, default=0)
    a2.add_argument("-o")
    a3 = sub.add_parser("estimate"); a3.add_argument("--ldlq", choices=("dense", "blocks"))
    a3.add_argument("--job-overhead", type=float, default=JOB_S, help="seconds per job beyond the tensors (default %(default)s)")
    a4 = sub.add_parser("req"); a4.add_argument("--ldlq", choices=("dense", "blocks"))
    a5 = sub.add_parser("post"); a5.add_argument("--pass", dest="which"); a5.add_argument("--levels")
    a5.add_argument("--export", default="EXPORT"); a5.add_argument("--model", default="MODEL"); a5.add_argument("--meta", default="SWJMETA")
    a5.add_argument("--scratch", default="SCRATCH"); a5.add_argument("--out", default="OUT")
    for x in (a1, a2, a3, a4, a5): x.add_argument("--config"); x.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)
    cfg = C.load(a.config)
    sets = list(a.set) + ([f"inputs.hout.ldlq={a.ldlq}"] if getattr(a, "ldlq", None) else [])
    if sets:
        from .__main__ import _setval
        cfg = C.with_overrides(cfg, **{s.split("=", 1)[0]: _setval(s.split("=", 1)[1]) for s in sets})
    if a.cmd == "levels":
        for lv in levels(cfg, a.which): print(json.dumps(lv))
    elif a.cmd == "jobs":
        man = json.load(open(a.manifest))["files"]
        lvs = levels(cfg, a.which)
        if a.levels: lvs = [v for v in lvs if v["name"] in a.levels.split(",") or fmt_k(v["K"]) + "-" + v["codebook"] in a.levels.split(",")]
        if not lvs: raise SystemExit("no level selected")
        fh = open(a.o, "w") if a.o else sys.stdout
        n = 0
        for lv in lvs:
            for jb in level_jobs(cfg, lv, a.model, a.rev, man, a.split, a.seed_offset, layers=a.layers):
                fh.write(json.dumps(jb) + "\n"); n += 1
        if a.o: fh.close(); print(f"{n} jobs of {len(lvs)} levels -> {a.o} (settings stamp {C.settings_stamp(cfg)})")
    elif a.cmd == "post":
        lvs = levels(cfg, a.which)
        if a.levels: lvs = [v for v in lvs if v["name"] in a.levels.split(",") or fmt_k(v["K"]) + "-" + v["codebook"] in a.levels.split(",")]
        for lv in lvs: sys.stdout.write(post_script(cfg, lv, a.export, a.model, a.meta, a.scratch, a.out))
    elif a.cmd == "estimate":
        JOB_S = a.job_overhead
        tot = {"levels": 0, "quant_h": 0.0, "swj_h": 0.0, "pack_cpu_h": 0.0, "jobs": 0}
        print(f"settings {C.settings_stamp(cfg)}, H_out ldlq {cfg['inputs']['hout'].get('ldlq', 'dense')}, {JOB_S:g} s a job overhead; "
              f"R9700-equivalent hours")
        for p, r in sorted(estimate(cfg).items()):
            print(f"pass {p:2s}: {r['levels']:2d} levels, {r['jobs']:6d} jobs, quant {r['quant_h']:7.1f} card-h, swj {r['swj_h']:5.1f} "
                  f"card-h (32 GB, ours), pack {r['pack_cpu_h']:5.1f} CPU-h; jobs by declared VRAM GB {dict(sorted(r['vram'].items()))}")
            for k in tot: tot[k] += r[k]
        print(f"total  : {tot['levels']} levels, {tot['jobs']} jobs, quant {tot['quant_h']:.1f} card-h, swj {tot['swj_h']:.1f}, "
              f"pack {tot['pack_cpu_h']:.1f} CPU-h")
    else:
        keys = [k for _, ks in qwen35.layer_plan(3) for k in ks] + [k for _, ks in qwen35.layer_plan(2) for k in ks
                                                                    if "linear_attn" in k]
        print(f"H_out {_mode(cfg)}; per tensor: modeled peak GiB / R9700 seconds")
        print("family        " + "  ".join(f"K{fmt_k(K):>5s} vram/s" for K in (0.125, 0.5, 1.5, 2.5, 4, 6, 7, 8)))
        for key in keys:
            print(f"{qwen35.family(key):12s}  " + "  ".join(
                f"{tensor_vram(cfg, key, K):5.2f}/{tensor_seconds(cfg, key, K):5.1f}" for K in (0.125, 0.5, 1.5, 2.5, 4, 6, 7, 8)))


if __name__ == "__main__":
    main()
