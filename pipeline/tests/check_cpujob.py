#!/usr/bin/env python3
"""check_cpujob.py FORK_DENSE FORK_BLOCKS [WORK] -- the quant job end to end on the CPU (rxq/cpukern.py's stand-in
kernels, the fork's real quantize.py), on a tiny two-layer model (layer 2 GDN + MLP, layer 3 attention + MLP; random
weights and Hessians, small shapes), under the V3 settings (H_out at every level, tile CD on attn_qg, C9 on the
writers, sub-1 damping, sigma 0.2 at K0.125):
  levels   K0.125 i8, K0.5 i8 (sub-1; i4 until #126), K2.5 i8, K6.5 mul1 (one process each: `rxq job` semantics, run_quant with
           save_wq), H_out dense (FORK_DENSE: the fork as is) and blocks (FORK_BLOCKS: + w81's patch 0008, inputs.hout.ldlq
           = blocks): every tensor of both layers, every job's outputs present
  decode   each t-file decoded by rxq.deq (trellis, suh, svh) == the quantizer's own dequantized weight (wq, f16) to
           f16 rounding: the codes the job stores are the weight it reports, through regularize, the Hadamards, the
           scale refit and the pack
  settings the log records: sigma (0.2 both groups at K0.125; 0.05 input group at K0.5; 0.005 / 0.025 at K2.5), H_out
           on every tensor, hout_blocks only in blocks runs, tile CD on q_proj only (tilecd.jsonl, "hout": true), C9
           files and records for out_proj (alpha 3) and o_proj (alpha -1), none for down (alpha 0)
  stamps   CPU stand-in metadata (rxq_kernels) on every tensor; assemble refuses the dir, --allow-standin assembles it
           (partial level), and a blocks tensor next to a dense one is refused (mixed settings stamps)
Last line: "CPUJOB CHECKS n/m PASS"."""
import json, os, re, shutil, subprocess, sys, tempfile

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, PIPE)
from rxq import config as C, deq, job, qwen35  # noqa: E402

if len(sys.argv) > 1 and sys.argv[1] == "--run":          # child: one job (one engine setup a process)
    spec = json.load(open(sys.argv[2]))
    job.run_quant(spec, sys.argv[3], device="cpu", save_wq=True)
    sys.exit(0)

FD, FB = sys.argv[1], sys.argv[2]
W = sys.argv[3] if len(sys.argv) > 3 else tempfile.mkdtemp(prefix="w86cpujob-")
res = []


def report(name, ok, why):
    res.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'} {name}: {why}", flush=True)


# ---- the tiny model: (out, in) per family
SH = {"ssm_qkv": (384, 256), "ssm_z": (256, 256), "ssm_out": (256, 256), "attn_qg": (256, 256), "attn_k": (128, 256),
      "attn_v": (128, 256), "attn_output": (256, 256), "ffn_gate": (384, 256), "ffn_up": (384, 256), "ffn_down": (256, 384)}
g = torch.Generator().manual_seed(86)
ck, hs, ho = f"{W}/ckpt", f"{W}/hess", f"{W}/hout/bn1920"
for d in (ck, hs, ho): os.makedirs(d, exist_ok=True)
tens, keys = {}, []
for l in (2, 3):
    for hname, ks in qwen35.layer_plan(l):
        k_in = SH[qwen35.family(ks[0])][1]
        A = torch.randn(4 * k_in, k_in, generator=g) * torch.linspace(0.3, 2.0, k_in)
        H = (A.T @ A).float(); iu = torch.triu_indices(k_in, k_in)
        save_file({"hessian_upper": H[iu[0], iu[1]].contiguous()}, f"{hs}/{hname}.safetensors", metadata={"k": str(k_in), "rows": str(4 * k_in)})
        for key in ks:
            n, k = SH[qwen35.family(key)]
            tens[f"{key}.weight"] = (torch.randn(n, k, generator=g) * 0.02).to(torch.bfloat16)
            save_file({"hout_diag": (torch.rand(n, generator=g) + 0.1).float()}, f"{ho}/{key}.safetensors")
            keys.append((l, hname, ks, key))
save_file(tens, f"{ck}/model.safetensors")

base = open(os.path.join(PIPE, "config", "v3.toml")).read()
cfgs = {m: re.sub(r'^ldlq = "(dense|blocks)"', f'ldlq = "{m}"', base, flags=re.M) for m in ("dense", "blocks")}
assert cfgs["blocks"] != cfgs["dense"]
for m, t in cfgs.items(): open(f"{W}/{m}.toml", "w").write(t)
LEVELS = [(0.125, "i8"), (0.5, "i8"), (2.5, "i8"), (6.5, "mul1")]
runs = {}
for mode, fork in (("dense", FD), ("blocks", FB)):
    cfg = C.load(f"{W}/{mode}.toml"); st = C.settings_stamp(cfg)
    for K, cb in LEVELS:
        out = f"{W}/out-{mode}-K{K}"
        groups = []
        for l in (2, 3): groups += [list(ks) for _, ks in qwen35.layer_plan(l)]
        ok = True
        for gi, ks in enumerate(groups):
            spec = {"type": "rxq.quant", "version": 1, "level": f"t-k{K}-{cb}", "K": K, "codebook": cb, "seed_offset": 0, "keys": ks,
                    "settings_stamp": st, "config": f"{W}/{mode}.toml",
                    "inputs": {"ckpt": ck, "hessians": hs, "hin": None, "wpre": None, "hout": ho}}
            sp = f"{W}/spec-{mode}-K{K}-{gi}.json"; json.dump(spec, open(sp, "w"))
            env = dict(os.environ, RXQ_CPU_STANDIN="1", RXQ_EXL3_SRC=f"{fork}/exllamav3/exllamav3_ext", PYTHONPATH=PIPE, OMP_NUM_THREADS="2")
            r = subprocess.run([sys.executable, __file__, "--run", sp, out], env=env, capture_output=True, text=True)
            if r.returncode: ok = False; print(r.stdout[-1500:], r.stderr[-3000:]); break
        runs[(mode, K)] = (out, ok)
        report(f"run-{mode}-K{K}", ok and all(os.path.exists(job.tensor_file(out, k)) for _, _, _, k in keys),
               f"{len(groups)} group jobs, {len(keys)} tensors at K{K} {cb}, H_out {mode}")

table = {cb: torch.from_numpy(deq.codebook_lut(cb).astype(np.float32)) for cb in ("i4", "i8", "mul1")}
for (mode, K), (out, ok) in runs.items():
    if not ok: continue
    cb = dict(LEVELS)[K]
    worst, recs = 0.0, {json.loads(x)["key"]: json.loads(x) for x in open(f"{out}/log.jsonl")}
    sig_ok, hb_ok, c9_ok, std_ok = True, True, True, True
    for l, hname, ks, key in keys:
        with safe_open(job.tensor_file(out, key), "pt") as f:
            t = {n.rsplit(".", 1)[1]: f.get_tensor(n) for n in f.keys()}; meta = f.metadata()
        Wd = deq.weight(t["trellis"], t["suh"], t["svh"], K, table[cb])                         # (k, n)
        with safe_open(f"{out}/wq/{key}.safetensors", "pt") as f: wq = f.get_tensor("weight").float().T
        worst = max(worst, float((Wd - wq).norm() / wq.norm()))
        r = recs[key]; inp = qwen35.is_input_group(hname)
        want = (0.2 if K <= 0.125 else (0.05 if inp and K < 1 else (0.005 if inp else 0.025)))
        sig_ok &= abs(r["sigma_reg"] - want) < 1e-12 and r["hout"] is True
        hb_ok &= (r.get("hout_blocks") == "group") == (mode == "blocks")
        fam = qwen35.family(key); al = {"ssm_out": 3.0, "attn_output": -1.0}.get(fam)
        c9_ok &= (os.path.exists(job.c9_path(out, key)) and r.get("c9", {}).get("alpha") == al) if al else \
            (not os.path.exists(job.c9_path(out, key)) and "c9" not in r)
        std_ok &= meta.get("rxq_kernels") == "cpu-standin"
    cd = [json.loads(x) for x in open(f"{out}/tilecd.jsonl")] if os.path.exists(f"{out}/tilecd.jsonl") else []
    cd_ok = [c["key"] for c in cd] == [k for _, _, _, k in keys if qwen35.family(k) == "attn_qg"] and all(c.get("hout") for c in cd)
    report(f"decode-{mode}-K{K}", worst < 2e-3, f"max |W_decoded - wq| / |wq| = {worst:.2e} over {len(keys)} tensors (f16 scales)")
    report(f"settings-{mode}-K{K}", sig_ok and hb_ok and c9_ok and cd_ok and std_ok,
           f"sigma / H_out {sig_ok}, hout_blocks {hb_ok}, C9 {c9_ok}, tile CD on q_proj only with H_out {cd_ok} "
           f"(ratio {[round(c['ratio'], 4) for c in cd]}), stand-in metadata {std_ok}")

out, ok = runs[("dense", 2.5)]
if ok:
    lv = f"{W}/level"
    try: job.assemble([out], lv, partial=True); refused = False
    except SystemExit as e: refused = "kernels" in str(e)
    st = job.assemble([out], lv, partial=True, allow_standin=True)
    shards = sorted(f for f in os.listdir(lv) if f.startswith("q-L"))
    c9 = os.path.exists(f"{lv}/c9/.done")
    mix = f"{W}/mixdir"; os.makedirs(mix, exist_ok=True)
    for f in os.listdir(out):
        if f.startswith(("t-", "c9-", "log")): os.symlink(f"{out}/{f}", f"{mix}/{f}")
    bk = keys[0][3]
    os.remove(f"{mix}/t-{bk}.safetensors"); os.symlink(job.tensor_file(runs[("blocks", 2.5)][0], bk), f"{mix}/t-{bk}.safetensors")
    try: job.assemble([mix], f"{W}/level-mixed", partial=True, allow_standin=True); mixed = False
    except SystemExit as e: mixed = "mixed stamps" in str(e)
    se = json.load(open(f"{lv}/stamp.json"))[-1]
    imp = (se.get("args") or {}).get("codebook", "mul1")     # rad-convert exl3i stamp_agrees: args.codebook, default mul1
    from rxq import mix as MX
    rec = MX.recipe(lv, *MX.level_info(lv))
    la, la_bad = job.levelargs(lv, C.load())
    stamp_ok = imp == "i8" and "codebook=mul1i8 rate=2.5" in rec and any("unknown ourq args" in b for b in la_bad)
    report("assemble", refused and shards == ["q-L02.safetensors", "q-L03.safetensors"] and c9 and mixed and stamp_ok,
           f"stand-in refused {refused}; --allow-standin -> {shards}, c9/.done {c9}, stamp {st}; blocks + dense tensors refused {mixed}; "
           f"stamp.json args.codebook {imp} (the exl3i importer's check), recipe mul1i8, rxq levelargs refuses an rxq level {stamp_ok}")
print(f"CPUJOB CHECKS {sum(res)}/{len(res)} PASS")
sys.exit(0 if all(res) else 1)
