#!/usr/bin/env python3
"""test_wpre.py ORIG_OURQ PATCHED_OURQ CKPT HESSDIR WORK -- CPU test of wpre.patch: the real ourq.py driver (original and
patched) on layer 0 of the 0.8B, with exllamav3's quantize_exl3 replaced by a deterministic fake (wq = w + 1e-3 x mean
diag H, so the Hessian and the target both show in the output) and the CUDA device mapped to the CPU (the same textual
edit on both copies). Cases:
  1. no --wpre: every output tensor (shard, wq, rowerr) identical to the original's; proxy.jsonl identical but for "s";
     stamp.json's argument dict identical (the stamp hash differs: it hashes the file);
  2. --wpre with identity transforms: tensors identical to case 1;
  3. --wpre with random transforms: each saved wq = (W F) + c in f16, W F computed here; lm_head untouched (not run);
  4. --wpre with one group's file missing: exits naming it."""
import glob, json, os, shutil, subprocess, sys, types
import numpy as np, torch
from safetensors.torch import save_file, load_file
ORIG, PAT, CKPT, HESS, WORK = sys.argv[1:6]
os.makedirs(WORK, exist_ok=True)

STUB = '''
import torch
LDLQ_DRIFT = {1: 1.08, 1.5: 1.035, 2: 1.018, 2.5: 1.009, 3: 1.004}
def ldlq_drift(K): return LDLQ_DRIFT.get(float(K), 1.0)
def block_trace(A, B, block_size=1024): return 1.0
def refit_scales(*a, **k): raise RuntimeError("not used by the fake")
def finalize_capture_H(*a, **k): raise RuntimeError("not used by the fake")
def unrotate_H(*a, **k): raise RuntimeError("not used by the fake")
def g_scale_gss(*a, **k): return 1.0, 0.0
def quantize_exl3(weight, H_data, qa, ret):
    c = 1e-3 * float(H_data["H"].diagonal().mean())
    wq = weight + c
    qa["g_scale"] = 1.0
    return wq, float(weight.square().mean()), {"trellis": wq.half(), "suh": H_data["H"].diagonal()[:16].half()}
'''
def prep(src, dst):                                    # CPU edit (both copies alike) + the exllamav3 stub package
    s = open(src).read()
    s = s.replace('torch.device("cuda", a.device)', 'torch.device("cpu")')
    s = s.replace("torch.cuda.synchronize(dev_)", "pass").replace("torch.cuda.empty_cache()", "pass")
    os.makedirs(os.path.dirname(dst), exist_ok=True); open(dst, "w").write(s)
pk = f"{WORK}/stub/exllamav3/modules/quant/exl3_lib"
os.makedirs(pk, exist_ok=True)
for d in ("exllamav3", "exllamav3/modules", "exllamav3/modules/quant", "exllamav3/modules/quant/exl3_lib"):
    open(f"{WORK}/stub/{d}/__init__.py", "w").close()
open(f"{pk}/quantize.py", "w").write(STUB)
prep(ORIG, f"{WORK}/a/ourq.py"); prep(PAT, f"{WORK}/b/ourq.py")
os.makedirs(f"{WORK}/rw/challenge/hessians", exist_ok=True)
if not os.path.exists(f"{WORK}/rw/challenge/hessians/calib"): os.symlink(HESS, f"{WORK}/rw/challenge/hessians/calib")
json.dump({"quantization_config": {"quant_method": "exl3"}}, open(f"{WORK}/template.json", "w"))
ENV = dict(os.environ, PYTHONPATH=f"{WORK}/stub", RXM_WORK=f"{WORK}/rw", RXM_CKPT=CKPT, RXM_TEMPLATE=f"{WORK}/template.json")
def run(which, out, *extra):
    shutil.rmtree(out, ignore_errors=True)
    return subprocess.run([sys.executable, f"{WORK}/{which}/ourq.py", "-K", "3", "-o", out, "--layers", "0-0", "--save-wq", *extra],
                          env=ENV, capture_output=True, text=True, cwd=WORK)
def tensors(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/q-*.safetensors") + glob.glob(f"{d}/wq/*.safetensors")):
        for k, v in load_file(f).items(): out[f"{os.path.relpath(f, d)}:{k}"] = v
    for f in sorted(glob.glob(f"{d}/wq/*.npz")):
        z = np.load(f)
        for k in z.files: out[f"{os.path.relpath(f, d)}:{k}"] = torch.from_numpy(z[k])
    return out
def same(A, B): return set(A) == set(B) and all(torch.equal(A[k], B[k]) for k in A)
def proxy(d): return [{k: v for k, v in json.loads(l).items() if k != "s"} for l in open(f"{d}/proxy.jsonl")]
ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name} {info}", flush=True)

r0 = run("a", f"{WORK}/o-orig"); r1 = run("b", f"{WORK}/o-off")
check("original runs", r0.returncode == 0, r0.stderr[-400:]); check("patched, option off, runs", r1.returncode == 0, r1.stderr[-400:])
T0, T1 = tensors(f"{WORK}/o-orig"), tensors(f"{WORK}/o-off")
check("1. option off: every tensor identical", len(T0) > 0 and same(T0, T1), f"({len(T0)} tensors)")
check("1. option off: proxy.jsonl identical (but timing)", proxy(f"{WORK}/o-orig") == proxy(f"{WORK}/o-off"))
s0, s1 = (json.load(open(f"{WORK}/{d}/stamp.json"))[-1] for d in ("o-orig", "o-off"))
check("1. option off: stamp argument dict identical", s0["args"] == s1["args"], f"(stamps {s0['stamp']} / {s1['stamp']})")

G = ["model.language_model.layers.0.input", "model.language_model.layers.0.linear_attn.out_proj.input",
     "model.language_model.layers.0.post_attention.input", "model.language_model.layers.0.mlp.down_proj.input"]
K = {G[0]: 1024, G[1]: 2048, G[2]: 1024, G[3]: 3584}
for nm, mk in (("eye", lambda k: torch.eye(k)), ("rand", lambda k: torch.eye(k) + 0.01 * torch.randn(k, k, generator=torch.Generator().manual_seed(k)))):
    os.makedirs(f"{WORK}/wp-{nm}", exist_ok=True)
    for g in G: save_file({"Ft": mk(K[g]).contiguous()}, f"{WORK}/wp-{nm}/{g}.safetensors")
r2 = run("b", f"{WORK}/o-eye", "--wpre", f"{WORK}/wp-eye")
check("2. identity transforms run", r2.returncode == 0, r2.stderr[-400:])
check("2. identity transforms: every tensor identical to the original", same(T0, tensors(f"{WORK}/o-eye")))
check("2. stamp args carry wpre", json.load(open(f"{WORK}/o-eye/stamp.json"))[-1]["args"].get("wpre") == f"{WORK}/wp-eye")
r3 = run("b", f"{WORK}/o-rand", "--wpre", f"{WORK}/wp-rand")
check("3. random transforms run", r3.returncode == 0, r3.stderr[-400:])
from safetensors import safe_open
idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
H = {g: None for g in G}
worst, n = 0.0, 0
plan = {G[0]: ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"], G[1]: ["linear_attn.out_proj"],
        G[2]: ["mlp.gate_proj", "mlp.up_proj"], G[3]: ["mlp.down_proj"]}
for g, ks in plan.items():
    with safe_open(f"{HESS}/{g}.safetensors", "pt") as f: up = f.get_tensor("hessian_upper"); rows = int(f.metadata()["rows"])
    k = K[g]; Hm = torch.zeros(k, k); iu = torch.triu_indices(k, k); Hm[iu[0], iu[1]] = up; Hm = Hm + Hm.triu(1).T
    c = 1e-3 * float(Hm.diagonal().mean())             # the fake's constant (ourq.py's H as loaded, before finalize)
    Ft = load_file(f"{WORK}/wp-rand/{g}.safetensors")["Ft"]
    for s_ in ks:
        key = f"model.language_model.layers.0.{s_}"
        with safe_open(f"{CKPT}/{idx[key + '.weight']}", "pt") as f: W = f.get_tensor(key + ".weight").float()
        want = (W @ Ft.T + c).half(); got = load_file(f"{WORK}/o-rand/wq/{key}.safetensors")["weight"]
        worst = max(worst, float((got.float() - want.float()).abs().max() / want.float().abs().max())); n += 1
check("3. random transforms: wq = W F + c (f16)", n == 6 and worst < 2e-3, f"({n} linears, max rel dev {worst:.1e})")
os.remove(f"{WORK}/wp-rand/{G[3]}.safetensors")
r4 = run("b", f"{WORK}/o-miss", "--wpre", f"{WORK}/wp-rand")
check("4. missing transform exits naming its group", r4.returncode != 0 and G[3] in r4.stderr, r4.stderr.strip().splitlines()[-1][:200] if r4.stderr else "")
print("ALL PASS" if ok else "SOME FAIL")
