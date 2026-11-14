#!/usr/bin/env python3
"""check_mad.py FORK [WORK] -- the mad codebook in rxq / ourq (#177-03, w151), on the CPU:
  table    rxq.deq's mad table == v3/e3/exl3hash values() == w139's exl3hashm values() (no map), A / B from the settings
  config   v3.toml: [quant.codebook.mad] (a, b) = deq.MAD_AB, every ladder level's codebook is mad and its name ends
           -mad; validate() refuses another mask / L / bias; the codes stamp differs from the same file with i8
  env      exl3hash.set_ab: an env EXL3HASH_A that disagrees with the explicit value is an error, an agreeing one is fine
  job      a K2.5 mad job on a tiny attention layer through rxq.job.run_quant with the CPU stand-in kernels (FORK = the
           exllamav3 fork with patch 0008): every tensor's codes decode (rxq.deq, mad table) to the job's own wq; the
           tensor metadata carries codebook mad + (a, b); assemble writes "mad" into stamp.json (top level and args);
           deq.Level reads the level with the mad table; the recipe says codebook=mad
  ourq     quantize/ourq.py parses --codebook mad (and keeps --mad-a / --mad-b out of other codebooks' stamps)
Last line: "MAD CHECKS n/m PASS"."""
import json, os, re, subprocess, sys, tempfile

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.normpath(os.path.join(HERE, ".."))
RXM = os.path.normpath(os.path.join(PIPE, ".."))
sys.path.insert(0, PIPE)
from rxq import config as C, deq, job, qwen35  # noqa: E402

if len(sys.argv) > 1 and sys.argv[1] == "--run":          # child: one job (one engine setup a process)
    job.run_quant(json.load(open(sys.argv[2])), sys.argv[3], device="cpu", save_wq=True)
    sys.exit(0)

FORK = sys.argv[1]
W = sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp(prefix="w151mad-")
os.makedirs(W, exist_ok=True)
res = []


def report(name, ok, why):
    res.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'} {name}: {why}", flush=True)


def load_mod(name, path):
    import importlib.util
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


# ---- table
cfg = C.load()
ab = C.mad_params(cfg)
H = load_mod("exl3hash", f"{RXM}/v3/e3/exl3hash/exl3hash.py")
HM = load_mod("exl3hashm", f"{RXM}/v3/e3/hash177/exl3hashm/exl3hashm.py")
for k in ("EXL3HASH_A", "EXL3HASH_B", "EXL3HASH_MAP"): os.environ.pop(k, None)
t = deq.lut("mad", ab)
report("table", ab == deq.MAD_AB and np.array_equal(t, H.values(*ab)) and np.array_equal(t, HM.values(*ab))
       and np.array_equal(t, deq.lut("mad")) and len(np.unique(t)) > 100,
       f"(a, b) = ({ab[0]:#x}, {ab[1]:#x}); deq == exl3hash == exl3hashm over 65,536 states; {len(np.unique(t))} values, "
       f"q range [{t.min() / t[t > 0].min():.0f}, {t.max() / t[t > 0].min():.0f}] steps")

# ---- config
from rxq import ladder  # noqa: E402
lv = ladder.levels(cfg)
src = open(C.DEFAULT).read()
i8 = src.replace('codebook = "mad" }', 'codebook = "i8" }')
assert i8 != src
open(f"{W}/i8.toml", "w").write(i8)
bad = []
for k, v in (("mask", "0x3F3F3F3E"), ("L", "17"), ("bias", "127")):
    open(f"{W}/bad.toml", "w").write(re.sub(rf"^{k} = .*$", f"{k} = {v}", src, count=1, flags=re.M))
    try: C.load(f"{W}/bad.toml"); bad.append(k)
    except SystemExit: pass
st, st8 = C.settings_stamp(cfg), C.settings_stamp(C.load(f"{W}/i8.toml"))
report("config", all(v["codebook"] == "mad" and v["name"].endswith("-mad") for v in lv) and not bad and st != st8,
       f"{len(lv)} levels all mad (first {lv[0]['name']}); bad mask / L / bias accepted: {bad}; codes stamp {st} vs i8 {st8}")

# ---- env
os.environ["EXL3HASH_A"] = "0xd0474c"
try: H.set_ab(*ab); env_bad = False
except SystemExit: env_bad = True
os.environ["EXL3HASH_A"] = hex(ab[0])
try: H.set_ab(*ab); env_ok = H.ab() == ab
except SystemExit: env_ok = False
os.environ.pop("EXL3HASH_A")
report("env", env_bad and env_ok, f"disagreeing env refused {env_bad}; agreeing env accepted {env_ok}")

# ---- job: one attention layer (layer 3), K2.5 mad, CPU stand-in
SH = {"attn_qg": (256, 256), "attn_k": (128, 256), "attn_v": (128, 256), "attn_output": (256, 256),
      "ffn_gate": (384, 256), "ffn_up": (384, 256), "ffn_down": (256, 384)}
g = torch.Generator().manual_seed(151)
ck, hs, ho = f"{W}/ckpt", f"{W}/hess", f"{W}/hout/bn1920"
for d in (ck, hs, ho): os.makedirs(d, exist_ok=True)
tens, keys, groups = {}, [], []
for hname, ks in qwen35.layer_plan(3):
    k_in = SH[qwen35.family(ks[0])][1]
    A = torch.randn(4 * k_in, k_in, generator=g) * torch.linspace(0.3, 2.0, k_in)
    Hm = (A.T @ A).float(); iu = torch.triu_indices(k_in, k_in)
    save_file({"hessian_upper": Hm[iu[0], iu[1]].contiguous()}, f"{hs}/{hname}.safetensors", metadata={"k": str(k_in), "rows": str(4 * k_in)})
    for key in ks:
        n, k = SH[qwen35.family(key)]
        tens[f"{key}.weight"] = (torch.randn(n, k, generator=g) * 0.02).to(torch.bfloat16)
        save_file({"hout_diag": (torch.rand(n, generator=g) + 0.1).float()}, f"{ho}/{key}.safetensors")
        keys.append(key)
    groups.append(list(ks))
save_file(tens, f"{ck}/model.safetensors")
out = f"{W}/out"; ok = True
for gi, ks in enumerate(groups):
    spec = {"type": "rxq.quant", "version": 1, "level": "t-k2.5-mad", "K": 2.5, "codebook": "mad", "seed_offset": 0, "keys": ks,
            "settings_stamp": st, "inputs": {"ckpt": ck, "hessians": hs, "hin": None, "wpre": None, "hout": ho}}
    sp = f"{W}/spec-{gi}.json"; json.dump(spec, open(sp, "w"))
    env = dict(os.environ, RXQ_CPU_STANDIN="1", RXQ_EXL3_SRC=f"{FORK}/exllamav3/exllamav3_ext", PYTHONPATH=PIPE, OMP_NUM_THREADS="2")
    r = subprocess.run([sys.executable, __file__, "--run", sp, out], env=env, capture_output=True, text=True)
    if r.returncode: ok = False; print(r.stdout[-1500:], r.stderr[-3000:]); break
if ok:
    worst, metas = 0.0, set()
    tb = torch.from_numpy(t)
    for key in keys:
        with safe_open(job.tensor_file(out, key), "pt") as f:
            tt = {n.rsplit(".", 1)[1]: f.get_tensor(n) for n in f.keys()}; m = f.metadata()
        metas.add((m["codebook"], m.get("mad")))
        Wd = deq.weight(tt["trellis"], tt["suh"], tt["svh"], 2.5, tb)
        with safe_open(f"{out}/wq/{key}.safetensors", "pt") as f: wq = f.get_tensor("weight").float().T
        worst = max(worst, float((Wd - wq).norm() / wq.norm()))
    # the same codes read with the i8 table must NOT match (the check would pass vacuously otherwise)
    with safe_open(job.tensor_file(out, keys[0]), "pt") as f: tt = {n.rsplit(".", 1)[1]: f.get_tensor(n) for n in f.keys()}
    with safe_open(f"{out}/wq/{keys[0]}.safetensors", "pt") as f: wq = f.get_tensor("weight").float().T
    w8 = float((deq.weight(tt["trellis"], tt["suh"], tt["svh"], 2.5, torch.from_numpy(deq.lut("i8"))) - wq).norm() / wq.norm())
    report("job-decode", worst < 2e-3 and w8 > 0.1 and metas == {("mad", f"{ab[0]:#x},{ab[1]:#x}")},
           f"{len(keys)} tensors: max |W_mad - wq| / |wq| = {worst:.2e} (i8 table on the same codes: {w8:.2f}); metadata {sorted(metas)}")
    lvd = f"{W}/level"
    job.assemble([out], lvd, partial=True, allow_standin=True)
    se = json.load(open(f"{lvd}/stamp.json"))[-1]
    L = deq.Level(lvd)
    Wl = L.get(keys[0]).T                      # (k, n) like deq.weight
    from rxq import mix as MX
    rec = MX.recipe(lvd, *MX.level_info(lvd))
    report("assemble", se["codebook"] == "mad" and se["args"]["codebook"] == "mad" and se.get("mad") == se["args"].get("mad")
           == {"a": f"{ab[0]:#x}", "b": f"{ab[1]:#x}"} and L.mad == ab and float((Wl - wq).norm() / wq.norm()) < 2e-3
           and "codebook=mad rate=2.5" in rec and se["rxq_settings"] == st,
           f"stamp.json codebook {se['codebook']} / args {se['args'].get('codebook')} / mad {se.get('mad')}; deq.Level mad "
           f"{L.mad}; recipe line '{rec.splitlines()[0]}'")
else:
    report("job", False, "a job failed (above)")

# ---- ourq argparse (no run: --help lists mad; the stamp args drop mad_a / mad_b for other codebooks)
oq = open(f"{RXM}/quantize/ourq.py").read()
report("ourq", '"mad")' in oq and 'k in ("mad_a", "mad_b") and a.codebook != "mad"' in oq and "exl3hash.install(a.frac_ext, True, a=a.mad_a, b=a.mad_b)" in oq,
       "ourq: --codebook mad installs exl3hash with explicit (a, b); --mad-a / --mad-b stamped with mad only")
print(f"MAD CHECKS {sum(res)}/{len(res)} {'PASS' if all(res) else 'FAIL'}")
sys.exit(0 if all(res) else 1)
