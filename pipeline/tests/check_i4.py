#!/usr/bin/env python3
"""check_i4.py [RXM] -- CPU checks of the i4 sub-1 path (w53's qtf2-i4.patch, applied 11-07) without a GPU or a build:
  build    exl3frac2.load(SRC, "i4") asks torch's extension builder for exl3frac2_i4 with -DEXL3_MUL1_I4, the define
           exl3cb.load(SRC, "i4") builds the old i4 kernels with (so both decode the same int4 grid); i8 / mul1 unchanged
  ourq     quantize/ourq.py's --qtf2 check accepts --codebook i4 (and still refuses an unknown codebook)
  engine   rxq.engine.setup(codebook="i4", qtf2=True) installs exl3frac, exl3cb "i4" and exl3frac2 "i4", in ourq's order
  plan     the ladder's sub-1 levels resolve to i4 + qtf2 + H_out (every level since 11-08), sigma 0.2 on both groups at
           K0.125 (#114-02, 11-08), sigma_reg_in 0.05 on the layer-input group only (#38, user
           11-07), 0.005 from K1, tile CD on attn_qg
Torch's builder, exllamav3 and the kernel modules are stubs: what is checked is what the code asks for.
Last line: "I4 CHECKS n/m PASS"."""
import importlib, os, re, sys, tempfile, types

HERE = os.path.dirname(os.path.abspath(__file__))
RXM = sys.argv[1] if len(sys.argv) > 1 else os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
res = []


def report(name, ok, why):
    res.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'} {name}: {why}", flush=True)


# ---- build: capture cpp_extension.load
import torch  # noqa: E402
from torch.utils import cpp_extension  # noqa: E402
calls = []
cpp_extension.load = lambda **kw: (calls.append(kw), types.SimpleNamespace(name=kw["name"]))[1]
F = types.ModuleType("exllamav3.util.cuda_flags"); F.cuda_cflags = lambda **kw: ["-cuda"]
for m in ("exllamav3", "exllamav3.util"): sys.modules.setdefault(m, types.ModuleType(m))
sys.modules["exllamav3.util.cuda_flags"] = F; sys.modules["exllamav3.util"].cuda_flags = F
with tempfile.TemporaryDirectory() as src:
    os.makedirs(f"{src}/quant/comp_units")
    open(f"{src}/quant/frac.cu", "w").write("// KA >= 1 && KA <= 7\n#include \"../util.h\"\n")
    for f in ["quantize.cu", "exl3_devctx.cu"] + [f"comp_units/quantize_tiles_inst_k{i}.cu" for i in range(1, 9)] + \
             [f"comp_units/quantize_tiles_frac_inst_a{i}.cu" for i in range(1, 8)]:
        open(f"{src}/quant/{f}", "w").close()
    os.environ["TORCH_EXTENSIONS_DIR"] = src
    sys.path[:0] = [f"{RXM}/v3/e3/exl3frac2", f"{RXM}/v3/e3/exl3cb", f"{RXM}/v3/e3/exl3frac"]
    gen = f"{RXM}/v3/e3/exl3frac2/qtf2_frac_gen.cu"
    keep = open(gen).read() if os.path.exists(gen) else None           # load() rewrites the generated copy: restore it
    try:
        import exl3frac2, exl3cb
        for v in ("i4", "i8", "mul1"): exl3frac2.load(src, v)
        exl3cb.load(src, "i4")
    finally:
        if keep is not None: open(gen, "w").write(keep)
q = {c["name"]: c for c in calls}
i4f, i4c = q.get("exl3frac2_i4"), q.get("exl3cb_i4")
ok = (i4f and "-DEXL3_MUL1_I4" in i4f["extra_cflags"] and "-DEXL3_MUL1_I4" in i4f["extra_cuda_cflags"]
      and i4c and "-DEXL3_MUL1_I4" in i4c["extra_cflags"]
      and "-DEXL3_MUL1_I8" in q["exl3frac2_i8"]["extra_cflags"] and not any(x.startswith("-DEXL3_MUL1") for x in q["exl3frac2_mul1"]["extra_cflags"]))
report("build", ok, f"builds {sorted(q)}; exl3frac2_i4 defines {[x for x in (i4f or {}).get('extra_cflags', []) if x.startswith('-D')]}, "
       f"exl3cb_i4 {[x for x in (i4c or {}).get('extra_cflags', []) if x.startswith('-D')]}")

# ---- ourq: the patched assert
src = open(f"{RXM}/quantize/ourq.py").read()
m = re.search(r'assert a\.frac_ext and a\.codebook in \(([^)]*)\), "--qtf2', src)
cbs = eval("(" + m.group(1) + ")") if m else ()
report("ourq", "i4" in cbs and set(cbs) == {"mul1", "i8", "i8c", "i4"}, f"--qtf2 accepts {cbs}")

# ---- engine: rxq.engine.setup with stub exllamav3 / kernels
log = []
Qm = types.ModuleType("exllamav3.modules.quant.exl3_lib.quantize")
Qm.block_trace = None; Qm.refit_scales = lambda *a, **k: None; Qm.LDLQ_DRIFT = {2: 0.9, 3: 0.95}; Qm.ldlq_drift = None
for m in ("exllamav3.modules", "exllamav3.modules.quant", "exllamav3.modules.quant.exl3_lib"): sys.modules[m] = types.ModuleType(m)
sys.modules["exllamav3.modules.quant.exl3_lib.quantize"] = Qm
sys.modules["exllamav3.modules.quant.exl3_lib"].quantize = Qm
for name in ("exl3frac", "exl3cb", "exl3frac2"):
    mod = types.ModuleType(name)
    mod.install = (lambda n: lambda *a, **k: log.append((n,) + a[1:]))(name)
    sys.modules[name] = mod
os.environ["RXQ_EXL3_SRC"] = "/stub/src"
from rxq import engine  # noqa: E402
engine._warm_refit = lambda d: None
engine.setup(torch.device("cpu"), codebook="i4", qtf2=True)
report("engine", log == [("exl3frac",), ("exl3cb", "i4"), ("exl3frac2", "i4")], f"installs {log}")

# ---- plan: the sub-1 levels
from rxq import config as C, qwen35  # noqa: E402
cfg = C.load(); bad = []
for K in [x / 8 for x in range(1, 9)]:
    cb = C.codebook_for(cfg, K)
    for l in (2, 3):
        for _, ks in qwen35.layer_plan(l):
            for key in ks:
                s = C.quant_settings(cfg, K, cb, key)
                inp = qwen35.is_input_group(qwen35.locate(key)[3])
                # 11-08 (user): H_out at every level (apply_k_min 0); sigma 0.2 on both groups at K0.125 (#114-02)
                want = (0.2 if K <= 0.125 else 0.05 if K < 1 else 0.005) if inp else (0.2 if K <= 0.125 else None)
                if cb != "i4" or not s["qtf2"] or "hout" not in s: bad.append((K, key, "codebook / qtf2 / hout"))
                if inp and s["sigma_reg_in"] != want: bad.append((K, key, s["sigma_reg_in"]))
                if not inp and want is not None and s["sigma_reg"] != want: bad.append((K, key, s["sigma_reg"]))
                if ("tile_cd" in s) != key.endswith("self_attn.q_proj"): bad.append((K, key, "tile_cd"))
off = C.with_overrides(cfg, **{"quant.apply_sub1_damping": False})
offv = C.quant_settings(off, 0.5, "i4", "model.language_model.layers.3.self_attn.q_proj")["sigma_reg_in"]
report("plan", not bad and offv == 0.005, f"K0.125-1 x layers 2 / 3: {len(bad)} wrong {bad[:3]}; switch off -> sigma_in {offv}")

print(f"I4 CHECKS {sum(res)}/{len(res)} PASS")
sys.exit(0 if all(res) else 1)
