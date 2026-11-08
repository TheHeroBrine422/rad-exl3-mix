#!/usr/bin/env python3
"""check_grouphead.py [QUANTIZE_PY] -- CPU regression test for gpucheck27's i4 (2) FAIL of 11-08 (w80): rxq vs ourq at
K0.5 i4 + sigma_in 0.05 differed on layer 39 k_proj / v_proj (suh, proxy_err, g_scale) but not on o_proj.
Cause: the reference was `ourq --only {k, v, o}`. ourq skips q_proj, so k_proj is the first tensor to finalize the
shared input Hessian and exllamav3 draws the group's sign vector su under k_proj's seed. The ladder (a whole-layer
ourq run) and rxq both draw it under q_proj's seed. The sub-1 damping is the same on both sides (0.05, layer-input
group only).
  rxq      rxq.quant.quantize_tensor of k_proj / v_proj == ourq's group run (q, k, v): the same su; the real
           finalize_capture_H of the exllamav3 fork (QUANTIZE_PY, default the VM's fork), ext stubbed, Hadamard
           Sylvester, tiny random Hessian
  only     ourq --only {k, v} (no q_proj): su differs from the group run (the 11-08 mechanism), for k and v alike
  damping  the su draw does not depend on sigma_reg (0.05 vs 0.005 under one seed: same su, different H), and the
           settings put 0.05 on q / k / v (and GDN qkv / z) at K0.5 (rxq's group pre-finalize included), 0.025 on
           o_proj / gate / up / down, 0.005 on the inputs from K1
  guard    rxq.qwen35.missing_group_heads flags k / v without q (and z without qkv, up without gate), nothing with the
           heads; check_quant.py prints BADREF for such keys
  source   ourq.quantize_group still skips --only misses before quantizing (the emulation below mirrors it), and
           tools/gpucheck27.sh's i4 reference list now holds q_proj
Last line: "GROUPHEAD CHECKS n/m PASS"."""
import importlib.util, json, math, os, re, struct, sys, tempfile, types

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
RXM = os.path.normpath(os.path.join(HERE, "..", ".."))
QPY = sys.argv[1] if len(sys.argv) > 1 else "/workspace/forks/exllamav3-frac/exllamav3/modules/quant/exl3_lib/quantize.py"
sys.path.insert(0, os.path.join(HERE, ".."))
from rxq import qwen35, quant, config, engine  # noqa: E402
res = []


def report(name, ok, why):
    res.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'} {name}: {why}", flush=True)


# ---- the fork's quantize.py with its extension and utilities stubbed (finalize_capture_H is plain torch)
def had(n):
    h = torch.ones(1, 1)
    while h.shape[0] < n: h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def stub(name, **kw):
    m = types.ModuleType(name); m.__path__ = []; m.__dict__.update(kw); sys.modules[name] = m; return m


stub("exllamav3"); stub("exllamav3.ext", exllamav3_ext=types.SimpleNamespace())
stub("exllamav3.util", cuda_sync_active=lambda *a, **k: None)
stub("exllamav3.util.progress", ProgressBar=object); stub("exllamav3.util.memory", free_mem=None, list_gpu_tensors=None)
stub("exllamav3.util.hadamard", get_hadamard_dt=lambda n, device, dtype, scale=1.0: had(n).to(device, dtype) * scale)
stub("exllamav3.util.tensor", save_tensor_image=None)
for p in ("exllamav3.modules", "exllamav3.modules.quant", "exllamav3.modules.quant.exl3_lib"): stub(p)
spec = importlib.util.spec_from_file_location("exllamav3.modules.quant.exl3_lib.quantize", QPY)
X = importlib.util.module_from_spec(spec); spec.loader.exec_module(X)


def fake_quantize_exl3(weight, H_data, qa, return_weight=False):
    """quantize_exl3's first steps verbatim (seed, finalize); 'suh' = the su it got"""
    if "seed" in qa: torch.manual_seed(qa["seed"])
    _, H, _, su, _ = X.finalize_capture_H(H_data, qa, False)
    return None, 0.0, {"suh": su.flatten().clone()}


Q = types.SimpleNamespace(finalize_capture_H=X.finalize_capture_H, quantize_exl3=fake_quantize_exl3,
                          ldlq_drift=lambda K: 1.0, g_scale_gss=None)
engine._STATE.update(gain=[1.0])                       # what engine.setup() leaves (no kernels here)

# ---- a tiny layer 39 (full attention): input Hessian 256 wide, random weights
K_IN, l = 256, 39
P = f"{qwen35.L}.{l}"
hname, gkeys = qwen35.layer_plan(l)[0]
assert gkeys == [f"{P}.self_attn.q_proj", f"{P}.self_attn.k_proj", f"{P}.self_attn.v_proj"]
g = torch.Generator().manual_seed(1)
A = torch.randn(4 * K_IN, K_IN, generator=g)
H0 = (A.T @ A).float()
tmp = tempfile.mkdtemp(prefix="w80gh-")
iu = torch.triu_indices(K_IN, K_IN)
save_file({"hessian_upper": H0[iu[0], iu[1]].contiguous()}, f"{tmp}/{hname}.safetensors", metadata={"k": str(K_IN), "rows": "1024"})
save_file({f"{k}.weight": torch.randn(64, K_IN, generator=g).to(torch.bfloat16) for k in gkeys}, f"{tmp}/model.safetensors")
ck = quant.Ckpt(tmp)
cfg = config.load(os.path.join(RXM, "pipeline", "config", "v3.toml"))


def hdata():
    return quant.hessian(f"{tmp}/{hname}.safetensors", hname, torch.device("cpu"))


def ourq_group(only, sigma):
    """ourq.quantize_group's loop: one fresh group Hessian, keys in plan order, --only misses skipped, seed0 + j"""
    Hd, out = hdata(), {}
    seed0 = l * 16 + 0 * 4
    for j, key in enumerate(gkeys):
        if only is not None and key not in only: continue
        torch.manual_seed(seed0 + j)
        out[key] = X.finalize_capture_H(Hd, {"sigma_reg": sigma, "scaled_hessian": True}, False)[3].flatten().clone()
    return out


def rxq_tensor(key, K=0.5):
    s = config.quant_settings(cfg, K, "i4", key)
    s["tile_cd"] = None
    t, rec, _, _ = quant.quantize_tensor(Q, ck, key, s, tmp, "cpu", tmp)
    return t[f"{key}.suh"], rec


# ---- rxq == ourq group run
grp = ourq_group(None, 0.05)
same = []
for key in gkeys[1:]:
    su, rec = rxq_tensor(key)
    same.append(torch.equal(su, grp[key]) and rec["sigma_reg"] == 0.05)
report("rxq", all(same) and torch.equal(grp[gkeys[1]], grp[gkeys[0]]),
       f"k / v su of rxq per-tensor jobs == ourq group run {same}; sigma 0.05 logged")

# ---- ourq --only without the head
only = ourq_group({gkeys[1], gkeys[2]}, 0.05)
fr = [(only[k] != grp[k]).float().mean().item() for k in gkeys[1:]]
report("only", all(f > 0.3 for f in fr), f"--only {{k, v}}: su differs from the group run in {[round(f, 3) for f in fr]} "
       f"of entries (the 11-08 FAIL; o_proj, a group of one, is unaffected)")

# ---- damping does not move the su draw; the settings' rule
a, b = hdata(), hdata()
torch.manual_seed(624); sa = X.finalize_capture_H(a, {"sigma_reg": 0.05, "scaled_hessian": True}, False)[3]
torch.manual_seed(624); sb = X.finalize_capture_H(b, {"sigma_reg": 0.005, "scaled_hessian": True}, False)[3]
rule = {}
for K in (0.5, 1.0):
    for key in (f"{P}.self_attn.q_proj", f"{P}.self_attn.k_proj", f"{P}.self_attn.v_proj", f"{P}.self_attn.o_proj",
                f"{P}.mlp.gate_proj", f"{P}.mlp.up_proj", f"{P}.mlp.down_proj", f"{qwen35.L}.40.linear_attn.in_proj_qkv",
                f"{qwen35.L}.40.linear_attn.in_proj_z", f"{qwen35.L}.40.linear_attn.out_proj"):
        s = config.quant_settings(cfg, K, "i4" if K < 1 else "i8", key)
        _, _, _, hn, gk = qwen35.locate(key)
        qa = quant.quant_args(s, key, 0, 0, tmp, hn)
        rule[(K, qwen35.family(key))] = qa["sigma_reg"]
want = {f: (0.05 if f in ("attn_qg", "attn_k", "attn_v", "ssm_qkv", "ssm_z") else 0.025) for _, f in rule}
ok_rule = all(rule[(0.5, f)] == want[f] for f in want) and \
    all(rule[(1.0, f)] == (0.005 if want[f] == 0.05 else 0.025) for f in want)
report("damping", torch.equal(sa, sb) and not torch.equal(a["H"], b["H"]) and ok_rule,
       f"su same under sigma 0.05 / 0.005: {torch.equal(sa, sb)}; K0.5 {dict((f, v) for (K, f), v in rule.items() if K == 0.5)}; "
       f"K1 inputs {sorted(set(v for (K, f), v in rule.items() if K == 1.0 and want[f] == 0.05))}")

# ---- the guard
m1 = qwen35.missing_group_heads(gkeys[1:] + [f"{P}.self_attn.o_proj"])
m2 = qwen35.missing_group_heads(gkeys + [f"{P}.self_attn.o_proj", "lm_head", "model.embed_tokens"])
m3 = qwen35.missing_group_heads([f"{qwen35.L}.40.linear_attn.in_proj_z", f"{P}.mlp.up_proj"])
ok = (m1 == {gkeys[1]: gkeys[0], gkeys[2]: gkeys[0]} and m2 == {}
      and m3 == {f"{qwen35.L}.40.linear_attn.in_proj_z": f"{qwen35.L}.40.linear_attn.in_proj_qkv", f"{P}.mlp.up_proj": f"{P}.mlp.gate_proj"})
cq = open(os.path.join(HERE, "check_quant.py")).read()
report("guard", ok and "missing_group_heads" in cq and "BADREF" in cq, f"{{k, v, o}} -> {len(m1)} flagged; with heads {len(m2)}; z / up {len(m3)}")

# ---- source
src = open(os.path.join(RXM, "quantize", "ourq.py")).read()
qg = src[src.index("def quantize_group"):src.index("_CUR[0] = key; wq, perr, t = quantize_exl3(w, Hk")]
gc = open(os.path.join(RXM, "pipeline", "tools", "gpucheck27.sh")).read()
i4 = gc[gc.index("i4_unit() {"):gc.index("c9_unit() {")]
mref = re.search(r"printf '%s\\n' ([^>]*)> \$O/i4-ref-keys.txt", i4)
report("source", "if ONLY is not None and key not in ONLY: continue" in qg and mref is not None
       and "$P.39.self_attn.q_proj" in mref.group(1) and "--only $O/i4-ref-keys.txt" in i4,
       "ourq skips --only misses before quantize_exl3; gpucheck27 i4 reference list holds q_proj")

import shutil; shutil.rmtree(tmp, ignore_errors=True)
print(f"GROUPHEAD CHECKS {sum(res)}/{len(res)} {'PASS' if all(res) else 'FAIL'}")
