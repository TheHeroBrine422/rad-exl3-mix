#!/usr/bin/env python3
"""test_moments.py MODEL META WQDIR WORK -- CPU test of moments.py on the real 0.8B (a few 512-token qad windows):
  A. noise mode, gamma 0: C = N = 0 exactly, and H equals an independent batch-1 hook collection on the same windows;
  B. noise mode, gamma 0.05: at layer 0's input group eta is the local noise alone, so diag(N)/n = sigma^2 (sampling
     error ~ sqrt(2/n) per channel) and C/n is near 0; later groups carry propagated noise (diag N/n > sigma^2);
  C. the same moments from layer ranges 0-1 + 2-3 as from one 0-3 run (noise seeded per window, layer, group);
  D. sib mode, siblings = the layer's own bf16 weights (written as bf16 wq files): C = N = 0 exactly and H = the clean
     collection; o / out and down groups only;
  E. sib mode, siblings = the K3 base quant's wq (layers 0 and 3): eta != 0, and layer 0's down input equals an
     independent recomputation act(Gq x) * (Uq x) from the clean post-attention input;
  F. quant mode, every linear of layers 0-3 = its own bf16 weight: C = N = 0 exactly, H = the clean collection; then with
     the K3 base quant's wq (only layers 0 and 3 present here, 1-2 stay bf16): layer 0's input group has eta = 0 exactly
     (nothing upstream is quantized), every later group eta != 0; trace shares printed (the real upstream error)."""
import glob, json, os, subprocess, sys
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file
MODEL, META, WQ, WORK = sys.argv[1:5]
HERE = os.path.dirname(os.path.abspath(__file__)); MOM = os.path.join(HERE, "..", "moments.py")
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
os.makedirs(WORK, exist_ok=True)
ENV = dict(os.environ, DEV="cpu")
ARGS = ["--tokens", "1500", "--win", "512", "--seed", "3"]
ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name} {info}", flush=True)
def run(out, *extra):
    r = subprocess.run([sys.executable, MOM, MODEL, META, out, *ARGS, *extra], env=ENV, capture_output=True, text=True)
    if r.returncode: print(r.stderr[-1500:])
    return r.returncode == 0
def load(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/*.input.safetensors")):
        with safe_open(f, "pt") as h: out[os.path.basename(f)[:-12]] = ({s: h.get_tensor(s).double() for s in "HCN"}, int(h.metadata()["n"]))
    return out
L = "model.language_model.layers"

# reference: plain clean collection (batch 1), the same windows as moments.py's
from common import load_model
model, dec, _ = load_model(MODEL, dev="cpu"); model.requires_grad_(False)
m = json.load(open(f"{META}/kld.json")); toks = np.memmap(f"{META}/tokens.i32", np.int32, "r")
pools = json.load(open(os.environ["POOLS"]))["pool"]; off = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
wins = []
for i, d in enumerate(m["docs"]):
    if pools[i] != "qad": continue
    if d["tokens"] < 512:
        if d["tokens"] >= 64: wins.append((int(off[i]), d["tokens"]))
    else: wins += [(int(off[i] + s), 512) for s in range(0, d["tokens"] - 511, 512)]
np.random.default_rng(3).shuffle(wins); cum = np.cumsum([n for _, n in wins]); wins = wins[:int(np.searchsorted(cum, 1500)) + 1]
REF, POST = {}, {}
def rh(name, out_side):
    def f(mod, *xs):
        x = (xs[1] if out_side else xs[0][0]).reshape(-1, (xs[1] if out_side else xs[0][0]).shape[-1]).double()
        REF[name] = REF.get(name, 0) + x.T @ x
        if name == f"{L}.0.post_attention.input": POST.setdefault("x", []).append(x)
    return f
hk = []
for l in range(4):
    ly = dec.layers[l]; full = hasattr(ly, "self_attn"); mix = ly.self_attn.o_proj if full else ly.linear_attn.out_proj
    mn = f"{L}.{l}.self_attn.o_proj.input" if full else f"{L}.{l}.linear_attn.out_proj.input"
    hk += [ly.input_layernorm.register_forward_hook(rh(f"{L}.{l}.input", True)), mix.register_forward_pre_hook(rh(mn, False)),
           ly.post_attention_layernorm.register_forward_hook(rh(f"{L}.{l}.post_attention.input", True)),
           ly.mlp.down_proj.register_forward_pre_hook(rh(f"{L}.{l}.mlp.down_proj.input", False))]
with torch.no_grad():
    for w0, n in wins: dec(input_ids=torch.from_numpy(np.asarray(toks[w0:w0 + n], np.int64))[None])
for h in hk: h.remove()
ntok = sum(n for _, n in wins)
print(f"reference: {len(wins)} windows, {ntok} tokens", flush=True)
def rel(a, b): return float((a - b).norm() / b.norm())

# A
check("A. noise gamma 0 runs", run(f"{WORK}/a", "--mode", "noise", "--gamma", "0", "--layers", "0-3"))
A = load(f"{WORK}/a")
check("A. 16 groups, n = tokens", len(A) == 16 and all(n == ntok for _, n in A.values()))
check("A. gamma 0: C = N = 0 exactly", all(float(v["C"].abs().max()) == 0 and float(v["N"].abs().max()) == 0 for v, _ in A.values()))
ra = {g: rel(A[g][0]["H"], REF[g]) for g in A}
check("A. H = the batch-1 clean collection (bf16 rounding of the batch-2 forward)", max(ra.values()) < 1e-2,
      "(rel by layer: " + ", ".join(f"{l} {max(v for g, v in ra.items() if f'layers.{l}.' in g):.1e}" for l in range(4)) + ")")
# B
check("B. noise gamma 0.05 runs", run(f"{WORK}/b", "--mode", "noise", "--layers", "0-3"))
B = load(f"{WORK}/b"); sig = torch.load(f"{WORK}/b/sigma.pt")
g0 = f"{L}.0.input"; Nm = B[g0][0]["N"] / ntok; s2 = (0.05 * sig[g0].double()) ** 2
live = s2 > 0; ratio = (Nm.diagonal()[live] / s2[live])
offd = float((Nm - torch.diag(Nm.diagonal())).norm() / Nm.diagonal().norm())
Cm = B[g0][0]["C"] / ntok; Hm = B[g0][0]["H"] / ntok
corr = float(Cm.norm() / (Hm.diagonal().sum() * Nm.diagonal().sum()).sqrt())
check("B. layer 0 input: diag N / n = sigma^2", abs(float(ratio.mean()) - 1) < 0.02, f"(mean {float(ratio.mean()):.4f}, channel sd {float(ratio.std()):.3f}, expected ~{(2 / ntok) ** 0.5:.3f})")
kk = Nm.shape[0]
check("B. layer 0 input: N's off-diagonal at its sampling level, C near 0", offd < 1.2 * (kk / ntok) ** 0.5 and corr < 0.1,
      f"(offdiag/diag {offd:.3f} vs sqrt(k/n) {(kk / ntok) ** 0.5:.3f}, |C|/sqrt(trH trN) {corr:.3f})")
prop = []
for g in B:
    if g == g0: continue
    s2g = (0.05 * sig[g].double()) ** 2; prop.append((g.split("layers.")[1], float((B[g][0]["N"].diagonal() / ntok).sum() / s2g.sum())))
check("B. later groups: propagated noise on top of the local one (trN / tr sigma^2 > 1)", all(r > 1 for _, r in prop),
      "; ".join(f"{g} {r:.2f}" for g, r in prop[:6]) + " ...")
print("   B. trace shares tr(N)/tr(H), tr(C+C^T)/tr(H): " + "; ".join(
    f"{g.split('layers.')[1]} {float(v['N'].trace() / v['H'].trace()):.3f} {float(2 * v['C'].trace() / v['H'].trace()):+.3f}" for g, (v, _) in B.items()))
# C
check("C. layer ranges 0-1 / 2-3 run", run(f"{WORK}/c", "--mode", "noise", "--layers", "0-1") and run(f"{WORK}/c", "--mode", "noise", "--layers", "2-3"))
Cc = load(f"{WORK}/c")
check("C. split runs = one run (bit-exact)", set(Cc) == set(B) and all(torch.equal(Cc[g][0][s], B[g][0][s]) for g in B for s in "HCN"))
# D
os.makedirs(f"{WORK}/selfwq", exist_ok=True); exact = True
for l in (0, 3):
    ly = dec.layers[l]
    for s in (["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] if hasattr(ly, "self_attn") else ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"]) + ["mlp.gate_proj", "mlp.up_proj"]:
        save_file({"weight": ly.get_submodule(s).weight.data.contiguous()}, f"{WORK}/selfwq/{L}.{l}.{s}.safetensors")
check("D. sib mode with own weights runs", run(f"{WORK}/d0", "--mode", "sib", "--wq", f"{WORK}/selfwq", "--layers", "0-0")
      and run(f"{WORK}/d3", "--mode", "sib", "--wq", f"{WORK}/selfwq", "--layers", "3-3"))
D = {**load(f"{WORK}/d0"), **load(f"{WORK}/d3")}
check("D. groups = o/out and down of layers 0, 3", sorted(D) == sorted([f"{L}.0.linear_attn.out_proj.input", f"{L}.0.mlp.down_proj.input",
                                                                   f"{L}.3.self_attn.o_proj.input", f"{L}.3.mlp.down_proj.input"]))
check("D. own weights: C = N = 0 exactly", all(float(v["C"].abs().max()) == 0 and float(v["N"].abs().max()) == 0 for v, _ in D.values()))
wd = max(rel(D[g][0]["H"], REF[g]) for g in D)
check("D. own weights: H = the clean collection", wd < 1e-6, f"(max rel {wd:.1e})")
# E
check("E. sib mode with the K3 base siblings runs", run(f"{WORK}/e0", "--mode", "sib", "--wq", WQ, "--layers", "0-0")
      and run(f"{WORK}/e3", "--mode", "sib", "--wq", WQ, "--layers", "3-3"))
E = {**load(f"{WORK}/e0"), **load(f"{WORK}/e3")}
print("   E. per group: tr(N)/tr(H), tr(C+C^T)/tr(H), ||H~ - H|| / ||H|| (H~ = H + C + C^T + N): " + "; ".join(
    f"{g.split('layers.')[1]} {float(v['N'].trace() / v['H'].trace()):.4f} {float(2 * v['C'].trace() / v['H'].trace()):+.4f} "
    f"{rel(v['H'] + v['C'] + v['C'].T + v['N'], v['H']):.4f}" for g, (v, _) in E.items()))
check("E. quantized siblings perturb the inputs", all(float(v["N"].trace()) > 0 for v, _ in E.values()))
def wqt(k):
    with safe_open(f"{WQ}/{k}.safetensors", "pt") as h: return h.get_tensor("weight").to(torch.bfloat16)
ly0 = dec.layers[0]; Gq, Uq = wqt(f"{L}.0.mlp.gate_proj"), wqt(f"{L}.0.mlp.up_proj")
Nref = 0; Cref = 0
for x in POST["x"]:
    xb = x.to(torch.bfloat16)
    xt = (ly0.mlp.act_fn(torch.nn.functional.linear(xb, Gq)) * torch.nn.functional.linear(xb, Uq)).double()
    xc = (ly0.mlp.act_fn(torch.nn.functional.linear(xb, ly0.mlp.gate_proj.weight)) * ly0.mlp.up_proj(xb)).double()
    Nref = Nref + (xt - xc).T @ (xt - xc); Cref = Cref + xc.T @ (xt - xc)
g = f"{L}.0.mlp.down_proj.input"
check("E. layer 0 down: N and C = an independent act(Gq x)(Uq x) recomputation", rel(E[g][0]["N"], Nref) < 1e-3 and rel(E[g][0]["C"], Cref) < 1e-3,
      f"(rel N {rel(E[g][0]['N'], Nref):.1e}, C {rel(E[g][0]['C'], Cref):.1e})")
# F
os.makedirs(f"{WORK}/allwq", exist_ok=True)
for l in range(4):
    for nm, mod in dec.layers[l].named_modules():
        if isinstance(mod, torch.nn.Linear) and not nm.endswith(("in_proj_a", "in_proj_b")):
            save_file({"weight": mod.weight.data.contiguous()}, f"{WORK}/allwq/{L}.{l}.{nm}.safetensors")
check("F. quant mode with own weights runs", run(f"{WORK}/f0", "--mode", "quant", "--wq", f"{WORK}/allwq", "--layers", "0-3"))
F0 = load(f"{WORK}/f0")
check("F. own weights: 16 groups, C = N = 0 exactly", len(F0) == 16 and all(float(v["C"].abs().max()) == 0 and float(v["N"].abs().max()) == 0 for v, _ in F0.values()))
wf = max(rel(F0[g][0]["H"], REF[g]) for g in F0)
check("F. own weights: H = the clean collection", wf < 1e-6, f"(max rel {wf:.1e})")
check("F. quant mode with the K3 base quant runs", run(f"{WORK}/f3", "--mode", "quant", "--wq", WQ, "--layers", "0-3"))
F3 = load(f"{WORK}/f3")
check("F. K3 base: layer 0 input eta = 0 exactly, later groups eta != 0",
      float(F3[f"{L}.0.input"][0]["N"].abs().max()) == 0 and all(float(v["N"].trace()) > 0 for g, (v, _) in F3.items() if g != f"{L}.0.input"))
print("   F. K3 base upstream error, tr(N)/tr(H), tr(C+C^T)/tr(H): " + "; ".join(
    f"{g.split('layers.')[1]} {float(v['N'].trace() / v['H'].trace()):.4f} {float(2 * v['C'].trace() / v['H'].trace()):+.4f}" for g, (v, _) in F3.items()))
print("ALL PASS" if ok else "SOME FAIL")
