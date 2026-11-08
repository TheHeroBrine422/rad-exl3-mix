#!/usr/bin/env python3
"""signchk84.py ORDER CKPT HESS OUT [SEED_OFFSET] -- gate G2 of ../w84.md (card 0): route A (the converted tensor,
perm117.py's own functions on W and the packed calib H, quantized as exllamav3 would quantize the converted model, with
the natural run's random signs permuted along and its g_scale tile sample mapped through P) against route B (the
natural tensor, LDLQ block order patched: blkorder84.py), and route A without the sign permutation (a plain converted
run: the sign trap). Tensors: layer 0 in_proj_qkv, in_proj_z, out_proj; layer 3 q / k / v_proj, o_proj; K2.5, rxq with
the plain ladder settings (tile CD off, H_out off), seeds as the base level (+ SEED_OFFSET x 100000).

What is compared (per tensor):
  su        the sign draw of the group (H_data["su"]): A == P(B)  (readers; writers: equal)
  ldlq_in   the rotated, regularized weight and L the LDLQ loop receives: A == B's (B permutes inside its wrapper)
  trellis   the stored codes: A == P(B) moved by whole tile rows (readers) / tile columns (writers)
  suh / svh A == P(B) up to the scale refit's float reordering (max |diff| in f16 ulps, count)
Writers in route B are the natural run (no reorder). Route A for writers permutes output rows / sv.
Last line: "SIGNCHK PASS|FAIL {json}" (PASS: trellis bit-identical for every tensor with signs handled)."""
import json, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import perm117, blkorder84
from rxq import config as C, engine, quant, qwen35

ORDER, CKPT, HESS, OUT = sys.argv[1:5]
SO = int(sys.argv[5]) if len(sys.argv) > 5 else 0
K, CB, D = 2.5, "i8", 5120
L = "model.language_model.layers"
TENS = [f"{L}.0.linear_attn.in_proj_qkv", f"{L}.0.linear_attn.in_proj_z", f"{L}.0.linear_attn.out_proj",
        f"{L}.3.self_attn.q_proj", f"{L}.3.self_attn.k_proj", f"{L}.3.self_attn.v_proj", f"{L}.3.self_attn.o_proj"]
os.makedirs(OUT, exist_ok=True)
so, idx = perm117.load_order(ORDER)
inv = torch.argsort(idx)
tidx = torch.cat([torch.arange(b * 8, (b + 1) * 8) for b in so])

cfg = C.with_overrides(C.load(), **{"quant.tile_cd.enable": False, "inputs.hout.enable": False})
dev = torch.device("cuda", 0)
s0 = C.quant_settings(cfg, K, CB, TENS[0], SO)
Q = engine.setup(dev, codebook=CB, qtf2=s0["qtf2"], drift=s0["drift"], block_trace=s0["block_trace"], tile_cd=None)

# ---- capture: the LDLQ loop's inputs (innermost wrapper), the group's sign vector (outermost)
CAP = {}
_ldlq = Q.ldlq
def ldlq_cap(weight, Lm, qa, pb=None):
    CAP["w"] = weight.detach().float().cpu().clone(); CAP["L"] = Lm.detach().float().cpu().clone()
    return _ldlq(weight, Lm, qa, pb)
Q.ldlq = ldlq_cap
blkorder84.install(Q, ORDER)                    # route B (outer wrapper over ldlq_cap)

# ---- route A's patches, active only while A["on"] (kind: reader = input axis permuted, writer = output axis)
#  signs: the natural run's sign draws (any 5120-long randn: su of a reader group, sv of a writer) permuted along
#  reg:   exllamav3's regularize (output / input channel RMS, the g_scale tile sample along a wrapped tile diagonal, which
#         is NOT permutation-equivariant) computed on the natural channel order and its outputs moved by P
#  finalize's mean diagonal (the damping) summed in the natural order (float reassociation only)
A = {"on": False, "kind": None, "signs": True, "reg": True}
_torch = torch
class _TorchShim:
    def __getattr__(self, n): return getattr(_torch, n)
    def randn(self, *a, **kw):
        r = _torch.randn(*a, **kw)
        if A["on"] and A["signs"] and r.dim() == 1 and r.numel() == D: r = r.index_select(0, idx.to(r.device))
        return r
    def diag(self, x, *a, **kw):
        r = _torch.diag(x, *a, **kw)
        if A["on"] and A["kind"] == "reader" and x.dim() == 2 and x.shape == (D, D): r = r.index_select(0, inv.to(r.device))
        return r
Q.torch = _TorchShim()
_reg = Q.regularize
def regularize(weight, su, sv, *a, **kw):
    if not (A["on"] and A["reg"]): return _reg(weight, su, sv, *a, **kw)
    ax = 0 if A["kind"] == "reader" else 1
    i, v = inv.to(weight.device), idx.to(weight.device)
    weight = weight.index_select(ax, i)
    if ax == 0: su = su.index_select(0, inv.to(su.device))
    else: sv = sv.index_select(1, inv.to(sv.device))
    aos, w, g, su, sv = _reg(weight, su, sv, *a, **kw)
    w = w.index_select(ax, v)
    if ax == 0: su = su.index_select(0, idx.to(su.device))
    else: sv = sv.index_select(1, idx.to(sv.device))
    return aos, w, g, su, sv
Q.regularize = regularize
_qx = Q.quantize_exl3
def qx_cap(weight, H_data, qa, *a, **kw):
    r = _qx(weight, H_data, qa, *a, **kw); CAP["su"] = H_data["su"].detach().flatten().cpu().clone(); return r
Q.quantize_exl3 = qx_cap


class PermCkpt(quant.Ckpt):
    def get(self, name):
        t = super().get(name); p = perm117.permute_tensor(name, t, idx); return t if p is None else p


# route A's Hessians: perm117's packed-upper re-indexing for the residual-input groups used here, links for the rest
HP = f"{OUT}/hess-perm"
os.makedirs(HP, exist_ok=True)
from safetensors import safe_open
from safetensors.torch import save_file
for f in os.listdir(HESS):
    dst = f"{HP}/{f}"
    if os.path.lexists(dst): continue
    name = f[:-len(".safetensors")]
    if perm117.hess_permuted(name) and name in (f"{L}.0.input", f"{L}.3.input"):
        meta = perm117._hdr(f"{HESS}/{f}")["__metadata__"]
        with safe_open(f"{HESS}/{f}", "pt", device="cpu") as fh: up = fh.get_tensor("hessian_upper")
        save_file({"hessian_upper": perm117.upper_permute(up, int(meta["k"]), idx)}, dst + ".part", metadata=meta)
        os.replace(dst + ".part", dst)
    else: os.symlink(os.path.abspath(f"{HESS}/{f}"), dst)


def run(key, route, signs=True, reg=True):
    fam = qwen35.family(key); writer = fam in qwen35.WRITERS
    s = C.quant_settings(cfg, K, CB, key, SO)
    if route == "B":
        blkorder84._ST["enabled"] = True; A["on"] = False
        ck, hd = quant.Ckpt(CKPT), HESS
    else:
        blkorder84._ST["enabled"] = False; A.update(on=True, kind="writer" if writer else "reader", signs=signs, reg=reg)
        ck, hd = PermCkpt(CKPT), HP
    CAP.clear(); t0 = time.time()
    try:
        tensors, rec, _, _ = quant.quantize_tensor(Q, ck, key, s, hd, dev, OUT)
    finally:
        A["on"] = False; blkorder84._ST["enabled"] = True
    out = {k.rsplit(".", 1)[1]: v for k, v in tensors.items()}
    return {"t": out, "w": CAP["w"], "L": CAP["L"], "su": CAP["su"], "proxy": rec["proxy_err"], "s": time.time() - t0}


def f16_ulps(a, b):
    ai = a.view(torch.int16).int(); bi = b.view(torch.int16).int()
    return int((ai - bi).abs().max()), int((ai != bi).sum())


def cmp(key, a, b):
    """a = route A (stored = permuted order), b = route B (natural order): moves b into A's order and compares"""
    writer = qwen35.family(key) in qwen35.WRITERS
    r = {"key": key, "writer": writer}
    if writer:                                   # output axis permuted: weight columns, sv / svh, trellis tile columns
        r["su"] = bool(torch.equal(a["su"], b["su"]))
        bw = b["w"].index_select(1, idx); bL = b["L"]
        bt = b["t"]["trellis"].index_select(1, tidx); bsuh = b["t"]["suh"]; bsvh = b["t"]["svh"].index_select(0, idx)
    else:                                        # input axis: B's ldlq input is already in the stored order
        r["su"] = bool(torch.equal(a["su"], b["su"].index_select(0, idx)))
        bw, bL = b["w"], b["L"]
        bt = b["t"]["trellis"].index_select(0, tidx); bsuh = b["t"]["suh"].index_select(0, idx); bsvh = b["t"]["svh"]
    r["ldlq_w_equal"] = bool(torch.equal(a["w"], bw)); r["ldlq_w_maxdiff"] = float((a["w"] - bw).abs().max())
    r["ldlq_L_equal"] = bool(torch.equal(a["L"], bL)); r["ldlq_L_maxdiff"] = float((a["L"] - bL).abs().max())
    at = a["t"]["trellis"]
    r["trellis_equal"] = bool(torch.equal(at, bt))
    tile_diff = (at != bt).flatten(2).any(-1)
    r["tiles_differ"] = f"{int(tile_diff.sum())}/{tile_diff.numel()}"
    r["suh_ulps_max"], r["suh_n_diff"] = f16_ulps(a["t"]["suh"], bsuh)
    r["svh_ulps_max"], r["svh_n_diff"] = f16_ulps(a["t"]["svh"], bsvh)
    r["proxy_A"], r["proxy_B"] = a["proxy"], b["proxy"]
    other = sorted(set(a["t"]) - {"trellis", "suh", "svh"})
    r["other_planes_equal"] = all(torch.equal(a["t"][p], b["t"][p]) for p in other)
    return r


res, ok = [], True
for key in TENS:
    b = run(key, "B"); a = run(key, "A")
    r = cmp(key, a, b); r["s"] = round(a["s"] + b["s"], 1)
    ok &= r["trellis_equal"] and r["su"] and r["other_planes_equal"]
    print(json.dumps(r), flush=True); res.append(r)
# the traps, on q / k / v + o_proj: route A without permuting the natural signs (a plain converted run), and route A
# with the signs permuted but exllamav3's regularize run as-is on the converted tensor (g_scale tile sample not mapped)
trap = []
for key in TENS[3:]:
    b = run(key, "B")
    for lab, kw in (("signs NOT permuted", {"signs": False}), ("signs permuted, regularize as-is", {"reg": False})):
        a = run(key, "A", **kw)
        r = cmp(key, a, b); r["route"] = "A, " + lab
        print(json.dumps(r), flush=True); trap.append(r)
json.dump({"order": ORDER, "seed_offset": SO, "checks": res, "sign_trap": trap}, open(f"{OUT}/signchk84.json", "w"), indent=1)
print(f"SIGNCHK {'PASS' if ok else 'FAIL'} " + json.dumps({"tensors": len(res), "trellis_equal": sum(r['trellis_equal'] for r in res),
      "ldlq_inputs_equal": sum(r['ldlq_w_equal'] and r['ldlq_L_equal'] for r in res),
      "trap_trellis_equal": sum(r['trellis_equal'] for r in trap), "trap_runs": len(trap)}))
