#!/usr/bin/env python3
"""jfit.py MODEL META SRC TRUNK OUT [--variants qk,vg,at,gz,gd,mA] [--windows 600] [--held 64] [--win 2048] [--seed 11]
[--lr 1e-2] -- d1113 (README.md): swj-style joint output-scale (svh) refits at the attention and GDN junctions of the
0.8B, codes frozen, one streaming pass.

Per layer and variant, log-scales s on the chosen tensors' output channels (W' = diag(e^s) W in (out, in) layout, i.e.
svh *= e^s) are fitted by Adam to the block output before its writer (attention: attn(q, k, v) * sigmoid(gate);
GDN: RMSNormGated(core, z); MLP: silu(g) u), quantized weights from TRUNK/wq, against the same fp32 code with the
bf16 weights, both on the bf16 model's own block inputs (forward pre-hooks) of qad-pool windows (META = calib-meta
tokens, pools-v3; moments.py's cut, shuffled with --seed). One Adam step per window and layer, lr cosine to 5%; s
clamped to log [0.5, 2]. The last --held windows of the order are not trained on: the per-layer relative residual
sum (y - t)^2 / sum t^2 there, before and after, goes to OUT/jfit.jsonl.
Variants (README table): qk q_proj query half + k_proj | vg q_proj gate half + v_proj | at q / k / v | gz in_proj_z |
gd in_proj_qkv + in_proj_z | mA gate_proj + up_proj (solver check vs swiglufit; replaces SRC's swj gate / up) |
aw = at, dw = gd with the loss taken after the bf16 writer (o_proj / out_proj): ||W_o (y - t)||^2 (amendment 1).
Out: OUT/<variant>/wq (fitted tensors written, the rest symlinks to SRC/wq's real files), OUT/<variant>/scales
(fsutil format: r = 1, c = e^s), OUT/<variant>/.done; each variant built as <variant>.part and renamed.
Env: VRAM_CAP_GB, POOLS."""
import argparse, json, math, os, shutil, sys, time
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "/workspace/rad-exl3-mix/proxy2b"); sys.path.insert(0, "/workspace/rad-exl3-mix/proxy2b/exp/w02-fixedscale")
import fsutil as U
from common import load_model

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("meta"); ap.add_argument("src"); ap.add_argument("trunk"); ap.add_argument("out")
ap.add_argument("--variants", default="qk,vg,at,gz,gd,mA,aw,dw")
ap.add_argument("--windows", type=int, default=600); ap.add_argument("--held", type=int, default=64)
ap.add_argument("--win", type=int, default=2048); ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--lr", type=float, default=1e-2); ap.add_argument("--pools", default="qad")
ap.add_argument("--check", action="store_true", help="first window: compare the fp32 targets with the model's own writer inputs")
a = ap.parse_args()
VAR = a.variants.split(","); assert set(VAR) <= {"qk", "vg", "at", "gz", "gd", "mA", "aw", "dw"}, VAR
dev = torch.device("cuda"); U.cap_vram()
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
LO, HI = math.log(0.5), math.log(2.0)

# ---- windows (moments.py's cut)
m = json.load(open(f"{a.meta}/kld.json")); toks = np.memmap(f"{a.meta}/tokens.i32", np.int32, "r")
pools = json.load(open(os.environ.get("POOLS", "/workspace/rad-exl3-mix/eval/pools/pools-v3.json")))["pool"]
assert len(pools) == len(m["docs"])
off = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])]); want = set(a.pools.split(","))
wins = []
for i, d in enumerate(m["docs"]):
    if pools[i] not in want: continue
    n = d["tokens"]
    if n < a.win:
        if n >= 64: wins.append((int(off[i]), n))
    else: wins += [(int(off[i] + s), a.win) for s in range(0, n - a.win + 1, a.win)]
np.random.default_rng(a.seed).shuffle(wins)
wins = wins[:a.windows]; NT = len(wins) - a.held; assert NT > 0
print(f"{len(wins)} windows ({NT} train, {a.held} held), {sum(n for _, n in wins)} tokens, pools {sorted(want)}", file=sys.stderr, flush=True)

model, dec, _ = load_model(a.model, dev="cuda"); model.requires_grad_(False)
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
NL = len(dec.layers); P = "model.language_model.layers"
KIND = {l: ("attn" if hasattr(dec.layers[l], "self_attn") else "gdn") for l in range(NL)}

def loadq(key):   # (out, in) f16 on the card
    from safetensors import safe_open
    with safe_open(f"{a.trunk}/wq/{key}.safetensors", "pt", device="cpu") as f: return f.get_tensor("weight").to(dev)

# ---- per variant: which (tensor, part) it fits; parts: "all" or q_proj "q" (query half) / "g" (gate half)
SPEC = {"qk": {"attn": [("self_attn.q_proj", "q"), ("self_attn.k_proj", "all")]},
        "vg": {"attn": [("self_attn.q_proj", "g"), ("self_attn.v_proj", "all")]},
        "at": {"attn": [("self_attn.q_proj", "all"), ("self_attn.k_proj", "all"), ("self_attn.v_proj", "all")]},
        "gz": {"gdn": [("linear_attn.in_proj_z", "all")]},
        "gd": {"gdn": [("linear_attn.in_proj_qkv", "all"), ("linear_attn.in_proj_z", "all")]},
        "mA": {"mlp": [("mlp.gate_proj", "all"), ("mlp.up_proj", "all")]}}
SPEC["aw"], SPEC["dw"] = SPEC["at"], SPEC["gd"]
WGT = {"aw", "dw"}                                   # loss after the bf16 writer
WRITER = {"attn": "self_attn.o_proj", "gdn": "linear_attn.out_proj"}
TENS = {"attn": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        "gdn": ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"], "mlp": ["mlp.gate_proj", "mlp.up_proj"]}
need = {k for v in VAR for k in SPEC[v]}
WQ = {}
for l in range(NL):
    for kind in need:
        if kind != "mlp" and kind != KIND[l]: continue
        for t in TENS[kind]: WQ[(l, t)] = loadq(f"{P}.{l}.{t}")
mod = lambda l, t: dec.layers[l].get_submodule(t)
NH, HD = 8, 256   # q heads, head dim (q_proj rows: per head [query 256 | gate 256])

class Fit:
    """log-scale params of one (variant, layer); full(t) -> the (out,) scale vector of tensor t (ones where not fitted)"""
    def __init__(s, v, l, kind):
        s.p, s.parts = {}, {}
        for t, part in SPEC[v][kind]:
            n = WQ[(l, t)].shape[0]
            if part == "all": s.p[t] = torch.zeros(n, device=dev, requires_grad=True)
            else: s.p[t + ":" + part] = torch.zeros(NH, HD, device=dev, requires_grad=True)
        s.opt = torch.optim.Adam(list(s.p.values()), lr=a.lr)
    def full(s, t, n):
        if t in s.p: return s.p[t].exp()
        if t == "self_attn.q_proj" and (t + ":q" in s.p or t + ":g" in s.p):
            z = torch.zeros(NH, HD, device=dev)
            return torch.cat([s.p.get(t + ":q", z), s.p.get(t + ":g", z)], 1).reshape(-1).exp()
        return None
    def clamp(s):
        with torch.no_grad():
            for x in s.p.values(): x.clamp_(LO, HI)

FITS = {(v, l): Fit(v, l, k) for v in VAR for l in range(NL) for k in SPEC[v] if k == "mlp" or k == KIND[l]}

def lin(x, W, c):   # x [T, in] fp32, W (out, in) any dtype, c (out,) or None
    y = x @ W.float().T
    return y if c is None else y * c

def attn_out(l, x, pe, W, C):
    am = mod(l, "self_attn"); T = x.shape[0]
    qg = lin(x, W["self_attn.q_proj"], C("self_attn.q_proj")).view(T, NH, 2 * HD)
    q, g = qg[..., :HD], qg[..., HD:].reshape(T, -1)
    qn = am.q_norm; kn = am.k_norm
    nrm = lambda z, w, eps: z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())
    q = nrm(q, qn.weight, qn.eps)
    k = nrm(lin(x, W["self_attn.k_proj"], C("self_attn.k_proj")).view(T, -1, HD), kn.weight, kn.eps)
    v = lin(x, W["self_attn.v_proj"], C("self_attn.v_proj")).view(T, -1, HD)
    q, k, v = q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None]
    cos, sin = pe; q, k = apply_rotary_pos_emb(q, k, cos.float(), sin.float())
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=am.scaling, enable_gqa=True)
    return o[0].transpose(0, 1).reshape(T, -1) * torch.sigmoid(g)

def gdn_out(l, x, W, C):
    gm = mod(l, "linear_attn"); T = x.shape[0]
    mixed = lin(x, W["linear_attn.in_proj_qkv"], C("linear_attn.in_proj_qkv"))          # [T, conv_dim]
    cw = gm.conv1d.weight.float()                                                       # (C, 1, K)
    y = F.conv1d(mixed.T[None], cw, padding=gm.conv_kernel_size - 1, groups=cw.shape[0])[..., :T]
    y = F.silu(y)[0].T                                                                  # [T, C]
    q, k, v = torch.split(y, [gm.key_dim, gm.key_dim, gm.value_dim], dim=-1)
    q = q.reshape(1, T, -1, gm.head_k_dim); k = k.reshape(1, T, -1, gm.head_k_dim); v = v.reshape(1, T, -1, gm.head_v_dim)
    b = x @ gm.in_proj_b.weight.float().T; aa = x @ gm.in_proj_a.weight.float().T
    beta = b.sigmoid()[None]; g = (-gm.A_log.float().exp() * F.softplus(aa + gm.dt_bias.float()))[None]
    rep = gm.num_v_heads // gm.num_k_heads
    if rep > 1: q = q.repeat_interleave(rep, dim=2); k = k.repeat_interleave(rep, dim=2)
    core, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    core = core.reshape(-1, gm.head_v_dim)
    z = lin(x, W["linear_attn.in_proj_z"], C("linear_attn.in_proj_z")).reshape(-1, gm.head_v_dim)
    nw = gm.norm
    h = core * torch.rsqrt(core.pow(2).mean(-1, keepdim=True) + nw.variance_epsilon) * nw.weight.float()
    return (h * F.silu(z)).reshape(T, -1)

def mlp_out(l, x, W, C):
    return F.silu(lin(x, W["mlp.gate_proj"], C("mlp.gate_proj"))) * lin(x, W["mlp.up_proj"], C("mlp.up_proj"))

def run(kind, l, x, pe, W, C):
    return attn_out(l, x, pe, W, C) if kind == "attn" else gdn_out(l, x, W, C) if kind == "gdn" else mlp_out(l, x, W, C)

ORIG = lambda l, kind: {t: mod(l, t).weight for t in TENS[kind]}
QW = lambda l, kind: {t: WQ[(l, t)] for t in TENS[kind]}
none = lambda t: None
st = {"phase": "train", "i": 0}
HOLD = {}            # (variant|"base:"kind, layer) -> [sum resid², sum t²] on held windows (float64)
TRL = {}             # (variant, layer) -> running train loss (last 10% of steps)
CHK = {}

def step(kind, l, x, pe):
    with torch.no_grad(): t = run(kind, l, x, pe, ORIG(l, kind), none)
    tt = t.pow(2).sum().double()
    Wo = mod(l, WRITER[kind]).weight if kind in WRITER else None
    wr = lambda y: y @ Wo.float().T                  # through the bf16 writer
    tw = ttw = None
    if Wo is not None and any(v in WGT and kind in SPEC[v] for v in VAR):
        with torch.no_grad(): tw = wr(t); ttw = tw.pow(2).sum().double()
    if st["phase"] == "held":
        with torch.no_grad():
            yb = run(kind, l, x, pe, QW(l, kind), none)
            r = (yb - t).pow(2).sum().double()
            h = HOLD.setdefault((f"base:{kind}", l), [0.0, 0.0]); h[0] += float(r); h[1] += float(tt)
            if tw is not None:
                r = (wr(yb) - tw).pow(2).sum().double()
                h = HOLD.setdefault((f"basew:{kind}", l), [0.0, 0.0]); h[0] += float(r); h[1] += float(ttw)
            for v in VAR:
                f = FITS.get((v, l)) if kind in SPEC[v] else None
                if f is None: continue
                n = lambda tn: f.full(tn, 0)
                y = run(kind, l, x, pe, QW(l, kind), n)
                r = ((wr(y) - tw) if v in WGT else (y - t)).pow(2).sum().double()
                h = HOLD.setdefault((v, l), [0.0, 0.0]); h[0] += float(r); h[1] += float(ttw if v in WGT else tt)
        return
    if st["phase"] == "check": CHK[(kind, l)] = t; return
    for v in VAR:
        f = FITS.get((v, l)) if kind in SPEC[v] else None
        if f is None: continue
        with torch.enable_grad():
            y = run(kind, l, x, pe, QW(l, kind), lambda tn: f.full(tn, 0))
            loss = ((wr(y) - tw).pow(2).sum() / ttw.float()) if v in WGT else ((y - t).pow(2).sum() / tt.float())
            f.opt.zero_grad(set_to_none=True); loss.backward()
        for g_ in f.opt.param_groups: g_["lr"] = st["lr"]
        f.opt.step(); f.clamp()
        if st["i"] >= 0.9 * NT: TRL.setdefault((v, l), []).append(float(loss))

def hook(kind, l):
    def h(module, args, kwargs):
        if kind == "attn": x, pe = kwargs["hidden_states"], kwargs["position_embeddings"]
        elif kind == "gdn": x, pe = kwargs["hidden_states"], None
        else: x, pe = args[0], None
        step(kind, l, x[0].float(), pe)
    return h
for l in range(NL):
    dec.layers[l].get_submodule("self_attn" if KIND[l] == "attn" else "linear_attn").register_forward_pre_hook(hook(KIND[l], l), with_kwargs=True)
    if "mlp" in need: dec.layers[l].mlp.register_forward_pre_hook(hook("mlp", l), with_kwargs=True)

def fwd(i):
    s, n = wins[i]
    with torch.no_grad(): dec(input_ids=torch.from_numpy(np.asarray(toks[s:s + n]).astype(np.int64))[None].to(dev))

if a.check:   # the fp32 targets vs the bf16 model's own writer inputs on window 0
    cap = {}
    def ch(key):
        def h(mo, args): cap[key] = args[0][0].float()
        return h
    hs = []
    for l in range(NL):
        w = "self_attn.o_proj" if KIND[l] == "attn" else "linear_attn.out_proj"
        hs.append(dec.layers[l].get_submodule(w).register_forward_pre_hook(ch((KIND[l], l))))
        if "mlp" in need: hs.append(dec.layers[l].mlp.down_proj.register_forward_pre_hook(ch(("mlp", l))))
    st["phase"] = "check"; fwd(0)
    for k_ in sorted(CHK, key=lambda z: (z[1], z[0])):
        e = float((CHK[k_] - cap[k_]).pow(2).sum() / cap[k_].pow(2).sum())
        print(f"CHECK {k_[0]} L{k_[1]:02d} rel err fp32 target vs model {e:.2e}", flush=True)
    for h_ in hs: h_.remove()
    CHK.clear()

t0 = time.time(); st["phase"] = "train"
for i in range(NT):
    st["i"] = i; st["lr"] = a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * i / NT)))
    fwd(i)
    if i % 50 == 0: print(f"train {i + 1}/{NT} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
st["phase"] = "held"
for i in range(NT, len(wins)): fwd(i)
print(f"fit + held-out pass {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

# ---- report + write
os.makedirs(a.out, exist_ok=True)
rows = []
for (v, l), f in sorted(FITS.items(), key=lambda z: (z[0][1], z[0][0])):
    kind = next(k for k in SPEC[v] if k == "mlp" or k == KIND[l])
    hb, hv = HOLD[(f"{'basew' if v in WGT else 'base'}:{kind}", l)], HOLD[(v, l)]
    sc = {t: f.full(t, 0).detach() for t in TENS[kind] if f.full(t, 0) is not None}
    allc = torch.cat([c for c in sc.values()])
    rec = {"variant": v, "layer": l, "kind": kind, "held_base": hb[0] / hb[1], "held_fit": hv[0] / hv[1],
           "held_ratio": hv[0] / hb[0], "train_end": float(np.mean(TRL.get((v, l), [float("nan")]))),
           "c_mean": float(allc.mean()), "c_sd": float(allc.std()), "c_min": float(allc.min()), "c_max": float(allc.max()),
           "c_edge": int(((allc <= 0.5001) | (allc >= 1.9999)).sum())}
    rows.append(rec)
    print(f"{v:3s} L{l:02d} {kind:4s} held rel resid base {rec['held_base']:.4e} fit {rec['held_fit']:.4e} "
          f"ratio {rec['held_ratio']:.4f} | c {rec['c_mean']:.3f}±{rec['c_sd']:.3f} [{rec['c_min']:.3f}, {rec['c_max']:.3f}] edge {rec['c_edge']}", flush=True)
with open(f"{a.out}/jfit.jsonl.part", "w") as fo:
    for r in rows: fo.write(json.dumps(r) + "\n")
os.replace(f"{a.out}/jfit.jsonl.part", f"{a.out}/jfit.jsonl")
for v in VAR:
    tot = {}
    for r in rows:
        if r["variant"] == v:
            hb = HOLD[(f"{'basew' if v in WGT else 'base'}:{r['kind']}", r["layer"])]; hv = HOLD[(v, r["layer"])]
            tot.setdefault(r["kind"], [0, 0]); tot[r["kind"]][0] += hb[0] / hb[1]; tot[r["kind"]][1] += hv[0] / hv[1]
    print(f"SUMMARY {v}: " + " ".join(f"{k} mean held rel resid {b / sum(1 for r in rows if r['variant'] == v and r['kind'] == k):.4e} -> "
                                      f"{f_ / sum(1 for r in rows if r['variant'] == v and r['kind'] == k):.4e} ({f_ / b:.4f})" for k, (b, f_) in tot.items()), flush=True)

keys = U.wq_keys(a.src)
for v in VAR:
    d = f"{a.out}/{v}"
    if os.path.exists(f"{d}/.done"): continue
    shutil.rmtree(f"{d}.part", ignore_errors=True); os.makedirs(f"{d}.part/wq")
    done = set()
    for (vv, l), f in FITS.items():
        if vv != v: continue
        for t in TENS["attn"] + TENS["gdn"] + TENS["mlp"]:
            if (l, t) not in WQ: continue
            c = f.full(t, 0)
            if c is None: continue
            key = f"{P}.{l}.{t}"; c = c.detach().float()
            U.write_wq(f"{d}.part", key, (WQ[(l, t)].float() * c[:, None]).T)    # write_wq takes (in, out)
            U.write_scales(f"{d}.part", key, torch.ones(WQ[(l, t)].shape[1]), c)
            done.add(key)
    for key in keys:
        if key not in done: os.symlink(os.path.realpath(f"{a.src}/wq/{key}.safetensors"), f"{d}.part/wq/{key}.safetensors")
    json.dump({"src": a.src, "trunk": a.trunk, "variant": v, "windows": len(wins), "held": a.held, "win": a.win,
               "seed": a.seed, "lr": a.lr, "fitted": len(done)}, open(f"{d}.part/jfit.json", "w"), indent=1)
    open(f"{d}.part/.done", "w").close(); shutil.rmtree(d, ignore_errors=True); os.replace(f"{d}.part", d)
    print(f"wrote {d}: {len(done)} fitted tensors, {len(keys) - len(done)} links", flush=True)
print(f"jfit done {time.time() - t0:.0f}s", file=sys.stderr)
