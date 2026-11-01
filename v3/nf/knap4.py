#!/usr/bin/env python3
"""knap4.py MODEL.json AQ.jsonl AQFACTORS.json MAP OUT.jsonl TARGETS LAMBDAS -- knap3 with the V3 menu (2026-11-01).
Switches (env, all default off = knap3's V2-like menu):
  KN_FRAC=1   ours at quarter steps K1.25..4.75 and half steps K5.5..7.5: bytes linear in K, error log-linear between
              the integer neighbours (CPU sweeps: fractional K on that line, 0.98-1.04); family of the lower neighbour.
              KN_HEAD_STEP=0.5 (default) restricts the head to half steps, 0.25 allows quarters.
  KN_EMBT=1   trellis embedding, one scale per row, quarter steps K1.5..6: its error per K (embmenu.jsonl) mapped onto
              the fitted Lloyd embedding costs through the Lloyd error curve (embtrellis.jsonl), log-log interpolation.
  KN_DAMP=1   attention inputs (q/k/v) requantized at sigma_reg 0.005: error x the CPU damping-sweep factor
              (K2 0.994, K4 0.981, K6 0.991, K8 0.955; log-linear in between).
  KN_MODES=f16  quality-only (no int compute modes).
  KN_EMBT=2   trellis embedding priced from REAL KLD (chain1, 2026-11-01: swaps into V2 6.00 on golden-ml, t2 / t2.5 / t3
              / t3.5 / t4 vs Lloyd 2-5): the ml increments are scaled to the pooled corpus by the Lloyd points
              (model e3-e5, e4-e5 vs the same on ml: 0.333, both agree to 1%); other K from the nearest measured
              point along the CPU error curve (embmenu.jsonl).
  KN_HEADSLAB=1  slab heads (3 row slabs, K3 floor; headsim on 32,768 calib-A head inputs): offered at K5 / K5.5 / K6
              at the conservative end of the 95% interval (head KL x0.69 / x0.70 / x0.71); NOT at K4 (gain n.s.,
              rare-token KL x2.5). Bytes + 1 MB (row permutation). Containers hdslab-K<k>.rad (to build).
  KN_LEVELS=DIR  measured fractional levels: DIR/K<k>/proxy.jsonl (ourq.py, the V3 ladder, int8 codebook) for k and
              both neighbouring integers -> a tensor's fractional error sits where the measurement puts it between
              the integers (log scale), instead of the midpoint interpolation. Tensors without all three: interpolated.
  F8 now uses the measured K7 x1.25 / K8 x4.5 (E3b full set), fractional K interpolated.
  KN_V3=1     the V3 option table (critique 2026-11-02 C3.2): only options with a container (built or queued):
              ours fractional / integer K = the ladder's levels (KN_LADDER, default 2..6 in quarter steps to 4.75, then
              5, 5.5, 6), i8 codebook, the only ones offered int8 modes; V2's mul1 ours-K1 / K7 / K8 stay f16-only;
              head = integer hd-K3..8 plus slab K5 / K6 (KN_HEADSLAB=1; no half-step or fractional heads); KN_M
              default 2048 (speedcal's M rows, other shapes scaled by the kernel's measured 2048 / 4096 ratio);
              containers: ladder levels RXM_V3/ours-K<k>.rad, trellis embeddings RXM_V3/emb-trellis-K<k>.rad; the
              writer refuses to write a set that names a missing file (KN_WRITE_CHECK=0 to override). KN_GROUPS=0,128
              restricts the int8 activation groups (0 = per token: what the real kernel runs today).
              Pricing caveat: ladder integers still carry V2's hproxy error x DAMPF (V3 ladder hproxy needs the ladder's
              .rad import); fractional levels sit at their measured ladder positions (KN_LEVELS).
The rest is knap3 (below).

Per tensor, the options are (compute mode, activation group, weight code):
  f16   any EXL3 option the model knows (ours-K1..8, turboderp sc/plain), or bf16      -- today's path
  a8    our trellis code K1..8 decoded to int8 (weight error x F8[K], E3b)  x  group {pt,1024,512,256,128,64}
  a4    our trellis code K1..3 on the int4 grid (x F4[K])                    x  same groups
Linears sharing an input (q/k/v; GDN qkv/z) take one compute mode + group (one quantized input); their K stay free.

KLD model:
  - weights: kfit2's fitted model (MODEL.json from nf/kfit2dump.py) -- per-tensor sensitivity s_w, base error eb_w,
    exponent q, family factors phi, depth compounding P_l = P_{l-1}(1 + kappa_l X_l) + X_l, embedding level costs.
  - activations: additive, k_family x rel_out(tensor, bits, group) (E1c proxy, AQFACTORS calibrated on E1b KLDs).
Time: GEMM time per prefill step of M = 4096 tokens (E2 table) + a weight-decode pre-pass; nonlinear ops are a fixed
  8% of today's all-f16 step (eval profile: 92% of the time in the EXL3 GEMMs). The LM head runs on the last token
  only: no prefill time.
  KN_SPEED=speedcal.jsonl: measured matmul cores instead of the E2 TOPS (v3/bench/speedcal.sh, TP1 shapes, M 4096,
    per-token; the group penalty stays E2's ratio; attn_k / attn_v (1024 rows, not measured) take gdn_z's rate).
  KN_A4=a4time.jsonl: measured whole paths (decode included) per shape and K (v3/bench/a4time.sh, M 4096): f16 = T2,
    a8 per-token = the NFQ int8 path; a8 groups add the matmul penalty; a4 = the int8 path with its matmul swapped
    for speedcal's iu4 core (no int4 kernel yet). K between measured points: linear; outside 2..6: clamped.
Objective per (target bytes, lambda): min KLD + lambda * time_ms (lambda in KLD per ms), bytes strictly under the target.
  The per-layer stages minimise g_l X_l + A_l + lambda T_l, with g_l = dKLD/dX_l at the lambda = 0 solution of the same
  target (two passes), then the layered DP over 1 MiB buckets carries (P, A, T) exactly.
KN_WRITE=DIR: also write each solution's validation set (DIR/NAME.mix for radmix: tensor -> EXL3 container with the
picked K, which matches the new format's weights to <= 0.4% for a8 and 1-7% for a4 K2-3; DIR/NAME.aq: the E1b hook's
EXL3_AQ_MASK and EXL3_AQ_GROUPS strings; NAME = nf-<target>-l<lambda>).
Output: one JSON line per (target, lambda): bpw, predicted KLD, linear time, estimated prefill speedup vs all-f16, the
mode mix.
"""
import collections, json, math, os, re, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e1b"))
import mkmask
# paths (README: Paths)
RXM_WORK = os.environ.get("RXM_WORK", "/root")
RXM_MODELS = os.environ.get("RXM_MODELS", f"{RXM_WORK}/models/final")
RXM_OURS = os.environ.get("RXM_OURS", "/data/ours")

MJ, AQJ, AQF, MAP, OUT = sys.argv[1:6]
TARGETS = [float(x) for x in sys.argv[6].split(",")]
LAMBDAS = [float(x) for x in sys.argv[7].split(",")]
V3 = os.environ.get("KN_V3") == "1"
M = int(os.environ.get("KN_M", 2048 if V3 else 4096))
RXM_V3 = os.environ.get("RXM_V3", "/data/v3levels/rad")
LADDER = [float(x) for x in os.environ.get("KN_LADDER", "2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5,4.75,5,5.5,6").split(",")]
BK = 1 << 20
G = [int(g) for g in os.environ.get("KN_GROUPS", "0,1024,512,256,128,64").split(",")]   # activation groups (0 = per token)
F8 = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.001, 5: 1.002, 6: 1.004, 7: 1.25, 8: 4.5}    # E3b full set (K7, K8 measured)
F4 = {1: 1.0, 2: 1.01, 3: 1.065, 4: 2.72}               # E3b int4grid; K >= 5 not offered (x10 and up)
# E2 (gfx1201, M = 4096, first version): TOPS by mode and group; per-token from the 128x256 tile, groups from 128x128.
# g128 / g64 are scaled from the microbench ratios (not yet measured in the real kernel).
TOPS = {"a8": {0: 254, 1024: 225, 512: 220, 256: 210, 128: 185, 64: 153},
        "a4": {0: 488, 1024: 390, 512: 375, 256: 346, 128: 274, 64: 205}}
TF16 = 135.0                                             # T2's f16 GEMM core, TFLOPS
DEC_NS = float(os.environ.get("KN_DEC_NS", 0.006))       # weight-decode pre-pass, ns per weight per step (placeholder)
FAMSHAPE = {"attn_qg": "attn_qg", "ssm_qkv": "gdn_qkv", "ssm_z": "gdn_z", "ffn_gate_up": "gate_up", "ffn_down": "down",
            "attn_output": "attn_o", "ssm_out": "ssm_out", "attn_k": "gdn_z", "attn_v": "gdn_z"}
CORE = {}                                                 # (kern f16/iu8/iu4, speedcal shape) -> TOPS
if os.environ.get("KN_SPEED"):
    _cm = collections.defaultdict(dict)
    for l in open(os.environ["KN_SPEED"]):
        d = json.loads(l)
        if "ms" in d and d["split"] == "tp1" and d["M"] in (M, 4096): _cm[(d["kern"], d["shape"])][d["M"]] = 2.0 * d["M"] * d["N"] * d["K"] / d["ms"] / 1e9
    _rat = collections.defaultdict(list)    # per kernel: TOPS at M / TOPS at 4096 where both are measured
    for (kn, sh), v in _cm.items():
        if M in v and 4096 in v: _rat[kn].append(v[M] / v[4096])
    for (kn, sh), v in _cm.items():
        if M in v: CORE[(kn, sh)] = v[M]
        elif 4096 in v: CORE[(kn, sh)] = v[4096] * (sorted(_rat[kn])[len(_rat[kn]) // 2] if _rat[kn] else 1.0)
    if M != 4096: print(f"speed at M {M}: measured/4096 ratios {({k: [round(x, 3) for x in v] for k, v in _rat.items()})}", file=sys.stderr)
PATH = {}                                                 # (path f16/nfq8, speedcal shape, bits) -> ms at M 4096, per weight
if os.environ.get("KN_A4"):
    for l in open(os.environ["KN_A4"]):
        d = json.loads(l)
        if "ms" in d and d["M"] == 4096: PATH[(d["path"], d["shape"], d["bits"])] = d["ms"] / (d["N"] * d["K"])
def path_ms(path, sh, bits, NK):   # measured whole-path time for N*K weights at bits (linear in K, clamped)
    ks = sorted(b for (p, s_, b) in PATH if p == path and s_ == sh)
    if not ks: return None
    b = min(max(bits, ks[0]), ks[-1]); lo = max(k for k in ks if k <= b); hi = min(k for k in ks if k >= b)
    v = PATH[(path, sh, lo)] if lo == hi else PATH[(path, sh, lo)] + (b - lo) / (hi - lo) * (PATH[(path, sh, hi)] - PATH[(path, sh, lo)])
    return v * NK

md = json.load(open(MJ))
HERE = os.path.dirname(os.path.abspath(__file__))
def lerp_log(tab, K):   # log-linear interpolation of a {int K: factor} table at a fractional K (None outside it)
    if K in tab: return tab[K]
    a, b = math.floor(K), math.ceil(K)
    if a not in tab or b not in tab: return None
    return math.exp(math.log(tab[a]) + (K - a) * (math.log(tab[b]) - math.log(tab[a])))
FRACS = [1.25, 1.5, 1.75, 2.25, 2.5, 2.75, 3.25, 3.5, 3.75, 4.25, 4.5, 4.75, 5.5, 6.5, 7.5]
HEAD_STEP = float(os.environ.get("KN_HEAD_STEP", 0.5))
if V3: FRACS = [k for k in LADDER if not float(k).is_integer()]; HEAD_STEP = 1.0
DAMPF = {2: 0.994, 4: 0.981, 6: 0.991, 8: 0.955}
def dampf(K):
    ks = sorted(DAMPF); K = min(max(K, ks[0]), ks[-1])
    lo = max(k for k in ks if k <= K); hi = min(k for k in ks if k >= K)
    return DAMPF[lo] if lo == hi else math.exp(math.log(DAMPF[lo]) + (K - lo) / (hi - lo) * (math.log(DAMPF[hi]) - math.log(DAMPF[lo])))
# ---- measured ladder levels (KN_LEVELS)
HF2KN = {"self_attn.q_proj": "attn_qg", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
         "self_attn.o_proj": "attn_output", "linear_attn.in_proj_qkv": "ssm_qkv", "linear_attn.in_proj_z": "ssm_z",
         "linear_attn.out_proj": "ssm_out", "mlp.gate_proj": "ffn_gate_up", "mlp.up_proj": "ffn_gate_up",
         "mlp.down_proj": "ffn_down"}
LEV = collections.defaultdict(dict)     # knap name -> K -> proxy error (gate / up: geometric mean)
if os.environ.get("KN_LEVELS"):
    import glob as _g
    for pj in _g.glob(f"{os.environ['KN_LEVELS']}/K*/proxy.jsonl"):
        acc = collections.defaultdict(list)
        for l in open(pj):
            d = json.loads(l); k = d["key"].split(".")
            if k[2] != "layers" or d.get("proxy_err") is None: continue
            n = HF2KN.get(".".join(k[4:6]))
            if n: acc[(f"blk.{k[3]}.{n}.weight", float(d["K"]))].append(d["proxy_err"])
        for (n, K), v in acc.items(): LEV[n][K] = math.exp(sum(map(math.log, v)) / len(v))
NLEV = [0]
def level_pos(w, K):
    m = LEV.get(w); a, b = float(math.floor(K)), float(math.ceil(K))
    if not m or K not in m or a not in m or b not in m or not (m[b] < m[K] < m[a]): return None
    return math.log(m[K] / m[a]) / math.log(m[b] / m[a])
for t in md["tensors"]:
    o = t["opts"]
    if os.environ.get("KN_FRAC") == "1":
        for K in FRACS:
            if t["w"] == "output.weight" and abs(K / HEAD_STEP - round(K / HEAD_STEP)) > 1e-9: continue
            a, b = f"ours-K{math.floor(K)}", f"ours-K{math.ceil(K)}"
            if a not in o or b not in o: continue
            f = K - math.floor(K); (ba, ea, _), (bb, eb_, _) = o[a], o[b]
            fm = level_pos(t["w"], K)
            if fm is not None: f_e = fm; NLEV[0] += 1
            else: f_e = f
            e = math.exp(math.log(ea) + f_e * (math.log(eb_) - math.log(ea))) if ea > 0 and eb_ > 0 else ea
            o[f"ours-K{K:g}"] = [ba + f * (bb - ba), e, K]; t["family_of"][f"ours-K{K:g}"] = t["family_of"][a]
    if os.environ.get("KN_DAMP") == "1" and t["w"].split(".")[-2] in ("attn_qg", "attn_k", "attn_v"):
        for c in list(o):
            if c.startswith("ours-K"): o[c] = [o[c][0], o[c][1] * dampf(float(c[6:])), o[c][2]]
if os.environ.get("KN_FRAC") == "1" and LEV: print(f"measured fractional positions: {NLEV[0]} tensor-levels", file=sys.stderr)
SLABF = {5: 0.69, 5.5: 0.70, 6: 0.71}
if V3:
    SLABF = {5: 0.69, 6: 0.71}       # measured budgets only: between them the slab is a staircase (critique C3.5)
    for t in md["tensors"]:          # head: the integer containers that exist (hd-K3..8)
        if t["w"] == "output.weight":
            for c in [c for c in t["opts"] if c.startswith("ours-K") and (not float(c[6:]).is_integer() or float(c[6:]) < 3)]:
                del t["opts"][c]
if os.environ.get("KN_HEADSLAB") == "1":
    for t in md["tensors"]:
        if t["w"] != "output.weight": continue
        o = t["opts"]
        for K, fK in SLABF.items():
            c = f"ours-K{K:g}"
            if c in o: o[f"ours-slabK{K:g}"] = [o[c][0] + (1 << 20), o[c][1] * fK, o[c][2]]; t["family_of"][f"ours-slabK{K:g}"] = t["family_of"][c]
if os.environ.get("KN_EMBT") == "1":
    E = md["emb"]; cb = E["bf16"]["cost"]
    lr = {}
    for l in open(f"{HERE}/../e3/data/embtrellis.jsonl"):
        d = json.loads(l)
        if d["method"] == "lloyd": lr[d["bits"]] = d["rel"]
    lr[6] = lr[5] ** 2 / lr[4]                                     # e6: one more step at the e4->e5 ratio
    pts = sorted((math.log(lr[B]), math.log(E[f"e{B}"]["cost"] - cb)) for B in (3, 4, 5, 6))
    def dcost(r):
        x = math.log(r); (x0, y0), (x1, y1) = (pts[0], pts[1]) if x <= pts[1][0] else (pts[-2], pts[-1]) if x >= pts[-2][0] else \
            next((p, q_) for p, q_ in zip(pts, pts[1:]) if p[0] <= x <= q_[0])
        return math.exp(y0 + (x - x0) / (x1 - x0) * (y1 - y0))
    V_, C_ = 248320, 5120
    for l in open(f"{HERE}/../e3/data/embmenu.jsonl"):
        d = json.loads(l)
        if d["group"] != 5120: continue
        E[f"t{d['bits']:g}"] = {"cost": cb + dcost(d["rel"]), "container": f"trellis-K{d['bits']:g}",
                                "bytes": V_ * C_ * d["bits"] / 8 + V_ * 2}
if os.environ.get("KN_EMBT") == "2":
    E = md["emb"]; cb = E["bf16"]["cost"]
    LLOYD_ML = {2: 0.0091770, 3: 0.0021311, 4: 0.00029939}              # vs Lloyd 5 (the base's), golden-ml
    TRE_ML = {2: 0.0047127, 2.5: 0.0022361, 3: 0.0011973, 3.5: 0.00051891, 4: 0.00017474}
    SC = ((E["e3"]["cost"] - E["e5"]["cost"]) / LLOYD_ML[3] * (E["e4"]["cost"] - E["e5"]["cost"]) / LLOYD_ML[4]) ** 0.5
    rel = {}
    for l in open(f"{HERE}/../e3/data/embmenu.jsonl"):
        d = json.loads(l)
        if d["group"] == 5120: rel[float(d["bits"])] = d["rel"]
    V_, C_ = 248320, 5120
    for B, r in rel.items():
        m = min(TRE_ML, key=lambda k: abs(k - B))
        dm = SC * TRE_ML[m] + (E["e5"]["cost"] - cb)          # damage vs a bf16 embedding at the measured point
        dmg = dm * (r / rel[float(m)] if B not in TRE_ML else 1.0)   # along the CPU error curve to B
        E[f"t{B:g}"] = {"cost": cb + dmg, "container": f"trellis-K{B:g}", "bytes": V_ * C_ * B / 8 + V_ * 2}
    print(f"real embedding costs: ml->pooled scale {SC:.3f}", file=sys.stderr)
if os.environ.get("KN_EMBT") in ("1", "2"):
    print("embedding menu:", {k: (round(v["bytes"] * 8 / V_ / C_, 3), f"{v['cost'] - cb:.2e}") for k, v in sorted(E.items(), key=lambda x: x[1]["bytes"])}, file=sys.stderr)
q, kap, NL = md["q"], np.array(md["kappa"]), md["NL"]
PARAMS, FIXED = md["params"], md["fixed_bytes"]
phi = md["phi"]
TS = md["tensors"]; WI = {t["w"]: i for i, t in enumerate(TS)}
kf = json.load(open(AQF))["factors"]

# activation proxy: hook index -> tensor name (issue order; within a layer the shapes tell q/k/v and qkv/z apart)
mp = sorted(tuple(map(int, l.split())) for l in open(MAP) if l.strip())
lab = mkmask.label(mp)
idx_name = {}
for l in range(64):
    ii = [i for i, (f, ll) in enumerate(lab) if ll == l]
    att = l % 4 == 3
    names = (["attn_qg", "attn_k", "attn_v", "attn_output", "ffn_gate_up", "ffn_down"] if att else
             ["ssm_qkv", "ssm_z", "ssm_out", "ffn_gate_up", "ffn_down"])
    for i, n in zip(ii, names): idx_name[i] = f"blk.{l}.{n}.weight"
idx_name[len(lab) - 1] = "output.weight"
AQ = collections.defaultdict(dict)        # name -> (bits, group) -> activation KLD cost
for l in open(AQJ):
    r = json.loads(l)
    if abs(r["clip"] - 1.0) > 1e-6: continue
    AQ[idx_name[r["i"]]][(r["bits"], r["group"])] = kf[r["fam"]] * r["rel_out"]
SHARE = {"attn_qg": "attn_in", "attn_k": "attn_in", "attn_v": "attn_in", "ssm_qkv": "gdn_in", "ssm_z": "gdn_in"}

def shape(t):   # (N, K) from the hook map via the name
    for i, n in idx_name.items():
        if n == t["w"]: return mp[i][1], mp[i][2]
    raise KeyError(t["w"])

def wcost(t, c, e):
    if t["eb"] <= 0: return 0.0
    fam = t["family_of"].get(c); ph = math.exp(phi.get(fam, 0.0)) if fam in phi else 1.0
    return t["s"] * t["eb"] * (ph * e / t["eb"]) ** q

def gemm_ms(N, K, mode, g, fam=None):
    fl = 2.0 * M * N * K
    sh = FAMSHAPE.get(fam)
    if CORE and sh:
        kern = {"f16": "f16", "a8": "iu8", "a4": "iu4"}[mode]
        tops = CORE[(kern, sh)] * (1.0 if mode == "f16" else TOPS[mode][g] / TOPS[mode][0])
        return fl / (tops * 1e12) * 1e3
    return fl / (TF16 * 1e12) * 1e3 if mode == "f16" else fl / (TOPS[mode][g] * 1e12) * 1e3

def time_ms(t, mode, g, bits):
    """prefill time of tensor t in a mode: measured whole path (KN_A4) when there is one, else core + decode estimate"""
    if t["w"] == "output.weight": return 0.0          # last token only
    N, K = shape(t); fam = t["w"].split(".")[-2]; sh = FAMSHAPE.get(fam)
    if PATH and sh:
        if mode == "f16":
            v = path_ms("f16", sh, bits, N * K)
        else:
            v = path_ms("nfq8", sh, bits, N * K)
            if v is not None:   # swap the per-token int8 matmul for this mode / group's
                v += gemm_ms(N, K, mode, g, fam) - gemm_ms(N, K, "a8", 0, fam)
        if v is not None: return v
    return gemm_ms(N, K, mode, g, fam) + N * K * DEC_NS * 1e-6

def options(t):
    """[(mode, group, label, bytes, weight cost, act cost, time ms)] for one tensor"""
    N, K = shape(t); o = []; fam = t["w"].split(".")[-2]
    for c, (b, e, bits) in t["opts"].items():
        o.append(("f16", 0, c, b, wcost(t, c, e), 0.0, time_ms(t, "f16", 0, bits)))
    if t["bf16_bytes"]: o.append(("f16", 0, "bf16", t["bf16_bytes"], 0.0, 0.0,
                                   0.0 if t["w"] == "output.weight" else gemm_ms(N, K, "f16", 0, fam)))
    for c, (b, e, bits) in t["opts"].items():
        m = re.fullmatch(r"ours-K([\d.]+)", c)
        if not m or os.environ.get("KN_MODES") == "f16": continue
        Kb = float(m.group(1)); Kb = int(Kb) if Kb.is_integer() else Kb
        if V3 and (float(Kb) not in LADDER or t["w"] == "output.weight"): continue   # int8 only on the i8-codebook ladder
        for mode, F, bits_a in (("a8", F8, 8), ("a4", F4, 4)):
            fK = lerp_log(F, Kb)
            if fK is None: continue
            for g in G:
                a = AQ[t["w"]].get((bits_a, g))
                if a is None: continue
                o.append((mode, g, c, b, wcost(t, c, e * fK), a, time_ms(t, mode, g, bits)))
    return o

T_F16 = sum(time_ms(t, "f16", 0, 4) for t in TS)   # today's path (all f16; decode at K4 when measured)
T_OTHER = T_F16 * 0.08 / 0.92

def units_of_layer(l):
    ts = [i for i, t in enumerate(TS) if t["layer"] == l]
    groups = collections.defaultdict(list)
    for i in ts:
        n = TS[i]["w"].split(".")[-2] if TS[i]["w"] != "output.weight" else "head"
        groups[SHARE.get(n, TS[i]["w"])].append(i)
    return list(groups.values())

OPT = [options(t) for t in TS]

def pareto(cands):
    """{bucket: (score, X, A, T, picks)} keeping, per bucket, the min score; then drop dominated buckets"""
    out = {}; best = math.inf
    for k in sorted(cands):
        if cands[k][0] < best - 1e-18: out[k] = cands[k]; best = cands[k][0]
    return out

def stage(l, gl, lam):
    """layer l: bucket -> (score = gl X + A + lam T, X, A, T, [(tensor, option)])"""
    cur = {0: (0.0, 0.0, 0.0, 0.0, [])}
    for unit in units_of_layer(l):
        modes = sorted({(o[0], o[1]) for i in unit for o in OPT[i]})
        un = {}
        for mg in modes:   # one compute mode + group for the whole unit; K free per member
            sub = {0: (0.0, 0.0, 0.0, 0.0, [])}
            for i in unit:
                nxt = {}
                for o in OPT[i]:
                    if (o[0], o[1]) != mg: continue
                    kb = int(round(o[3] / BK)); sc = gl * o[4] + o[5] + lam * o[6]
                    for b0, (s0, x0, a0, t0, pk) in sub.items():
                        nb = b0 + kb
                        if nb not in nxt or s0 + sc < nxt[nb][0]:
                            nxt[nb] = (s0 + sc, x0 + o[4], a0 + o[5], t0 + o[6], pk + [(i, o)])
                sub = pareto(nxt)
            for k, v in sub.items():
                if k not in un or v[0] < un[k][0]: un[k] = v
        un = pareto(un); nxt = {}
        for b0, (s0, x0, a0, t0, pk) in cur.items():
            for k, (s1, x1, a1, t1, pk1) in un.items():
                nb = b0 + k
                if nb not in nxt or s0 + s1 < nxt[nb][0]: nxt[nb] = (s0 + s1, x0 + x1, a0 + a1, t0 + t1, pk + pk1)
        cur = pareto(nxt)
    return cur

def solve(cap, gls, lam):
    E = md["emb"]
    V = np.full(cap + 1, np.inf); P = np.full(cap + 1, np.inf); A = np.zeros(cap + 1); T = np.zeros(cap + 1)
    ech = np.full(cap + 1, -1); levels = list(E)
    for j, lv in enumerate(levels):
        nb = int(round(E[lv]["bytes"] / BK))
        if nb > cap: continue
        p = md["c0"] + E[lv]["cost"]; sl = slice(nb, None); better = p < V[sl]
        V[sl] = np.where(better, p, V[sl]); P[sl] = np.where(better, p, P[sl]); ech[sl] = np.where(better, j, ech[sl])
    args = []; stages = []
    for l in range(NL + 1):
        st = stage(l, gls[l], lam); stages.append(st)
        NV = np.full(cap + 1, np.inf); NP = np.full(cap + 1, np.inf); NA = np.zeros(cap + 1); NT = np.zeros(cap + 1)
        ch = np.full(cap + 1, -1, dtype=np.int64)
        for k, (s, x, a, t, _) in st.items():
            if k > cap: continue
            cp = np.full(cap + 1, np.inf); cp[k:] = P[:cap + 1 - k] * (1 + kap[l] * x) + x
            ca = np.zeros(cap + 1); ca[k:] = A[:cap + 1 - k] + a
            ct = np.zeros(cap + 1); ct[k:] = T[:cap + 1 - k] + t
            cv = cp + ca + lam * ct
            better = cv < NV
            NV = np.where(better, cv, NV); NP = np.where(better, cp, NP); NA = np.where(better, ca, NA)
            NT = np.where(better, ct, NT); ch = np.where(better, k, ch)
        V, P, A, T = NV, NP, NA, NT; args.append(ch)
    B = int(np.argmin(V))
    if not np.isfinite(V[B]): return None
    picks = []; Bb = B
    for l in range(NL, -1, -1):
        k = int(args[l][Bb]); picks += stages[l][k][4]; Bb -= k
    lv = levels[int(ech[Bb])]
    real = FIXED + E[lv]["bytes"] + sum(o[3] for _, o in picks)
    return dict(kld=float(P[B] + A[B]), kld_w=float(P[B]), kld_a=float(A[B]), t_lin=float(T[B]), picks=picks, emb=lv,
                bytes=real)

def grads(sol):
    """dKLD/dX_l at a solution (weights' compounding model)"""
    X = np.zeros(NL + 1)
    for i, o in sol["picks"]: X[TS[i]["layer"]] += o[4]
    Ps = [md["c0"] + md["emb"][sol["emb"]]["cost"]]
    for l in range(NL + 1): Ps.append(Ps[-1] * (1 + kap[l] * X[l]) + X[l])
    g = np.zeros(NL + 1); d = 1.0
    for l in range(NL, -1, -1):
        g[l] = d * (1 + kap[l] * Ps[l]); d *= 1 + kap[l] * X[l]
    return g

def run(target, lam):
    limit = target * PARAMS / 8 - 1
    gls = np.ones(NL + 1); sol = None
    for it in range(2):   # pass 1: g = 1; pass 2: g from pass 1's solution
        cap = int((limit - FIXED) / BK)
        while cap > 0:
            sol = solve(cap, gls, lam)
            if sol is None or sol["bytes"] <= limit: break
            cap -= max(1, int((sol["bytes"] - limit) / BK))
        if sol is None: return None
        gls = grads(sol)
    return sol

RADF = os.environ.get("KN_F", RXM_MODELS)
def rad(c, w):
    if V3 and c.startswith("trellis-K"): return f"{RXM_V3}/emb-{c}.rad"
    if V3 and re.fullmatch(r"ours-K[\d.]+", c) and float(c[6:]) in LADDER and w != "output.weight": return f"{RXM_V3}/{c}.rad"
    if not c.startswith("ours-"): return f"{RADF}/q38-27b-{c}.rad"
    if c.startswith("ours-slab"): return f"{RXM_OURS}/hdslab-{c[9:]}.rad"
    return f"{RXM_OURS}/hd-{c[5:]}.rad" if w == "output.weight" else f"{RXM_OURS}/{c}.rad"
name_idx = {n: i for i, n in idx_name.items()}
GCODE = {0: "0", 1024: "1", 512: "2", 256: "3", 128: "4", 64: "5"}
def write_set(dirn, name, sol):
    if os.environ.get("KN_WRITE_CHECK", "1") == "1":
        need = {rad(o[2], TS[i]["w"]) for i, o in sol["picks"]} | {rad(md["emb"][sol["emb"]]["container"], "token_embd.weight")}
        miss = sorted(p for p in need if not os.path.exists(p))
        if miss: print(f"{name}: NOT written, {len(miss)} containers missing: {miss[:6]}", file=sys.stderr); return
    os.makedirs(dirn, exist_ok=True)
    mask = ["0"] * len(lab); grp = ["0"] * len(lab)
    with open(f"{dirn}/{name}.mix", "w") as f:
        for i, o in sol["picks"]:
            w = TS[i]["w"]; f.write(f"{w} {rad(o[2], w)}\n")
            if o[0] != "f16": mask[name_idx[w]] = o[0][1]; grp[name_idx[w]] = GCODE[o[1]]
        f.write(f"token_embd.weight {rad(md['emb'][sol['emb']]['container'], 'token_embd.weight')}\n")
    with open(f"{dirn}/{name}.aq", "w") as f:
        f.write(f"EXL3_AQ_MASK={''.join(mask)}\nEXL3_AQ_GROUPS={''.join(grp)}\n")
WDIR = os.environ.get("KN_WRITE")

with open(OUT, "a") as f:
    for tg in TARGETS:
        for lam in LAMBDAS:
            s = run(tg, lam)
            if s is None: print(tg, lam, "nothing fits", file=sys.stderr); continue
            modes = collections.Counter(f"{o[0]}{'' if o[0] == 'f16' else '-g' + str(o[1] or 'pt')}" for _, o in s["picks"])
            flops = collections.Counter()
            for i, o in s["picks"]:
                if TS[i]["w"] == "output.weight": continue   # last token only: not prefill FLOPs
                N, K = shape(TS[i]); flops[o[0]] += 2.0 * N * K
            tot = sum(flops.values())
            r = {"target": tg, "lambda": lam, "bpw": s["bytes"] * 8 / PARAMS, "kld": s["kld"], "kld_w": s["kld_w"],
                 "kld_a": s["kld_a"], "t_lin_ms": s["t_lin"], "t_f16_ms": T_F16,
                 "prefill_speedup": (T_F16 + T_OTHER) / (s["t_lin"] + T_OTHER), "emb": s["emb"],
                 "flop_share": {k: round(v / tot, 3) for k, v in flops.items()}, "modes": dict(modes.most_common())}
            ps = collections.Counter()
            for i, o in s["picks"]: N, K = shape(TS[i]); ps[o[2]] += N * K
            pt = sum(ps.values()); r["param_share"] = {k: round(v / pt, 4) for k, v in ps.most_common()}
            r["name"] = f"nf-{tg:.2f}-l{lam:.0e}"
            if WDIR: write_set(WDIR, r["name"], s)
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"{tg:5.2f} lam {lam:8.1e}: bpw {r['bpw']:.3f} KLD {r['kld']:.5f} (w {r['kld_w']:.5f} a {r['kld_a']:.5f}) "
                  f"lin {r['t_lin_ms']:7.1f} ms  speedup x{r['prefill_speedup']:.2f}  flops {r['flop_share']}", flush=True)
