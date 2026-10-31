#!/usr/bin/env python3
"""knap3.py MODEL.json AQ.jsonl AQFACTORS.json MAP OUT.jsonl TARGETS LAMBDAS -- the new-format mix solver: bytes x time
x KLD frontier (first version, 2026-10-30).

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
  8% of today's all-f16 step (eval profile: 92% of the time in the EXL3 GEMMs).
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
M = int(os.environ.get("KN_M", 4096))
BK = 1 << 20
G = [0, 1024, 512, 256, 128, 64]                         # activation scale groups (0 = per token)
F8 = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.001, 5: 1.002, 6: 1.004, 7: 1.02, 8: 1.06}   # E3b; K1, K7, K8 extrapolated
F4 = {1: 1.0, 2: 1.01, 3: 1.065, 4: 2.72}               # E3b int4grid; K >= 5 not offered (x10 and up)
# E2 (gfx1201, M = 4096, first version): TOPS by mode and group; per-token from the 128x256 tile, groups from 128x128.
# g128 / g64 are scaled from the microbench ratios (not yet measured in the real kernel).
TOPS = {"a8": {0: 254, 1024: 225, 512: 220, 256: 210, 128: 185, 64: 153},
        "a4": {0: 488, 1024: 390, 512: 375, 256: 346, 128: 274, 64: 205}}
TF16 = 135.0                                             # T2's f16 GEMM core, TFLOPS
DEC_NS = float(os.environ.get("KN_DEC_NS", 0.006))       # weight-decode pre-pass, ns per weight per step (placeholder)

md = json.load(open(MJ))
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

def gemm_ms(N, K, mode, g):
    fl = 2.0 * M * N * K
    return fl / (TF16 * 1e12) * 1e3 if mode == "f16" else fl / (TOPS[mode][g] * 1e12) * 1e3

def options(t):
    """[(mode, group, label, bytes, weight cost, act cost, time ms)] for one tensor"""
    N, K = shape(t); o = []
    dec = N * K * DEC_NS * 1e-6
    for c, (b, e, bits) in t["opts"].items():
        o.append(("f16", 0, c, b, wcost(t, c, e), 0.0, gemm_ms(N, K, "f16", 0) + dec))
    if t["bf16_bytes"]: o.append(("f16", 0, "bf16", t["bf16_bytes"], 0.0, 0.0, gemm_ms(N, K, "f16", 0)))
    for c, (b, e, bits) in t["opts"].items():
        m = re.fullmatch(r"ours-K(\d)", c)
        if not m: continue
        Kb = int(m.group(1))
        for mode, F, bits_a in (("a8", F8, 8), ("a4", F4, 4)):
            if Kb not in F: continue
            for g in G:
                a = AQ[t["w"]].get((bits_a, g))
                if a is None: continue
                o.append((mode, g, c, b, wcost(t, c, e * F[Kb]), a, gemm_ms(N, K, mode, g) + dec))
    return o

T_F16 = sum(gemm_ms(*shape(t), "f16", 0) + shape(t)[0] * shape(t)[1] * DEC_NS * 1e-6 for t in TS)   # today's path
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
    if not c.startswith("ours-"): return f"{RADF}/q38-27b-{c}.rad"
    return f"{RXM_OURS}/hd-{c[5:]}.rad" if w == "output.weight" else f"{RXM_OURS}/{c}.rad"
name_idx = {n: i for i, n in idx_name.items()}
GCODE = {0: "0", 1024: "1", 512: "2", 256: "3", 128: "4", 64: "5"}
def write_set(dirn, name, sol):
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
                N, K = shape(TS[i]); flops[o[0]] += 2.0 * N * K
            tot = sum(flops.values())
            r = {"target": tg, "lambda": lam, "bpw": s["bytes"] * 8 / PARAMS, "kld": s["kld"], "kld_w": s["kld_w"],
                 "kld_a": s["kld_a"], "t_lin_ms": s["t_lin"], "t_f16_ms": T_F16,
                 "prefill_speedup": (T_F16 + T_OTHER) / (s["t_lin"] + T_OTHER), "emb": s["emb"],
                 "flop_share": {k: round(v / tot, 3) for k, v in flops.items()}, "modes": dict(modes.most_common())}
            r["name"] = f"nf-{tg:.2f}-l{lam:.0e}"
            if WDIR: write_set(WDIR, r["name"], s)
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"{tg:5.2f} lam {lam:8.1e}: bpw {r['bpw']:.3f} KLD {r['kld']:.5f} (w {r['kld_w']:.5f} a {r['kld_a']:.5f}) "
                  f"lin {r['t_lin_ms']:7.1f} ms  speedup x{r['prefill_speedup']:.2f}  flops {r['flop_share']}", flush=True)
