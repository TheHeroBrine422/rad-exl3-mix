#!/usr/bin/env python3
"""report115.py OUT [W114_LV] -- #152-01 / #153-01 (../w115.md): checks c1-c3, the f2 ratio tables, quant times, gates.

OUT/<unit>/ = one rxq quant unit (N-K<K>-s<S>, R-K<K>-s<S>, C-<fam>-K<K>-s<S>): t-<key>.safetensors, log.jsonl, wq/,
f2.json (f2calc98.py). W114_LV = w114's rxrun level root (c2: N's seed-0 t-files == the level's at K2.5 / K4), or
(amendment 1, box2) c2hash115.py's JSON of those levels' tensor hashes. Times are box2 card seconds; card-h per ladder is
also given in gpubox card-0 equivalents (/ 1.39, w117).
Last line: the gate summary."""
import glob, json, math, os, re, sys

import torch
from safetensors import safe_open

OUT = sys.argv[1]; LV = sys.argv[2] if len(sys.argv) > 2 else None
STAMP = "4f8319840e59c5d0"
FAMS = ("ssm_qkv", "ssm_z", "ffn_gate", "ffn_up", "ffn_down", "attn_v")
FAM = (("self_attn.q_proj", "attn_qg"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
       ("self_attn.o_proj", "attn_output"), ("linear_attn.in_proj_qkv", "ssm_qkv"), ("linear_attn.in_proj_z", "ssm_z"),
       ("linear_attn.out_proj", "ssm_out"), ("mlp.gate_proj", "ffn_gate"), ("mlp.up_proj", "ffn_up"), ("mlp.down_proj", "ffn_down"))
TRUNK_N = {"ssm_qkv": 48, "ssm_z": 48, "attn_v": 16, "ffn_gate": 64, "ffn_up": 64, "ffn_down": 64}
LEVELS = 50
fam = lambda k: next(f for s, f in FAM if k.endswith(s))
short = lambda k: f"L{k.split('.')[3]}.{fam(k)}"
geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def unit(name):
    d = f"{OUT}/{name}"
    if not os.path.exists(f"{d}/.done"): return None
    f2 = json.load(open(f"{d}/f2.json")) if os.path.exists(f"{d}/f2.json") else {}
    recs = {json.loads(x)["key"]: json.loads(x) for x in open(f"{d}/log.jsonl")}
    t = {}
    for k, r in recs.items():
        st = r["rxq"].get("stages") or {}
        t[k] = r["rxq"]["wall"] - st.get("save_wq", 0.0) - r["rxq"].get("save", 0.0)
    stamps = set()
    for tf in glob.glob(f"{d}/t-*.safetensors"):
        with safe_open(tf, "pt") as f: stamps.add(f.metadata().get("rxq_settings"))
    return {"dir": d, "f2": f2, "rec": recs, "t": t, "stamps": stamps}


U = {os.path.basename(d): unit(os.path.basename(d)) for d in sorted(glob.glob(f"{OUT}/*")) if os.path.isdir(d)}
U = {k: v for k, v in U.items() if v}
print(f"units done: {len(U)}: {' '.join(U)}")
bad = []

# c1: stamps
cfgst = json.load(open(f"{OUT}/../cfg/stamps.json")) if os.path.exists(f"{OUT}/../cfg/stamps.json") else {}
for n, u in U.items():
    want = STAMP if n.startswith("N-") else cfgst.get("R" if n.startswith("R-") else n.split("-")[1])
    if u["stamps"] != {want}: bad.append(f"c1 {n}: stamps {u['stamps']} != {want}")
print(f"c1 stamps: {'PASS' if not [b for b in bad if b.startswith('c1')] else 'FAIL'} (N {STAMP}; arms {cfgst})")

# c2: N seed 0 vs w114's levels
c2 = []


def thash(p):
    import hashlib, struct
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8 + n; hdr.pop("__metadata__", None); out = {}
        for k, v in sorted(hdr.items()):
            a, b = v["data_offsets"]; f.seek(base + a)
            out[k] = [v["dtype"], v["shape"], hashlib.sha256(f.read(b - a)).hexdigest()]
    return out


if LV and LV.endswith(".json"):
    man = json.load(open(LV))
    for K in ("2.5", "4"):
        u = U.get(f"N-K{K}-s0"); m = man.get(K, {})
        if not u or not m: c2.append(f"K{K} n/a"); continue
        same = diff = miss = 0
        for tf in glob.glob(f"{u['dir']}/t-*.safetensors"):
            k = os.path.basename(tf)[2:-len(".safetensors")]
            r = m.get(k)
            if not r or r["seed_offset"] != 0 or r["stamp"] != STAMP: miss += 1; continue
            ok = thash(tf) == r["t"]; same += ok; diff += not ok
        c2.append(f"K{K} {same} identical / {diff} differ / {miss} missing")
        if diff: bad.append(f"c2 K{K}: {diff} differ")
elif LV:
    for K in ("2.5", "4"):
        u = U.get(f"N-K{K}-s0"); lv = glob.glob(f"{LV}/q27v3-{STAMP[:8]}-k{K}-i8")
        if not u or not lv: c2.append(f"K{K} n/a"); continue
        same = diff = miss = 0
        for tf in glob.glob(f"{u['dir']}/t-*.safetensors"):
            k = os.path.basename(tf)[2:-len(".safetensors")]
            ref = glob.glob(f"{lv[0]}/out/*/t-{k}.safetensors")
            if not ref: miss += 1; continue
            with safe_open(tf, "pt") as a, safe_open(ref[0], "pt") as b:
                ok = set(a.keys()) == set(b.keys()) and all(torch.equal(a.get_tensor(x), b.get_tensor(x)) for x in a.keys())
            same += ok; diff += not ok
        c2.append(f"K{K} {same} identical / {diff} differ / {miss} missing")
        if diff: bad.append(f"c2 K{K}: {diff} differ")
print(f"c2 N == w114 levels (seed 0): {'; '.join(c2) or 'n/a (no level root given)'}")

# c3: R pass 0 == N; C: tile CD lines
for n, u in U.items():
    if n.startswith("R-"):
        N = U.get("N-" + n[2:])
        if not N: bad.append(f"c3 {n}: no N unit"); continue
        for k, r in u["rec"].items():
            p0 = (r.get("reldlq") or [{}])[0].get("proxy_err")
            if p0 is None or len(r.get("reldlq", [])) != 2: bad.append(f"c3 {n} {short(k)}: reldlq record {r.get('reldlq')}"); continue
            rel = abs(p0 - N["rec"][k]["proxy_err"]) / N["rec"][k]["proxy_err"]
            if rel > 1e-6: bad.append(f"c3 {n} {short(k)}: pass-0 proxy {p0:.6e} vs N {N['rec'][k]['proxy_err']:.6e} (rel {rel:.1e})")
    if n.startswith("C-"):
        lp = f"{u['dir']}/tilecd.jsonl"
        got = {json.loads(x)["key"] for x in open(lp)} if os.path.exists(lp) else set()
        if got != set(u["rec"]): bad.append(f"c3 {n}: tile CD lines {len(got)} vs {len(u['rec'])} tensors")
print(f"c3: {'PASS' if not [b for b in bad if b.startswith('c3')] else 'FAIL'}")
for b in bad: print("  CHECK FAIL", b)


def ratios(arm_prefix, K, keys=None, stat="f2"):
    """[(key, S, rho)] for arm units <arm_prefix>-K<K>-s<S> vs N"""
    out = []
    for S in (0, 1):
        a, N = U.get(f"{arm_prefix}-K{K}-s{S}"), U.get(f"N-K{K}-s{S}")
        if not a or not N: continue
        for k in a["rec"]:
            if keys and k not in keys: continue
            if k in a["f2"] and k in N["f2"]: out.append((k, S, a["f2"][k][stat] / N["f2"][k][stat]))
    return out


def times(arm_prefix, K):
    """[(key, S, t_arm, t_N)]"""
    out = []
    for S in (0, 1):
        a, N = U.get(f"{arm_prefix}-K{K}-s{S}"), U.get(f"N-K{K}-s{S}")
        if a and N: out += [(k, S, a["t"][k], N["t"][k]) for k in a["t"] if k in N["t"]]
    return out


print("\n## #152-01 re-LDLQ (R) vs N: f2 ratio rho per tensor (seeds 0 / 1), geo; quant time R / N")
gate152 = None
for K in ("1.5", "2.5", "4"):
    rs = ratios("R", K)
    if not rs: print(f"K{K}: no data"); continue
    g = geo([r for _, _, r in rs]); g1 = geo([r for _, _, r in ratios("R", K, stat="f1")])
    tt = times("R", K); tr = sum(a for *_, a, _ in tt) / max(sum(n for *_, n in tt), 1e-9)
    print(f"K{K}: geo rho f2 {g:.4f} (n {len(rs)}), f1 {g1:.4f}; time R/N {tr:.2f}x ({sum(a for *_, a, _ in tt):.0f} s vs {sum(n for *_, n in tt):.0f} s)")
    by = {}
    for k, S, r in rs: by.setdefault(k, {})[S] = r
    for k in sorted(by, key=lambda k: (int(k.split('.')[3]), fam(k))):
        print(f"   {short(k):22s} " + " ".join(f"s{S} {by[k][S]:.4f}" for S in sorted(by[k])))
    # report only: the pass the job's own proxy prefers
    bo = []
    for S in (0, 1):
        a, N = U.get(f"R-K{K}-s{S}"), U.get(f"N-K{K}-s{S}")
        if not a or not N: continue
        for k, r in a["rec"].items():
            if k not in a["f2"] or k not in N["f2"]: continue
            p = r.get("reldlq") or []
            pick1 = len(p) == 2 and p[1]["proxy_err"] < p[0]["proxy_err"]
            bo.append(a["f2"][k]["f2"] / N["f2"][k]["f2"] if pick1 else 1.0)
    print(f"   best-of by the job's own proxy (report only): geo {geo(bo):.4f}, pass 1 picked {sum(x != 1.0 for x in bo)}/{len(bo)}")
    if K == "2.5": gate152 = (g, len(rs))
print("\n## #153-01 tile CD on family F (C_F) vs N: f2 ratio per tensor / K / seed; geo over K2.5 + K4 x seeds; added time")
gates153 = {}
for F in FAMS:
    rs, rows, dt = [], [], {}
    for K in ("2.5", "4"):
        r = ratios(f"C-{F}", K); rs += r; rows += [(K, *x) for x in r]
        tt = times(f"C-{F}", K)
        if tt: dt[K] = sum(a - n for _, _, a, n in tt) / len(tt)
    if not rs: print(f"{F}: no data"); continue
    g = geo([x for *_, x in rs])
    ch = TRUNK_N[F] * sum(dt.values()) / len(dt) * LEVELS / 3600 if dt else float("nan")
    gates153[F] = (g, len(rs), ch)
    print(f"{F:9s} geo {g:.4f} (n {len(rs)}) {'PASS' if g <= 0.99 else 'fail'}; added s/tensor " +
          " ".join(f"K{K} {v:+.1f}" for K, v in dt.items()) + f"; ~{ch:.2f} box2 card-h per ladder = ~{ch / 1.39:.2f} card-0 h (serial, {TRUNK_N[F]} tensors x {LEVELS} levels)")
    print("   " + " ".join(f"K{K}/{short(k)}/s{S} {x:.4f}" for K, k, S, x in rows))

print()
if bad: print("CHECKS FAILED: gates not read")
g = f"#152 K2.5 geo {gate152[0]:.4f} (n {gate152[1]}) {'PASS' if gate152[0] <= 0.99 else 'FAIL'}" if gate152 else "#152 no K2.5 data"
p = [F for F, (x, _, _) in gates153.items() if x <= 0.99]
print(f"GATES{' (CHECKS FAILED)' if bad else ''}: {g}; #153 pass: {','.join(p) or 'none'} "
      f"({', '.join(f'{F} {x:.4f}' for F, (x, _, _) in gates153.items())})")
