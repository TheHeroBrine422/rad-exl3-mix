#!/usr/bin/env python3
"""fit17602.py -- #176-02 (w159): decode prices per (tensor, option) from end-to-end server runs on ds8.

  python3 fit17602.py fit RES_DIR OUT.json [--report OUT.txt]
  python3 fit17602.py annotate PRICES.json MENU.json OUT.json [--suffix _i8]   every menu option gets "dec_us" (ctx 0) and
                                                                    "dec_us_8k" (with --suffix _i8: dec_us_i8 / dec_us_i8_8k)
  fit ... --prefix i8                                               the int8-decode column (rows i8u<K>-*, i8m-*)
  python3 fit17602.py predict PRICES.json ASSIGN.json               predicted ms / step (ctx 0, 8k) of an assignment
                                                                    ({w: option} as the search's / fin133's picks)

Method. Batch-1 decode runs every op of a step in order on one queue (RADIANCE_AQL_STAMP shows it: 741 dispatches a pass,
no overlap), and the k-th exl3i GEMV of a pass is menu.trunk_names()[k] (then the head, if it is exl3i). So a stamped
served pass gives every menu tensor's own device time. Prices:
  raw(type, level) = median over the type's layers of its GEMV time in the uniform level's stamped run (u<K>-s);
  raw(head plan)   = the head GEMV in h-<plan>-s (bf16 head: logits_gemm in the uniform runs);
  raw(emb option)  = the embedding gather kernel(s); raw(a/b) = the 48 ssm_ab GEMVs vs the 96 bf16 ssm_a / ssm_b GEMMs.
End to end: server ms per decode step (engine decode-step seconds / steps; median of the measured runs of a row; rows of
the same pack in different rounds averaged) = F_ctx + alpha_ctx * sum(raw of the pack's menu tensors) / 1000, fitted by
least squares on the uniform packs (two unstamped rounds, ABBA order). The table's prices are alpha_ctx * raw (us a step);
F_ctx is everything outside the menu (attention, GDN recurrence, norms, launch gaps, host). Held-out check: mixed packs
(m-*) predicted from the table vs measured (the rule: |error| < 2 %). Standard library only. (w159)"""
import glob, json, os, re, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "search", "v3"))
import menu as MN  # noqa: E402

TYPES = ["ssm_qkv", "ssm_z", "ssm_out", "attn_qg", "attn_k", "attn_v", "attn_output", "ffn_gate_up", "ffn_down"]
NAMES = MN.trunk_names()
NT = {t: sum(1 for w in NAMES if w.split(".")[2] == t) for t in TYPES}
LEVEL = "q27v3-635bac02-k{}-mad"
GDN = [l for l in range(64) if not MN.is_full_attention(l)]


def ktag(k):
    return f"{float(k):g}"


PREFIX = ""    # --prefix i8: the int8-decode column (rows i8u<K>-*, i8m-*; heads / embedding / a-b from the f16 rows)
STAMPS = None  # --stamps DIR: the reduced stamp files from another directory (e.g. the card-7 round, a cross-card check)
STAT = "min"   # RULES amendment 1 (decided 1:00 PM CST on the uniform fit: min): med (median of runs) | min (minimum of runs) | gap (median inter-chunk gap, median of runs)


def rows(res, stat=None):
    """tag -> {ctx: server ms / step of the row (statistic STAT over its measured runs)}, the spread, the row meta"""
    stat = stat or STAT
    out = {}
    for f in glob.glob(f"{res}/*.jsonl"):
        tag = os.path.basename(f)[:-6]
        by, gp = {}, {}
        meta = None
        for l in open(f):
            r = json.loads(l)
            meta = meta or r
            if r["warmup"] or r["srv_ms_per_decode_step"] is None: continue
            by.setdefault(r["ctx"], []).append(r["srv_ms_per_decode_step"])
            if r.get("client_med_gap_ms"): gp.setdefault(r["ctx"], []).append(r["client_med_gap_ms"])
        if stat == "gap":
            if not gp: continue
            ms = {c: st.median(v) for c, v in gp.items()}
        else:
            ms = {c: (st.median(v) if stat == "med" else min(v)) for c, v in by.items()}
        out[tag] = {"ms": ms, "n": {c: len(v) for c, v in by.items()},
                    "spread": {c: (max(v) - min(v)) / min(v) for c, v in by.items()},
                    "pack": meta.get("pack"), "stamp": meta.get("stamp"), "utc": meta.get("utc")}
    return out


def stamp(path):
    """a reduced stamp file -> {name: us} for the menu tensors + 'other', 'wall', 'busy', and a check string"""
    d = json.load(open(path))
    S = d["seqs"]
    gemv = [s for s in S if "exl3i_gemm" in s["kernel"] or "exl3i_mad_gemv" in s["kernel"]]
    n = len(gemv)
    has_ab = n in (336 + 48, 336 + 48 + 1)
    has_head = n in (337, 336 + 48 + 1)
    if n not in (336, 337, 384, 385): raise ValueError(f"{path}: {n} exl3i GEMV launches")
    order = []
    for w in NAMES:
        order.append(w)
        if has_ab and w.endswith(".ssm_qkv.weight"):   # where the bf16 ssm_a / ssm_b ran (after qkv); checked below
            order.append(w.replace("ssm_qkv", "ssm_ab"))
    if has_head: order.append(MN.HEAD)
    out = {"tensors": {w: s["us"] for w, s in zip(order, gemv)}}
    # int8 decode rows (EXL3I_MAD_GEMV=1): a GEMV without a producer's xb rows (ssm_out, at K <= 2) quantises its rows in
    # an exl3i_mad_rows pre-pass launched right before it: that time is the tensor's own
    pre, carry, gi = {}, 0.0, 0
    gid = {id(s): w for w, s in zip(order, gemv)}
    for s_ in S:
        if "exl3i_mad_rows" in s_["kernel"]: carry += s_["us"]
        elif id(s_) in gid:
            if carry: out["tensors"][gid[id(s_)]] += carry; pre[gid[id(s_)]] = carry
            carry = 0.0
    pre_ids = {id(s_) for s_ in S if "exl3i_mad_rows" in s_["kernel"]}
    if has_ab:   # the a/b GEMV must be the smallest of its GDN layer
        for l in GDN:
            ab = out["tensors"][f"blk.{l}.ssm_ab.weight"]
            if any(out["tensors"][f"blk.{l}.{t}.weight"] <= ab for t in ("ssm_qkv", "ssm_z", "ssm_out")):
                raise ValueError(f"{path}: layer {l} a/b GEMV is not the smallest: order wrong")
    ids = {id(s) for s in gemv} | pre_ids
    other = [s for s in S if id(s) not in ids]
    lg = [s["us"] for s in other if "logits_gemm" in s["kernel"]]
    if not has_head:
        if len(lg) != 1: raise ValueError(f"{path}: bf16 head kernel count {len(lg)}")
        out["head_bf16"] = lg[0]
    bf = [s["us"] for s in other if "r4d_gemm_bf16_nt" in s["kernel"]]
    out["ab_bf16"] = sum(bf) if bf else None
    out["ab_bf16_n"] = len(bf)
    em = [s for s in other if "embed" in s["kernel"]]
    out["emb"] = sum(s["us"] for s in em); out["emb_kernels"] = sorted({s["kernel"][:60] for s in em})
    out["other"] = sum(s["us"] for s in other if "logits_gemm" not in s["kernel"] and "r4d_gemm_bf16_nt" not in s["kernel"]
                       and "embed" not in s["kernel"])
    out["wall"], out["busy"], out["passes"] = d["wall_us"], d["busy_us"], d["passes"]
    return out


def per_type(t):
    by = {}
    for w, us in t["tensors"].items():
        if w.startswith("blk."): by.setdefault(w.split(".")[2], []).append(us)
    return {k: st.median(v) for k, v in by.items()}


def ols(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x); sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    b = sxy / sxx; a = my - b * mx
    res = [yy - (a + b * xx) for xx, yy in zip(x, y)]
    se_b = (sum(r * r for r in res) / (n - 2) / sxx) ** 0.5 if n > 2 else float("nan")
    return a, b, se_b, res


def fit(res, out, report=None):
    R = rows(res)
    lines0 = [f"e2e statistic: {STAT}" + (f"; rows with prefix '{PREFIX}'" if PREFIX else "")]
    sdir = STAMPS or res
    ST = {os.path.basename(f)[3:-5]: stamp(f) for f in glob.glob(f"{sdir}/st-*.json")}
    lines = lines0
    P = lambda s: lines.append(s)
    # trunk raw prices from the uniform stamped rounds (u<K>-s; u<K>-r1 if present too, averaged)
    uni = {}
    for tag, s in ST.items():
        m = re.fullmatch(PREFIX + r"u([0-9.]+)-(s|r1)", tag)
        if m: uni.setdefault(ktag(m.group(1)), []).append(s)
    raw = {t: {} for t in TYPES}
    base = {"head_bf16": [], "ab_bf16": [], "emb_bf16": [], "other": []}
    for k, ss in sorted(uni.items(), key=lambda kv: float(kv[0])):
        pts = [per_type(s) for s in ss]
        for t in TYPES: raw[t][k] = st.mean(p[t] for p in pts)
        for s in ss:
            base["head_bf16"].append(s["head_bf16"]); base["ab_bf16"].append(s["ab_bf16"]); base["emb_bf16"].append(s["emb"])
            base["other"].append(s["other"])
    levels = sorted(raw["ffn_gate_up"], key=float)
    head_bf16 = st.median(base["head_bf16"]); ab_bf16 = st.median(base["ab_bf16"]); emb_bf16 = st.median(base["emb_bf16"])
    def trunk_sum(k): return sum(NT[t] * raw[t][k] for t in TYPES)
    P(f"uniform levels with stamps: {len(levels)} ({levels[0]}..{levels[-1]}); bf16 head {head_bf16:.1f} us, bf16 a/b "
      f"{ab_bf16:.1f} us (96 GEMMs), bf16 embedding {emb_bf16:.1f} us, other kernels {st.median(base['other']) / 1e3:.2f} ms")
    # e2e of the uniform packs, unstamped rounds a / b (ABBA); stamped rows as a check
    E = {}
    for tag, r in R.items():
        m = re.fullmatch(PREFIX + r"u([0-9.]+)-(a|b|c|s|r1)", tag)
        if m: E.setdefault(ktag(m.group(1)), {})[m.group(2)] = r["ms"]
    fitp = {}
    for ctx in ("0", "8k"):
        x, y, ks = [], [], []
        for k in levels:
            e = E.get(k, {})
            v = [e[r][ctx] for r in ("a", "b", "c") if r in e and ctx in e[r]]   # amendment 2: median of a / b (/ c)
            if not v: continue
            x.append((trunk_sum(k) + head_bf16 + ab_bf16 + emb_bf16) / 1e3); y.append(st.median(v)); ks.append(k)
        if len(x) < 3: P(f"ctx {ctx}: {len(x)} uniform e2e points, no fit"); continue
        a, b, se, rs = ols(x, y)
        rel = [r / yy for r, yy in zip(rs, y)]
        fitp[ctx] = {"F_ms": a, "alpha": b, "alpha_se": se, "n": len(x), "rms_rel": (sum(q * q for q in rel) / len(rel)) ** 0.5,
                     "max_rel": max(rel, key=abs)}
        P(f"ctx {ctx}: e2e = {a:.3f} ms + {b:.4f} (se {se:.4f}) x stamped menu time; n {len(x)}, residual rms "
          f"{100 * fitp[ctx]['rms_rel']:.2f} %, max {100 * fitp[ctx]['max_rel']:+.2f} %")
        for k, xx, yy, rr in zip(ks, x, y, rs):
            e = E[k]; ab = [e[r][ctx] for r in ("a", "b", "c") if r in e and ctx in e[r]]
            P(f"   K{k:>6s}: stamped menu {xx:7.3f} ms  e2e {yy:7.3f} (rounds {' / '.join(f'{q:.3f}' for q in ab)})  fit {a + b * xx:7.3f}  "
              f"res {100 * rr / yy:+.2f} %" + (f"  stamped-run e2e {e['s'][ctx]:.3f}" if 's' in e and ctx in e['s'] else ""))
    for k in levels:   # amendment 2: packs whose rounds a / b differ by > 3 % (they get a round c)
        e = E.get(k, {})
        for ctx in ("0", "8k"):
            if "a" in e and "b" in e and ctx in e["a"] and ctx in e["b"] and abs(e["a"][ctx] / e["b"][ctx] - 1) > 0.03:
                P(f"   a/b disagree > 3 %: u{k}-{ctx} a {e['a'][ctx]:.3f} b {e['b'][ctx]:.3f}" + (" (c present)" if "c" in e else " (needs round c)"))
    al = {c: fitp[c]["alpha"] for c in fitp}
    # heads / embedding / a-b from their packs
    head = {"bf16": head_bf16}; emb = {"bf16": emb_bf16}; ab = {"bf16": ab_bf16}; extra = {}
    for tag, s in ST.items():
        m = re.fullmatch(r"h-(.+)-s|h-(.+)", tag)
        if m and MN.HEAD in s["tensors"]: head[m.group(1) or m.group(2)] = s["tensors"][MN.HEAD]
        m = re.fullmatch(r"e-(.+)", tag)
        if m: emb[m.group(1)] = s["emb"]; extra[tag] = s["emb_kernels"]
        m = re.fullmatch(r"ab-(T[0-9]+)", tag)
        if m: ab[m.group(1)] = sum(us for w, us in s["tensors"].items() if ".ssm_ab." in w)
    P(f"heads priced: {len(head) - 1}; embedding options: {sorted(emb)} ({extra}); a/b: {ab}")
    def scaled(v, c): return v * al[c]
    prices = {"about": __doc__.split("\n\n")[0].strip(), "unit": "us per batch-1 decode step (alpha x stamped device time)",
              "stat": STAT, "prefix": PREFIX, "fit": fitp, "levels": [LEVEL.format(k) for k in levels], "level_K": levels,
              "raw": {"trunk": raw, "head": head, "emb": emb, "ab": ab}, "n_per_type": NT}
    for c in al:
        prices[f"ctx{c}"] = {"F_ms": fitp[c]["F_ms"], "alpha": al[c],
                             "trunk": {t: {LEVEL.format(k): scaled(raw[t][k], c) for k in levels} for t in TYPES},
                             "head": {h: scaled(v, c) for h, v in head.items()}, "emb": {e: scaled(v, c) for e, v in emb.items()},
                             "ab": {a_: scaled(v, c) for a_, v in ab.items()}}
    # held-out mixes and every other non-uniform row: predicted from the table vs measured
    hold = {}
    for tag, r in R.items():
        m = re.fullmatch(PREFIX + r"m-(.+?)-([abcs])", tag)
        if not m or m.group(2) == "s": continue
        hold.setdefault(m.group(1), []).append(r)
    prices["heldout"] = {}
    for name, rs in sorted(hold.items()):
        af = f"{res}/../recipes/m-{name}.assign.json"
        if not os.path.exists(af): P(f"held-out {name}: no assignment file {af}"); continue
        asg = json.load(open(af))
        try: predict(prices, asg, next(iter(al)))
        except KeyError as e: P(f"held-out {name}: not priced yet ({e})"); continue
        row = {}
        for c in al:
            pred = predict(prices, asg, c)
            meas = st.median(r["ms"][c] for r in rs if c in r["ms"]) if any(c in r["ms"] for r in rs) else None
            row[c] = {"pred_ms": pred, "meas_ms": meas, "err": (pred - meas) / meas if meas else None,
                      "rounds": [r["ms"].get(c) for r in rs]}
        sm = ST.get(f"{PREFIX}m-{name}-s")
        if sm:   # additivity: the table's raw sum vs this mix's own stamps
            rt = raw_sum(prices, asg); own = sum(sm["tensors"].values()) + sm.get("head_bf16", 0) + (sm["ab_bf16"] or 0) + sm["emb"]
            row["stamp_additivity"] = {"table_raw_ms": rt / 1e3, "own_stamp_ms": own / 1e3, "err": (rt - own) / own}
        prices["heldout"][name] = row
        P(f"held-out {name}: " + "  ".join(f"ctx {c}: pred {v['pred_ms']:.3f} meas {v['meas_ms']:.3f} ({100 * v['err']:+.2f} %)"
                                           for c, v in row.items() if c in al and v["meas_ms"])
          + (f"  | stamps: table {row['stamp_additivity']['table_raw_ms']:.3f} vs own {row['stamp_additivity']['own_stamp_ms']:.3f} ms "
             f"({100 * row['stamp_additivity']['err']:+.2f} %)" if "stamp_additivity" in row else ""))
    json.dump(prices, open(out, "w"), indent=1)
    txt = "\n".join(lines); print(txt)
    if report: open(report, "w").write(txt + "\n")


def emb_interp(table, o):
    """an embedding option not measured: linear in K between (or beyond) the measured options of its codebook (t<K> i8,
    t<K>m mul1); the whole spread is ~3 us a step, so this only keeps the knapsack from seeing a hole"""
    m = re.fullmatch(r"t([0-9.]+)(m?)", o)
    if not m: return None
    k, fam = float(m.group(1)), m.group(2)
    pts = sorted((float(q[1:].rstrip("m")), v) for q, v in table.items() if re.fullmatch(r"t[0-9.]+" + ("m" if fam else ""), q))
    if len(pts) < 2: return pts[0][1] if pts else None
    lo, hi = (pts[0], pts[1]) if k <= pts[0][0] else (pts[-2], pts[-1]) if k >= pts[-1][0] else \
        next((a, b) for a, b in zip(pts, pts[1:]) if a[0] <= k <= b[0])
    return lo[1] + (hi[1] - lo[1]) * (k - lo[0]) / (hi[0] - lo[0])


def opt_price(prices, c, w, o):
    T = prices[f"ctx{c}"]
    if w == MN.HEAD: return T["head"].get(o)
    if w == MN.EMB:
        o = "bf16" if o == "base" else o
        return T["emb"][o] if o in T["emb"] else emb_interp(T["emb"], o)
    if w.endswith(".ssm_ab.weight") or w == "ab": return T["ab"].get(o)
    return T["trunk"][w.split(".")[2]].get(o)


def raw_sum(prices, asg):
    tot = 0.0; R = prices["raw"]
    for w, o in asg.items():
        if w == MN.HEAD: tot += R["head"][o]
        elif w == MN.EMB: o = "bf16" if o == "base" else o; tot += R["emb"][o] if o in R["emb"] else emb_interp(R["emb"], o)
        elif w == "ab": tot += R["ab"][o]
        else: tot += R["trunk"][w.split(".")[2]][re.search(r"-k([0-9.]+)-", o).group(1)]
    return tot


def predict(prices, asg, c="0"):
    """ms / step: F + sum of the assignment's prices; asg covers the 336 trunk tensors, the head, the embedding
    ("base" = bf16) and "ab" (bf16 / T4 / T5); missing head / embedding / ab count as bf16"""
    a = dict(asg); a.setdefault(MN.HEAD, "bf16"); a.setdefault(MN.EMB, "bf16"); a.setdefault("ab", "bf16")
    tot = prices[f"ctx{c}"]["F_ms"] * 1e3
    for w, o in a.items():
        p = opt_price(prices, c, w, o)
        if p is None: raise KeyError(f"no price for {w} = {o}")
        tot += p
    return tot / 1e3


def annotate(pp, mp, out):
    prices = json.load(open(pp)); m = json.load(open(mp)); miss = 0; n = 0; interp = 0
    for t in m["tensors"]:
        for o in t["opts"]:
            for c, key in (("0", "dec_us" + SUF), ("8k", "dec_us" + SUF + "_8k")):
                if f"ctx{c}" not in prices: continue
                p = opt_price(prices, c, t["w"], o["o"])
                if p is None: miss += c == "0"; o.pop(key, None)
                else: o[key] = round(p, 3)
            if t["w"] == MN.EMB and o["o"] not in prices["ctx0"]["emb"] and o["o"] != "base": o["dec_interp"] = True; interp += 1
            n += 1
    m.setdefault("built", {})["decode_prices"] = pp
    json.dump(m, open(out, "w"))
    print(f"{out}: {n} options, {miss} without a decode price, {interp} embedding options interpolated (dec_interp)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if "--prefix" in sys.argv: PREFIX = sys.argv[sys.argv.index("--prefix") + 1]
    SUF = sys.argv[sys.argv.index("--suffix") + 1] if "--suffix" in sys.argv else ""
    if "--stamps" in sys.argv: STAMPS = sys.argv[sys.argv.index("--stamps") + 1]
    if "--stat" in sys.argv: STAT = sys.argv[sys.argv.index("--stat") + 1]
    if cmd == "fit":
        rep = sys.argv[sys.argv.index("--report") + 1] if "--report" in sys.argv else None
        fit(sys.argv[2], sys.argv[3], rep)
    elif cmd == "annotate": annotate(*sys.argv[2:5])
    elif cmd == "predict":
        pr = json.load(open(sys.argv[2])); asg = json.load(open(sys.argv[3]))
        print(json.dumps({c: predict(pr, asg, c) for c in ("0", "8k") if f"ctx{c}" in pr}))
