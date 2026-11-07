#!/usr/bin/env python3
"""r75.py CMD O [...] -- g75.sh's readouts (../w75.md #112-01 and #106-01), from O/result.txt's lines ("TAG: \"kld\": .."
and "pair A -> B: \"ratio\": r, \"diff\": d, \"se\": s, \"z\": z"); "w64:TAG" names a w64 score file.
  pick O         #112-01's α*_new per (stack, K) from the seed-0 fit screen -> JSON {"u3": 3.0, ...} (exit 1 if incomplete)
  needclamp O T  exit 0 if any row multiplier of fit arm T (O/stats/T.jsonl) exceeds 2 (before any clamp)
  a27 O          #112-02 arm B's gdn_out α ("4.5") or "none", by the rule (pl RAISE at >= 1 K)
  rel O          #106-01 (a): rel = tr(E H E^T) / tr(W H W^T) of each o_proj, natural vs perm, raw calib H -> O/rel06.jsonl
  report O       markdown readout
  done O         exit 0 when every arm the rules call for is scored"""
import json, math, os, re, sys

cmd, O = sys.argv[1], sys.argv[2]
W64 = "/data/yb/w64"
P, KL = {}, {}
for l in open(f"{O}/result.txt") if os.path.exists(f"{O}/result.txt") else []:
    m = re.search(r"pair (\S+) -> (\S+): \"ratio\": ([0-9.]+), \"diff\": ([0-9.e-]+), \"se\": ([0-9.e-]+), \"z\": ([0-9.e-]*)", l)
    if m:
        P[(m.group(1), m.group(2))] = {"r": float(m.group(3)), "d": float(m.group(4)), "se": float(m.group(5)),
                                       "z": float(m.group(6)) if m.group(6) else 0.0}
        continue
    m = re.search(r"^\S+ (\S+): \"kld\": ([0-9.e-]+)", l)
    if m: KL[m.group(1)] = float(m.group(2))
sfx = lambda s: "" if s == 0 else f"_s{s}"
geo = lambda v: math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")
GRID = (3, 4, 5, 6, 8)
SK = [(st, k) for st in "up" for k in ("3", "2")]
gs = lambda a: f"{a:g}"                                     # alpha -> tag text (3 -> "3", 4.75 -> "4.75")
L06 = [3, 7, 11, 15, 19, 23]


def solve3(A, b):
    A = [r[:] for r in A]; b = list(b); n = 3
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(A[r][i])); A[i], A[p] = A[p], A[i]; b[i], b[p] = b[p], b[i]
        for r in range(i + 1, n):
            f = A[r][i] / A[i][i]
            for c in range(i, n): A[r][c] -= f * A[i][c]
            b[r] -= f * b[i]
    x = [0.0] * n
    for i in reversed(range(n)): x[i] = (b[i] - sum(A[i][c] * x[c] for c in range(i + 1, n))) / A[i][i]
    return x


def pick1(st, k):
    b = f"fit-{st}{k}"
    pts = [(g, P[(f"w64:{b}", f"{b}-g{g}")]["d"]) for g in GRID if (f"w64:{b}", f"{b}-g{g}") in P]
    if len(pts) < len(GRID): return None, {}
    sx = [sum(p[0] ** j for p in pts) for j in range(5)]; sy = [sum(p[1] * p[0] ** j for p in pts) for j in range(3)]
    c = solve3([[sx[i + j] for j in range(3)] for i in range(3)], sy)
    best = min(pts, key=lambda p: p[1])
    v = max(3.0, min(8.0, -c[1] / (2 * c[2]))) if c[2] > 0 else float(best[0])
    v = round(v * 4) / 4
    zb = P.get((f"{b}-g3", f"{b}-g{best[0]}"), {}).get("z", 0.0) if best[0] != 3 else 0.0
    a = v if (best[0] != 3 and zb <= -2) else 3.0
    return a, {"quad": c, "vertex_or_best": v, "best_g": best[0], "z_best_vs_g3": zb, "pts": pts}


def pick():
    out = {}
    for st, k in SK:
        a, _ = pick1(st, k)
        if a is None: return None
        out[f"{st}{k}"] = a
    return out


def alpha():
    try: return json.load(open(f"{O}/alpha75.json"))
    except (OSError, ValueError): return None


def confirm1(st, k, a):
    rs, vs, r0 = [], [], []
    for s in (0, 1, 2):
        x = sfx(s); t = f"{st}{k}{x}g{gs(a)}"
        p = P.get((f"w64:{st}{k}{x}c9", t)); q = P.get((f"w64:vf-{st}{k}{x}c9", f"vf-{t}")); p0 = P.get((f"w64:{st}{k}{x}", t))
        rs.append(p["r"] if p else None); vs.append(q["r"] if q else None); r0.append(p0["r"] if p0 else None)
    ok = all(v is not None for v in rs + vs)
    if not ok: return None
    ls, lv = sum(map(math.log, rs)) / 3, sum(map(math.log, vs)) / 3
    keep = lv / ls if ls < 0 else float("nan")
    raise_ = geo(rs) <= 0.995 and all(r < 1 for r in rs) and ls < 0 and keep >= 2 / 3
    return {"rs": rs, "vs": vs, "r0": r0, "geo": geo(rs), "vgeo": geo(vs), "keep": keep, "raise": raise_}


def a27():
    al = alpha()
    if al is None: return None
    ap = {}
    for k in ("3", "2"):
        a = al[f"p{k}"]
        if a == 3: ap[k] = 3.0; continue
        c = confirm1("p", k, a)
        if c is None: return None
        ap[k] = a if c["raise"] else 3.0
    if ap["3"] == 3 and ap["2"] == 3: return "none"
    return gs(round((ap["3"] + ap["2"]) / 2 * 2) / 2)


def stats(t):
    f = f"{O}/stats/{t}.jsonl"
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else []


def spread(t):
    out = {}
    for fam in ("gdn_out", "attn_o", "down"):
        x = [o for o in stats(t) if o["fam"] == fam]
        if not x: continue
        n = sum(o.get("rows", 0) for o in x)
        out[fam] = dict(c_mean=sum(o["c_mean"] * o.get("rows", 1) for o in x) / max(n, 1), c_sd_max=max(o["c_sd"] for o in x),
                        c_max=max(o.get("c_max_unclamped", o["c_max"]) for o in x), c_min=min(o.get("c_min", 0) for o in x),
                        gt15=sum(o.get("n_gt15", 0) for o in x) / max(n, 1), gt2=sum(o.get("n_gt2", 0) for o in x) / max(n, 1),
                        e_max=max(o["e_max"] for o in x), clipped=sum(o["clipped"] for o in x))
    return out


# ---- #106-01 ------------------------------------------------------------------------------------------------------
def rel06():
    import torch
    from safetensors import safe_open
    sys.path.insert(0, "/workspace/rad-exl3-mix/proxy2b/exp/w02-fixedscale")
    import fsutil as U
    ck = U.Ckpt("/data/models/qwen35-08b")
    with open(f"{O}/rel06.jsonl.part", "w") as fh:
        for k in ("3", "2"):
            for s in (0, 1, 2):
                x = sfx(s); nat = f"{O}/dec-on{k}{x}" if os.path.exists(f"{O}/noop-{k}{x}.fail") else f"{W64}/dec-k{k}ste{x}"
                for l in L06:
                    key = f"model.language_model.layers.{l}.self_attn.o_proj"
                    W = ck.get(key + ".weight").double()
                    H = U.hessian(f"/data/h08/calib/{U.group(key)}.safetensors", "cpu").double()
                    ref = float(((W @ H) * W).sum()); r = {}
                    for arm, d in (("natural", nat), ("perm", f"{O}/dec-opn{k}{x}")):
                        with safe_open(f"{d}/{key}.safetensors", "pt") as f: Q = f.get_tensor("weight").double()
                        E = Q - W; r[arm] = float(((E @ H) * E).sum()) / ref
                    fh.write(json.dumps({"K": k, "seed": s, "layer": l, **r, "ratio": r["perm"] / r["natural"]}) + "\n")
    os.replace(f"{O}/rel06.jsonl.part", f"{O}/rel06.jsonl")
    print("REL06 DONE")


def item106():
    L = ["## #106-01 · o_proj head act order (heaviest stored last) on the real 0.8B quantizer", ""]
    g = [l for l in open(f"{O}/result.txt")] if os.path.exists(f"{O}/result.txt") else []
    for l in g:
        if "BF16 GATE" in l or "noop gate" in l or "plumbing" in l: L.append("- " + l.strip())
    pj = f"{O}/m08perm/perm06.json"
    if os.path.exists(pj):
        L += ["", "| layer | head traces (natural head order) | stored order (groups, heads) |", "|---|---|---|"]
        for l, p in json.load(open(pj))["layers"].items():
            L.append(f"| {l} | {', '.join(f'{x:.0f}' for x in p['head_trace'])} | {p['group_order']}, {p['head_order']} |")
    R = [json.loads(l) for l in open(f"{O}/rel06.jsonl")] if os.path.exists(f"{O}/rel06.jsonl") else []
    res = {}
    L += ["", "| K | seed | (a) o_proj rel perm / natural (6-layer geo) | per layer | (b) search KLD ratio (z) | verify ratio |",
          "|---|---|---|---|---|---|"]
    for k in ("3", "2"):
        a_all, a_seed, b, v = [], [], [], []
        for s in (0, 1, 2):
            x = sfx(s); rr = [o["ratio"] for o in R if o["K"] == k and o["seed"] == s]
            ctl = f"pn{k}{x}" if os.path.exists(f"{O}/noop-{k}{x}.fail") else f"w64:p{k}{x}"
            vctl = f"vf-pn{k}{x}" if ctl.startswith("pn") else f"w64:vf-p{k}{x}"
            p = P.get((ctl, f"perm{k}{x}")); q = P.get((vctl, f"vf-perm{k}{x}"))
            if rr: a_all += rr; a_seed.append(geo(rr))
            if p: b.append(p["r"])
            if q: v.append(q["r"])
            L.append(f"| K{k} | {s} | {geo(rr):.4f} | {' '.join(f'{r:.3f}' for r in rr)} | "
                     + (f"{p['r']:.4f} ({p['z']:+.1f})" if p else "–") + " | " + (f"{q['r']:.4f}" if q else "–") + " |")
        L.append(f"| K{k} | **geo** | **{geo(a_all):.4f}** | | **{geo(b):.4f}** | {geo(v):.4f} |")
        res[k] = (geo(a_all), a_seed, geo(b), len(a_seed) == 3 and len(b) == 3)
    if all(r[3] for r in res.values()):
        A = {k: r[0] for k, r in res.items()}; B = {k: r[2] for k, r in res.items()}
        if all(A[k] <= 0.985 and all(x < 1 for x in res[k][1]) for k in A) and all(B[k] < 1 for k in B): v = "PASS → propose a 27B pair"
        elif all(A[k] > 0.995 for k in A) or any(B[k] > 1.005 for k in B): v = "FAIL"
        else: v = "PARTIAL"
        L += ["", f"**Rule (w75.md #106-01): {v}** ((a) geo K3 {A['3']:.4f}, K2 {A['2']:.4f}; (b) search geo K3 {B['3']:.4f}, K2 {B['2']:.4f})"]
    else: L += ["", "(#106-01 rule pending)"]
    return L


def item112():
    L = ["## #112-01 · gdn_out α beyond 3 (others at w64's α*)", "", "### Screen (seed 0, fit): ratio (z) vs α = 0; [vs g 3]", "",
         "| stack K | " + " | ".join(f"g {g}" for g in GRID) + " | α*_new | quad / best g / z(best vs 3) |", "|---" * (len(GRID) + 3) + "|"]
    for st, k in SK:
        b = f"fit-{st}{k}"; cells = []
        for g in GRID:
            p = P.get((f"w64:{b}", f"{b}-g{g}")); q = P.get((f"{b}-g3", f"{b}-g{g}"))
            cells.append((f"{p['r']:.4f} ({p['z']:+.1f})" if p else "–") + (f" [{q['r']:.4f}, z {q['z']:+.1f}]" if q else ""))
        a, info = pick1(st, k)
        L.append(f"| {'st' if st == 'u' else 'pl'} K{k} | " + " | ".join(cells) + f" | **{a if a is None else gs(a)}** | "
                 + (f"vertex/best {info['vertex_or_best']:g}, best g {info['best_g']}, z {info['z_best_vs_g3']:+.1f}" if info else "") + " |")
    al = alpha() or {}
    L += ["", "### Multiplier spread (seed 0; gdn_out / attn_o): c mean, max sd per tensor, c min..max, share > 1.5 / > 2, e_n max", "",
          "| arm | family | c mean | sd | min..max | > 1.5 | > 2 | e max | clipped |", "|---|---|---|---|---|---|---|---|---|"]
    for st, k in SK:
        tags = [f"fit-{st}{k}-g{g}" for g in GRID]
        if al.get(f"{st}{k}") is not None and al[f"{st}{k}"] not in GRID: tags.append(f"fit-{st}{k}-g{gs(al[f'{st}{k}'])}")
        for t in tags:
            for fam, s in spread(t).items():
                L.append(f"| {t} | {fam} | {s['c_mean']:.4f} | {s['c_sd_max']:.4f} | {s['c_min']:.3f}..{s['c_max']:.3f} | "
                         f"{s['gt15']:.2%} | {s['gt2']:.2%} | {s['e_max']:.3f} | {s['clipped']} |")
    cl = [(a, b_) for (a, b_) in P if b_.endswith("-cm2")]
    if cl:
        L += ["", "Clamp check (c ≤ 2, fit, seed 0, report only): " + "; ".join(f"{b_}: {P[(a, b_)]['r']:.4f} (z {P[(a, b_)]['z']:+.1f}) vs unclamped" for a, b_ in cl)]
    L += ["", "### Confirm (seeds 0–2): α*_new vs g 3 (w64's c9 arm), search; verify; and vs α = 0", "",
          "| stack K | α*_new | s0 | s1 | s2 | geo | verify geo | keeps | vs α = 0 (s0, s1, s2) | verdict |", "|---|---|---|---|---|---|---|---|---|---|"]
    for st, k in SK:
        a = al.get(f"{st}{k}")
        if a is None: continue
        if a == 3: L.append(f"| {'st' if st == 'u' else 'pl'} K{k} | 3 | | | | | | | | KEEP 3 (screen) |"); continue
        c = confirm1(st, k, a)
        if c is None: L.append(f"| {'st' if st == 'u' else 'pl'} K{k} | {gs(a)} | pending | | | | | | | |"); continue
        L.append(f"| {'st' if st == 'u' else 'pl'} K{k} | {gs(a)} | " + " | ".join(f"{r:.4f}" for r in c["rs"])
                 + f" | {c['geo']:.4f} | {c['vgeo']:.4f} | {c['keep']:.2f} | " + ", ".join(f"{r:.4f}" for r in c["r0"] if r)
                 + f" | **{'RAISE' if c['raise'] else 'KEEP 3'}** |")
    L += ["", f"27B arm B (pl rule): gdn_out α_27 = **{a27()}**"]
    return L


if cmd == "pick":
    a = pick()
    if a is None: sys.exit(1)
    print(json.dumps(a)); sys.exit(0)
if cmd == "needclamp":
    t = sys.argv[3]; s = stats(t)
    sys.exit(0 if s and max(o.get("c_max_unclamped", o["c_max"]) for o in s) > 2 else 1)
if cmd == "a27":
    v = a27(); print(v); sys.exit(0 if v is not None else 1)
if cmd == "rel": rel06(); sys.exit(0)
if cmd == "report":
    L = ["# w75 4070 readout (g75.sh)", ""] + item112() + [""] + item106()
    L += ["", "## KLD per arm", "", "| tag | KLD |", "|---|---|"] + [f"| {t} | {v:.5f} |" for t, v in sorted(KL.items())]
    print("\n".join(L)); sys.exit(0)
if cmd == "done":
    miss = []
    al = alpha()
    if al is None: miss.append("alpha75.json")
    else:
        for st, k in SK:
            if al[f"{st}{k}"] != 3 and confirm1(st, k, al[f"{st}{k}"]) is None: miss.append(f"confirm {st}{k}")
    if not os.path.exists(f"{O}/alpha27.txt"): miss.append("alpha27.txt")
    if not os.path.exists(f"{O}/.bf16fail"):
        for k in ("3", "2"):
            for s in (0, 1, 2):
                for v in ("", "vf-"):
                    if not os.path.exists(f"{O}/{v}perm{k}{sfx(s)}.f32"): miss.append(f"{v}perm{k}{sfx(s)}")
        if not os.path.exists(f"{O}/rel06.jsonl"): miss.append("rel06.jsonl")
    print(f"missing: {miss}" if miss else "complete"); sys.exit(1 if miss else 0)
