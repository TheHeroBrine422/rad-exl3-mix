#!/usr/bin/env python3
"""longmine.py MANIFEST.tsv RADLS.jsonl OUT.json [--ref TAG] [--boot 1000] [--block 1024] -- ranked.md I1 (measure
L3): mine the existing long-trace evals of the 27B (golden-long2 and its capped variants: the two 250k verify traces
905 / 2675; also golden-ver3-long / ver4-long rows if listed) for family x position effects. CPU only, no GPU.

MANIFEST, one arm per line (tab-separated, '#' comments): TAG  SCORE.f32  GOLDEN_DIR  BITS_SOURCE
  SCORE.f32: per-row scores [kl, top1, gen, x] f32 (radiance gold runs / evalq.py, rows in GOLDEN_DIR's order);
  BITS_SOURCE: a .mix file (lines "<gguf tensor> <container>"), or one container path (a uniform quant).
Rows are keyed by (orig doc, context position), so capped goldens (golden-long2-cCAP) pair with the full one.

Per arm, on the gen rows it shares with the reference arm (--ref; default the first line):
  - mean KL by position band 0-2k / 2-8k / 8-32k / 32-64k / 64-128k / 128k+, and the ratio to the reference's;
  - compounding index c = ln(ratio over positions >= 32k / ratio over positions < 8k)  (> 0: the arm's error grows
    with context faster than the reference's);
  - per-family mean bits (parameter-weighted; container bytes from RADLS, ours-K<k> / hd-K<k> containers at K):
    attn_in (q/k/v), attn_out, gdn_in (ssm qkv / z), gdn_out, ffn (gate_up + down), head, emb.
Across arms: least squares c ~ b0 + b_t * trunk_bits + b_attn * (attn_bits - trunk_bits) + b_gdn * (gdn_bits -
trunk_bits) (attention / GDN = in + out; the only contrasts 17 strongly collinear mixes can carry), with CIs from a
block bootstrap (BLOCK-position blocks resampled within each document's short / mid / long region; every arm uses
the same resample: paired).
Decision (README): a contrast whose exp(beta) per bit has a 95% CI excluding 1 and a point estimate beyond x1.25 or
below x0.8 -> long-position family probe (I2 / measure L1) is worth its card-hour. Writes OUT.json; prints tables."""
import json, math, os, re, sys
import numpy as np

a = sys.argv[1:]
def opt(k, d):
    if k in a: i = a.index(k); v = a[i + 1]; del a[i:i + 2]; return v
    return d
REF = opt("--ref", None); NB = int(opt("--boot", 1000)); BLK = int(opt("--block", 1024))
man, radls, outp = a[:3]
EDG = [0, 2048, 8192, 32768, 65536, 131072, 1 << 40]; BN = ["0-2k", "2-8k", "8-32k", "32-64k", "64-128k", "128k+"]
SHORT, LONG = 8192, 32768
FAM = {"attn_qg": "attn_in", "attn_k": "attn_in", "attn_v": "attn_in", "attn_output": "attn_out", "ssm_qkv": "gdn_in",
       "ssm_z": "gdn_in", "ssm_out": "gdn_out", "ffn_gate_up": "ffn", "ffn_down": "ffn"}
TRUNK = ("attn_in", "attn_out", "gdn_in", "gdn_out", "ffn")

# ---- bits per tensor
RL = {}
for l in open(radls):
    d = json.loads(l); RL[d["c"]] = d["w"]
bf = next(v for c, v in RL.items() if c.endswith("bf16.rad"))
PAR = {w: b // 2 for w, b in bf.items()}
p4 = next((v for c, v in RL.items() if "plain4.00" in c), {})   # bf16 fuses GDN qkv/z/a/b as ssm_in: params of the
PAR.update({w: b * 2 for w, b in p4.items() if w not in PAR})   # quantized split tensors from their K4 bytes
def fam(w):
    if w == "output.weight": return "head"
    if w == "token_embd.weight": return "emb"
    p = w.split(".")
    return FAM.get(p[2]) if len(p) == 4 and p[0] == "blk" and p[3] == "weight" else None
def tbits(w, c):
    b = os.path.basename(c)
    if c in RL and w in RL[c]: return RL[c][w] * 8 / PAR[w]
    m = re.search(r"(?:ours|hd|hdslab)-(?:slab)?K([\d.]+)\.rad$", b)
    if m: return float(m.group(1))
    raise SystemExit(f"no bits for {w} in {c} (add the container to RADLS)")
def fambits(src):
    mix = {}
    if src.endswith(".mix"):
        for l in open(src):
            p = l.split()
            if len(p) == 2: mix[p[0]] = p[1]
    else: mix = {w: src for w in PAR}
    acc = {}
    for w, c in mix.items():
        f = fam(w)
        if f is None: continue
        s = acc.setdefault(f, [0.0, 0]); s[0] += tbits(w, c) * PAR[w]; s[1] += PAR[w]
    out = {f: s[0] / s[1] for f, s in acc.items()}
    tw = sum(acc[f][1] for f in TRUNK if f in acc)
    out["trunk"] = sum(acc[f][0] for f in TRUNK if f in acc) / tw
    for g, ms in (("attn", ("attn_in", "attn_out")), ("gdn", ("gdn_in", "gdn_out"))):
        out[g] = sum(acc[f][0] for f in ms) / sum(acc[f][1] for f in ms)
    return out

# ---- rows keyed by (orig doc, position)
_gold = {}
def golden(gd):
    if gd not in _gold:
        m = json.load(open(f"{gd}/kld.json")); R = m["rows"]; gen = np.fromfile(f"{gd}/gen.u8", np.uint8).astype(bool)
        key = np.zeros(R, np.int64)
        for j, d in enumerate(m["docs"]):
            n = d["tokens"] - 1 - d["score_from"]
            key[d["row0"]:d["row0"] + n] = d.get("orig", j) * (1 << 24) + d["score_from"] + np.arange(n)
        _gold[gd] = (key, gen, R)
    return _gold[gd]
arms = []
for l in open(man):
    if not l.strip() or l.startswith("#"): continue
    tag, f, gd, src = l.rstrip("\n").split("\t")
    key, gen, R = golden(gd)
    kl = np.fromfile(f, np.float32).reshape(-1, 4)[:, 0].astype(np.float64); assert len(kl) == R, (f, len(kl), R)
    arms.append({"tag": tag, "kl": (key[gen], kl[gen]), "bits": fambits(src)})
REF = REF or arms[0]["tag"]
ref = next(x for x in arms if x["tag"] == REF)
rk, rv = ref["kl"]; order = np.argsort(rk); rk, rv = rk[order], rv[order]
rdoc, rpos = rk >> 24, rk & ((1 << 24) - 1)
rband = np.digitize(rpos, EDG) - 1
# resampling blocks: (doc, pos // BLK); a replicate draws blocks with replacement within each stratum (doc x region
# short < 8k / mid / long >= 32k), so every replicate keeps each document's short and long rows
blk_id = rdoc * (1 << 20) + rpos // BLK
ub, inv = np.unique(blk_id, return_inverse=True)
bpos = (ub % (1 << 20)) * BLK; strat = (ub >> 20) * 3 + (bpos >= SHORT).astype(np.int64) + (bpos >= LONG)
rng = np.random.default_rng(0)
W = np.ones((NB + 1, len(ub)))
for d in np.unique(strat):
    ix = np.where(strat == d)[0]
    for r in range(1, NB + 1): W[r, ix] = np.bincount(rng.integers(0, len(ix), len(ix)), minlength=len(ix))
def aligned(x):
    k, v = x["kl"]; o = np.argsort(k); k, v = k[o], v[o]
    i = np.searchsorted(k, rk); hit = (i < len(k)) & (k[np.minimum(i, len(k) - 1)] == rk)
    out = np.full(len(rk), np.nan); out[hit] = v[i[hit]]; return out
def wsum(v, sel):               # [replicate] weighted sums over rows sel (row weight = its block's weight)
    b = np.bincount(inv[sel], weights=v[sel], minlength=len(ub)); return W @ b
res = {}
for x in arms:
    v = aligned(x); ok = ~np.isnan(v)
    r = {"bits": x["bits"], "rows": int(ok.sum()), "bands": {}}
    for k, n in enumerate(BN):
        s = ok & (rband == k)
        if s.sum() >= 1000: r["bands"][n] = {"rows": int(s.sum()), "kl": float(v[s].mean()), "ratio": float(v[s].sum() / rv[s].sum())}
    sS, sL = ok & (rpos < SHORT), ok & (rpos >= LONG)
    if sS.sum() >= 1000 and sL.sum() >= 1000:
        cr = np.log(wsum(np.nan_to_num(v), sL) / wsum(rv, sL)) - np.log(wsum(np.nan_to_num(v), sS) / wsum(rv, sS))
        r["c"] = float(cr[0]); r["c_boot"] = cr[1:]
    res[x["tag"]] = r
# ---- regression across arms (reference excluded: its c is 0 by construction)
use = [t for t, r in res.items() if "c" in r and t != REF]
reg = None
if len(use) >= 6:
    Xm = np.array([[1, res[t]["bits"]["trunk"], res[t]["bits"]["attn"] - res[t]["bits"]["trunk"],
                    res[t]["bits"]["gdn"] - res[t]["bits"]["trunk"]] for t in use])
    Y = np.array([[res[t]["c"]] + list(res[t]["c_boot"]) for t in use])      # [arm, replicate]
    beta = np.linalg.lstsq(Xm, Y, rcond=None)[0]                            # [4, replicate]
    names = ["const", "trunk_bits", "attn_minus_trunk", "gdn_minus_trunk"]
    reg = {}
    for i, n in enumerate(names):
        lo, hi = np.percentile(beta[i, 1:], [2.5, 97.5])
        reg[n] = {"beta": float(beta[i, 0]), "ci95": [float(lo), float(hi)],
                  "x_per_bit": float(math.exp(beta[i, 0])), "x_ci95": [math.exp(lo), math.exp(hi)]}
        if n in ("attn_minus_trunk", "gdn_minus_trunk"):
            e = math.exp(beta[i, 0])
            reg[n]["flag"] = bool((lo > 0 or hi < 0) and (e > 1.25 or e < 0.8))
    cond = float(np.linalg.cond(Xm - np.r_[0, Xm[:, 1:].mean(0)]))
    reg["n_arms"] = len(use); reg["design_condition"] = cond
    sd = Xm[:, 2:].std(0); reg["contrast_sd_bits"] = {"attn": float(sd[0]), "gdn": float(sd[1])}
print(f"reference {REF}; positions short < {SHORT}, long >= {LONG}; {len(arms)} arms")
print(f"{'arm':28s} {'trunk':>5s} {'attn':>5s} {'gdn':>5s} | ratio to ref by band " + " ".join(f"{n:>7s}" for n in BN) + " |      c   95% CI")
for t, r in res.items():
    b = r["bits"]; rb = " ".join(f"{r['bands'][n]['ratio']:7.3f}" if n in r["bands"] else "      -" for n in BN)
    ci = f"{r['c']:+.3f} [{np.percentile(r['c_boot'], 2.5):+.3f}, {np.percentile(r['c_boot'], 97.5):+.3f}]" if "c" in r else "-"
    print(f"{t[:28]:28s} {b['trunk']:5.2f} {b['attn']:5.2f} {b['gdn']:5.2f} | {'':20s} {rb} | {ci}")
if reg:
    for n in ("trunk_bits", "attn_minus_trunk", "gdn_minus_trunk"):
        g = reg[n]; print(f"c ~ {n:18s}: x{g['x_per_bit']:.3f} per bit, 95% CI x{g['x_ci95'][0]:.3f}..x{g['x_ci95'][1]:.3f}"
                          + (f"  FLAG" if g.get("flag") else ""))
    print(f"design: {reg['n_arms']} arms, condition {reg['design_condition']:.1f}, contrast SD attn {reg['contrast_sd_bits']['attn']:.2f} "
          f"/ gdn {reg['contrast_sd_bits']['gdn']:.2f} bits")
    fl = [n for n in ("attn_minus_trunk", "gdn_minus_trunk") if reg[n]["flag"]]
    print("VERDICT: " + (f"family x position effect ({', '.join(fl)}): run the long-position family probe (I2)" if fl else
                         "no family x position effect this data can see: I2 not justified by I1"))
else: print("VERDICT: too few arms with long rows for the regression")
for r in res.values(): r.pop("c_boot", None)
json.dump({"ref": REF, "arms": res, "regression": reg}, open(outp, "w"), indent=1)
