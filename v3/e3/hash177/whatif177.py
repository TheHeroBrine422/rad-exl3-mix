#!/usr/bin/env python3
"""whatif177.py EPROF.json PICKDIR [--f K:F,...] [--scale S] -- first-order what-if for the hash codebook as a per-tensor
option (lead 11-13: estimate before any requant). Hash and mul1i8 options have equal bytes at equal K, so with each
size's pick (w133's pickT133-<size>.json: tensor -> ladder level) held fixed the codebook choice decouples per tensor:
  dt_t = t_i8(fam, K_t) - t_hash(fam, K_t)  (decode us a step; eprof-p1's in-situ per-op rows, family sum / layers, linear
         in K between the profiled K 2.5 / 4.125 / 6 and flat outside; profiled times x S, default 0.90 ~ unprofiled)
  dKL_t / KL ~ (f(K_t) - 1) * w_t  (w_t = the tensor's byte share of the trunk: a crude stand-in for its KL share)
Prints per size: tensors / bytes by K band, the decode saving if every tensor with dt > 0 goes hash, if all go hash, and
the KL factor that buys, for each f given. A threshold policy (hash below K*) is what any (lambda_d) solve reduces to
when f(K) is flat. (w139, #177)"""
import json, os, re, sys, glob
from collections import defaultdict

argv = sys.argv[1:]
FS = [{"2.5": 1.0, "4.125": 1.0}]
scale = 0.90
if "--f" in argv: i = argv.index("--f"); FS = [dict(x.split(":") for x in s.split(",")) for s in argv[i + 1].split(";")]; del argv[i:i + 2]
if "--scale" in argv: i = argv.index("--scale"); scale = float(argv[i + 1]); del argv[i:i + 2]
prof = json.load(open(argv[0])); pdir = argv[1]
PK = {"v3-k2.5-r16k2.5": 2.5, "v3-k4.125-r16k4": 4.125, "q27v3-144dd6e1-k6-i8": 6.0}
NL = {"attn_qg": 16, "attn_k": 16, "attn_v": 16, "attn_output": 16, "ffn_gate_up": 64, "ffn_down": 64,
      "ssm_qkv": 48, "ssm_z": 48, "ssm_out": 48}
rows = defaultdict(dict)           # fam -> K -> (t_i8 us / layer, t_hash us / layer)
for p, K in PK.items():
    for fam, v in prof[p].items():
        if fam in NL: rows[fam][K] = (v["base3"] / NL[fam] * scale, v["hq"] / NL[fam] * scale)


def interp(fam, K):
    ks = sorted(rows[fam]); K = min(max(K, ks[0]), ks[-1])
    for a, b in zip(ks, ks[1:]):
        if a <= K <= b:
            u = (K - a) / (b - a); return tuple((1 - u) * rows[fam][a][j] + u * rows[fam][b][j] for j in range(2))
    return rows[fam][ks[-1]]


def fval(f, K):
    pts = sorted((float(k), float(v)) for k, v in f.items())
    if K <= pts[0][0]: return pts[0][1]
    if K >= pts[-1][0]: return pts[-1][1]
    for (a, x), (b, y) in zip(pts, pts[1:]):
        if a <= K <= b: return x + (y - x) * (K - a) / (b - a)


SHAPE = {"attn_qg": 12288 * 5120, "attn_k": 1024 * 5120, "attn_v": 1024 * 5120, "attn_output": 5120 * 6144,
         "ffn_gate_up": 34816 * 5120, "ffn_down": 5120 * 17408, "ssm_qkv": 10240 * 5120, "ssm_z": 6144 * 5120, "ssm_out": 5120 * 6144}
print("| size | trunk bytes by K: <3 / 3-4.5 / >4.5 | step i8 ms (est.) | dt if hash where faster | dt if all hash | "
      + " | ".join(f"KL x (f {'/'.join(f'{k}:{v}' for k, v in f.items())}) hash-where-faster" for f in FS) + " |")
print("|---" * (5 + len(FS)) + "|")
for fp in sorted(glob.glob(os.path.join(pdir, "pickT133-*.json")), key=lambda x: float(re.search(r"-([\d.]+)\.json", x).group(1))):
    size = re.search(r"-([\d.]+)\.json", fp).group(1); pk = json.load(open(fp))
    band = [0, 0, 0]; tot_b = 0; t_i8 = dt_pos = dt_all = 0.0; kl = [0.0] * len(FS)
    items = []
    for n, lv in pk.items():
        m = re.match(r"blk\.\d+\.(\w+)\.weight", n); k = re.search(r"-k([\d.]+)-i8$", lv)
        if not m or not k or m.group(1) not in NL: continue
        fam, K = m.group(1), float(k.group(1)); b = SHAPE[fam] * K / 8; tot_b += b
        band[0 if K < 3 else 1 if K <= 4.5 else 2] += b
        ti, th = interp(fam, K); t_i8 += ti; dt_all += ti - th
        items.append((fam, K, b, ti - th))
    for fam, K, b, d in items:
        if d > 0:
            dt_pos += d
            for j, f in enumerate(FS): kl[j] += (fval(f, K) - 1) * b / tot_b
    print(f"| {size} | " + " / ".join(f"{100 * x / tot_b:.0f}%" for x in band) + f" | {t_i8 / 1000:.1f} | "
          f"−{dt_pos / 1000:.2f} ms | {'−' if dt_all > 0 else '+'}{abs(dt_all) / 1000:.2f} ms | " + " | ".join(f"{1 + x:.4f}" for x in kl) + " |")
