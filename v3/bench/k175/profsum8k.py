#!/usr/bin/env python3
"""profsum8k.py RES ROW [ROW ...] -- #175 (w137) per-op breakdown of one warm 8k prefill from two --profile-ops servers
(pf.sh prof1 / prof5: 1 and 5 prefill-only 8k requests): each op row's total ms in prof5 minus prof1 (matched by op /
geometry text), / 4. Categories: GEMM by weight family (exl3i_dense, gemm_nt*, the head), activation quant (V3's
exl3i_actq with its producer folded in; IQR's fused norm / gate + Hadamard + int8 quant), norms, elementwise, GDN,
attention, other. Also the measured request wall (prof5's requests 2-5, profiled) minus the ops: gaps. One column a ROW."""
import re, sys
from collections import defaultdict

res, rows = sys.argv[1], sys.argv[2:]
RX = re.compile(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\w+)\s+(\S+)\s+([\d.]+|-)\s+(.*)")


def table(path):
    lines = open(path, errors="replace").read().splitlines()
    st = [i for i, l in enumerate(lines) if l.startswith("per-op timing")]
    out = {}
    for l in lines[st[-1]:] if st else []:
        m = RX.match(l)
        if m:
            k = m.group(7).strip(); c, t = out.get(k, (0, 0.0)); out[k] = (c + int(m.group(1)), t + float(m.group(2)))
    return out


def cat(op):
    name = op.split()[0]
    w = re.search(r"blk\.\d+\.(\S+?)\.weight", op)
    if name in ("logits_gemm_vocab",) or re.search(r"(?<![\w.])output\.weight", op):
        return "GEMM head"
    if name in ("exl3i_dense", "gemm_nt", "gemm_nt_q", "gemm_nt_q_gated"):
        return "GEMM " + (w.group(1) if w else "?")
    if name == "exl3i_actq" or name.endswith("quant_i8g"):
        return "actq+producer " + name
    if name.startswith("gdn_"):
        return "GDN " + name
    if name in ("rope", "rope_table", "kv_store", "attn_paged", "qk_norm_rope_gate_kv_store"):
        return "attention " + name
    if name == "rmsnorm":
        return "norm rmsnorm"
    if name in ("add", "mul", "silu_mul", "sigmoid"):
        return "elementwise " + name
    return "other " + name


def walls(path):
    return [float(m.group(1)) for m in re.finditer(r"tokens, ([\d.]+) s", open(path).read())]


S, top = {}, {}
for r in rows:
    a, b = table(f"{res}/serve-prof1-{r}.log"), table(f"{res}/serve-prof5-{r}.log")
    d = defaultdict(float)
    for k, (c, t) in b.items():
        c0, t0 = a.get(k, (0, 0.0))
        if c - c0 > 0:
            d[cat(k)] += (t - t0) / 4
    w = walls(f"{res}/bench-prof5-{r}.txt")[1:]
    d["(request wall, profiled)"] = 1e3 * sum(w) / len(w)
    S[r] = d
fam = lambda k: k.split()[0]
keys = sorted({k for d in S.values() for k in d if not k.startswith("(")}, key=lambda k: (fam(k), -max(S[r].get(k, 0) for r in rows)))
print(f"{'ms per 8k prefill':44s}" + "".join(f"{r:>12s}" for r in rows))
last = None
for k in keys:
    if fam(k) != last:
        last = fam(k)
        tot = [sum(v for kk, v in S[r].items() if fam(kk) == last and not kk.startswith("(")) for r in rows]
        print(f"{last.upper() + ' total':44s}" + "".join(f"{t:12.2f}" for t in tot))
    print(f"  {k[len(last) + 1:]:42s}" + "".join(f"{S[r].get(k, 0):12.2f}" for r in rows))
ops = [sum(v for k, v in S[r].items() if not k.startswith("(")) for r in rows]
print(f"{'ops total':44s}" + "".join(f"{t:12.2f}" for t in ops))
print(f"{'request wall (profiled)':44s}" + "".join(f"{S[r]['(request wall, profiled)']:12.2f}" for r in rows))
print(f"{'gaps (wall - ops)':44s}" + "".join(f"{S[r]['(request wall, profiled)'] - o:12.2f}" for r, o in zip(rows, ops)))
