#!/usr/bin/env python3
"""cmp.py REF TEST -- precision noise of collect.py's rows: TEST (e.g. bf16 backward) against REF (fp32 backward), same
sequences, positions and labels (TEST run with YAQA_LABELS=REF, both YAQA_ROWS_F32=1).

Per linear: row error |r_test - r_ref| / |r_ref| (Frobenius over all rows), H_out error |H_test - H_ref| / |H_ref| and
the cosine of the two H_out diagonals. Yardstick: |H(first half of seqs) - H(second half)| / |H| within REF, the
sampling noise of an H_out from half the sequences. Prints means by family and by depth band."""
import os, sys, numpy as np, collections

REF, TEST = sys.argv[1], sys.argv[2]
def rows(d):
    fs = sorted(f for f in os.listdir(d) if f.endswith((".f32", ".bf16")))
    out = {}
    for f in fs:
        if f.endswith(".f32"): r = np.fromfile(f"{d}/{f}", np.float32)
        else: r = (np.fromfile(f"{d}/{f}", np.uint16).astype(np.uint32) << 16).view(np.float32)
        out[f[:5]] = r.astype(np.float64)
    return out
fro = np.linalg.norm
stats = collections.defaultdict(list)
for name in sorted(os.listdir(REF)):
    if name.startswith(".") or not os.path.isdir(f"{TEST}/{name}"): continue
    A, B = rows(f"{REF}/{name}"), rows(f"{TEST}/{name}")
    ks = sorted(set(A) & set(B))
    if len(ks) < 2: continue
    n = 32
    RA = np.concatenate([A[k].reshape(n, -1) for k in ks]); RB = np.concatenate([B[k].reshape(n, -1) for k in ks])
    HA, HB = RA.T @ RA, RB.T @ RB
    h = len(ks) // 2
    R1 = np.concatenate([A[k].reshape(n, -1) for k in ks[:h]]); R2 = np.concatenate([A[k].reshape(n, -1) for k in ks[h:]])
    H1, H2 = R1.T @ R1 / h, R2.T @ R2 / (len(ks) - h)
    dA, dB = np.diag(HA), np.diag(HB)
    r = dict(row=fro(RB - RA) / fro(RA), H=fro(HB - HA) / fro(HA),
             diagcos=dA @ dB / (fro(dA) * fro(dB)), halves=fro(H1 - H2) / fro((H1 + H2) / 2))
    l = int(name.split(".")[0]); fam = name.split(".")[-1]
    for key in (f"fam {fam}", f"layers {l // 16 * 16:2d}-{l // 16 * 16 + 15}", "all"):
        stats[key].append(r)
    if os.environ.get("CMP_VERBOSE"): print(name, {k: round(v, 4) for k, v in r.items()})
print(f"{'group':28s} {'n':>4s} {'row err':>8s} {'H err':>8s} {'diag cos':>9s} {'halves (sampling)':>18s}")
for k in sorted(stats):
    v = stats[k]
    print(f"{k:28s} {len(v):4d} " + " ".join(f"{np.mean([x[m] for x in v]):{w}.4f}" for m, w in
                                            (("row", 8), ("H", 8), ("diagcos", 9), ("halves", 18))))
