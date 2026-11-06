#!/usr/bin/env python3
"""planes.py [W34DIR] -- review idea 5: decode-plane bandwidth arithmetic per trunk shape at M 512 / 2048 (markdown).

Inputs (w34's card runs, gfx1201 card 0, 250 W, plugin flags; mul1i4 weights): W34DIR/full-k2-2048.txt and
full-k3-2048.txt (every path and pass alone at M 2048), full-k2-sweep.csv (f16 / a8x / a4x totals per M).
Default W34DIR: local/workers/instr3/w34-results.

Model (estimates; what is measured is said so):
- Bytes a step: trellis read W*K/8, plane write W*{2, 1, 0.5} (f16 / int8 / packed int4), the GEMM's plane read >= once.
- BW: the decode passes' effective bandwidth on the large shapes (gate_up, down: measured wdec8 / wdec4 bytes / time).
- Decode pass of a plane: wdec8 / wdec4 measured; T2's f16 pass (not timed alone) = wdec8 + W / BW (one more byte a
  weight written; an upper bound on the small, latency-bound shapes).
- In-tile conversion, if it does not overlap the WMMAs (w34: the A4 fold's VALU did not): c ns per wave instruction
  per SIMD, calibrated on the fold (A4 tile no-fold vs g64b1: +693 us for 178M fold FMAs on gate_up, VOPD on: 0.50 ns;
  plugin flags: 0.64 ns), times the conversion instructions:
    accurate (T2, int8 plane -> f16 fragments): 8 ops per 16x16 fragment (v_perm + v_pk_add_f16 per 2 values; plane
    biased by 128), each fragment shared by MT = 4 WMMAs, fragments = (M / 64) * (N / 16) * (K / 16);
    i4 under A8 (A8 tile, packed int4 plane -> int8 fragments): 3 ops per fragment (and, shift, and; unsigned-B WMMA,
    the -7.5 * sum(x) row term as A4's R), each shared by 4 WMMAs (wave = 2 weight x 4 activation tiles),
    fragments = (M / 64) * (N / 16) * (K / 16).
- Scenarios: pess = decode-pass saving - serialized conversion (c = 0.64), no GEMM read saving;
  mid = decode-pass saving, conversion hidden; opt = mid + the halved plane read once at BW.
"""
import csv, os, sys
from collections import defaultdict

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
D = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/local/workers/instr3/w34-results"
SHAPES = [("gate_up", 34816, 5120), ("down", 5120, 17408), ("gdn_qkv", 10240, 5120), ("gdn_z", 6144, 5120),
          ("attn_qg", 12288, 5120), ("attn_k", 1024, 5120), ("attn_v", 1024, 5120), ("attn_o", 5120, 6144),
          ("ssm_out", 5120, 6144)]
GDN = ["gate_up", "down", "gdn_qkv", "gdn_z", "ssm_out"]
ATT = ["gate_up", "down", "attn_qg", "attn_k", "attn_v", "attn_o"]
NSIMD, C_LO, C_HI = 128, 0.50e-3, 0.64e-3   # us per wave instruction per SIMD


def load_txt(p):
    t = {}
    for l in open(p):
        f = l.split()
        if len(f) == 8 and f[3].isdigit(): t[(f[0], f[4])] = float(f[5])
    return t


P2 = {2: load_txt(f"{D}/full-k2-2048.txt"), 3: load_txt(f"{D}/full-k3-2048.txt")}
SW = defaultdict(dict)
for r in csv.DictReader(open(f"{D}/full-k2-sweep.csv")): SW[(r["shape"], r["mode"])][int(r["M"])] = float(r["us_median"])

# bandwidth from the large shapes' measured decode passes
bw = []
for K, t in P2.items():
    for s, n, k in SHAPES[:2]:
        W = n * k
        for mode, pb in (("wdec8", 1.0), ("wdec4", 0.5)):
            bw.append((W * K / 8 + W * pb) / (t[(s, mode)] * 1e-6) / 1e9)
BW = sum(bw) / len(bw)
print(f"decode-pass bandwidth on gate_up / down (K2, K3; wdec8, wdec4): {min(bw):.0f}-{max(bw):.0f} GB/s, mean {BW:.0f}\n")


def tot(s, mode, M):   # path total at M (2048: the pass-split run; 512: the sweep, A8 forced)
    if M == 2048: return P2[2][(s, mode)]
    return SW[(s, {"a8": "a8x", "a4": "a4x"}.get(mode, mode))][M]


def conv_us(M, n, k, ops):
    return ops * (M / 64) * (n / 16) * (k / 16) / NSIMD   # wave instructions per SIMD; x c gives us


rows_acc, rows_i4 = {}, {}
print("### Bytes a prefill step per trunk shape (MB; plane bytes do not depend on M)\n")
print("| shape | N×K | trellis K2 / K4 | f16 plane | int8 plane | int4 plane | f16 plane write+read (≥) |")
print("|---|---|---|---|---|---|---|")
for s, n, k in SHAPES:
    W = n * k
    print(f"| {s} | {n}×{k} | {W*2/8/1e6:.0f} / {W*4/8/1e6:.0f} | {2*W/1e6:.0f} | {W/1e6:.0f} | {W/2e6:.0f} | {4*W/1e6:.0f} |")
print()

for M in (2048, 512):
    print(f"### M {M}: accurate path (f16 activations, i8 tensors), int8 plane converted in the T2 tile\n")
    print("| shape | f16 path µs | est. f16 decode pass µs (share) | Δ decode (1 B/w less written) | conv. if serialized | "
          "gain pess / mid / opt |")
    print("|---|---|---|---|---|---|")
    for s, n, k in SHAPES:
        W = n * k
        T = tot(s, "f16", M)
        dw = W / (BW * 1e9) * 1e6                     # us: one byte a weight at BW
        f16dec = P2[2][(s, "wdec8")] + dw
        cv = conv_us(M, n, k, 8)
        g = (dw - cv * C_HI, dw, 2 * dw)   # opt: + the halved plane read (1 B/w less) once
        rows_acc[(s, M)] = (T, g)
        print(f"| {s} | {T:.0f} | {f16dec:.0f} ({f16dec / T:.0%}) | {dw:.0f} | {cv * C_LO:.0f}–{cv * C_HI:.0f} | "
              f"{g[0] / T:+.1%} / {g[1] / T:+.1%} / {g[2] / T:+.1%} |")
    print()
    print(f"### M {M}: i4 tensors under A8, packed int4 plane unpacked in the A8 tile (K2; wdec8 → wdec4 measured)\n")
    print("| shape | A8 path µs (mul1i4 K2) | wdec8 / wdec4 µs K2 (K3) | Δ decode measured | conv. if serialized | "
          "gain pess / mid / opt |")
    print("|---|---|---|---|---|---|")
    for s, n, k in SHAPES:
        W = n * k
        T = tot(s, "a8", M)
        d8, d4 = P2[2][(s, "wdec8")], P2[2][(s, "wdec4")]
        e8, e4 = P2[3].get((s, "wdec8")), P2[3].get((s, "wdec4"))
        dd = d8 - d4
        cv = conv_us(M, n, k, 3)
        rd = W / 2 / (BW * 1e9) * 1e6
        g = (dd - cv * C_HI, dd, dd + rd)
        rows_i4[(s, M)] = (T, g)
        k3 = f" ({e8:.0f} / {e4:.0f})" if e8 else ""
        print(f"| {s} | {T:.0f} | {d8:.0f} / {d4:.0f}{k3} | {dd:+.0f} | {cv * C_LO:.0f}–{cv * C_HI:.0f} | "
              f"{g[0] / T:+.1%} / {g[1] / T:+.1%} / {g[2] / T:+.1%} |")
    print()


def layer(rows, M, names_sel=None):
    out = []
    for i in range(3):
        tg = tt = 0.0
        for grp, w in ((GDN, 48), (ATT, 16)):
            for s in grp:
                if names_sel and s not in names_sel: continue
                T, g = rows[(s, M)]
                tt += w * T; tg += w * g[i]
        out.append(tg / tt)
    return out


print("### Layer (48 GDN : 16 attention), every tensor on the variant\n")
print("| M | accurate (int8 plane in T2) pess / mid / opt | i4 under A8 (int4 plane) pess / mid / opt | i4 under A8, gate_up + down only |")
print("|---|---|---|---|")
for M in (2048, 512):
    a, b, c = layer(rows_acc, M), layer(rows_i4, M), layer(rows_i4, M, {"gate_up", "down"})
    print(f"| {M} | {a[0]:+.1%} / {a[1]:+.1%} / {a[2]:+.1%} | {b[0]:+.1%} / {b[1]:+.1%} / {b[2]:+.1%} | "
          f"{c[0]:+.1%} / {c[1]:+.1%} / {c[2]:+.1%} |")
