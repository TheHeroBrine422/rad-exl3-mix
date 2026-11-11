#!/usr/bin/env python3
"""proj111.py BENCH.csv [DENOM.csv] -- w111 #146-01 (../w111.md): the kernel bar from bench_p16's CSV. Per remapped shape (gdn_qkv,
gdn_z, attn_qg, attn_v) x rate x M: GEMM ratio on / off (ratio of medians, median of paired ratios), A8 op per call ratio;
per M and rate the projected change of the prefill gemm time: sum over 48 x (gdn_qkv + gdn_z) + 16 x (attn_qg + attn_v)
of (gemm_on - gemm_off) over the all-trunk A8 op sum (48 GDN layers: gdn_qkv, gdn_z, ssm_out, gate_up, down; 16
attention layers: attn_qg, attn_k, attn_v, attn_o, gate_up, down). Bar: C1 / C2 everywhere, projected <= +0.5% at M
2048 and 8192 (both rates), no remapped GEMM ratio > 1.03 at M >= 2048. Last line = the verdict. DENOM.csv: the all-trunk
A8 op times from another run of the same build and card (a run without --all-shapes-off)."""
import csv, sys

R = list(csv.DictReader(open(sys.argv[1])))
G = {(r["shape"], r["rate"], int(r["M"])): r for r in R}
D = {(r["shape"], r["rate"], int(r["M"])): r for r in csv.DictReader(open(sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]))}
REM = {"gdn_qkv": 48, "gdn_z": 48, "attn_qg": 16, "attn_v": 16}
ALL = {"gdn_qkv": 48, "gdn_z": 48, "ssm_out": 48, "gate_up": 64, "down": 64, "attn_qg": 16, "attn_k": 16, "attn_v": 16, "attn_o": 16}
ok_c = all(r["c1"] == "1" for r in R) and all(r["c2"] == "1" for r in R if r["shape"] in REM)
rates = sorted({r["rate"] for r in R}, key=float); Ms = sorted({int(r["M"]) for r in R})
print("shape     rate     M   gemm_off   gemm_on  r_med   r_pair  r_call")
worst = 0.0
for s in REM:
    for k in rates:
        for M in Ms:
            r = G[(s, k, M)]
            print(f"{s:8s} {k:>4s} {M:5d} {float(r['gemm_off_us']):9.1f} {float(r['gemm_on_us']):9.1f} "
                  f"{float(r['ratio_med']):.4f} {float(r['ratio_paired']):.4f} {float(r['call_ratio']):.4f}")
            if M >= 2048: worst = max(worst, float(r["ratio_med"]))
proj = {}
print("\nprojected prefill gemm change (A8 everywhere; 48 GDN + 16 attention layers):")
for k in rates:
    for M in Ms:
        tot = sum(n * float(D[(s, k, M)]["call_off_us"]) for s, n in ALL.items())
        rem = sum(n * float(G[(s, k, M)]["call_off_us"]) for s, n in REM.items())
        d = sum(n * (float(G[(s, k, M)]["gemm_on_us"]) - float(G[(s, k, M)]["gemm_off_us"])) for s, n in REM.items())
        dp = sum(n * (float(G[(s, k, M)]["ratio_paired"]) - 1) * float(G[(s, k, M)]["gemm_off_us"]) for s, n in REM.items())
        proj[(k, M)] = d / tot
        print(f"  K{k:>4s} M {M:5d}: all-trunk {tot / 1e3:8.1f} ms, remapped share {rem / tot * 100:5.1f}%, "
              f"delta {d:+8.1f} us = {d / tot * 100:+.3f}% (from paired ratios {dp / tot * 100:+.3f}%)")
pb = all(proj[(k, M)] <= 0.005 for k in rates for M in Ms if M in (2048, 8192))
v = ok_c and pb and worst <= 1.03
print(f"\nKERNEL BAR {'PASS' if v else 'FAIL'}: C1/C2 {'ok' if ok_c else 'FAIL'}, projected max at M 2048 / 8192 "
      f"{max(proj[(k, M)] for k in rates for M in Ms if M in (2048, 8192)) * 100:+.3f}% (<= +0.5%), worst remapped GEMM ratio "
      f"at M >= 2048 {worst:.4f} (<= 1.03)")
