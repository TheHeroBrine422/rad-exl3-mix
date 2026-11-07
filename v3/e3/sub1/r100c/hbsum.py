#!/usr/bin/env python3
"""hbsum.py RES -- #100c speed tables (r100c.md rules) from hashbench.sh's clean passes RES/p<i>/<v>-K<k>-{2048,gemv}.csv
(w29's builds: old = mul1i8 today, cf = closed form, cfz, iqrh = hash, abl = no decode ALU). Per number the median
over passes of bench_twopass's us_median. Layer = 48 GDN : 16 attention (bench_twopass's trunk shapes). Prints
markdown: validity (K4 cf GEMV attn_k M 1 in 50-75 us), the GEMV table (decides: t_cf / t_iqrh at M 1 >= 1.15 at K2-4),
wdec8 / A8 (prefill), decode shares vs abl, and decode throughput (values/s = N K / t)."""
import csv, glob, os, statistics, sys, collections
RES = sys.argv[1]
V = ['old', 'cf', 'cfz', 'iqrh', 'abl']
GDN = ['gdn_qkv', 'gdn_z', 'ssm_out', 'gate_up', 'down']
ATT = ['attn_qg', 'attn_k', 'attn_v', 'attn_o', 'gate_up', 'down']
T = collections.defaultdict(list); NK = {}
passes = sorted(d for d in glob.glob(f"{RES}/p[0-9]") if os.path.exists(f"{d}/.ok"))
for p in passes:
    for f in glob.glob(f"{p}/*.csv"):
        v, k, _ = os.path.basename(f)[:-4].split('-')
        for r in csv.DictReader(open(f)):
            if int(r['rc']): continue
            T[(v, int(k[1:]), r['shape'], int(r['M']), r['mode'])].append(float(r['us_median']))
            NK[r['shape']] = int(r['N']) * int(r['K'])
t = lambda v, k, s, M, m: statistics.median(T[(v, k, s, M, m)]) if T.get((v, k, s, M, m)) else None
def layer(v, k, M, m):
    g = [t(v, k, s, M, m) for s in GDN]; a = [t(v, k, s, M, m) for s in ATT]
    return None if None in g + a else (48 * sum(g) + 16 * sum(a)) / 64
def lvals():   # values in the mean layer
    return (48 * sum(NK[s] for s in GDN) + 16 * sum(NK[s] for s in ATT)) / 64
fm = lambda x: '—' if x is None else f"{x:.0f}" if x >= 100 else f"{x:.1f}"
sp = lambda a, b: '—' if a is None or b is None else f"{a / b:.3f}×"
sh = lambda x, a: '—' if x is None or a is None else f"{100 * (x - a) / x:.0f}%"
Ks = sorted({k for (_, k, *_) in T})
print(f"clean passes: {len(passes)} ({', '.join(os.path.basename(p) for p in passes)})\n")
c = t('cf', 4, 'attn_k', 1, 'f16')
print(f"validity: K4 cf GEMV attn_k M 1 = {fm(c)} µs (valid if 50-75): {'OK' if c and 50 <= c <= 75 else 'NOT VALID'}\n")
print("### GEMV (f16, nt_pass) layer µs and speedups (decode path)\n")
print("| K | M | old | cf | iqrh | abl | t_cf / t_iqrh | t_old / t_iqrh | decode share old / cf / iqrh | Gvalues/s cf / iqrh |")
print("|---|---|---|---|---|---|---|---|---|---|")
for k in Ks:
    for M in (1, 4, 16):
        L = {v: layer(v, k, M, 'f16') for v in V}
        gv = lambda x: '—' if x is None else f"{lvals() / x / 1e3:.0f}"
        print(f"| {k} | {M} | {fm(L['old'])} | {fm(L['cf'])} | {fm(L['iqrh'])} | {fm(L['abl'])} | **{sp(L['cf'], L['iqrh'])}** | "
              f"{sp(L['old'], L['iqrh'])} | {sh(L['old'], L['abl'])} / {sh(L['cf'], L['abl'])} / {sh(L['iqrh'], L['abl'])} | "
              f"{gv(L['cf'])} / {gv(L['iqrh'])} |")
r1 = [None if None in (layer('cf', k, 1, 'f16'), layer('iqrh', k, 1, 'f16')) else layer('cf', k, 1, 'f16') / layer('iqrh', k, 1, 'f16') for k in (2, 3, 4)]
sv = 'INCOMPLETE' if None in r1 else 'PASS' if all(x >= 1.15 for x in r1) else 'FAIL'
print(f"\n**Speed verdict (t_cf / t_iqrh ≥ 1.15 at M 1, K2-4):** {sv} ({', '.join('—' if x is None else f'{x:.3f}' for x in r1)})\n")
print("### Prefill: wdec8 pass alone and the whole A8 path at M 2048, layer µs\n")
print("| K | wdec8 old / cf / iqrh / abl | wdec8 cf / iqrh | wdec8 Gvalues/s cf / iqrh | A8 old / cf / iqrh | A8 cf / iqrh |")
print("|---|---|---|---|---|---|")
for k in Ks:
    w = {v: layer(v, k, 2048, 'wdec8') for v in V}; a = {v: layer(v, k, 2048, 'a8') for v in V}
    gv = lambda x: '—' if x is None else f"{lvals() / x / 1e3:.0f}"
    print(f"| {k} | {' / '.join(fm(w[v]) for v in ('old', 'cf', 'iqrh', 'abl'))} | {sp(w['cf'], w['iqrh'])} | {gv(w['cf'])} / {gv(w['iqrh'])} | "
          f"{' / '.join(fm(a[v]) for v in ('old', 'cf', 'iqrh'))} | {sp(a['cf'], a['iqrh'])} |")
print("\n### Per shape, K4 (µs old / cf / iqrh / abl)\n")
for M, m in ((1, 'f16'), (2048, 'wdec8')):
    print(f"- {m} M {M}: " + "; ".join(f"{s} {'/'.join(fm(t(v, 4, s, M, m)) for v in ('old', 'cf', 'iqrh', 'abl'))}"
                                     for s in ('gate_up', 'down', 'gdn_qkv', 'attn_o', 'attn_k')))
print("\nPass spread (max/min of the per-pass layer time, GEMV M 1, cf / iqrh): " + ", ".join(
    f"K{k} " + "/".join(
        (lambda xs: '—' if len(xs) < 2 else f"{max(xs) / min(xs):.3f}")(
            [(48 * sum(T[(v, k, s, 1, 'f16')][i] for s in GDN) + 16 * sum(T[(v, k, s, 1, 'f16')][i] for s in ATT)) / 64
             for i in range(min(len(T[(v, k, s, 1, 'f16')]) for s in GDN + ATT))]) for v in ('cf', 'iqrh')) for k in Ks))
