#!/usr/bin/env python3
"""extrap.py -- w209 speed extrapolations from the measured layer-3 times (prod-new.*.jsonl, old-K*/proxy.jsonl) and the
stage-1 tile rates (stage1-tiles.jsonl new, stage1-qtf2-speed.jsonl old). Model: time(tensor, K) = tiles / rate(K) x c +
rest, c = the measured in-pipeline / stage-1 rate ratio at K4, rest = the measured non-search time. Rough by design."""
import glob, json, os
H = os.path.dirname(os.path.abspath(__file__))
rd = lambda p: [json.loads(l) for l in open(p) if l.startswith("{")]
rn = {str(d["K"]): d["tiles_per_s"] for d in rd(f"{H}/stage1-tiles.jsonl")}
ro = {str(d["K"]): d["old_tiles_per_s"] for d in rd(f"{H}/stage1-qtf2-speed.jsonl")}
new = {}; old = {}
for f in glob.glob(f"{H}/prod-new.*.jsonl"):
    for d in rd(f): new[(d["key"], str(d["K"]))] = d
for f in glob.glob(f"{H}/old-K*/proxy.jsonl"):
    for d in rd(f): old[(d["key"], str(d["K"]))] = d
keys = sorted({k for k, _ in new})
tiles = sum(new[(k, "4")]["n"] * new[(k, "4")]["k"] for k in keys) / 256
print(f"layer 3: {len(keys)} tensors, {tiles * 256 / 1e6:.1f}M weights, {tiles / 1e6:.3f}M tiles")
print("| K | new layer s | of which search | ourq layer s (sum of tensors) | new / ourq |\n|---|---|---|---|---|")
for K in ("2", "4", "4.25", "6"):
    tn = sum(new[(k, K)]["sec"]["quantize"] for k in keys); tv = sum(new[(k, K)]["sec"]["viterbi"] for k in keys)
    to = sum(old[(k, K)]["s"] for k in keys) if all((k, K) in old for k in keys) else None
    print(f"| {K} | {tn:.0f} | {tv:.0f} | {to if to is None else round(to)} | {'—' if to is None else '%.1fx' % (tn / to)} |")
# the model's constants at K4
tn4 = sum(new[(k, "4")]["sec"]["quantize"] for k in keys); tv4 = sum(new[(k, "4")]["sec"]["viterbi"] for k in keys)
gs4 = sum(new[(k, "4")]["sec"]["gscale"] for k in keys)
cn = (tv4 + gs4) / (tiles / rn["4"]); restn = tn4 - tv4 - gs4
to4 = sum(old[(k, "4")]["s"] for k in keys); resto = to4 - tiles / ro["4"]
print(f"\nmodel: new search+gscale = {cn:.2f} x tiles / stage-1 rate, rest {restn:.0f} s a layer; ourq = tiles / old kernel rate + {resto:.0f} s a layer")
# the V3 ladder's levels (docs/design/v3.md: K0.125-5 in 1/8 steps, 5.25-7 in quarters, 7.25-8 in quarters = 55 levels incl. the old mul1 top)
lv = [i / 8 for i in range(1, 41)] + [5.25 + i / 4 for i in range(0, 12)]
def rate(tab, K):   # nearest measured rate at or below K's integer part / sub-1 class
    ks = sorted(float(k) for k in tab); c = max([k for k in ks if k <= K] or [ks[0]]); return tab[("%g" % c)]
L = 64
tnew = sum(L * (cn * tiles / rate(rn, K) + restn) for K in lv); told = sum(L * (tiles / rate(ro, K) + resto) for K in lv)
print(f"27B trunk (64 layers ~ layer 3's size, 448 tensors), {len(lv)} levels: new {tnew / 3600:.0f} card-h, ourq (plain settings) {told / 3600:.0f} card-h; one level at K4: new {L * tn4 / 60:.0f} min, ourq {L * to4 / 60:.0f} min")
# DS4.1-F: 15,360 experts x (gate, up: 2304 x 5120; down: 5120 x 2304), one K, 4 cards; trunk ~7B
et = 15360 * 3 * 2304 * 5120 / 256; tt = 7e9 / 256
for K in ("2", "3", "4"):
    a = cn * (et + tt) / rn[K] / 4 / 3600; b = (et + tt) / ro[K] / 4 / 3600
    print(f"DS4.1-F K{K}: {et / 1e9:.2f}B expert tiles + {tt / 1e6:.0f}M trunk tiles: search alone on 4 cards: this kernel {a:.1f} h, at the old kernel's rate {b:.1f} h")
