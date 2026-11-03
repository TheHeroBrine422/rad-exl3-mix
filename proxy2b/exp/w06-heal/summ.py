#!/usr/bin/env python3
"""summ.py O GOLDEN [Y] -- the README's pass rules on the arms' evalq.py rows (O/<tag>.f32; controls from Y=/data/yb),
via pcmp.py. Writes O/summary.json and prints one decision line per rule. Missing arms: the rule says 'pending'."""
import json, os, subprocess, sys
O, G = sys.argv[1:3]; Y = sys.argv[3] if len(sys.argv) > 3 else "/data/yb"
PC = "/workspace/rad-exl3-mix/proxy2b/pcmp.py"
def f(t):
    for d in (O, Y):
        if os.path.exists(f"{d}/{t}.f32"): return f"{d}/{t}.f32"
def pair(a, b):
    fa, fb = f(a), f(b)
    if not (fa and fb): return None
    return json.loads(subprocess.run([sys.executable, PC, G, fa, fb], capture_output=True, text=True, check=True).stdout)
S, out = {}, []
def P(a, b):
    S[f"{a}->{b}"] = r = pair(a, b); return r
def gain(r): return None if r is None else r["kld_A"] - r["kld_B"]
def say(s): out.append(s); print(s)

# jitter control: re-rounding every weight (evalq casts to bf16)
j = P("k3base", "k3jit")
say("jitter: pending" if j is None else f"jitter: ratio {j['ratio']:.4f} z {j['z']:+.1f} -> "
    + ("ok" if abs(j["ratio"] - 1) <= 0.005 else "FLAG: re-rounding noise; arm differences of this size are unreadable"))
# H1(a) oracle ceiling and the realistic heal
r = {k: P(f"k{k}base", f"k{k}orc") for k in (3, 2)}; h = {k: P(f"k{k}base", f"k{k}all") for k in (3, 2)}
if r[3] is None or h[3] is None: say("H1a: pending")
else:
    ro, ra, za = r[3]["ratio"], h[3]["ratio"], h[3]["z"]
    d = "DEFER S27 (H3 / E3 first)" if ro >= 0.95 else "BUILD S27 now" if ro <= 0.85 else \
        ("BUILD S27 now" if ra <= 0.97 and za <= -3 else "DEFER S27 (H3 / E3 first)")
    say(f"H1a: K3 oracle {ro:.4f}, heal-all {ra:.4f} (z {za:+.1f}); K2 oracle "
        f"{r[2]['ratio'] if r[2] else float('nan'):.4f}, heal-all {h[2]['ratio'] if h[2] else float('nan'):.4f} -> {d}")
    bar = [k for k in (3, 2) if h[k] and h[k]["ratio"] <= 0.95 and h[k]["z"] <= -3]
    say(f"H2 0.8B pilot bar (heal-all >= 5%, z <= -3): {'met at K' + ','.join(map(str, bar)) if bar else 'not met'}; "
        f"generalised share of the ceiling K3 {gain(h[3]) / gain(r[3]) if gain(r[3]) else float('nan'):.2f}")
# H1(b) last quarter vs all layers
lv = {k: P(f"k{k}all", f"k{k}last") for k in (3, 2)}; lb = {k: P(f"k{k}base", f"k{k}last") for k in (3, 2)}
if None in lv.values() or None in h.values(): say("H1b: pending")
else:
    rep = all(lv[k]["diff"] <= 0 for k in (3, 2)); strong = rep and any(lv[k]["z"] <= -2 for k in (3, 2))
    worse = all(lv[k]["z"] >= 3 for k in (3, 2))
    sh = {k: gain(lb[k]) / gain(h[k]) if gain(h[k]) and gain(h[k]) > 0 else float("nan") for k in (3, 2)}
    d = ("REPRODUCES (strong)" if strong else "REPRODUCES") + " -> H5 depth curriculum: 27B healing starts on the last 16 layers" if rep \
        else "ALL LAYERS (last quarter worse at both K)" if worse else "INCONCLUSIVE -> all layers by default"
    say(f"H1b: last vs all K3 {lv[3]['ratio']:.4f} (z {lv[3]['z']:+.1f}), K2 {lv[2]['ratio']:.4f} (z {lv[2]['z']:+.1f}); "
        f"last's share of all's gain K3 {sh[3]:.2f} K2 {sh[2]:.2f} -> {d}; "
        f"backward truncation to the last quarter {'ALLOWED' if min(sh.values()) >= 0.7 else 'not allowed'} (share >= 0.70 at both K)")
# H1(c) gradient precision
for k in (3, 2):
    p = f"{O}/gc-k{k}/gradcos.txt"
    say(f"H1c K{k}: " + (open(p).read().strip().splitlines()[-1] if os.path.exists(p) else "pending"))
# stacking on the diagonal-H_out quant (reported, no rule)
s = P("k3bdiag", "k3dall")
if s and h[3]: say(f"stack: heal on bdiag K3 {s['ratio']:.4f} (z {s['z']:+.1f}) vs heal on base {h[3]['ratio']:.4f}; "
                   f"gain kept {gain(s) / gain(h[3]):.2f}")
# H4 level -> mix transfer (overall = the weaker of the two mixes)
H4 = {}
for m in ("F", "D"):
    x, hh, w = P(f"mix{m}", f"mix{m}x"), P(f"mix{m}", f"mix{m}h"), P(f"mix{m}", f"mix{m}w")
    if x is None or hh is None: say(f"H4 mix{m}: pending"); continue
    S[f"mix{m}h->mix{m}x"] = P(f"mix{m}h", f"mix{m}x")
    if hh["z"] > -3: say(f"H4 mix{m}: mix healing {hh['ratio']:.4f} (z {hh['z']:+.1f}) is no gain -> uninformative"); continue
    sh = gain(x) / gain(hh); shw = gain(w) / gain(hh) if w else float("nan")
    d = "LIBRARY" if sh >= 0.70 and x["z"] <= -3 else "LIBRARY + POLISH" if shw >= 0.90 else "PER-MIX"; H4[m] = d
    say(f"H4 mix{m}: transfer {x['ratio']:.4f} (z {x['z']:+.1f}), mix-specific {hh['ratio']:.4f}, polish "
        f"{w['ratio'] if w else float('nan'):.4f}; transfer share {sh:.2f}, polish share {shw:.2f} -> {d}")
ORD = ["LIBRARY", "LIBRARY + POLISH", "PER-MIX"]
say("H4 overall: " + (max(H4.values(), key=ORD.index) if len(H4) == 2 else "pending (needs both mixes informative)"))
json.dump({"pairs": S, "decisions": out}, open(f"{O}/summary.json", "w"), indent=1)
