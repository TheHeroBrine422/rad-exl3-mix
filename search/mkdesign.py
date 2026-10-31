#!/usr/bin/env python3
"""mkdesign.py DIR MIX NAME N [FRAC] [SEED] -- budget-neutral random perturbations of a mix (R3 of
research-mix-allocation.md): each design point moves a random FRAC (default 0.4) of the trunk tensors one bit up or
down (random sign; the option one bit away with the lowest proxy error among SC/plain), then flips/drops moves until
the size is within 0.005 bpw of the mix. Pairs: point 2k+1 is the antithetic of 2k (all moves reversed, re-balanced).
Head and embedding stay as in the mix. Writes DIR/mixes/dz-NAME-NN.mix, prints queue lines (base sc4.00-e4)."""
import json, re, sys, glob, random, collections
import os
# paths (README: Paths)
RXM_WORK = os.environ.get("RXM_WORK", "/root")
RXM_MODELS = os.environ.get("RXM_MODELS", f"{RXM_WORK}/models/final")
D, MIX, NAME, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
FRAC = float(sys.argv[5]) if len(sys.argv) > 5 else 0.4
rng = random.Random(int(sys.argv[6]) if len(sys.argv) > 6 else 0)
F, BASE, PARAMS = RXM_MODELS, "sc4.00-e4", 26895998464
H, E = "output.weight", "token_embd.weight"
opt = collections.defaultdict(dict)
for p in glob.glob(f"{D}/hproxy-c*.jsonl"):
    for l in open(p):
        r = json.loads(l); opt[r["w"]][r["c"]] = (r["bytes"], r["err"], r["bits"])
W = sorted(w for w in opt if w != H)
cur, extra = {w: BASE for w in W}, []
for l in open(MIX):
    w, p = l.split()
    if w in cur: cur[w] = re.search(r"q38-27b-(.+)\.rad$", p).group(1)
    else: extra.append(l.rstrip("\n"))        # head / embedding lines: kept as they are
def ladder(w, c, d):                          # best option with bits = bits(c) + d
    b = opt[w][c][2] + d
    cand = [x for x in opt[w] if opt[w][x][2] == b]
    return min(cand, key=lambda x: opt[w][x][1]) if cand else None
def size(a): return sum(opt[w][a[w]][0] for w in W)
S0 = size(cur); tol = 0.005 * PARAMS / 8
def balance(moves):
    a = dict(cur)
    for w, d in moves.items():
        c = ladder(w, cur[w], d)
        if c: a[w] = c
    keys = list(moves); rng.shuffle(keys)
    for w in keys:                             # undo moves in the direction of the imbalance until within tol
        dS = size(a) - S0
        if abs(dS) <= tol: break
        if (dS > 0) == (moves[w] > 0) and a[w] != cur[w]: a[w] = cur[w]
    return a if abs(size(a) - S0) <= tol else None
pts = []
while len(pts) < N:
    mv = {w: rng.choice((-1, 1)) for w in W if rng.random() < FRAC}
    a = balance(mv); b = balance({w: -d for w, d in mv.items()})
    if a and b: pts += [a, b]
for k, a in enumerate(pts[:N]):
    nm = f"dz-{NAME}-{k:02d}"
    with open(f"{D}/mixes/{nm}.mix", "w") as f:
        for w in W:
            if a[w] != BASE: f.write(f"{w} {F}/q38-27b-{a[w]}.rad\n")
        for l in extra: f.write(l + "\n")
    n = sum(a[w] != cur[w] for w in W)
    print(f"a-{nm} {F}/q38-27b-{BASE}.rad {RXM_WORK}/knap/mixes/{nm}.mix")
    print(f"{nm}: {n} tensors moved, d bpw {(size(a) - S0) * 8 / PARAMS:+.4f}", file=sys.stderr)
