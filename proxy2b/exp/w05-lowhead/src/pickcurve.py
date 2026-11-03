#!/usr/bin/env python3
"""pickcurve.py SIM.json OUT.json [GUARD_X] -- the slab-head curve knap4 prices (KN_HEADCURVE) from one headsim run
(headslab.py plan with HS_FLOOR_ALL=1: per budget B one plan s<B>-f<K> per tail floor, plus uniform plans u<K>).

  anchors: every integer K with a uniform plan u<K> -> its head-only KL (the curve is priced relative to these);
  per budget B: candidates = the floor plans s<B>-f* and a uniform u<B> if that donor exists; admissible if the plan's
  rare-token guard KL <= GUARD_X (default 1.3) x the guard KL of u<lo>, lo = the largest uniform K <= B; pick the
  admissible plan with the lowest KL. Budgets with no admissible plan, or not below the previous pick's KL, are left
  out (written to "dropped" with the reason)."""
import json, sys

sim, out = sys.argv[1:3]; GX = float(sys.argv[3]) if len(sys.argv) > 3 else 1.3
P = json.load(open(sim))["plans"]
uni = {float(n[1:]): r for n, r in P.items() if n.startswith("u")}
anchor = {str(int(k)): r["kl"] for k, r in uni.items() if float(k).is_integer()}
budgets = sorted({float(n[1:].split("-")[0]) for n in P if n.startswith("s")} | {k for k in uni if not float(k).is_integer()})
res, dropped, last = {}, {}, float("inf")
for B in budgets:
    lo = max((k for k in uni if k <= B + 1e-9 and k < B), default=None)
    if lo is None: dropped[f"{B:g}"] = "no uniform plan below"; continue
    gref = uni[lo]["guard_kl"]
    cand = {n: r for n, r in P.items() if n.startswith(f"s{B:g}-")}
    if B in uni: cand[f"u{B:g}"] = uni[B]
    ok = {n: r for n, r in cand.items() if r["guard_kl"] is not None and r["guard_kl"] <= GX * gref}
    if not ok: dropped[f"{B:g}"] = f"guard: none of {sorted(cand)} <= {GX} x u{lo:g}"; continue
    n = min(ok, key=lambda n: ok[n]["kl"]); r = ok[n]
    if r["kl"] >= last: dropped[f"{B:g}"] = f"{n} KL {r['kl']:.6f} not below the previous budget's"; continue
    last = r["kl"]
    res[f"{B:g}"] = {"plan": n, "kl": r["kl"], "guard_kl": r["guard_kl"], "avg_bits": r["avg_bits"], "top1": r["top1"],
                     "vs_u_lo": f"u{lo:g}", "ratio": r["kl"] / uni[lo]["kl"], "guard_ratio": r["guard_kl"] / gref}
    print(f"B {B:<6g} {n:12s} KL {r['kl']:.6f} (x{r['kl'] / uni[lo]['kl']:.3f} u{lo:g}) guard x{r['guard_kl'] / gref:.2f}")
for B, why in dropped.items(): print(f"B {B:<6s} dropped: {why}")
json.dump({"source": sim, "guard_x": GX, "anchor": anchor, "budgets": res, "dropped": dropped}, open(out, "w"), indent=1)
