#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""pull199.py DRIVER.json CONSDIR OUTDIR [--n 12,24] [--sizes ...] -- RULES-sizes199 amendment 4 (11-16): partial pull
toward the cross-size consensus. For each size: the live incumbent's assignment, and the consensus assignment
CONSDIR/<size>.json (consensus192 output). The N trunk tensors whose K differs most from the consensus take the
consensus option; nothing else changes, except a byte rebalance into [size - tol, size): over budget, the unchosen trunk
tensor with the smallest menu err increase per byte saved steps down one option (same family); under the floor, the
one with the largest err decrease per byte added steps up. Head / embedding stay the incumbent's. Writes inbox decide files."""
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import menu as MN  # noqa: E402


def fam(o): return "mad" if o.startswith("q27v3") else ("bf16" if o == "bf16" else o[0] + ("m" if o.endswith("m") else ""))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("conf"); ap.add_argument("cons"); ap.add_argument("out")
    ap.add_argument("--n", default="12,24"); ap.add_argument("--sizes", default="")
    a = ap.parse_args()
    c = json.load(open(a.conf)); M = MN.Menu.load(c["menu"]); R = c["root"]; tol = float(c.get("size_tol", 0.02))
    st = [json.loads(l) for l in open(f"{R}/state.jsonl")]
    inc = {}
    for e in st:
        if e.get("ev") == "incumbent": inc[float(e["size"])] = e["tag"]
    sizes = [float(x) for x in a.sizes.split(",")] if a.sizes else sorted(inc)
    os.makedirs(a.out, exist_ok=True)
    for s in sizes:
        cf = next((p for p in (f"{a.cons}/{s:g}.json", f"{a.cons}/{s:g}-decide.json") if os.path.exists(p)), "")
        if s not in inc or not os.path.exists(cf): continue
        A0 = json.load(open(f"{R}/assign/{inc[s]}.json")); C = json.load(open(cf))["assign"]
        trunk = [w for w in A0 if w.startswith("blk.")]
        dev = []
        for w in trunk:
            if fam(A0[w]) != fam(C[w]): continue
            dk = M.opt(w, C[w])["K"] - M.opt(w, A0[w])["K"]
            if abs(dk) > 1e-9: dev.append((abs(dk), w))
        dev.sort(reverse=True)
        for n in [int(x) for x in a.n.split(",")]:
            A = dict(A0); chosen = {w for _, w in dev[:n]}
            for w in chosen: A[w] = C[w]
            lo, hi = s - tol + 0.003, s - 0.0005
            free = [w for w in trunk if w not in chosen]
            for _ in range(2000):
                b = M.bpw(A)
                if lo <= b <= hi: break
                best = None
                for w in free:
                    t = M.T[M.WI[w]]; opts = [o for o in t["opts"] if fam(o["o"]) == fam(A[w])]
                    j = [o["o"] for o in opts].index(A[w])
                    if b > hi and j > 0:
                        o2 = opts[j - 1]; d = (o2["err"] - opts[j]["err"]) / max(1, opts[j]["bytes"] - o2["bytes"])
                    elif b < lo and j + 1 < len(opts):
                        o2 = opts[j + 1]; d = (o2["err"] - opts[j]["err"]) / max(1, o2["bytes"] - opts[j]["bytes"])
                    else: continue
                    if best is None or d < best[0]: best = (d, w, o2["o"])
                if best is None: break
                A[best[1]] = best[2]
            b = M.bpw(A); ok = lo - 0.003 <= b <= s
            ch = sum(1 for w in A if A[w] != A0[w])
            print(f"pull {s:g} n{n}: {len(chosen)} pulled, {ch} changed, bpw {b:.4f} {'OK' if ok else 'OUT'}")
            if not ok or ch == 0: continue
            json.dump({"size": s, "mode": "decide", "assign": A, "extra_slices": [],
                       "note": f"RULES-sizes199 am. 4: pull n{n} toward consensus ({os.path.basename(a.cons)}) from {inc[s]}"},
                      open(f"{a.out}/pull-{s:g}-n{n}.json", "w"))


if __name__ == "__main__":
    main()
