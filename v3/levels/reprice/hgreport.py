#!/usr/bin/env python3
"""report.py OUTDIR [MODEL.json] -- w88 head-grid solves vs today's menu: KLD, head pick, head bytes moved to / from the trunk."""
import json, sys, os, collections
D = sys.argv[1]; V, H = 248320, 5120
MD = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None   # kfit2-model-final.json (menu head cost; q ~ 1)
def ld(n):
    p = f"{D}/{n}.jsonl"; return json.loads(open(p).readline()) if os.path.exists(p) else None
def trunk_k(r):   # parameter-weighted mean K over trunk picks (option name -> K)
    import re
    return None
for var, arms, lam in (("acc", ["menu", "grid", "gridmean", "gridg1"], None), ("f", ["menu", "grid"], "3e-07")):
    print(f"== {'accurate (f16, lambda 0)' if var == 'acc' else 'fast (a8-g128 allowed, lambda 3e-7)'}")
    print(f"{'size':>5} {'arm':9} {'bpw':>6} {'KLD':>9} {'vs menu':>8} {'head pick':>14} {'head bits':>9} {'head cost':>9}  emb    plan")
    for s in ("2", "2.5", "3", "4"):
        base = None
        for a in arms:
            r = ld(f"{a}-{var}-{s}" + (f"-l{lam}" if lam else ""))
            if r is None: continue
            hp = r["picks"]["output.weight"][0]
            hb = r.get("head", {}).get("bits") or float(hp.split("K")[-1])
            if base is None: base = r
            hc = r.get("head", {}).get("cost")
            if hc is None and MD:                          # menu arm: the head's model cost from MODEL.json
                t = [t for t in MD["tensors"] if t["w"] == "output.weight"][0]; c = hp.replace("slab", "")
                import math
                hc = t["s"] * math.exp(MD["phi"]["ours_head"]) * t["opts"][c][1] * (0.69 if hp == "ours-slabK5" else 0.71 if hp == "ours-slabK6" else 1.0)
            print(f"{s:>5} {a:9} {r['bpw']:6.3f} {r['kld']:9.6f} {r['kld'] / base['kld'] - 1:+8.2%} {hp:>14} {hb:9.4g} "
                  f"{'' if hc is None else f'{hc:9.2e}'}  {r['emb']:6} {r.get('head', {}).get('plan') or ''}")
            if a != "menu":
                ch = [w for w in r["picks"] if r["picks"][w] != base["picks"][w] and w != "output.weight"]
                def kk(c):
                    try: return float(c.split("K")[-1])
                    except ValueError: return None
                up = sum(1 for w in ch if None not in (kk(r["picks"][w][0]), kk(base["picks"][w][0])) and kk(r["picks"][w][0]) > kk(base["picks"][w][0]))
                dn = sum(1 for w in ch if None not in (kk(r["picks"][w][0]), kk(base["picks"][w][0])) and kk(r["picks"][w][0]) < kk(base["picks"][w][0]))
                hb0 = float(base["picks"]["output.weight"][0].split("K")[-1])
                print(f"{'':>5} {'':9} head {hb - hb0:+.4g} bit ({(hb - hb0) * V * H / 8 / 2**20:+.0f} MiB); trunk picks changed {len(ch)} "
                      f"({up} up, {dn} down, {len(ch) - up - dn} other family); emb {base['emb']} -> {r['emb']}")
