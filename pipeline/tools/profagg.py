#!/usr/bin/env python3
"""profagg.py LEVELDIR... -- per-level stage profile from rxq job logs (<level>/*/log.jsonl, rxq.stages): per family
mean wall s per tensor and the level's summed card time, and the stage shares over the whole level."""
import glob, json, sys, collections
for L in sys.argv[1:]:
    fam = collections.defaultdict(list); st = collections.Counter(); wall = 0.0; K = None; n = 0
    for f in glob.glob(f"{L}/*/log.jsonl"):
        for line in open(f):
            r = json.loads(line); x = r.get("rxq") or {}
            if "wall" not in x: continue
            K = r.get("K", K); n += 1; wall += x["wall"]; fam[x.get("family", "?")].append(x["wall"])
            for k, v in (x.get("stages") or {}).items(): st[k] += v
    if not n: print(f"{L}: no records"); continue
    tot = sum(st.values()) or 1
    grp = collections.Counter()
    for k, v in st.items():
        g = ("trellis" if k.startswith("trellis") else "ldlq+hout" if k in ("ldlq", "prepare_hout", "hout_load") else
             "refit" if k == "refit" else "c9/post" if "c9" in k else "prep" if k in ("finalize", "regularize", "scaled_ldl", "scale_search") else "load/other")
        grp[g] += v
    print(f"== {L}  K{K}  {n} tensors  sum wall {wall/3600:.2f} h")
    print("   per tensor (s): " + "  ".join(f"{k} {sum(v)/len(v):.1f}" for k, v in sorted(fam.items())))
    print("   shares: " + "  ".join(f"{k} {v/tot*100:.0f}%" for k, v in grp.most_common()))
