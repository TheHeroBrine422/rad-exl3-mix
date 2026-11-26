# w233: per-step wall (trainer's run_group wall) vs log-timestamp step-to-step elapsed, pipe_s averages
import json, sys, re, datetime
for d in sys.argv[1:]:
    ts = []; recs = []
    for ln in open(f"{d}/run.log"):
        m = re.match(r"(\d\d):(\d\d):(\d\d) (\{.*\"what\": \"main\".*)", ln)
        if m:
            t = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]); r = json.loads(m[4]); ts.append(t); recs.append(r)
    if len(recs) < 3: print(d, "few", len(recs)); continue
    el = [(b - a) % 86400 for a, b in zip(ts, ts[1:])]
    w = [r["wall"] for r in recs[1:]]
    pos = [sum(r["n"].values()) for r in recs[1:]]
    k = len(el)
    print(f"{d}: {k} steps, mean wall {sum(w)/k:.1f} s, mean elapsed {sum(el)/k:.1f} s (overhead {100*(sum(el)-sum(w))/sum(el):.1f}%), rows/elapsed-s {sum(pos)/sum(el):.0f}")
    ps = [r.get("pipe_s") for r in recs[1:] if r.get("pipe_s")]
    if ps: print("   pipe_s mean:", {k_: round(sum(p[k_] for p in ps) / len(ps), 1) for k_ in ps[0]})
    rw = [r.get("rank_wall") for r in recs[1:] if r.get("rank_wall")]
    if rw: print("   rank_wall mean:", [round(sum(x[i] for x in rw)/len(rw),1) for i in range(len(rw[0]))])
    pt = [r.get("prefix_tok", 0) for r in recs[1:]]; print("   prefix_tok total", sum(pt))
