# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""t183_193.py [SIZES] -- #193 report only (RULES-193 §0: t183 / t186n are never training, selection or tuning data):
the tracker's whole-slice KLD (trk183-* / trk186-* results) of each size's incumbents in order, with the candidate each
incumbent came from; my inbox candidates are marked with their file name. Read-only."""
import hashlib, json, math, sys

RUN = "/workspace/rad-local/workers/lead-srch/run"


def main(sizes):
    c = json.load(open(f"{RUN}/driver.json")); st = [json.loads(l) for l in open(f"{c['root']}/state.jsonl")]
    trk = {}
    for l in open(c["results"]):
        if '"trk18' not in l: continue
        r = json.loads(l); t = r["tag"]; k = r.get("kld") or (r.get("r") or {}).get("kld")
        if k: trk[t] = k
    sub = {e["tag"]: e for e in st if e.get("ev") == "submit"}; mine = {e["tag"]: e["file"] for e in st if e.get("ev") == "inbox"}
    byhash = {}
    for t, e in sub.items():
        if e["kind"] not in ("incumbent", "confirm"): byhash.setdefault(e["hash"], t)
    for s in sizes:
        print(f"== {s:g}"); last = {}; seenh = None
        for e in st:
            if e.get("ev") != "incumbent" or float(e["size"]) != s: continue
            t = e["tag"]; h = sub[t]["hash"] if t in sub else None
            src = byhash.get(h, t); row = [e.get("t", ""), t, "from " + src + (f" [{mine[src]}]" if src in mine else "")]
            for p in ("trk183", "trk186"):
                k = trk.get(f"{p}-{t}")
                if k:
                    row.append(f"{p[3:]} {k:.5f}" + (f" ({math.log(k / last[p]) * 100:+.2f} % vs prev read)" if p in last and h != seenh else "")); last[p] = k
            if len(row) > 3 or src in mine: print("  " + " | ".join(row))
            seenh = h


if __name__ == "__main__":
    main([float(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "2,2.5,3,3.5,4,4.5,5,6").split(",")])
