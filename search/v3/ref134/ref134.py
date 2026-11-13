#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""ref134.py HOSTCONF OUTDIR SLICE=GOLDEN[,SLICE=GOLDEN...] TAG=CONTAINER[@BPW] [...] -- reference containers (V2 at
every size, IQR-V4) scored on the monitor golden and on a search slice for the V3 convergence plot (#171, w134).
ref171.py's procedure (the host's own score_cmd, one free card lock held while it scores, ram.lock shared, run with
PYTHONPATH=the staged pipeline), plus:
- several goldens per container (each scored once; a scored (tag, slice) is skipped: restartable);
- per-part rows for two-part search slices (golden kld.json docs with "part": c / x), beside "m" = all documents;
- @BPW: the bpw to plot at, when the file carries more than the language model (IQR-V4 containers hold the DFlash2
  drafter and the vision tower: the published LM-only bpw is used; "bytes" stays the file size, "bpw_file" its bpw).
Lines in rxrun.multi's format: tag, slice, bytes, bpw, positions, kld, p99, top1, parts, host, id, t."""
import json, os, sys, time

import numpy as np
from rxrun import hostd as HD


def part_stats(golden, rows):
    """per document KL sums -> {"m": stats, part: stats}: KLD = mean over the gen positions of that part's documents"""
    m = json.load(open(f"{golden}/kld.json")); gen = np.fromfile(f"{golden}/gen.u8", np.uint8).astype(bool)
    K = np.fromfile(rows, np.float32).reshape(-1, 4)
    ks = {}; docs = []
    for d in m["docs"]:
        s = slice(d["row0"], d["row0"] + d["tokens"] - 1 - d["score_from"]); g = gen[s]
        k = K[s, 0][g].astype(np.float64); docs.append([round(float(k.sum()), 6), int(g.sum())])
        ks.setdefault("m", []).append(k)
        if d.get("part"): ks.setdefault(d["part"], []).append(k)
    out = {}
    for p, v in ks.items():
        k = np.concatenate(v) if v else np.zeros(0)
        out[p] = {"positions": int(len(k)), "kld": float(k.mean()) if len(k) else None,
                  "p99": float(np.percentile(k, 99)) if len(k) else None}
    return out, docs


def main(hc, od, goldens, specs):
    H = HD.Host(hc); E = H.hc["eval"]; os.makedirs(f"{od}/docs", exist_ok=True)
    gl = [g.split("=", 1) for g in goldens.split(",")]
    res = f"{od}/results.jsonl"
    done = {(j["tag"], j["slice"]) for j in map(json.loads, open(res))} if os.path.exists(res) else set()
    rc_all = 0
    for spec in specs:
        tag, path = spec.split("=", 1); bpw = None
        if "@" in path: path, b = path.rsplit("@", 1); bpw = float(b)
        for sl, golden in gl:
            if (tag, sl) in done: continue
            if not os.path.exists(path): print(f"{tag}: {path} missing", file=sys.stderr); rc_all = 1; break
            o = f"{od}/{tag}-{sl}-m"; log = f"{od}/{tag}-{sl}.log"
            if not os.path.exists(o + ".json.rows"):
                t0 = time.time()
                with HD.CardLock(E) as cl, HD.Lock(E.get("ram_lock"), False):
                    rc = HD.sh(HD.fill(E["score_cmd"], mix=path, golden=golden, out=o, card=cl.card), log, nice=0)
                if rc or not os.path.exists(o + ".json.rows"):
                    print(f"{tag} {sl}: score rc {rc} ({log})", file=sys.stderr, flush=True); rc_all = 1; continue
                print(f"{tag} {sl}: scored in {time.time() - t0:.0f} s", flush=True)
            j = json.load(open(o + ".json"))["all"]; parts, docs = part_stats(golden, o + ".json.rows")
            parts["m"]["top1"] = j.get("top1_agreement"); parts["m"]["positions_all"] = j.get("positions")
            nb = os.path.getsize(path); bf = nb * 8 / HD.PARAMS
            r = {"tag": tag, "slice": sl, "bytes": nb, "positions": parts["m"]["positions"], "kld": parts["m"]["kld"],
                 "p99": parts["m"]["p99"], "top1": parts["m"]["top1"], "parts": parts, "bpw": bpw if bpw else bf,
                 "bpw_file": bf, "host": H.name, "id": f"ref134-{tag}-{sl}", "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            json.dump({"m": docs}, open(f"{od}/docs/{tag}-{sl}.json", "w"))
            with open(res, "a") as f: f.write(json.dumps(r) + "\n")
            print(f"{tag}: KLD {r['kld']:.5f} at {r['bpw']:.4f} bpw on {sl} "
                  + " ".join(f"{p} {v['kld']:.5f}" for p, v in parts.items() if p != "m"), flush=True)
    return rc_all


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]))
