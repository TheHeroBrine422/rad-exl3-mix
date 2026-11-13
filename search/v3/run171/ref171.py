#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""ref171.py HOSTCONF SLICE GOLDEN OUTDIR TAG=CONTAINER [...] -- the monitor's V2 references (#171): each container
scored once on the monitor golden with the host's own eval.score_cmd, card locks and per-document sums (rxrun.hostd's
functions; run with PYTHONPATH=the staged pipeline), because a hostd eval needs trunk levels and cannot score a bare
container. Writes OUTDIR/results.jsonl (rxrun.multi's line format: tag, slice, bytes, bpw, positions, kld, p99, top1,
parts, host, id, t) and OUTDIR/docs/<tag>-<slice>.json; the VM appends the lines to the coordinator's results.jsonl
(before it starts), where convplot3 finds them (--ref-re ^ref-V2-). Restartable: a scored tag is skipped."""
import json, os, sys, time

import numpy as np
from rxrun import hostd as HD


def main(hc, sl, golden, od, specs):
    H = HD.Host(hc); E = H.hc["eval"]; os.makedirs(f"{od}/docs", exist_ok=True)
    done = {json.loads(l)["tag"] for l in open(f"{od}/results.jsonl")} if os.path.exists(f"{od}/results.jsonl") else set()
    for spec in specs:
        tag, path = spec.split("=", 1)
        if tag in done: continue
        o = f"{od}/{tag}-{sl}-m"; log = f"{od}/{tag}.log"
        if not os.path.exists(o + ".json.rows"):
            with HD.CardLock(E) as cl, HD.Lock(E.get("ram_lock"), False):
                rc = HD.sh(HD.fill(E["score_cmd"], mix=path, golden=golden, out=o, card=cl.card), log, nice=0)
            if rc or not os.path.exists(o + ".json.rows"): print(f"{tag}: score rc {rc} ({log})", file=sys.stderr); return 1
        j = json.load(open(o + ".json"))["all"]; docs, k = HD.doc_sums(golden, o + ".json.rows")
        nb = os.path.getsize(path)
        part = {"positions": int(len(k)), "kld": float(k.mean()), "p99": float(np.percentile(k, 99)), "top1": j.get("top1_agreement"),
                "positions_all": j.get("positions")}
        r = {"tag": tag, "slice": sl, "bytes": nb, "positions": int(len(k)), "kld": float(k.mean()), "p99": part["p99"], "top1": part["top1"],
             "parts": {"m": part}, "bpw": nb * 8 / HD.PARAMS, "host": H.name, "id": f"ref171-{tag}", "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        json.dump({"m": docs}, open(f"{od}/docs/{tag}-{sl}.json", "w"))
        with open(f"{od}/results.jsonl", "a") as f: f.write(json.dumps(r) + "\n")
        print(f"{tag}: KLD {r['kld']:.5f} at {r['bpw']:.4f} bpw on {sl}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]))
