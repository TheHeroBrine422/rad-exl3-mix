#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""calsubmit.py PLAN.json -- the embedding calibration evals (w135, #174) through the running coordinator, one at a time
(one slot): each = an incumbent's eval line (evals.jsonl) with token_embd.weight from an embedding donor, scored on the
incumbent's own slice, whole (no early stop). The driver ignores the tags (not its submissions); fit rows are built
afterwards (calobs.py). Waits for each result before the next; skips tags already in results.jsonl (restartable); waits
for a donor to exist on the host (ssh test -f) before submitting it.
PLAN: {"evals": EVALS, "results": RESULTS, "host": [ssh argv...], "runs": [{"tag", "of", "emb", "donor"}, ...]}"""
import json, os, subprocess, sys, time

CST = -6 * 3600


def log(m): print(time.strftime("%m-%d %H:%M:%S", time.gmtime(time.time() + CST)) + " CST " + m, flush=True)


def results(p):
    out = {}
    for l in open(p):
        try: r = json.loads(l)
        except ValueError: continue
        out[r.get("tag")] = r
    return out


def line_of(p, tag):
    last = None
    for l in open(p):
        if f'"tag": "{tag}"' in l:
            r = json.loads(l)
            if r["tag"] == tag: last = r
    return last


def main(argv):
    P = json.load(open(argv[0]))
    for run in P["runs"]:
        t = run["tag"]
        if t in results(P["results"]): log(f"{t}: already scored"); continue
        if line_of(P["evals"], t) is None:
            while subprocess.run(P["host"] + ["test", "-f", run["donor"]]).returncode != 0:
                log(f"{t}: waiting for {run['donor']}"); time.sleep(120)
            src = line_of(P["evals"], run["of"])
            if src is None: raise SystemExit(f"{run['of']}: no eval line")
            line = {"tag": t, "mix": dict(src["mix"], donors=dict(src["mix"]["donors"], **{"token_embd.weight": run["donor"]})),
                    "slices": src["slices"]}
            with open(P["evals"], "a") as f: f.write(json.dumps(line) + "\n")
            log(f"{t}: submitted ({run['of']} + {run['emb']} on {src['slices'][0]['name']})")
        while t not in results(P["results"]): time.sleep(30)
        r = results(P["results"])[t]
        log(f"{t}: {'FAILED ' + str(r.get('err')) if r.get('fail') else 'KLD %.5f, %s bytes' % (r['kld'], r.get('bytes'))}")
    log("all done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
