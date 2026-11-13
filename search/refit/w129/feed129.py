#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""feed129.py SPECS.jsonl EVALS.jsonl RESULTS.jsonl [--once] -- #169's queue feeder (VM, light; RULES-prior.md).
Every POLL s: asks ds8 (ssh line in $SSH_DS8, default local/bin/ssh-ds8.sh) which ladder levels are final
(tools/curlevels.py) and packed (packs/<level>[-s<scales8>].rad, hostd's name); starts tools/packall.sh 4 there when a
final level has no pack and no packall runs; appends a spec to the coordinator's EVALS only once every level its mix
uses is final and packed (the coordinator then hands it to ds8); deletes a finished w129 mix container
(/data/q27/mixes/<tag>.rad) once all its slices are in RESULTS (disk: ~10-25 GB a mix). Exits when every spec has a
result for every slice (or has failed twice). Log: stdout."""
import json, os, subprocess, sys, time

SSH = os.environ.get("SSH_DS8", "/workspace/rad-local/bin/ssh-ds8.sh")
POLL = int(os.environ.get("POLL", 120))
PROBE = r'''cd /data/q27
cur=$(/data/q27/venv/bin/python tools/curlevels.py)
for l in $cur; do s=$(/data/q27/venv/bin/python -c "import json; print(json.load(open('ladder/$l/DONE')).get('scales') or '')")
  n=packs/$l${s:+-s${s:0:8}}.rad; [ -f $n ] && echo "P $l" || echo "C $l"; done
pgrep -f "^/bin/bash tools/packall.sh" > /dev/null && echo PACKALL
df -BG --output=avail /data | tail -1 | tr -d ' G' | sed 's/^/DF /'
'''


def ssh(cmd, inp=None):
    r = subprocess.run([SSH, "bash", "-s"], input=cmd if inp is None else inp, capture_output=True, text=True, timeout=600)
    return r.stdout


def log(m): print(time.strftime("%H:%M:%S", time.localtime()) + " " + m, flush=True)


def main(specs, evals, results, once=False):
    S = [json.loads(l) for l in open(specs)]
    while True:
        out = ssh(PROBE).split("\n")
        packed = {l[2:] for l in out if l.startswith("P ")}; cur = packed | {l[2:] for l in out if l.startswith("C ")}
        pk_run = "PACKALL" in out; df = next((int(l[3:]) for l in out if l.startswith("DF ")), None)
        if cur - packed and not pk_run:
            ssh("cd /data/q27 && setsid nohup bash tools/packall.sh 4 > /dev/null 2>&1 < /dev/null & echo ok")
            log(f"packall started ({len(cur - packed)} final levels without a pack)")
        queued = set()
        if os.path.exists(evals):
            for l in open(evals):
                try: queued.add(json.loads(l)["tag"])
                except (ValueError, KeyError): pass
        res, fail = {}, {}
        if os.path.exists(results):
            for l in open(results):
                try: r = json.loads(l)
                except ValueError: continue
                if not str(r.get("tag", "")).startswith("w129-"): continue
                if r.get("fail"): fail[r["tag"]] = fail.get(r["tag"], 0) + 1
                else: res.setdefault(r["tag"], set()).add(r["slice"])
        new = []
        for s in S:
            if s["tag"] in queued: continue
            need = set(s["mix"]["levels"].values())
            if need <= packed: new.append(s)
        if new:
            if df is not None and df < 120:
                log(f"HOLD: ds8 /data has {df} GB free (< 120): not queueing {len(new)} more")
            else:
                with open(evals, "a") as f:
                    for s in new: f.write(json.dumps(s) + "\n")
                log(f"queued {len(new)}: {[s['tag'] for s in new]}")
        done = [s["tag"] for s in S if res.get(s["tag"], set()) >= {x["name"] for x in s["slices"]}]
        if done:
            rm = " ".join(f"/data/q27/mixes/{t}.rad" for t in done)
            ssh(f"for f in {rm}; do [ -f $f ] && rm -f $f && echo $f; done", None)
        dead = [s["tag"] for s in S if fail.get(s["tag"], 0) >= 2 and s["tag"] not in done]
        log(f"final {len(cur)} packed {len(packed)} packall {'on' if pk_run else 'off'} df {df} GB | queued {len(queued) + len(new)} "
            f"done {len(done)} / {len(S)} dead {len(dead)}")
        if once or len(done) + len(dead) == len(S): return 0
        time.sleep(POLL)


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4], once="--once" in sys.argv))
