#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""feed129.py SPECS.jsonl EVALS.jsonl RESULTS.jsonl [--once] -- #169's queue feeder (VM, light; RULES-prior.md).
Every POLL s: asks ds8 (ssh line in $SSH_DS8, default local/bin/ssh-ds8.sh) which ladder levels are final
(tools/curlevels.py) and packed (packs/<level>[-s<scales8>].rad, hostd's name); starts w129/pack129.sh 3 there for the
final levels the specs use that have no pack, when no packall / pack129 runs; appends a spec to the coordinator's EVALS only once every level its mix
uses is final and packed (the coordinator then hands it to ds8); deletes a finished w129 mix container
(/data/q27/mixes/<tag>.rad) once all its slices are in RESULTS (disk: ~10-25 GB a mix). Exits when every spec has a
result for every slice (or has failed twice). Log: stdout."""
import json, os, subprocess, sys, time

SSH = os.environ.get("SSH_DS8", "/workspace/rad-local/bin/ssh-ds8.sh")
POLL = int(os.environ.get("POLL", 120))
PACK_MIN_GB = int(os.environ.get("PACK_MIN_GB", 50))          # ds8 /data free space needed to start a pack (~9-19 GB each)
QUEUE_MIN_GB = int(os.environ.get("QUEUE_MIN_GB", 80))        # ... to queue more evals (a mix container is ~9-25 GB)
HOLD_GB = int(os.environ.get("HOLD_GB", 60))                  # below: touch the coordinator's ROOT/HOLD (no new units)
RETRIES = 3                                                   # re-queues of a dead spec as TAG.rN (11-13: disk-full refusals)
import re
base_tag = lambda t: re.sub(r"\.r\d+$", "", t)
PROBE = r'''cd /data/q27
cur=$(/data/q27/venv/bin/python tools/curlevels.py)
for l in $cur; do s=$(/data/q27/venv/bin/python -c "import json; print(json.load(open('ladder/$l/DONE')).get('scales') or '')")
  n=packs/$l${s:+-s${s:0:8}}.rad; [ -f $n ] && echo "P $l" || echo "C $l"; done
pgrep -f "packall[.]sh|pack129[.]sh" > /dev/null && echo PACKALL
df -BG --output=avail /data | tail -1 | tr -d ' G' | sed 's/^/DF /'
'''


def ssh(cmd, inp=None, timeout=600):
    try: r = subprocess.run([SSH, "bash", "-s"], input=cmd if inp is None else inp, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired: return ""                       # the packall start keeps the session open: let it go
    return r.stdout


def log(m): print(time.strftime("%H:%M:%S", time.localtime()) + " " + m, flush=True)


def main(specs, evals, results, once=False):
    S = [json.loads(l) for l in open(specs)]
    while True:
        out = ssh(PROBE).split("\n")
        packed = {l[2:] for l in out if l.startswith("P ")}; cur = packed | {l[2:] for l in out if l.startswith("C ")}
        pk_run = "PACKALL" in out; df = next((int(l[3:]) for l in out if l.startswith("DF ")), None)
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
                if r.get("fail"): fail[r["tag"]] = fail.get(r["tag"], 0) + 1      # one fail line per dead unit
                else: res.setdefault(base_tag(r["tag"]), set()).add(r["slice"])
        use = {}                                                      # level -> number of PENDING specs using it (most-used first)
        for s in S:
            if s["tag"] in res and res[s["tag"]] >= {x["name"] for x in s["slices"]}: continue
            for l in set(s["mix"]["levels"].values()): use[l] = use.get(l, 0) + 1
        need = sorted(set(use) & (cur - packed), key=lambda l: -use[l])
        if need and not pk_run and df is not None and df >= PACK_MIN_GB:   # one pack at a time, only with room (pack129.sh)
            ssh(f"cd /data/q27 && setsid nohup bash w129/pack129.sh 1 {need[0]} > /dev/null 2>&1 < /dev/null & echo ok", timeout=20)
            log(f"pack129 started: {need[0]} ({len(need)} final levels without a pack, df {df} GB)")
        hold = os.path.join(os.path.dirname(evals), "HOLD")
        if df is not None and df < HOLD_GB and not os.path.exists(hold): open(hold, "w").write(f"feed129: ds8 df {df} GB\n"); log(f"HOLD on (df {df} GB)")
        elif df is not None and df >= QUEUE_MIN_GB and os.path.exists(hold) and open(hold).read().startswith("feed129"): os.remove(hold); log(f"HOLD off (df {df} GB)")
        new = []
        for s in S:
            if s["tag"] in res and res[s["tag"]] >= {x["name"] for x in s["slices"]}: continue
            tags = [s["tag"]] + [f"{s['tag']}.r{k}" for k in range(1, RETRIES + 1)]
            q = [t for t in tags if t in queued]
            if q and not fail.get(q[-1]): continue                  # queued, not dead
            if len(q) == len(tags): continue                        # out of retries
            need = set(s["mix"]["levels"].values())
            if need <= packed: new.append(dict(s, tag=tags[len(q)]))
        if new:
            if df is not None and df < QUEUE_MIN_GB:
                log(f"HOLD: ds8 /data has {df} GB free (< {QUEUE_MIN_GB}): not queueing {len(new)} more"); new = []
            else:
                with open(evals, "a") as f:
                    for s in new: f.write(json.dumps(s) + "\n")
                log(f"queued {len(new)}: {[s['tag'] for s in new]}")
        done = [s["tag"] for s in S if res.get(s["tag"], set()) >= {x["name"] for x in s["slices"]}]
        if done:
            rm = " ".join(f"/data/q27/mixes/{t}{x}.rad" for t in done for x in [""] + [f".r{k}" for k in range(1, RETRIES + 1)])
            ssh(f"for f in {rm}; do [ -f $f ] && rm -f $f && echo $f; done", None)
        dead = [s["tag"] for s in S if fail.get(f"{s['tag']}.r{RETRIES}") and s["tag"] not in done]
        log(f"final {len(cur)} packed {len(packed)} packall {'on' if pk_run else 'off'} df {df} GB | queued {len(queued) + len(new)} "
            f"done {len(done)} / {len(S)} dead {len(dead)}")
        if once or len(done) + len(dead) == len(S): return 0
        time.sleep(POLL)


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4], once="--once" in sys.argv))
