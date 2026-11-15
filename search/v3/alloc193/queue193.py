# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""queue193.py ROUND GENDIR... -- RULES-193 amendment 11: merge a round's generated files into my feeder queue
(workers/w172/queue.txt). Drops files whose assignment is already in the run, waiting in the inbox or queued; gives a
file whose name is taken the suffix -rROUND; orders the whole queue round-robin over allocators in the order of
confirmed / decided so far (PRIO), dropped allocators removed."""
import glob, hashlib, itertools, json, os, sys

W = "/workspace/rad-local/workers/w172"; D = "/workspace/rad-local/workers/lead-srch/run/drv"
LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "alloc193", "submitted.jsonl")
PRIO = ["anom", "xplant", "xavg", "anomall", "rankreg", "ens", "smooth", "floor", "tr", "recal", "pavg"]
DROP = {"dxp", "mom", "spread", "cap"}


def h(p):
    d = json.load(open(p)); return hashlib.sha1(json.dumps(d.get("assign", d), sort_keys=True).encode()).hexdigest()


def main(rnd, dirs):
    seen = {h(p) for p in glob.glob(D + "/assign/*.json")} | {h(p) for p in glob.glob(D + "/inbox/*.json")}
    names = {json.loads(l)["name"] for l in open(LEDGER)}
    tok = lambda p: os.path.basename(p).split("-")[0]
    q = []
    for p in [l.strip() for l in open(W + "/queue.txt") if l.strip()]:
        if tok(p) in DROP or not os.path.exists(p): continue
        hh = h(p)
        if hh in seen and not p.endswith("-re.json"): continue
        seen.add(hh); q.append(p)
    qn = {os.path.basename(p) for p in q}; stage = f"{W}/gen/stage-r{rnd}"; os.makedirs(stage, exist_ok=True); n = 0
    for dr in dirs:
        for p in sorted(glob.glob(dr + "/*.json")):
            hh = h(p)
            if hh in seen: continue
            b = os.path.basename(p)
            if "w172-" + b in names or b in qn: b = b[:-5] + f"-r{rnd}.json"
            if "w172-" + b in names or b in qn: continue
            seen.add(hh); s = f"{stage}/{b}"; json.dump(json.load(open(p)), open(s, "w")); q.append(s); qn.add(b); n += 1
    by = {}
    for p in q: by.setdefault(tok(p), []).append(p)
    order = [k for k in PRIO if k in by] + [k for k in by if k not in PRIO]
    fin = [x for t in itertools.zip_longest(*[by[k] for k in order]) for x in t if x]
    tmp = W + "/queue.txt.tmp"; open(tmp, "w").write("".join(p + "\n" for p in fin)); os.replace(tmp, W + "/queue.txt")
    print(f"round {rnd}: {n} new, queue {len(fin)}: " + ", ".join(f"{k} {len(by[k])}" for k in order))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
