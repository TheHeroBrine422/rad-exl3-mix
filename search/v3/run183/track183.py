#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""track183.py DRIVER.json SLICE GOLDEN [--once] -- #183 tracking slice (user 11-14: a fixed slice the search never
decides on, so progress reads as raw KL on the same documents). Every POLL s: each "incumbent" event in the driver's
state.jsonl (the first and then every TRACK_EVERY-th (5) of each size: user 11-14) whose tag has no tracking eval yet gets one eval line (tag trk183-<tag>, the incumbent's assignment, the
driver's base and act mode, the whole SLICE golden, no early stop) appended to the coordinator's evals.jsonl. Results
land in the coordinator's results.jsonl like any eval; nothing reads them back into the search."""
import json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import menu as MN  # noqa: E402

EVERY = int(os.environ.get("TRACK_EVERY", 5))


def main(conf, slice_name, golden, once=False):
    c = json.load(open(conf)); M = MN.Menu.load(c["menu"]); R = c["root"]; act = c.get("act") or "f16"
    done = set()
    if os.path.exists(c["evals"]):
        done = {json.loads(l)["tag"] for l in open(c["evals"]) if l.strip() and '"trk183-' in l}
    while True:
        new = []; nth = {}
        for l in open(f"{R}/state.jsonl"):
            e = json.loads(l)
            if e.get("ev") != "incumbent": continue
            k = nth[e["size"]] = nth.get(e["size"], -1) + 1           # (user 11-14: every EVERY-th incumbent of a size)
            if k % EVERY: continue
            t = f"trk183-{e['tag']}"
            if t in done or not os.path.exists(f"{R}/assign/{e['tag']}.json"): continue
            mx = M.eval_mix(json.load(open(f"{R}/assign/{e['tag']}.json")), c["base"])
            if act != "f16": mx["act"] = act
            new.append({"tag": t, "mix": mx, "slices": [{"name": slice_name, "parts": {"m": golden}}],
                        "track": {"of": e["tag"], "size": e["size"]}})
            done.add(t)
        if new:
            with open(c["evals"], "a") as f:
                for x in new: f.write(json.dumps(x) + "\n")
            print(time.strftime("%H:%M:%S"), "queued", " ".join(x["tag"] for x in new), flush=True)
        if once: return
        time.sleep(60)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(a[0], a[1], a[2], "--once" in sys.argv)
