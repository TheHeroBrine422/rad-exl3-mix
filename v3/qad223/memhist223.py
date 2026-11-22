#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""memhist223.py (w223) -- where the allocated-memory peak of a recorded run comes from: replays the allocator trace of
OUT/memhist.pickle (a8int8.py's memhist.request hook = torch.cuda.memory._dump_snapshot), finds the moment of the largest
allocated total within the trace and prints the biggest live blocks then, each with its python frames (innermost few), plus
the largest single allocations of the whole trace.
  python memhist223.py OUT/memhist.pickle [--top 25] [--frames 6]"""
import argparse, pickle


def fr(e, k):
    out = []
    for f in e.get("frames", []):
        fn = f.get("filename", "")
        if not fn.endswith(".py"): continue                       # C++ frames (unwind, allocator) first: keep python
        if any(s in fn for s in ("torch/utils/_", "torch/autograd/graph", "<frozen", "torch/_ops", "contextlib")): continue
        out.append(f"{fn.rsplit('/', 2)[-2] + '/' + fn.rsplit('/', 1)[-1] if '/' in fn else fn}:{f.get('line')} {f.get('name')}")
        if len(out) >= k: break
    return " <- ".join(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("snap"); ap.add_argument("--top", type=int, default=25); ap.add_argument("--frames", type=int, default=6)
    a = ap.parse_args()
    s = pickle.load(open(a.snap, "rb"))
    tr = [e for d in s["device_traces"] for e in d]
    live, cur, peak, at, best = {}, 0, 0, -1, None
    for i, e in enumerate(tr):
        act = e["action"]
        if act == "alloc":
            live[e["addr"]] = e; cur += e["size"]
            if cur > peak: peak, at, best = cur, i, dict(live)
        elif act in ("free_completed",) and e["addr"] in live:
            cur -= live.pop(e["addr"])["size"]
    print(f"{len(tr)} trace events; peak of the traced allocations {peak / 2**30:.2f} GiB above the trace start at event {at} "
          f"(allocations alive before recording are not counted)")
    print(f"\n## live blocks at the peak (top {a.top} by size)")
    for e in sorted(best.values(), key=lambda e: -e["size"])[:a.top]:
        print(f"{e['size'] / 2**20:9.1f} MiB  {fr(e, a.frames)}")
    agg = {}
    for e in best.values():
        k = fr(e, 2) or "(no python frames: C++ / autograd engine thread)"
        n, t = agg.get(k, (0, 0)); agg[k] = (n + 1, t + e["size"])
    print(f"\n## live at the peak, summed by the two innermost python frames (top {a.top})")
    for k, (n, t) in sorted(agg.items(), key=lambda x: -x[1][1])[:a.top]:
        print(f"{t / 2**20:9.1f} MiB  {n:5d} blocks  {k}")
    print(f"\n## the largest allocations of the whole trace")
    for e in sorted((e for e in tr if e["action"] == "alloc"), key=lambda e: -e["size"])[:a.top]:
        print(f"{e['size'] / 2**20:9.1f} MiB  {fr(e, a.frames)}")


if __name__ == "__main__":
    main()
