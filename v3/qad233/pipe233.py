#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""pipe233.py OUT <qad220 flags> -- w233: qad220.main() with the optimizer steps listed in W233_PROF_STEPS (logged step numbers, comma list) run under
torch.profiler (device activity only): per card the kernel busy time (union of kernel intervals), the step's span, and device time by kernel family
-> OUT/prof233-step<N>[-r<rank>].json. Other steps run unchanged (qad220's own train.jsonl wall / pipe_s)."""
import json, os, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qad220")); sys.path.insert(0, HERE)
import qad220 as T  # noqa: E402
from prof233 import family  # noqa: E402

WANT = {int(x) for x in os.environ.get("W233_PROF_STEPS", "").split(",") if x}
_run_group = T.Trainer.run_group


def busy(iv):
    """total length of the union of intervals [(start, end)] (us)"""
    tot, end = 0.0, -1e30
    for s, e in sorted(iv):
        if e <= end: continue
        tot += e - max(s, end); end = e
    return tot


def run_group(self, ss, lr):
    step = ss[-1] + 1
    if step not in WANT: return _run_group(self, ss, lr)
    from torch.profiler import profile, ProfilerActivity
    t0 = time.time()
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        r = _run_group(self, ss, lr)
        for d_ in dict.fromkeys(self.stu.devs): torch.cuda.synchronize(d_)
    wall = time.time() - t0
    per = {}
    for e in pr.events():
        if e.device_type.name != "CUDA" or e.time_range.elapsed_us() <= 0: continue
        z = per.setdefault(e.device_index, {"iv": [], "fam": {}})
        z["iv"].append((e.time_range.start, e.time_range.end)); f = family(e.name); z["fam"][f] = z["fam"].get(f, 0) + e.time_range.elapsed_us()
    out = {"step": step, "wall_s": wall, "devices": {}}
    for k, z in sorted(per.items()):
        s0 = min(a for a, _ in z["iv"]); s1 = max(b for _, b in z["iv"])
        out["devices"][str(k)] = {"busy_s": busy(z["iv"]) / 1e6, "span_s": (s1 - s0) / 1e6, "kernels": len(z["iv"]),
                                  "family_s": {f: round(v / 1e6, 2) for f, v in sorted(z["fam"].items(), key=lambda x: -x[1])}}
    out["rank"] = self.d.rank
    json.dump(out, open(os.path.join(self.a.out, f"prof233-step{step}" + (f"-r{self.d.rank}" if self.d.world > 1 else "") + ".json"), "w"), indent=1)
    self.log(f"w233 profile step {step}: wall {wall:.1f} s, " + ", ".join(f"card {k}: busy {v['busy_s']:.1f} s of span {v['span_s']:.1f}" for k, v in out["devices"].items()))
    return r


T.Trainer.run_group = run_group
if __name__ == "__main__":
    T.main()
