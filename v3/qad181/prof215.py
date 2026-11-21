#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""prof215.py OUT [qad181.py args] -- qad181.main() unchanged, plus (w215): PROF215_STEP=k profiles optimizer step k
(torch.profiler, CPU + device) -> OUT/profile-step.txt (top ops by device time), OUT/profile-shapes.txt (by input shape) and OUT/trace-step.json; at exit
OUT/speed215.json: per-step wall, positions / s over the steps after the first (compile / warm-up), peak VRAM."""
import json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402
import qad181  # noqa: E402

PSTEP = int(os.environ.get("PROF215_STEP", "-1"))
walls, rows, poss = [], [], []
_run = qad181.Trainer.run_step


def run_step(self, s, lr):
    dev = self.dev
    torch.cuda.synchronize(dev); t0 = time.time()
    if s == PSTEP:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as P:
            out = _run(self, s, lr); torch.cuda.synchronize(dev)
        ka = P.key_averages()
        key = "self_device_time_total" if hasattr(ka[0], "self_device_time_total") else "self_cuda_time_total"
        open(os.path.join(self.a.out, "profile-step.txt"), "w").write(ka.table(sort_by=key, row_limit=80))
        open(os.path.join(self.a.out, "profile-shapes.txt"), "w").write(
            P.key_averages(group_by_input_shape=True).table(sort_by=key, row_limit=150, max_name_column_width=40, max_shapes_column_width=90))
        P.export_chrome_trace(os.path.join(self.a.out, "trace-step.json"))
    else:
        out = _run(self, s, lr); torch.cuda.synchronize(dev)
    walls.append(time.time() - t0); rows.append(self.step_rows(s))
    poss.append(sum(b - a for st in self.steps[s] for (_, a, b) in st))
    json.dump({"walls": walls, "rows": rows, "pos": poss,
               "pos_s_after_first": (sum(poss[1:]) / sum(walls[1:])) if len(walls) > 1 else None,
               "rows_s_after_first": (sum(rows[1:]) / sum(walls[1:])) if len(walls) > 1 else None,
               "peak_gib": torch.cuda.max_memory_allocated(dev) / 2**30},
              open(os.path.join(self.a.out, "speed215.json"), "w"))
    return out


qad181.Trainer.run_step = run_step

if __name__ == "__main__":
    sys.argv[0] = qad181.__file__
    qad181.main()
