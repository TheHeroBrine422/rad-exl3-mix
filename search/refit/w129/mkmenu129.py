#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""mkmenu129.py DIAGSUM.jsonl MKMENU_ARGS... -- search/v3/mkmenu.py with the --diag level's diag files replaced by their
diagsum.py summary (R, E1, E2 per HF key): the ladder's out/ dirs (diag files) were deleted on ds8 on 11-13 to free
disk; R is level-independent and h̄ = E2 / E1 is what mkmenu derives from the same sums (#169, w129)."""
import glob, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "v3"))
import mkmenu as MK  # noqa: E402

by = {}
for l in open(sys.argv[1]):
    r = json.loads(l); by[r["key"]] = r
_glob = glob.glob
glob.glob = lambda p, *a, **k: sorted(by) if "/out/*/diag-" in p else _glob(p, *a, **k)
MK.DS.summarize = lambda k: {"key": k, "R": by[k]["R"], "E1": by[k]["E1"], "E2": by[k]["E2"]}
sys.exit(MK.main(sys.argv[2:]))
