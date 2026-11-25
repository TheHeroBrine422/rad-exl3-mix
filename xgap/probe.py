#!/usr/bin/env python3
"""probe.py REPO CONFIG|- SPLIT [DATA_FILE_GLOB] -- print the first record's fields (truncated) of a streamed HF dataset."""
import sys, json
from datasets import load_dataset
repo, cfg, split = sys.argv[1:4]
kw = dict(split=split, streaming=True)
if len(sys.argv) > 4: kw["data_files"] = {split: sys.argv[4]}
ds = load_dataset(repo, None if cfg == "-" else cfg, **kw)
for i, r in enumerate(ds):
    print(repo, {k: (str(v)[:150] + (f"...[{len(str(v))}]" if len(str(v)) > 150 else "")) for k, v in r.items()})
    if i >= 1: break
