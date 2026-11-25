#!/usr/bin/env python3
"""stackedu_langs.py OUT.json -- which language each common-pile/stackv2_edu_filtered shard holds (first record of each shard)."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
def head(i):
    f = f"stack-edu-{i:04d}.json.gz"
    try:
        ds = load_dataset("common-pile/stackv2_edu_filtered", split="train", streaming=True, data_files={"train": f})
        return f, next(iter(ds))["metadata"]["language"]
    except Exception as e: return f, "ERR " + str(e)[:80]
with ThreadPoolExecutor(8) as ex: res = dict(ex.map(head, range(97)))
json.dump(res, open(sys.argv[1], "w"), indent=0)
import collections; print(collections.Counter(res.values()))
