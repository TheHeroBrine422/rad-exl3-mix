#!/usr/bin/env python3
"""mktext.py OUT -- the format-matched text-only control for #182 (w156): 120 prompts from databricks-dolly-15k
(CC BY-SA 3.0, pinned revision), seed 182, stratified over its categories; context (if any) before the instruction.
OUT/text-items.jsonl in mkset.py's form (images: [])."""
import json, random, sys
from collections import defaultdict
from huggingface_hub import hf_hub_download, HfApi
OUT = sys.argv[1]
rev = HfApi().dataset_info("databricks/databricks-dolly-15k").sha
p = hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset", revision=rev)
by = defaultdict(list)
for i, l in enumerate(open(p)):
    d = json.loads(l); d["i"] = i; by[d["category"]].append(d)
rng = random.Random(182); out = []
cats = sorted(by)
per = 120 // len(cats)
for c in cats:
    rs = by[c][:]; rng.shuffle(rs)
    for d in rs[:per]:
        q = (d["context"].strip() + "\n\n" if d["context"].strip() else "") + d["instruction"].strip()
        out.append({"id": f"dolly-{d['i']}", "kind": "text", "source": "dolly", "licence": f"databricks-dolly-15k CC BY-SA 3.0 (rev {rev[:10]})",
                    "images": [], "question": q, "category": c})
with open(f"{OUT}/text-items.jsonl", "w") as f:
    for it in out: f.write(json.dumps(it) + "\n")
print(len(out), rev)
