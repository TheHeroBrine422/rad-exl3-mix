#!/usr/bin/env python3
"""tokenize.py DIR [PAR] -- DIR/*.jsonl (build.py's output) into the challenge golden's token layout, for the proxy
scripts: DIR/meta/tokens.i32 (every document's tokens, concatenated, no BOS) and DIR/meta/kld.json (format
"xcorp-tokens", docs: {source, category, tokens, file, line}; no reference distributions, no gen mask -- the proxy
scripts that read only kld.json docs + tokens.i32, e.g. hcollect.py, take it). Documents in a fixed shuffled order
(seed 0) so any prefix is a mix of every source."""
import glob, json, os, random, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
D = sys.argv[1]; PAR = int(sys.argv[2]) if len(sys.argv) > 2 else 16
TOK = os.environ.get("XCORP_TOKENIZER", "/data/models/qwen38-27b-bf16")
def one(f):
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(TOK); out = []
    for i, l in enumerate(open(f)):
        d = json.loads(l); ids = np.asarray(t(d["prompt"], add_special_tokens=False)["input_ids"], np.int32)
        out.append(({"source": d["source"], "category": d["category"], "tokens": len(ids), "file": os.path.basename(f), "line": i}, ids))
    return out
if __name__ == "__main__":
    files = sorted(glob.glob(f"{D}/*.jsonl"))
    with ProcessPoolExecutor(PAR) as ex: res = [x for r in ex.map(one, files) for x in r]
    random.Random(0).shuffle(res)
    os.makedirs(f"{D}/meta", exist_ok=True)
    np.concatenate([ids for _, ids in res]).tofile(f"{D}/meta/tokens.i32")
    json.dump({"format": "xcorp-tokens", "tokenizer": TOK, "rows": 0, "docs": [d for d, _ in res]}, open(f"{D}/meta/kld.json", "w"))
    print(f"{len(res)} docs, {sum(d['tokens'] for d, _ in res)} tokens")
