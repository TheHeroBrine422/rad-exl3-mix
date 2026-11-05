#!/usr/bin/env python3
"""sgserve.py PROMPTS.npy OUTDIR --url URL [--limit N] [--conc 16] [--max-new 768] [--batch 16] -- d1108 arm D (27B):
the served model's own continuations of sgprep.py's prompts (token ids, decoded to text and sent as raw
/v1/completions prompts, which end in "<|im_start|>assistant\\n"), with the Qwen family's sampling (T 1.0, top_p
0.95, top_k 20), stopping at <|im_end|>. Writes sggen.py's format (OUTDIR/b<i>.npz: gen [B, max_new] int32 padded
with -1, len, finished; the completion re-tokenized, <|im_end|> appended when it stopped there), so sgbuild.py reads
it unchanged. Resumable per batch. Private input: our machines only."""
import argparse, json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("prompts"); ap.add_argument("out"); ap.add_argument("--url", required=True)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--conc", type=int, default=16)
ap.add_argument("--max-new", type=int, default=768); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--tokenizer", default=os.environ.get("XCORP_TOKENIZER", "/data/workspace/models/qwen38-27b-bf16"))
a = ap.parse_args()
P = np.load(a.prompts); P = P[:a.limit] if a.limit else P
tok = AutoTokenizer.from_pretrained(a.tokenizer); END = tok.convert_tokens_to_ids("<|im_end|>")
os.makedirs(a.out, exist_ok=True)

def one(i):
    body = {"prompt": tok.decode(P[i].tolist()), "max_tokens": a.max_new, "temperature": 1.0, "top_p": 0.95,
            "top_k": 20, "stop": ["<|im_end|>"], "seed": 1000003 + int(i)}
    for t in range(5):
        try:
            r = urllib.request.Request(a.url.rstrip("/") + "/v1/completions", json.dumps(body).encode(),
                                       {"Content-Type": "application/json"})
            c = json.loads(urllib.request.urlopen(r, timeout=1800).read())["choices"][0]
            ids = tok(c["text"], add_special_tokens=False)["input_ids"][:a.max_new]
            fin = c.get("finish_reason") == "stop"
            if fin and len(ids) < a.max_new: ids = ids + [END]
            return ids, fin
        except Exception as e:
            print(f"request {i} attempt {t}: {e}", flush=True)
            if "400" in str(e): body.pop("seed", None); body.pop("top_k", None)   # a server without these fields
            time.sleep(10)
    return [], False

t0, ntok = time.time(), 0
with ThreadPoolExecutor(a.conc) as ex:
    for b in range(0, len(P), a.batch):
        f = f"{a.out}/b{b // a.batch:05d}.npz"
        if os.path.exists(f): continue
        res = list(ex.map(one, range(b, min(b + a.batch, len(P)))))
        G = np.full((len(res), a.max_new), -1, np.int32); L = np.zeros(len(res), np.int32); F = np.zeros(len(res), bool)
        for j, (ids, fin) in enumerate(res): G[j, :len(ids)] = ids; L[j] = len(ids); F[j] = fin
        np.savez(f + ".part.npz", gen=G, len=L, finished=F); os.replace(f + ".part.npz", f)
        ntok += int(L.sum())
        print(f"sgserve: batch {b // a.batch}: {int(L.sum())} tokens, {F.mean():.2f} finished, {ntok / (time.time() - t0):.0f} tok/s", flush=True)
print("SGSERVE DONE", flush=True)
