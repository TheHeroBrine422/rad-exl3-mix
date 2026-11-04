#!/usr/bin/env python3
"""sgprep.py TRACES.jsonl OUTDIR [--plen 3072] [--ntrain 2304] [--neval 256] [--seed 0] -- d1108 step 1 (CPU): prompts
for self-generation and raw-trace windows, from the user's agent sessions (pisess.py output; private, our machines).

Sessions are split by a hash of their id: 1 in 5 sessions -> eval, the rest -> train (no session in both).
Prompts (self-generation, arm C): at assistant-turn starts, the session's system block + the most recent context, cut to
exactly PLEN tokens and ending in "<|im_start|>assistant\\n" (every prompt the same length: batched generation with no
padding). Spread evenly over each session's assistant turns, shuffled, NTRAIN train + NEVAL eval.
Raw windows (arm B): 2048-token windows of the train sessions, each with the mask of assistant-turn tokens (what the
other model wrote), at most 8 a session piece, shuffled, NTRAIN of them.
Writes OUTDIR/{prompts-train.npy, prompts-eval.npy, raw-train.npy, raw-train-mask.npy, prep.json}."""
import argparse, hashlib, json, os, random
import numpy as np
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("traces"); ap.add_argument("out"); ap.add_argument("--plen", type=int, default=3072)
ap.add_argument("--ntrain", type=int, default=2304); ap.add_argument("--neval", type=int, default=256)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tokenizer", default=os.environ.get("XCORP_TOKENIZER", "/data/models/qwen38-27b-bf16"))
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.tokenizer)
IM, END = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
ASSIST = tok("assistant\n", add_special_tokens=False)["input_ids"]
SYS = tok("system", add_special_tokens=False)["input_ids"][0]
rng = random.Random(a.seed)
pr = {"train": [], "eval": []}; raw = []
for line in open(a.traces):
    d = json.loads(line); sess = str(d.get("session", ""))
    split = "eval" if int(hashlib.sha1(sess.encode()).hexdigest(), 16) % 5 == 0 else "train"
    ids = tok(d["prompt"], add_special_tokens=False)["input_ids"]
    st = [k for k, x in enumerate(ids) if x == IM]
    sys_end = next((s for s in st if ids[s + 1] != SYS), None)
    if sys_end is None or sys_end > a.plen // 3: continue
    asst = [s for s in st if s > sys_end and ids[s + 1:s + 1 + len(ASSIST)] == ASSIST]
    tail = a.plen - sys_end - 1 - len(ASSIST)
    for s in asst[:: max(1, len(asst) // 48)]:                      # ~48 boundaries a session piece, evenly spread
        if s - sys_end < tail: continue                              # not enough context yet: skip (equal lengths)
        p = ids[:sys_end] + ids[s - tail:s] + [IM] + ASSIST
        assert len(p) == a.plen
        pr[split].append(p)
    if split == "train" and len(ids) >= 2048:                         # raw windows: what the other model wrote
        m = np.zeros(len(ids), np.uint8)
        for s in asst:
            e = next((k for k in range(s + 1 + len(ASSIST), len(ids)) if ids[k] == END), len(ids) - 1)
            m[s + 1 + len(ASSIST):e + 1] = 1                          # the turn's tokens through its <|im_end|>
        starts = list(range(0, len(ids) - 2048 + 1, 2048)); rng.shuffle(starts)
        for w in starts[:8]:
            if m[w + 1:w + 2048].sum() >= 64: raw.append((ids[w:w + 2048], m[w:w + 2048]))
for k in pr: rng.shuffle(pr[k])
rng.shuffle(raw)
os.makedirs(a.out, exist_ok=True)
np.save(f"{a.out}/prompts-train.npy", np.array(pr["train"][:a.ntrain], np.int32))
np.save(f"{a.out}/prompts-eval.npy", np.array(pr["eval"][:a.neval], np.int32))
np.save(f"{a.out}/raw-train.npy", np.array([r[0] for r in raw[:a.ntrain]], np.int32))
np.save(f"{a.out}/raw-train-mask.npy", np.array([r[1] for r in raw[:a.ntrain]], np.uint8))
info = {"plen": a.plen, "train_prompts": min(a.ntrain, len(pr["train"])), "eval_prompts": min(a.neval, len(pr["eval"])),
        "available": {k: len(v) for k, v in pr.items()}, "raw_windows": min(a.ntrain, len(raw)), "raw_available": len(raw)}
json.dump(info, open(f"{a.out}/prep.json", "w"))
print("SGPREP DONE", json.dumps(info))
