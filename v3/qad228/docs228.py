#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""docs228.py RENDER GEN OUT_PREFIX [--tok DIR] [--cap 8192] [--max-gen 6000] -- w228 step 4: the 8k training documents of the on-policy
generations (gen228.py's GEN/gen/<cut>.json), as a radiance --kld-corpus for the bf16 golden.

Per cut: [the session's rendered system turn (tools block; tokens 0 .. the second <|im_start|>)] + [the prefix's tail from the
earliest turn start q such that system + tokens q..p + the generation prompt + the generation <= --cap] + the generation prompt
(`<|im_start|>assistant\\n<think>\\n`) + the generated ids (ending in <|im_end|> unless the turn hit n_predict). Dropped: a generation
over --max-gen tokens, or a last turn so long that no turn start fits.
The corpus is text (radiance --kld-corpus tokenises `prompt` with control tokens): the document is re-tokenised here with the model's
tokenizer (radiance's encoder matches HF tokenizers, docs/TOOLS.md) and the generation starts at the first token that ends after the
prompt part's last character (a sampled id sequence that is not the canonical tokenisation, or a first generated "\\n" that merges
with the prompt's "<think>\\n", moves a boundary by a token: counted, `retok_same` false). `score_from` = that token - 1, so the
golden holds the generated rows only (row j predicts token j + 1): its gen mask is all ones.
Writes OUT_PREFIX.jsonl ({prompt, source: pi-onpol-<kind>, score_from}), OUT_PREFIX.jsonl.meta.jsonl ({doc, cut, sidx, kind, p,
depth, q, tokens, score_from, gen_rows, gen_sampled, retok_same, stop_type}) and OUT_PREFIX.stats.json.
PRIVATE (the user's sessions): stays on box2."""
import argparse, collections, json, os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("render"); ap.add_argument("gen"); ap.add_argument("out")
ap.add_argument("--tok", default="/data/q27/models/qwen38-27b-bf16")
ap.add_argument("--cap", type=int, default=8192); ap.add_argument("--max-gen", type=int, default=6000)
a = ap.parse_args()
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(a.tok)
GP = [248045, 74455, 198, 248068, 198]
assert tk.encode("<|im_start|>assistant\n<think>\n", add_special_tokens=False) == GP
cuts = {json.loads(l)["cut"]: json.loads(l) for l in open(f"{a.render}/cuts.jsonl")}
gens = sorted(int(f[:-5]) for f in os.listdir(f"{a.gen}/gen") if f.endswith(".json"))
st = collections.Counter(); sess = {}
fo = open(a.out + ".jsonl.part", "w"); fm = open(a.out + ".jsonl.meta.jsonl.part", "w"); n = 0
for k in gens:
    d = json.load(open(f"{a.gen}/gen/{k}.json")); c = cuts[k]; g = d["gen"]
    st["cuts"] += 1; st["gen_sampled"] += len(g)
    if len(g) > a.max_gen: st["drop_gen_long"] += 1; continue
    if not g: st["drop_empty"] += 1; continue
    if c["sidx"] not in sess:
        z = np.load(f"{a.render}/sess/{c['sidx']}.npz"); sess = {c["sidx"]: (z["tokens"], z["starts"])}
    toks, starts = sess[c["sidx"]]; h = c["header"]; p = c["p"]
    budget = a.cap - h - len(GP) - len(g)
    qs = [int(q) for q in starts if h <= q <= p and p - q <= budget]
    if not qs: st["drop_no_turn_fits"] += 1; continue
    q = min(qs)
    pre = tk.decode(np.concatenate([toks[:h], toks[q:p], GP]).tolist(), skip_special_tokens=False)
    text = pre + tk.decode(g, skip_special_tokens=False)
    enc = tk(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    g0 = next(i for i, (s, e) in enumerate(offs) if e > len(pre))
    same = ids[:g0] == np.concatenate([toks[:h], toks[q:p], GP]).tolist() and ids[g0:] == g
    st["retok_same"] += same; st["prompt_retok_same"] += ids[:g0] == np.concatenate([toks[:h], toks[q:p], GP]).tolist()
    if len(ids) > a.cap + 8: st["drop_retok_long"] += 1; continue
    sf = g0 - 1; rows = len(ids) - 1 - sf
    fo.write(json.dumps({"prompt": text, "source": f"pi-onpol-{c['kind']}", "score_from": sf}, ensure_ascii=False) + "\n")
    fm.write(json.dumps({"doc": n, "cut": k, "sidx": c["sidx"], "kind": c["kind"], "p": p, "depth": c["depth"], "q": q,
                         "tokens": len(ids), "score_from": sf, "gen_rows": rows, "gen_sampled": len(g), "retok_same": bool(same),
                         "stop_type": d["stop_type"]}) + "\n")
    n += 1; st["docs"] += 1; st["tokens"] += len(ids); st["gen_rows"] += rows; st[f"stop_{d['stop_type']}"] += 1
    st[f"kind_{c['kind']}"] += 1
fo.close(); fm.close()
os.replace(a.out + ".jsonl.part", a.out + ".jsonl"); os.replace(a.out + ".jsonl.meta.jsonl.part", a.out + ".jsonl.meta.jsonl")
st["sessions"] = len({json.loads(l)["sidx"] for l in open(a.out + ".jsonl.meta.jsonl")})
json.dump(dict(st), open(a.out + ".stats.json", "w"), indent=1); print(json.dumps(dict(st), indent=1))
