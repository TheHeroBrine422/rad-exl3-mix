#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""docs230.py NAME RENDER GEN OUTDIR [--chunks K] [--tok DIR] [--max-tokens 135000] -- w230 step 4: the golden documents of one teacher
set (NAME = t1 / t1-judge / t2 / t2-judge; RENDER = cuts230.py's dir, GEN = gen230.py's dir) as radiance --kld-corpus files.
A document = the WHOLE prompt (RENDER/sess/<sidx>.npz tokens[:p] + the cut's generation header `gp`) + the generated ids (ending in
<|im_end|> unless the turn hit max_new), no 8k trim; score_from = the first generated token - 1, so the golden rows are exactly the
generated tokens (row j predicts token j + 1; w228's docs228.py boundary logic: the document is re-tokenised as text, radiance
tokenises text, and the generation starts at the first token that ends after the prompt part's last character; `retok_same` false =
the sampled ids are not the canonical tokenisation). Dropped: an empty generation, a document over --max-tokens after re-tokenising.
Outputs: OUTDIR/NAME.cK.jsonl (+ .meta.jsonl): K chunks of ~equal tokens, the deepest documents first (the recorders run one chunk
at a time on a TP2 pair; chunk k's max_tokens sizes its max-model-len), OUTDIR/NAME.jsonl (+ .meta.jsonl): the chunks concatenated in
chunk order (= the merged golden's document order), OUTDIR/NAME.chunks.json, OUTDIR/NAME.stats.json. Public data only."""
import argparse, collections, json, os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("render"); ap.add_argument("gen"); ap.add_argument("outdir")
ap.add_argument("--chunks", type=int, default=1); ap.add_argument("--tok", default="/data/q27/models/qwen38-27b-bf16")
ap.add_argument("--max-tokens", type=int, default=135000)
a = ap.parse_args()
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(a.tok)
cuts = {json.loads(l)["cut"]: json.loads(l) for l in open(f"{a.render}/cuts.jsonl")}
gens = sorted(int(f[:-5]) for f in os.listdir(f"{a.gen}/gen") if f.endswith(".json"))
st = collections.Counter(); docs = []; cache = (None, None)
for k in gens:
    d = json.load(open(f"{a.gen}/gen/{k}.json")); c = cuts[k]; g = d["gen"]
    assert d["p"] == c["p"], (k, d["p"], c["p"])
    st["cuts"] += 1; st["gen_sampled"] += len(g)
    if not g: st["drop_empty"] += 1; continue
    if cache[0] != c["sidx"]: cache = (c["sidx"], np.load(f"{a.render}/sess/{c['sidx']}.npz")["tokens"])
    pids = cache[1][:c["p"]].tolist() + c["gp"]
    pre = tk.decode(pids, skip_special_tokens=False)
    text = pre + tk.decode(g, skip_special_tokens=False)
    e = tk(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = e["input_ids"], e["offset_mapping"]
    g0 = next(i for i, (s, t) in enumerate(offs) if t > len(pre))
    prompt_same = ids[:g0] == pids; same = prompt_same and ids[g0:] == g
    st["retok_same"] += same; st["prompt_retok_same"] += prompt_same
    if len(ids) > a.max_tokens: st["drop_too_long"] += 1; continue
    sf = g0 - 1
    docs.append({"text": text, "tokens": len(ids), "score_from": sf, "rows": len(ids) - 1 - sf,
                 "meta": {"cut": k, "sidx": c["sidx"], "kind": c["kind"], "origin": c["origin"], "src_doc": c["doc"], "setting": c["setting"],
                          "p": c["p"], "band": c.get("band"), "follower": c.get("follower"), "holdout": c.get("holdout"), "batch": a.name,
                          "tokens": len(ids), "score_from": sf, "gen_rows": len(ids) - 1 - sf, "gen_sampled": len(g), "retok_same": bool(same),
                          "prompt_retok_same": bool(prompt_same), "stop_type": d["stop_type"]}})
    st["docs"] += 1; st["tokens"] += len(ids); st["gen_rows"] += len(ids) - 1 - sf; st[f"stop_{d['stop_type']}"] += 1
    st[f"setting_{c['setting']}"] += 1
# chunks: deepest first, ~equal tokens (a document never splits)
order = sorted(range(len(docs)), key=lambda i: (-docs[i]["tokens"], docs[i]["meta"]["cut"]))
tot = sum(x["tokens"] for x in docs); chunks = [[] for _ in range(a.chunks)]; acc = 0
for i in order:
    kk = min(int(acc * a.chunks / max(tot, 1)), a.chunks - 1); chunks[kk].append(i); acc += docs[i]["tokens"]
os.makedirs(a.outdir, exist_ok=True); info = []; n = 0
fo = open(f"{a.outdir}/{a.name}.jsonl.part", "w"); fm = open(f"{a.outdir}/{a.name}.jsonl.meta.jsonl.part", "w")
for kk, ch in enumerate(chunks):
    ch = sorted(ch, key=lambda i: docs[i]["meta"]["cut"])
    if not ch: continue
    with open(f"{a.outdir}/{a.name}.c{kk}.jsonl", "w") as f1, open(f"{a.outdir}/{a.name}.c{kk}.jsonl.meta.jsonl", "w") as f2:
        for i in ch:
            x = docs[i]; kind = x["meta"]["kind"]
            line = json.dumps({"prompt": x["text"], "source": f"{a.name}-{kind}", "score_from": x["score_from"]}, ensure_ascii=False) + "\n"
            m = dict(x["meta"], doc=n, chunk=kk); n += 1
            ml = json.dumps(m) + "\n"
            f1.write(line); f2.write(ml); fo.write(line); fm.write(ml)
    info.append({"k": kk, "docs": len(ch), "tokens": sum(docs[i]["tokens"] for i in ch), "rows": sum(docs[i]["rows"] for i in ch),
                 "max_tokens": max(docs[i]["tokens"] for i in ch), "file": f"{a.name}.c{kk}.jsonl"})
fo.close(); fm.close()
os.replace(f"{a.outdir}/{a.name}.jsonl.part", f"{a.outdir}/{a.name}.jsonl")
os.replace(f"{a.outdir}/{a.name}.jsonl.meta.jsonl.part", f"{a.outdir}/{a.name}.jsonl.meta.jsonl")
json.dump(info, open(f"{a.outdir}/{a.name}.chunks.json", "w"), indent=1)
json.dump(dict(st), open(f"{a.outdir}/{a.name}.stats.json", "w"), indent=1)
print(json.dumps(dict(st), indent=1)); print(json.dumps(info))
