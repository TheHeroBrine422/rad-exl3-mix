#!/usr/bin/env python3
"""fakemkgold.py -- stand-in for eval/gold27/mkgold27.py in agent32.sh's dry run (no card, no model): same arguments,
same output layout (kld.json, tokens.i32 from the real tokenizer, gen.u8 all 1, top.bin / rows.f32 with valid but fake
rows: head log p sorted, head + tail = 1), scratch timing.jsonl with a fake peak. A32_FAKE_CFG="N:AC" listed in
A32_FAKE_OOM, or a document longer than A32_FAKE_MAXTOK, fails like a torch out-of-memory error. Lines with chunk 0
(the published golden has their context) get the published rows (a correct reference), unless A32_FAKE_CFG is listed
in A32_FAKE_BAD (then fake rows: the gate must fail)."""
import argparse, json, os, sys, time
import numpy as np
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("corpus"); ap.add_argument("out")
ap.add_argument("--scratch", required=True); ap.add_argument("--stages", type=int, default=2)
ap.add_argument("--scratch-dtype", default="f32"); ap.add_argument("--attn-chunk", type=int, default=0)
ap.add_argument("--model-name", default=None)
a = ap.parse_args()
cfg = os.environ.get("A32_FAKE_CFG", "")
if cfg and cfg in os.environ.get("A32_FAKE_OOM", "").split():
    print(f"torch.OutOfMemoryError: HIP out of memory. Tried to allocate 48.00 GiB (fake, config {cfg})", file=sys.stderr)
    sys.exit(1)
tk = Tokenizer.from_file(f"{a.model}/tokenizer.json")
docs, T, r, pub = [], [], 0, []
for l in open(a.corpus):
    if not l.strip(): continue
    j = json.loads(l); ids = np.array(tk.encode(j["prompt"], add_special_tokens=False).ids, np.int32)
    if len(ids) > int(os.environ.get("A32_FAKE_MAXTOK", 10**9)):
        print(f"torch.OutOfMemoryError: HIP out of memory ({len(ids)} tokens, fake A32_FAKE_MAXTOK)", file=sys.stderr); sys.exit(1)
    if j.get("chunk") == 0 and cfg not in os.environ.get("A32_FAKE_BAD", "").split(): pub.append((r, j))
    sf = int(j.get("score_from", 0)); docs.append({"source": j.get("source", ""), "tokens": len(ids), "score_from": sf, "row0": r})
    r += len(ids) - 1 - sf; T.append(ids)
os.makedirs(a.out, exist_ok=True); os.makedirs(a.scratch, exist_ok=True)
np.concatenate(T).tofile(f"{a.out}/tokens.i32"); np.ones(r, np.uint8).tofile(f"{a.out}/gen.u8")
p = 0.98 ** np.arange(512); p = p / p.sum() * (1 - 1e-4)
row = np.empty(512, np.dtype([("id", "<i4"), ("lp", "<f4")])); row["id"] = np.arange(512); row["lp"] = np.log(p)
piece = np.tile(row, (4096, 1)).tobytes()
rng = np.random.default_rng(0)
with open(f"{a.out}/top.bin", "wb") as f:
    for j in range(0, r, 4096): f.write(piece[:min(4096, r - j) * 512 * 8])
rw = np.stack([rng.exponential(1, r), np.ones(r), np.full(r, 1e-4)], 1).astype(np.float32)
if pub:                                            # a correct reference where the published golden has the context
    cg = "/data/challenge/golden/calib"; cm = json.load(open(f"{cg}/kld.json"))
    ctb = np.memmap(f"{cg}/top.bin", np.uint8, "r").reshape(cm["rows"], 512 * 8)
    crw = np.memmap(f"{cg}/rows.f32", np.float32, "r").reshape(cm["rows"], 3)
    with open(f"{a.out}/top.bin", "r+b") as f:
        for r0, j in pub:
            n = j["to"] - j["from"]; c0 = cm["docs"][j["orig"]]["row0"] + j["from"] - 1
            f.seek(r0 * 512 * 8); f.write(np.asarray(ctb[c0:c0 + n]).tobytes()); rw[r0:r0 + n] = crw[c0:c0 + n]
rw.tofile(f"{a.out}/rows.f32")
with open(f"{a.scratch}/timing.jsonl", "a") as f:
    for s in range(a.stages):
        f.write(json.dumps({"stage": s, "tokens": int(sum(d["tokens"] for d in docs)), "run_s": 7.5, "peak_gib": 20.0 + s,
                            "load_s": 1.0, "fake": True}) + "\n")
m = {"format": "radiance-kld-top-1", "model": a.model_name or a.model, "corpus": a.corpus, "corpus_fnv1a64": "fake",
     "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_vocab": 248320, "top": 512, "rows": r, "docs": docs,
     "maker": {"tool": "fakemkgold.py", "stages": a.stages, "attn_chunk": a.attn_chunk}}
json.dump(m, open(f"{a.out}/kld.json", "w"), indent=1)
print(f"{time.strftime('%H:%M:%S')} DONE {a.out}: {r} rows (fake)", file=sys.stderr)
