#!/usr/bin/env python3
"""mkgold.py MODEL FULL OUT DOCIDX... -- a reference golden of the 2B (or any transformers model) in the challenge's
format: FULL is the published calib (or public) golden's directory, of which only kld.json, tokens.i32 and gen.u8 are
read (the documents and which rows are scored are model-independent); DOCIDX as golden_pick.py (idx, idx:cap,
idx:cap:from). Writes OUT/{kld.json, tokens.i32, gen.u8, top.bin, rows.f32}, rows in document order. bf16 model,
the LM head and the log-softmax in float64 a chunk of rows at a time."""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model, doc_args, ref_rows, doc_logsoftmax, TOP

model_path, full, od = sys.argv[1:4]
want = doc_args(sys.argv[4:])
m = json.load(open(f"{full}/kld.json"))
toks = np.memmap(f"{full}/tokens.i32", np.int32, "r")
gen = np.fromfile(f"{full}/gen.u8", np.uint8)
toff = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
os.makedirs(od, exist_ok=True)
_, dec, head = load_model(model_path)
ft, fg = open(f"{od}/top.bin", "wb"), open(f"{od}/rows.f32", "wb")
T, G, docs, r, t0 = [], [], [], 0, time.time()
for k, (i, c, f) in enumerate(want):
    d = m["docs"][i]; nt = min(d["tokens"], c) if c else d["tokens"]; sf = max(d["score_from"], f); n = nt - 1 - sf
    r0 = d["row0"] + sf - d["score_from"]; assert n > 0, (i, nt, sf)
    tk = np.asarray(toks[toff[i]:toff[i] + nt]); T.append(tk); G.append(gen[r0:r0 + n])
    nxt = torch.from_numpy(tk[sf + 1:nt].astype(np.int64)).cuda()
    for j, lq in doc_logsoftmax(dec, head, tk, sf):
        ids, lp, ent, tail = ref_rows(lq)
        nll = -lq.gather(1, nxt[j:j + lq.shape[0], None])[:, 0]
        rec = np.empty((lq.shape[0], TOP), np.dtype([("id", "<i4"), ("lp", "<f4")]))
        rec["id"], rec["lp"] = ids.cpu().numpy(), lp.cpu().numpy(); rec.tofile(ft)
        np.stack([nll.float().cpu().numpy(), ent.float().cpu().numpy(), tail.float().cpu().numpy()], 1).astype(np.float32).tofile(fg)
    docs.append({"source": d["source"], "tokens": nt, "score_from": sf, "row0": r, "orig": i}); r += n
    print(f"\r{k + 1}/{len(want)} docs, {r} rows, {r / (time.time() - t0):.0f} rows/s", end="", file=sys.stderr, flush=True)
print(file=sys.stderr)
ft.close(); fg.close()
np.concatenate(T).tofile(f"{od}/tokens.i32"); np.concatenate(G).astype(np.uint8).tofile(f"{od}/gen.u8")
mm = {k: v for k, v in m.items() if k != "docs"}
mm.update(model=model_path, rows=r, docs=docs, created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
json.dump(mm, open(f"{od}/kld.json", "w"))
