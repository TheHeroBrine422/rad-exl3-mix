#!/usr/bin/env python3
"""chunkfloor.py MODEL GOLDEN CHUNK... -- the bf16 model itself scored against its own (whole-document) bf16 golden,
with the prefill run in chunks of CHUNK tokens through the cache (0 = whole document, the golden's own path: the
control, ~0). Measures the numerics noise floor that chunking alone puts on KLD / top-1 (a Discord report:
0.025 KLD between chunk 2048 and 64 on another Qwen3.8 model with sparse attention). One JSON line per CHUNK."""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model, Golden, score_rows
from transformers import DynamicCache
mp, gd = sys.argv[1:3]; CH = [int(x) for x in sys.argv[3:]]
model, dec, head = load_model(mp); g = Golden(gd)
@torch.no_grad()
def hidden(t, c):
    if c == 0: return dec(input_ids=t).last_hidden_state[0]
    cache = DynamicCache(config=model.config) if "config" in DynamicCache.__init__.__code__.co_varnames else DynamicCache()
    hs = []
    for c0 in range(0, t.shape[1], c):
        o = dec(input_ids=t[:, c0:c0 + c], past_key_values=cache, use_cache=True); cache = o.past_key_values
        hs.append(o.last_hidden_state[0])
    return torch.cat(hs)
for c in CH:
    KL, T1, t0 = [], [], time.time()
    for i, d in enumerate(g.docs):
        tk, sf = g.doc_tokens(i), d["score_from"]
        t = torch.from_numpy(tk.astype(np.int64))[None].cuda()
        h = hidden(t, c)[sf:len(tk) - 1]
        for j in range(0, h.shape[0], 256):
            lq = torch.log_softmax(head(h[j:j + 256]).float(), 1); r0 = d["row0"] + j; n = lq.shape[0]; top = g.top[r0:r0 + n]
            kl, t1 = score_rows(lq, torch.from_numpy(np.ascontiguousarray(top["id"])).cuda(), torch.from_numpy(np.ascontiguousarray(top["lp"])).cuda(),
                                torch.from_numpy(np.asarray(g.rows[r0:r0 + n, 2], np.float64)).cuda())
            KL.append(kl.cpu().numpy()); T1.append(t1.cpu().numpy())
    kl, t1 = np.concatenate(KL), np.concatenate(T1); gen = g.gen[:len(kl)]
    print(json.dumps({"chunk": c, "positions": int(gen.sum()), "kld": float(kl[gen].mean()), "kl99": float(np.quantile(kl[gen], 0.99)),
                      "kl999": float(np.quantile(kl[gen], 0.999)), "top1": float(t1[gen].mean()), "sec": round(time.time() - t0, 1)}), flush=True)
