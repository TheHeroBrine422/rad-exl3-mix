#!/usr/bin/env python3
"""halves.py MODEL GOLDEN DTYPE [SPLIT] -- the 2B at DTYPE (bf16 | fp32) in two halves, for a model that does not fit
whole: pass A runs embedding + layers [0, SPLIT) over every golden document and keeps the hidden states on the host;
pass B frees A and runs layers [SPLIT, n) + norm + head from them (inputs_embeds; config.layer_types shifted with the
layers, as the decoder picks each layer's mask by list position). Scored against GOLDEN with common.score_rows.
bf16 is the control: it must reproduce a bf16 golden made by one whole forward (KL ~0), else the split is wrong.
Prints gen-masked mean KL, p99, top1, and the mean KL by position band."""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Golden, score_rows
from transformers import AutoModelForCausalLM
mp, gd, dts = sys.argv[1:4]; SPLIT = int(sys.argv[4]) if len(sys.argv) > 4 else 12
DT = {"bf16": torch.bfloat16, "fp32": torch.float32}[dts]
g = Golden(gd)
model = AutoModelForCausalLM.from_pretrained(mp, dtype=DT).eval()          # on the host
dec, head = model.get_decoder(), model.get_output_embeddings()
layers, types, norm = dec.layers, list(dec.config.layer_types), dec.norm
t0 = time.time()
with torch.no_grad():
    # pass A
    dec.layers = torch.nn.ModuleList(layers[:SPLIT]); dec.config.layer_types = types[:SPLIT]; dec.norm = torch.nn.Identity()
    dec.embed_tokens.cuda(); dec.rotary_emb.cuda(); dec.layers.cuda()
    H = []
    for i, d in enumerate(g.docs):
        t = torch.from_numpy(g.doc_tokens(i).astype(np.int64))[None].cuda()
        H.append(dec(input_ids=t, use_cache=False).last_hidden_state.cpu())
    dec.embed_tokens.cpu(); dec.layers.cpu(); torch.cuda.empty_cache()
    ta = time.time() - t0
    # pass B
    dec.layers = torch.nn.ModuleList(layers[SPLIT:]); dec.config.layer_types = types[SPLIT:]; dec.norm = norm
    dec.layers.cuda(); norm.cuda(); head.cuda()
    KL, T1 = [], []
    for i, d in enumerate(g.docs):
        sf = d["score_from"]; n = d["tokens"] - 1 - sf
        h = dec(inputs_embeds=H[i].cuda(), use_cache=False).last_hidden_state[0, sf:sf + n]; H[i] = None
        for j in range(0, n, 256):
            lq = torch.log_softmax(head(h[j:j + 256]).float(), 1); r0 = d["row0"] + j; m = lq.shape[0]; top = g.top[r0:r0 + m]
            kl, t1 = score_rows(lq, torch.from_numpy(np.ascontiguousarray(top["id"])).cuda(),
                                torch.from_numpy(np.ascontiguousarray(top["lp"])).cuda(),
                                torch.from_numpy(np.asarray(g.rows[r0:r0 + m, 2], np.float64)).cuda())
            KL.append(kl.cpu().numpy()); T1.append(t1.cpu().numpy())
kl, t1 = np.concatenate(KL), np.concatenate(T1); gen = g.gen[:len(kl)]
pos = np.concatenate([np.arange(d["tokens"] - 1 - d["score_from"]) for d in g.docs])
bands = [(0, 128), (128, 512), (512, 1024), (1024, 2048)]
bs = "  ".join(f"[{a},{b}) {kl[gen & (pos >= a) & (pos < b)].mean():.2e}" for a, b in bands)
print(f"{dts} split {SPLIT}: {int(gen.sum())} gen rows, KL vs golden mean {kl[gen].mean():.3e}, p99 {np.quantile(kl[gen], .99):.2e}, "
      f"top1 {t1[gen].mean():.4f} | by position {bs} | pass A {ta:.0f}s, total {time.time() - t0:.0f}s")
