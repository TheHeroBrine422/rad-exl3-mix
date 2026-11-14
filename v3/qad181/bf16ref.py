#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""bf16ref.py GOLDEN DOC[,DOC..] --model BF16 [--cap N] [--stu STU.npz] -- an independent bf16 reference for a golden's
rows (#181 check): the bf16 checkpoint run in torch (transformers' Qwen3.5 decoder layers, bf16, sdpa, shift conv),
one layer at a time on one card (weights streamed from the checkpoint, so the 27B model needs no second card), over
the first CAP tokens of each document; per row KL(golden || torch bf16) with the scorer's formula, and (with --stu)
next to the quantized student's per-row KL from tscore181.py, so a golden row the bf16 model itself does not
reproduce shows up as a large KL(golden || bf16)."""
import argparse, json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "qad"))
import data181 as D  # noqa: E402
import qmodel  # noqa: E402
from qad import chunk_kl  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("golden"); ap.add_argument("docs"); ap.add_argument("--model", required=True)
ap.add_argument("--cap", type=int, default=2048); ap.add_argument("--stu"); ap.add_argument("--out")
a = ap.parse_args()
from safetensors import safe_open
from transformers import AutoConfig
import transformers.models.qwen3_5.modeling_qwen3_5 as q35
qmodel.use_shift_conv()
dev = torch.device("cuda:0"); BF = torch.bfloat16
cfg = AutoConfig.from_pretrained(a.model); cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
cfg._attn_implementation = "sdpa"
wm = json.load(open(f"{a.model}/model.safetensors.index.json"))["weight_map"]
LP = "model.language_model." if any(k.startswith("model.language_model.") for k in wm) else "model."
fh = {}
def get(k):
    f = wm[k]
    if f not in fh: fh[f] = safe_open(f"{a.model}/{f}", "pt", device="cpu")
    return fh[f].get_tensor(k).to(BF)
g = D.Gold(a.golden); res = {}
st = np.load(a.stu) if a.stu else None
for i in [int(x) for x in a.docs.split(",")]:
    t0 = time.time(); n = min(a.cap, g.ntok(i))
    ids = torch.from_numpy(g.ids(i, 0, n))
    x = get(f"{LP}embed_tokens.weight")[ids][None].to(dev)
    pid = torch.arange(n, device=dev).view(1, 1, -1).expand(4, 1, -1)
    rot = q35.Qwen3_5TextRotaryEmbedding(config=cfg).to(dev); pe = rot(x, pid[1:])
    mk = {"config": cfg, "inputs_embeds": x, "attention_mask": None, "past_key_values": None, "position_ids": pid[0]}
    masks = {"full_attention": q35.create_causal_mask(**mk), "linear_attention": q35.create_recurrent_attention_mask(**mk)}
    with torch.no_grad():
        for l in range(cfg.num_hidden_layers):
            pre = f"{LP}layers.{l}."
            with torch.device("meta"): ly = q35.Qwen3_5DecoderLayer(cfg, l)
            ly.load_state_dict({k[len(pre):]: get(k) for k in wm if k.startswith(pre)}, strict=True, assign=True)
            ly = ly.to(dev).eval()
            x = ly(x, position_embeddings=pe, attention_mask=masks[cfg.layer_types[l]], position_ids=pid[0])
            del ly
        fn = q35.Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps); fn.weight.data = get(f"{LP}norm.weight"); fn = fn.to(dev)
        h = fn(x)[0]
        W = get("lm_head.weight" if "lm_head.weight" in wm else f"{LP}embed_tokens.weight").to(dev)
        j, r = g.scored(i, 0, n)
        if len(j) == 0: print(f"doc {i}: no scored rows", flush=True); continue
        tid, tlp, pt = g.targets(r, dev); hh = h[torch.from_numpy(j).to(dev)]
        kl = torch.cat([chunk_kl(hh[c:c + 256], W, tid[c:c + 256], tlp[c:c + 256], pt[c:c + 256], True) for c in range(0, len(j), 256)]).double().cpu().numpy()
        del W
    line = f"doc {i}: {len(j)} rows, KL(golden || torch bf16) mean {kl.mean():.5f} median {np.median(kl):.6f} p99 {np.percentile(kl, 99):.3f} max {kl.max():.2f}, rows > 1 nat {int((kl > 1).sum())}"
    if st is not None:
        sk = st["kl"][st["doc"] == i][:len(kl)]
        big = sk > 10
        line += f"; student rows > 10 nats {int(big.sum())}: bf16-ref KL there mean {kl[big].mean() if big.any() else float('nan'):.3f}, student KL elsewhere {sk[~big].mean():.4f}"
    print(line + f" ({time.time() - t0:.0f} s)", flush=True); res[i] = kl
if a.out: np.savez(a.out, **{str(k): v for k, v in res.items()})
