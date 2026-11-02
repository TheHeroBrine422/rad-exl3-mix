#!/usr/bin/env python3
"""lencheck.py MODEL META DOC CUT -- causality check: the first CUT rows of a document run alone vs the same rows of
the whole document, hidden state after every layer (rel |dh|), to find the layer where they part."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model
mp, meta, di, C = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
model, dec, _ = load_model(mp)
m = json.load(open(f"{meta}/kld.json")); toks = np.memmap(f"{meta}/tokens.i32", np.int32, "r")
off = sum(d["tokens"] for d in m["docs"][:di]); n = m["docs"][di]["tokens"]
tk = torch.from_numpy(np.asarray(toks[off:off + n], np.int64))[None].cuda()
with torch.no_grad():
    a = dec(input_ids=tk, use_cache=False, output_hidden_states=True).hidden_states
    b = dec(input_ids=tk[:, :C], use_cache=False, output_hidden_states=True).hidden_states
types = model.config.get_text_config().layer_types
for i, (x, y) in enumerate(zip(a, b)):
    x, y = x[0, :C].float(), y[0].float()
    r = ((x - y).norm(dim=1) / x.norm(dim=1).clamp_min(1e-6))
    print(f"{'embed' if i == 0 else f'L{i - 1:02d} {types[i - 1][:6]}'}: rel |dh| mean {r.mean():.2e} max {r.max():.2e}")
