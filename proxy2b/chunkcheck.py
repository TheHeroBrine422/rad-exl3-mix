#!/usr/bin/env python3
"""chunkcheck.py MODEL META DOC CHUNK -- chunked prefill through the cache (hcollect.py's long-document path) against
one whole-document forward: max |dh| / rms(h) of the final hidden states, per chunk boundary region."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model
mp, meta, di, C = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
_, dec, _ = load_model(mp)
m = json.load(open(f"{meta}/kld.json")); toks = np.memmap(f"{meta}/tokens.i32", np.int32, "r")
off = sum(d["tokens"] for d in m["docs"][:di]); n = m["docs"][di]["tokens"]
tk = torch.from_numpy(np.asarray(toks[off:off + n], np.int64))[None].cuda()
with torch.no_grad():
    hw = dec(input_ids=tk).last_hidden_state[0].float()
    cache, parts = None, []
    for s in range(0, n, C):
        o = dec(input_ids=tk[:, s:s + C], past_key_values=cache, use_cache=True); cache = o.past_key_values
        parts.append(o.last_hidden_state[0].float())
    hc = torch.cat(parts)
rel = ((hc - hw).norm(dim=1) / hw.norm(dim=1)).cpu().numpy()
print(f"doc {di}: {n} tokens, chunk {C}: rel |dh| mean {rel.mean():.2e}, first chunk {rel[:C].mean():.2e}, "
      f"later chunks {rel[C:].mean():.2e}, max {rel.max():.2e}")
