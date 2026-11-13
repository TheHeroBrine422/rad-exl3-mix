#!/usr/bin/env python3
"""embsrc.py LEVEL -- (w144, #174-02) ourq.py's head-path output for the token table (LEVEL/q-head.safetensors, keys
lm_head.*) -> LEVEL/src/, the exl3i importer's source for `token_embd.weight exl3i src=LEVEL/src`: the same tensors
under model.language_model.embed_tokens.* (q-embed.safetensors, bytes unchanged), its index, config.json and
stamp.json (the importer checks the codebook and the row order against it). Also prints the row error from
LEVEL/wq/lm_head.rowerr.npz (identity Hessian: err = sum (w - wq)^2 a row, ref = sum w^2): rel = sum err / sum ref,
w135's measure (embprices174.json). Writes .part then renames; skips a finished src."""
import json, os, shutil, sys
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file

L = sys.argv[1]; S = f"{L}/src"
if not os.path.exists(f"{S}/q-embed.safetensors"):
    os.makedirs(S, exist_ok=True)
    with safe_open(f"{L}/q-head.safetensors", "pt") as f:
        t = {k.replace("lm_head.", "model.language_model.embed_tokens.", 1): f.get_tensor(k) for k in f.keys()}
        meta = dict(f.metadata() or {})
    assert sorted(t) == [f"model.language_model.embed_tokens.{k}" for k in ("mul1", "suh", "svh", "trellis")], sorted(t)
    meta["renamed_from"] = "lm_head (ourq --only-head on the token table, w144 #174-02)"
    save_file(t, f"{S}/q-embed.safetensors.part", metadata=meta)
    os.replace(f"{S}/q-embed.safetensors.part", f"{S}/q-embed.safetensors")
    json.dump({"metadata": {}, "weight_map": {k: "q-embed.safetensors" for k in t}}, open(f"{S}/model.safetensors.index.json", "w"), indent=1)
    for f in ("config.json", "stamp.json"): shutil.copy(f"{L}/{f}", f"{S}/{f}")
r = np.load(f"{L}/wq/lm_head.rowerr.npz")
print(json.dumps({"level": L, "rel": float(r["err"].sum() / r["ref"].sum()), "rows": int(len(r["err"])),
                  "bytes_src": os.path.getsize(f"{S}/q-embed.safetensors")}))
