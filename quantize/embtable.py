#!/usr/bin/env python3
"""embtable.py CKPT_DIR OUT.bf16 -- the checkpoint's token table (model.language_model.embed_tokens.weight,
[248320][5120] bf16) as a raw file: embtier.py's input at offset 0 (the CPU preps on other machines)."""
import json, struct, sys
ck, out = sys.argv[1:3]
idx = json.load(open(f"{ck}/model.safetensors.index.json"))["weight_map"]
name = next(n for n in idx if n.endswith("embed_tokens.weight"))
with open(f"{ck}/{idx[name]}", "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); a, b = h[name]["data_offsets"]
    assert h[name]["dtype"] == "BF16" and h[name]["shape"] == [248320, 5120], h[name]
    f.seek(8 + n + a); left = b - a
    with open(out, "wb") as o:
        while left: c = f.read(min(left, 1 << 26)); o.write(c); left -= len(c)
