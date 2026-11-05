# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""safetensors without torch: copy named tensors byte for byte from checkpoint shards into one new file."""
import json, os, struct
from .util import st_header


def copy_tensors(ckpt_dir, weight_map, names, out, metadata=None):
    """names (checkpoint keys) -> one safetensors file at out (.part, then renamed); bytes copied unchanged"""
    src, hdrs = {}, {}
    for n in names:
        f = weight_map[n]
        if f not in hdrs: hdrs[f] = st_header(os.path.join(ckpt_dir, f))
        src[n] = (f, hdrs[f][n])
    hdr, off = {}, 0
    if metadata: hdr["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    for n in names:                                      # keep the caller's order
        e = src[n][1]; ln = e["data_offsets"][1] - e["data_offsets"][0]
        hdr[n] = {"dtype": e["dtype"], "shape": e["shape"], "data_offsets": [off, off + ln]}; off += ln
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(out + ".part", "wb") as fo:
        fo.write(struct.pack("<Q", len(hb))); fo.write(hb)
        for n in names:
            f, e = src[n]
            base = 8 + struct.unpack("<Q", open(os.path.join(ckpt_dir, f), "rb").read(8))[0]
            a, b = e["data_offsets"]
            with open(os.path.join(ckpt_dir, f), "rb") as fi:
                fi.seek(base + a); left = b - a
                while left:
                    chunk = fi.read(min(left, 1 << 24)); fo.write(chunk); left -= len(chunk)
    os.replace(out + ".part", out)
