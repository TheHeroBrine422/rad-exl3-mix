#!/usr/bin/env python3
"""skeleton.py SRC_DIR OUT_DIR [KEEP_REGEX] -- a sparse copy of a safetensors checkpoint directory.

Every *.safetensors file keeps its header byte for byte and its exact size; a tensor whose name
matches KEEP_REGEX has its bytes copied, every other tensor is a hole (zeros, no disk, sent as a
hole by `tar -S`). Every other file (config, tokenizer, index) is copied as is. Without KEEP_REGEX
nothing is kept: a headers-only skeleton.

For `rad-convert --reuse`: the weights a recipe copies from the earlier container are never read
from the checkpoint, so only the tensors that are converted again need their bytes. Check the
result against the --reuse container (every copied weight byte-identical): a hole that IS read
becomes zeros silently.
"""
import json, os, re, shutil, struct, sys

src, out = sys.argv[1], sys.argv[2]
keep = re.compile(sys.argv[3]) if len(sys.argv) > 3 else None
os.makedirs(out, exist_ok=True)
kept = holed = 0
for name in sorted(os.listdir(src)):
    p, q = os.path.join(src, name), os.path.join(out, name)
    if os.path.isdir(p):
        continue
    if not name.endswith(".safetensors"):
        shutil.copyfile(p, q)
        continue
    size = os.path.getsize(p)
    with open(p, "rb") as f, open(q, "wb") as g:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = f.read(n)
        g.write(struct.pack("<Q", n)); g.write(hdr)
        base = 8 + n
        for t, v in json.loads(hdr).items():
            if t == "__metadata__":
                continue
            a, b = v["data_offsets"]
            if keep and keep.search(t):
                f.seek(base + a); g.seek(base + a)
                left = b - a
                while left:
                    chunk = f.read(min(left, 64 << 20)); g.write(chunk); left -= len(chunk)
                kept += b - a
            else:
                holed += b - a
        g.truncate(size)
print(f"{out}: kept {kept / 2**30:.2f} GiB, holes {holed / 2**30:.2f} GiB")
