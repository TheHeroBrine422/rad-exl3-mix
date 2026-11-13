#!/usr/bin/env python3
"""abpatch.py CONTAINER.rad AB.safetensors -- write an arm's a/b values (bf16 [48, 5120] per blk.N.ssm_{a,b}.weight) over
the container's own a/b planes, in place (w142, #178). Every target must be a plain bf16 weight of the same shape with
one plane of exactly those bytes; anything else is refused before a byte is written. Use it on a reflinked copy: the
write unshares only the a/b extents, the donors are untouched. Prints the count and a digest of what was written."""
import hashlib, os, struct, sys
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.environ.get("RXQ_RXM", "/data/q27/stage/code"), "store"))
from census import Rad, describe   # noqa: E402

rad, src = sys.argv[1], sys.argv[2]
R = Rad(rad)
todo = []
with safe_open(src, "pt") as f:
    for n in sorted(f.keys()):
        t = f.get_tensor(n)
        if n not in R.w: sys.exit(f"{n}: not in {rad}")
        w = R.w[n]; sch, _, _, det = describe(w["enc"])
        b = t.contiguous().view(-1).view(dtype=__import__("torch").int16).numpy().tobytes()
        if sch != "plain" or not det.endswith("bf16") or w["shape"] != list(t.shape) or len(w["planes"]) != 1 or w["planes"][0][1] != len(b):
            sys.exit(f"{n}: container holds {sch} {det} {w['shape']} planes {w['planes']}, arm {tuple(t.shape)} {t.dtype}")
        todo.append((n, w["planes"][0][0], b))
R.f.close()
h = hashlib.sha256()
fd = os.open(rad, os.O_RDWR)
for n, off, b in todo:
    assert os.pwrite(fd, b, off) == len(b); h.update(n.encode()); h.update(b)
os.fsync(fd); os.close(fd)
print(f"abpatch {os.path.basename(rad)}: {len(todo)} weights from {os.path.basename(src)}, sha256 {h.hexdigest()[:16]}")
