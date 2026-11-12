#!/usr/bin/env python3
"""mixhash122.py E3D_DIR -- w122 E3 (d) confounder check (RULES.md): sha256 of each arm container with output.weight's
byte range skipped, per trunk, against the trunk + bf16 head base the arms were copied from. Equal = the arms differ
only in output.weight."""
import hashlib, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from headslab import entry

E = sys.argv[1]


def h(path):
    off, nbytes, _ = entry(path); s = hashlib.sha256(); f = open(path, "rb"); size = os.path.getsize(path)
    for a, b in ((0, off), (off + nbytes, size)):
        f.seek(a); left = b - a
        while left > 0:
            c = f.read(min(left, 64 << 20)); s.update(c); left -= len(c)
    return s.hexdigest()[:16], off, nbytes, size


ok = True
for t, arms in (("t3", ["U3", "S3", "U4", "S4"]), ("t4", ["U4", "S4", "U5", "S5"])):
    base = h(f"{E}/trunks/{t}-bf16head.rad"); print(f"{t} base {base}")
    for a in arms:
        x = h(f"{E}/mix/{t}-{a}.rad"); same = x == base; ok &= same
        print(f"{t}-{a} {x} {'SAME' if same else 'DIFFERS'}")
print("MIXHASH", "PASS: arms differ only in output.weight" if ok else "FAIL")
