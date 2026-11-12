#!/usr/bin/env python3
"""fetch119.py URL WEIGHT OUT.npz [--info] -- one weight of a public RAD2 container read by HTTP range requests (w119 E9:
deadcode's int8 lm_head as shipped in StillDeadcode/qwen3.8-27b-iq8r-dflash), without downloading the container.
Header layout as store/census.py's Rad. For an int8 g128 head: OUT.npz codes int8 [rows, cols], scale f32 [rows, cols/128]."""
import struct, sys, urllib.request
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/../../../store")
from census import HDR, ENT, KV, ENC_SIZE, PLANE, MAGIC, VER, cstr, DT   # noqa: E402

URL, NAME, OUT = sys.argv[1:4]; INFO = "--info" in sys.argv


def rng(o, n):
    out = bytearray()
    while len(out) < n:
        a = o + len(out); b = min(o + n, a + (256 << 20)) - 1
        r = urllib.request.Request(URL, headers={"Range": f"bytes={a}-{b}"})
        with urllib.request.urlopen(r, timeout=600) as f:
            assert f.status == 206, f.status
            out += f.read()
    assert len(out) == n
    return bytes(out)


h = HDR.unpack(rng(0, HDR.size)); assert h[0] == MAGIC and h[1] == VER
(so, sn, mo, mn, do, dn, po, pn, eo, en) = h[8:18]
S, D, P, E = rng(so, sn), rng(do, dn * ENT.size), rng(po, pn * 16), rng(eo, en * ENC_SIZE)
w = None
for i in range(dn):
    e = ENT.unpack_from(D, i * ENT.size)
    if cstr(S, e[0]) != NAME: continue
    eb = E[e[3] * ENC_SIZE:(e[3] + 1) * ENC_SIZE]
    w = dict(quant=cstr(S, e[1]), opts=cstr(S, e[2]), shape=list(e[5:5 + e[4]]), enc=eb,
             planes=[struct.unpack_from("<QQ", P, (e[15] + p) * 16) for p in range(e[14])])
assert w, f"{NAME} not in the container"
scheme = w["enc"][:24].split(b"\0")[0].decode(); tr = w["enc"][24:48].split(b"\0")[0].decode()
npl = struct.unpack_from("<I", w["enc"], 48)[0]
pl = [PLANE.unpack_from(w["enc"], 56 + 64 * i) for i in range(npl)]
print(f"{NAME}: quant {w['quant']!r} opts {w['opts']!r} shape {w['shape']} scheme {scheme!r} transform {tr!r}")
for (o, n_), p in zip(w["planes"], pl):
    print(f"  plane {p[0].split(b'\\0')[0].decode()!r} dtype {DT.get(p[1], p[1])} dims {p[3:]} bytes {n_} at {o}")
if INFO: sys.exit(0)
rows, cols = w["shape"][-2], w["shape"][-1]
roles = [p[0].split(b"\0")[0].decode() for p in pl]
ci = next(i for i, r in enumerate(roles) if "code" in r or r in ("q", "qs", "weight"))
si = next(i for i, r in enumerate(roles) if "scale" in r)
assert DT.get(pl[ci][1]) == "i8" and DT.get(pl[si][1]) == "bf16", (roles, [pl[i][1] for i in (ci, si)])
codes = np.frombuffer(rng(*w["planes"][ci]), np.int8)
sb = np.frombuffer(rng(*w["planes"][si]), np.uint16)
assert codes.size == rows * cols and sb.size == rows * cols // 128, (codes.size, sb.size)
scale = (sb.astype(np.uint32) << 16).view(np.float32)
np.savez(OUT, codes=codes.reshape(rows, cols), scale=scale.reshape(rows, cols // 128), roles=np.array(roles))
print(f"saved {OUT}: codes {rows}x{cols}, scale {rows}x{cols // 128}, |code| max {np.abs(codes).max()}")
