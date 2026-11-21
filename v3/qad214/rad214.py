# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""rad214.py -- a RAD2 container's directory (the layout store/scan.py and store/tdverify.py read) as Python objects:
`Rad(path).entries[name]` = Entry(name, scheme, transform, rows/shape, planes {role: (offset, nbytes)}).
CLI: `rad214.py PACK.rad [--grep SUBSTR]` lists entries."""
import struct, sys

HDR = struct.Struct("<IIQ5Q" + "Q" * 18)
ENT = struct.Struct("<QQQII6qiiiIQQQ2Q")
ENC = 24 + 24 + 4 + 4 + 8 * 64
MAGIC, VER = 0x31444152, 2


def cstr(blob, off):
    return "" if off == 0 else blob[off:blob.index(b"\0", off)].decode()


class Entry:
    def __init__(self, name, quant, shape, scheme, transform, planes):
        self.name, self.quant, self.shape, self.scheme, self.transform, self.planes = name, quant, shape, scheme, transform, planes

    def __repr__(self):
        return f"Entry({self.name} {self.scheme} {self.transform} {self.shape} {self.planes})"


class Rad:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            h = HDR.unpack(f.read(HDR.size))
            assert h[0] == MAGIC and h[1] == VER, f"{path}: not RAD2"
            str_off, str_n, _, _, dir_off, dir_n, pl_off, pl_n, enc_off, enc_n = h[8:18]
            f.seek(str_off); S = f.read(str_n); f.seek(dir_off); D = f.read(dir_n * ENT.size)
            f.seek(pl_off); P = f.read(pl_n * 16); f.seek(enc_off); E = f.read(enc_n * ENC)
        self.entries = {}
        for i in range(dir_n):
            e = ENT.unpack_from(D, i * ENT.size)
            name, quant = cstr(S, e[0]), cstr(S, e[1])
            shape = list(e[5:5 + e[4]])
            eb = E[e[3] * ENC:(e[3] + 1) * ENC]
            scheme, transform = cstr(eb, 0) if False else eb[:24].split(b"\0")[0].decode(), eb[24:48].split(b"\0")[0].decode()
            planes = {}
            for p in range(e[14]):
                role = eb[56 + 64 * p:56 + 64 * p + 24].split(b"\0")[0].decode()
                po, pn = struct.unpack_from("<QQ", P, (e[15] + p) * 16)
                planes[role] = (po, pn)
            self.entries[name] = Entry(name, quant, shape, scheme, transform, planes)

    def read(self, name, role):
        po, pn = self.entries[name].planes[role]
        with open(self.path, "rb") as f:
            f.seek(po); return f.read(pn)


if __name__ == "__main__":
    r = Rad(sys.argv[1])
    g = sys.argv[sys.argv.index("--grep") + 1] if "--grep" in sys.argv else ""
    for n, e in r.entries.items():
        if g in n: print(e)
