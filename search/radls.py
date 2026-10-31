#!/usr/bin/env python3
"""radls.py CONTAINER.rad... -- every entry's name and bytes, one JSON line per container: {"c": path, "total": bytes,
"w": {name: bytes}}."""
import json, struct, sys
MAXR = int(__import__("os").environ.get("RAD_MAX_RANK", "6"))
for p in sys.argv[1:]:
    f = open(p, "rb"); h = f.read(4096)
    magic, ver, fbytes = struct.unpack_from("<IIQ", h, 0)
    o = 16 + 5 * 8
    str_off, str_bytes, meta_off, meta_cnt, dir_off, dir_cnt = struct.unpack_from("<6Q", h, o)
    f.seek(str_off); S = f.read(str_bytes)
    es = 3 * 8 + 4 + 4 + 8 * MAXR + 4 * 3 + 4 + 8 + 8 + 8 + 16
    f.seek(dir_off); D = f.read(es * dir_cnt)
    w = {}
    for i in range(dir_cnt):
        e = D[i * es:(i + 1) * es]
        name = struct.unpack_from("<Q", e, 0)[0]
        nbytes = struct.unpack_from("<Q", e, es - 16 - 8)[0]
        w[S[name:S.index(b"\0", name)].decode()] = nbytes
    print(json.dumps({"c": p, "total": fbytes, "w": w}))
