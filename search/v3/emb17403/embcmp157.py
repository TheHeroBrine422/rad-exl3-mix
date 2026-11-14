#!/usr/bin/env python3
"""embcmp157.py A.rad [B.rad] -- (w157, #174-03) a container's token_embd.weight entry: bytes (the menu's exact bytes:
every plane of the entry) and sha256; with B, whether the two entries are byte-identical (RAD2 directory reader of
w144's embref.py / v3/yaqa/headslab.py).  embcmp157.py --names A.rad: every weight name of A, one a line."""
import hashlib, json, struct, sys


def names(path):
    f = open(path, "rb"); h = f.read(4096)
    str_off, str_bytes, meta_off, meta_cnt, dir_off, dir_cnt = struct.unpack_from("<6Q", h, 16 + 5 * 8)
    f.seek(str_off); S = f.read(str_bytes)
    es = 3 * 8 + 4 + 4 + 8 * 6 + 4 * 3 + 4 + 8 + 8 + 8 + 16
    f.seek(dir_off); D = f.read(es * dir_cnt)
    for i in range(dir_cnt):
        name = struct.unpack_from("<Q", D, i * es)[0]
        yield S[name:S.index(b"\0", name)].decode()


def entry(path, want="token_embd.weight"):
    f = open(path, "rb"); h = f.read(4096)
    str_off, str_bytes, meta_off, meta_cnt, dir_off, dir_cnt = struct.unpack_from("<6Q", h, 16 + 5 * 8)
    f.seek(str_off); S = f.read(str_bytes)
    es = 3 * 8 + 4 + 4 + 8 * 6 + 4 * 3 + 4 + 8 + 8 + 8 + 16
    f.seek(dir_off); D = f.read(es * dir_cnt)
    for i in range(dir_cnt):
        e = D[i * es:(i + 1) * es]
        name = struct.unpack_from("<Q", e, 0)[0]
        if S[name:S.index(b"\0", name)].decode() != want: continue
        off, nbytes = struct.unpack_from("<QQ", e, es - 32)
        return off, nbytes, dir_cnt
    raise KeyError(want)


def digest(path):
    off, n, cnt = entry(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(off); left = n
        while left: b = f.read(min(left, 1 << 24)); h.update(b); left -= len(b)
    return {"file": path, "bytes": n, "sha256": h.hexdigest(), "weights": cnt}


if sys.argv[1] == "--names":
    for n in names(sys.argv[2]): print(n)
    sys.exit(0)
r = [digest(p) for p in sys.argv[1:]]
out = {"a": r[0]}
if len(r) > 1: out["b"] = r[1]; out["identical"] = r[0]["bytes"] == r[1]["bytes"] and r[0]["sha256"] == r[1]["sha256"]
print(json.dumps(out))
