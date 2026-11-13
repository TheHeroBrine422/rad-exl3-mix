#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""radinspect.py SRC... [--weight token_embd.weight] [--hash-mb N] -- one weight's directory entry (quantizer, options,
encoding scheme / transform, bytes) in .rad containers, local paths or https URLs (HTTP range reads: only the header,
string blob, directory, encoding table and N MiB from each end of the weight's data are fetched). The head and tail
hash tells identical encodings apart (w135, #174: V2's embedding options). JSON lines on stdout."""
import argparse, hashlib, json, struct, sys, urllib.request

ES = 136                     # RadFileEntry (RAD_MAX_RANK 6)
ENC_STR = 24
ENC_SZ = 2 * 24 + 8 + 8 * 64      # RadEncoding (RAD_ENC_MAX_PLANES 8, RadEncPlane 64 bytes)


def reader(src):
    if src.startswith("http"):
        def rd(off, n):
            q = urllib.request.Request(src, headers={"Range": f"bytes={off}-{off + n - 1}"})
            with urllib.request.urlopen(q, timeout=120) as r: return r.read()
        return rd
    f = open(src, "rb")
    def rd(off, n): f.seek(off); return f.read(n)
    return rd


def inspect(src, weight, hash_mb):
    rd = reader(src); h = rd(0, 4096)
    magic, ver, fbytes = struct.unpack_from("<IIQ", h, 0)
    (str_off, str_bytes, meta_off, meta_cnt, dir_off, dir_cnt, plane_off, plane_cnt, enc_off, enc_cnt) = struct.unpack_from("<10Q", h, 56)
    S = rd(str_off, str_bytes); D = rd(dir_off, ES * dir_cnt)
    s = lambda o: S[o:S.index(b"\0", o)].decode() if o else ""
    out = {"src": src.rsplit("/", 1)[-1], "file_bytes": fbytes, "weight": weight}
    for i in range(dir_cnt):
        e = D[i * ES:(i + 1) * ES]; name, qz, opt = struct.unpack_from("<3Q", e, 0)
        if s(name) != weight: continue
        enc, rank = struct.unpack_from("<II", e, 24); shape = struct.unpack_from("<6q", e, 32)[:rank]
        off, nb = struct.unpack_from("<QQ", e, 104)
        out.update(quantizer=s(qz), options=s(opt), enc=enc, shape=list(shape), offset=off, bytes=nb)
        r = rd(enc_off + enc * ENC_SZ, ENC_SZ)
        out["scheme"] = r[:ENC_STR].split(b"\0")[0].decode(errors="replace")
        out["transform"] = r[ENC_STR:2 * ENC_STR].split(b"\0")[0].decode(errors="replace")
        npl = struct.unpack_from("<i", r, 2 * ENC_STR)[0]
        out["planes"] = [r[56 + 64 * j:56 + 64 * j + ENC_STR].split(b"\0")[0].decode() for j in range(npl)]
        n = min(hash_mb << 20, nb // 2)
        out["hash"] = hashlib.sha256(rd(off, n) + rd(off + nb - n, n)).hexdigest()[:16]
        return out
    out["missing"] = True
    return out



def main(argv=None):
    ap = argparse.ArgumentParser(prog="radinspect.py")
    ap.add_argument("src", nargs="+"); ap.add_argument("--weight", default="token_embd.weight"); ap.add_argument("--hash-mb", type=int, default=4)
    a = ap.parse_args(argv)
    for p in a.src: print(json.dumps(inspect(p, a.weight, a.hash_mb)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
