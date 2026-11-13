"""radents.py A.rad [B.rad] -- a container's weight directory (the layout headslab.py's entry() reads): name, stored bytes,
shape; with B, every entry's payload sha256 in both, and the names whose bytes differ (the confounder check: two packs that
should differ only in output.weight). Prints JSON. (w136, #164-02)"""
import hashlib, json, struct, sys

ES = 3 * 8 + 4 + 4 + 8 * 6 + 4 * 3 + 4 + 8 + 8 + 8 + 16


def entries(path):
    f = open(path, "rb"); h = f.read(4096)
    str_off, str_bytes, meta_off, meta_cnt, dir_off, dir_cnt = struct.unpack_from("<6Q", h, 16 + 5 * 8)
    f.seek(str_off); S = f.read(str_bytes)
    f.seek(dir_off); D = f.read(ES * dir_cnt)
    out = {}
    for i in range(dir_cnt):
        e = D[i * ES:(i + 1) * ES]
        name = struct.unpack_from("<Q", e, 0)[0]
        rank = struct.unpack_from("<I", e, 28)[0]; shape = struct.unpack_from("<6q", e, 32)[:rank]
        off, nbytes = struct.unpack_from("<QQ", e, ES - 32)
        out[S[name:S.index(b"\0", name)].decode()] = (off, nbytes, list(shape))
    return out


def digest(path, off, n):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(off)
        while n > 0:
            b = f.read(min(n, 1 << 24)); h.update(b); n -= len(b)
    return h.hexdigest()


def main(a, b=None):
    A = entries(a)
    r = {"file": a, "weights": len(A), "output.weight": A.get("output.weight"),
         "payload_bytes": sum(v[1] for v in A.values())}
    if b:
        B = entries(b)
        diff = [k for k in sorted(set(A) | set(B))
                if k not in A or k not in B or A[k][1] != B[k][1] or digest(a, A[k][0], A[k][1]) != digest(b, B[k][0], B[k][1])]
        r.update({"other": b, "other_weights": len(B), "other_output.weight": B.get("output.weight"), "differ": diff})
    return r


if __name__ == "__main__":
    print(json.dumps(main(*sys.argv[1:3])))
