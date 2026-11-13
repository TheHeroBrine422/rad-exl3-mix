#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""radfetch.py URL OUT.rad [--weight token_embd.weight] -- a sparse local copy of a remote .rad container holding only
what radmix reads from a donor for one weight (w135, #174: HF's xet download of whole V2 containers stalled on ds8):
the region before the data section (header, strings, metadata, tables, vocab), anything after the data section (radmix
appends its tables there), and the weight's own byte range; the rest of the data section stays a hole (reads as
zeros). HTTP range requests in 64 MiB chunks with retries; restartable (a chunk list in OUT.part.done). Only for a
radmix donor of that weight: never load the result as a model.
  --plan: print only the regions [[offset, bytes], ...] and the file size (JSON), for a relay that streams the ranges
  from a faster host (relay174.sh: the VM fetches, ds8 writes; ds8's own HF reads ran at 0.4 MiB/s)."""
import argparse, json, os, struct, sys, time, urllib.request

ES = 136
CH = 64 << 20


def get(url, off, n, tries=8):
    for k in range(tries):
        try:
            q = urllib.request.Request(url, headers={"Range": f"bytes={off}-{off + n - 1}"})
            with urllib.request.urlopen(q, timeout=300) as r: b = r.read()
            if len(b) == n: return b
            raise IOError(f"short read {len(b)} / {n}")
        except Exception as e:                                          # noqa: BLE001
            print(f"retry {k + 1} at {off}: {e}", file=sys.stderr, flush=True); time.sleep(5 * (k + 1))
    raise SystemExit(f"range {off}+{n} failed")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="radfetch.py")
    ap.add_argument("url"); ap.add_argument("out"); ap.add_argument("--weight", default="token_embd.weight")
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args(argv)
    h = get(a.url, 0, 4096)
    fbytes = struct.unpack_from("<Q", h, 8)[0]
    str_off, str_bytes, _, _, dir_off, dir_cnt = struct.unpack_from("<6Q", h, 56)
    data_off, data_bytes = struct.unpack_from("<2Q", h, 56 + 14 * 8)
    S = get(a.url, str_off, str_bytes); D = get(a.url, dir_off, ES * dir_cnt)
    ent = None
    for i in range(dir_cnt):
        e = D[i * ES:(i + 1) * ES]; o = struct.unpack_from("<Q", e, 0)[0]
        if S[o:S.index(b"\0", o)].decode() == a.weight: ent = struct.unpack_from("<QQ", e, 104)
    if ent is None: raise SystemExit(f"{a.weight} not in {a.url}")
    regions = [(0, data_off), (data_off + data_bytes, fbytes - data_off - data_bytes), ent]
    if a.plan: print(json.dumps({"file_bytes": fbytes, "regions": [list(r) for r in regions]})); return 0
    part = a.out + ".part"; donef = part + ".done"
    done = set(json.load(open(donef))) if os.path.exists(donef) and os.path.exists(part) else set()
    with open(part, "ab"): pass
    os.truncate(part, fbytes)
    tot = sum(n for _, n in regions); t0 = time.time(); got = 0
    with open(part, "r+b") as f:
        for off, n in regions:
            for c in range(off, off + n, CH):
                m = min(CH, off + n - c)
                if c in done: got += m; continue
                f.seek(c); f.write(get(a.url, c, m)); f.flush(); os.fsync(f.fileno())
                done.add(c); json.dump(sorted(done), open(donef, "w")); got += m
                print(f"{got >> 20} / {tot >> 20} MiB ({got / max(time.time() - t0, 1e-9) / 2 ** 20:.1f} MiB/s)", file=sys.stderr, flush=True)
    os.replace(part, a.out); os.remove(donef)
    print(json.dumps({"out": a.out, "file_bytes": fbytes, "fetched": tot, "weight": a.weight, "entry": list(ent)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
