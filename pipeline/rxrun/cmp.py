# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""python -m rxrun.cmp A B [--tensors] [--names L08-g0,...] -- byte identity of rxq job outputs (w105).

A, B: dirs of job dirs (OUT/<job>/t-<key>.safetensors, c9-<key>.safetensors, diag-<key>.safetensors), or job dirs, or assembled levels
(q-L*.safetensors, c9/scales/*). Every such file of A must exist in B. Default: every tensor's bytes, dtype, shape
and offsets, and the metadata as a dict (settings stamp, code stamp, K, codebook, level) — i.e. the file's bytes up
to the order of the header's __metadata__ keys, which safetensors writes from a Rust HashMap: random on every write,
even within one process (w105, 11-11), so two `rxq job` runs never give equal whole-file sha256. --raw: whole-file
sha256. --tensors: tensor bytes only, for runs whose code stamps differ by files that do not decide codes; metadata
differences are listed.
Last line: "CMP SAME n files" or "CMP DIFF d of n files"."""
import argparse, hashlib, json, os, struct, sys


def files(root, names=None):
    out = {}
    for d, _, fs in os.walk(root):
        rel = os.path.relpath(d, root)
        if names and rel != "." and rel.split(os.sep)[0] not in names: continue
        for f in fs:
            if f.endswith(".safetensors") and (f.startswith(("t-", "c9-", "diag-", "q-L", "q-head")) or os.path.basename(d) == "scales"):
                out[os.path.normpath(os.path.join(rel, f))] = os.path.join(d, f)
    return out


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()


def tensors(p):
    """({name: (dtype, shape, sha256 of bytes)}, metadata)"""
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8 + n
        meta = hdr.pop("__metadata__", {}); out = {}
        for k, v in hdr.items():
            a, b = v["data_offsets"]; f.seek(base + a)
            out[k] = (v["dtype"], tuple(v["shape"]), (a, b), hashlib.sha256(f.read(b - a)).hexdigest())
    return out, meta


def compare(a, b, tensors_only=False, names=None, quiet=False, raw=False):
    fa, fb = files(a, names), files(b, names)
    if not fa: raise SystemExit(f"{a}: no job / level files")
    diff, meta_diff = [], set()
    for rel, pa in sorted(fa.items()):
        pb = fb.get(rel)
        if pb is None: diff.append((rel, "missing in B")); continue
        if raw:
            if sha(pa) != sha(pb): diff.append((rel, "bytes differ"))
            continue
        ta, ma = tensors(pa); tb, mb = tensors(pb)
        if not tensors_only and ma != mb: diff.append((rel, f"metadata differs: {sorted(k for k in set(ma) | set(mb) if ma.get(k) != mb.get(k))}"))
        if ta != tb:
            bad = sorted(k for k in set(ta) | set(tb) if ta.get(k) != tb.get(k))
            diff.append((rel, f"tensors differ: {bad[:4]}"))
        for k in sorted(set(ma) | set(mb)):
            if ma.get(k) != mb.get(k): meta_diff.add(f"{k}: {ma.get(k)} vs {mb.get(k)}")
    if not quiet:
        for rel, why in diff[:30]: print(f"DIFF {rel}: {why}")
        if tensors_only:
            for m in sorted(meta_diff)[:10]: print(f"metadata (not compared): {m}")
        print(f"CMP SAME {len(fa)} files" if not diff else f"CMP DIFF {len(diff)} of {len(fa)} files")
    return len(fa), diff, sorted(meta_diff)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rxrun.cmp")
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--tensors", action="store_true"); ap.add_argument("--raw", action="store_true")
    ap.add_argument("--names", help="only these job dirs (comma list)")
    x = ap.parse_args(argv)
    n, diff, _ = compare(x.a, x.b, x.tensors, x.names.split(",") if x.names else None, raw=x.raw)
    sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
