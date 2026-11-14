#!/usr/bin/env python3
"""whash.py RAD_INFO A.rad [B.rad] [--prefix P ...] -- per-weight byte hashes of .rad containers (w153, #180).

Reads `rad-info -v` for each weight's absolute data offset, hashes [offset, next offset) of the file (the weight's bytes
plus its alignment padding) and prints one line per weight whose name starts with a --prefix (default: every weight):
name, bytes, sha256[:16]. With two containers: the weights both hold, how many are byte-identical, and every one that
differs or is held by one side only. Also prints the bytes per prefix (what a section adds to a container)."""
import hashlib, re, subprocess, sys

args = sys.argv[1:]
prefixes = []
while "--prefix" in args:
    i = args.index("--prefix"); prefixes.append(args[i + 1]); del args[i:i + 2]
info, files = args[0], args[1:]
LINE = re.compile(r"^\s+(\S+)\s+.*?\s@(\d+)(?:\s|$)")


def table(path):
    out = subprocess.run([info, "-v", path], capture_output=True, text=True, check=True).stdout
    ws = sorted({(int(m.group(2)), m.group(1)) for m in map(LINE.match, out.splitlines()) if m and m.group(1) != "plane"})
    with open(path, "rb") as f:
        f.seek(0, 2); end = f.tell()
        res = {}
        for k, (off, name) in enumerate(ws):
            hi = ws[k + 1][0] if k + 1 < len(ws) else end
            if prefixes and not any(name.startswith(p) for p in prefixes):
                continue
            f.seek(off); h = hashlib.sha256(); n = hi - off
            while n > 0:
                b = f.read(min(n, 1 << 24)); h.update(b); n -= len(b)
            res[name] = (hi - off, h.hexdigest()[:16])
    return res


T = [table(p) for p in files]
for p, t in zip(files, T):
    per = {}
    for name, (n, _) in t.items():
        key = next((q for q in prefixes if name.startswith(q)), "*")
        per[key] = per.get(key, 0) + n
    print(f"{p}: {len(t)} weight(s); " + ", ".join(f"{k} {v:,} B ({v / 2**30:.3f} GiB)" for k, v in sorted(per.items())))
if len(T) == 1:
    for name, (n, h) in sorted(T[0].items()):
        print(f"  {name} {n} {h}")
else:
    a, b = T
    both = sorted(set(a) & set(b))
    same = [k for k in both if a[k] == b[k]]
    print(f"common {len(both)}, byte-identical {len(same)}, differ {len(both) - len(same)}, "
          f"only A {len(set(a) - set(b))}, only B {len(set(b) - set(a))}")
    for k in both:
        if a[k] != b[k]:
            print(f"  DIFF {k} A {a[k]} B {b[k]}")
    for k in sorted(set(a) ^ set(b)):
        print(f"  ONLY {'A' if k in a else 'B'} {k}")
