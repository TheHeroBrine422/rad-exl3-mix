#!/usr/bin/env python3
"""#185 w162: compare two `rad-info --weights-dump` files weight by weight.

  cmp_dumps.py A.dump B.dump

Records are matched by name (directory order may differ: a pack's and a manifest's directories are not in the same
order). Per weight: the header line (quantizer, options with src= masked, encoding, shape, layer / expert / slot, plane
roles and sizes) must be equal, and every plane's bytes must hash equal (sha256). Prints the counts and the first
mismatches; exit 0 only when every name in both files is identical and neither file has a name the other lacks.
"""
import hashlib
import sys


def records(path):
    out = {}
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith(b"W "):
                raise SystemExit(f"{path}: a record does not start with 'W ': {line[:40]!r}")
            text = line.decode().rstrip("\n")
            name = text[2:].split("\t")[0]
            planes = [(p.split(":")[0], int(p.split(":")[1])) for p in text.split("\t")[-1].split(" ")[1:]]
            hs = []
            for role, n in planes:
                h = hashlib.sha256()
                left = n
                while left:
                    chunk = f.read(min(left, 1 << 24))
                    if not chunk:
                        raise SystemExit(f"{path}: short read in {name} {role}")
                    h.update(chunk)
                    left -= len(chunk)
                hs.append((role, n, h.hexdigest()))
            if name in out:
                raise SystemExit(f"{path}: {name} twice")
            out[name] = (text, hs)
    return out


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    a, b = records(argv[1]), records(argv[2])
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    both = sorted(set(a) & set(b))
    bad = []
    planes_total = 0
    for n in both:
        (ta, ha), (tb, hb) = a[n], b[n]
        if ta != tb:
            bad.append((n, "header", ta, tb))
            continue
        planes_total += len(ha)
        if ha != hb:
            bad.append((n, "planes", [x for x, y in zip(ha, hb) if x != y][:1], None))
    print(f"A {argv[1]}: {len(a)} weights; B {argv[2]}: {len(b)} weights; both {len(both)}")
    print(f"only in A: {len(only_a)} {only_a[:3]}; only in B: {len(only_b)} {only_b[:3]}")
    print(f"planes compared: {planes_total}; identical weights: {len(both) - len(bad)} of {len(both)}")
    for x in bad[:5]:
        print("MISMATCH", x)
    ok = not bad and not only_a and not only_b and both
    print("IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
