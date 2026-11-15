#!/usr/bin/env python3
"""#185 w173: the copy / plane order of the 49 mad levels (the lead's order: K >= 6 first, then sub-1, then the middle).

  make_order.py COUNTS OUT

COUNTS is `uniq -c` output of the pack names used by the newest 300 live mixes (ds8 srch183/mixes). Inside each group the
least-used level comes first (ties: lower K). Sub-1 levels have no uses in the newest 300 mixes and sort by K.
"""
import re
import sys


def level_k(name):
    return float(re.search(r"-k([0-9.]+)-mad$", name).group(1))


def main(argv):
    counts = {}
    for line in open(argv[1]):
        parts = line.split()
        if len(parts) == 2:
            counts[parts[1]] = int(parts[0])
    names = [f"q27v3-635bac02-k{k}-mad" for k in
             ["0.125", "0.25", "0.375", "0.5", "0.625", "0.75", "0.875", "1", "1.125", "1.25", "1.375", "1.5", "1.625",
              "1.75", "1.875", "2", "2.125", "2.25", "2.375", "2.5", "2.625", "2.75", "2.875", "3", "3.125", "3.25",
              "3.375", "3.5", "3.625", "3.75", "3.875", "4", "4.125", "4.25", "4.375", "4.5", "4.625", "4.75", "4.875",
              "5", "5.25", "5.5", "5.75", "6", "6.25", "6.5", "6.75", "7", "7.5"]]
    assert len(names) == 49 and len(set(names)) == 49
    for n in names:
        counts.setdefault(n, 0)
    hi = [n for n in names if level_k(n) >= 6]
    sub1 = [n for n in names if level_k(n) < 1]
    mid = [n for n in names if 1 <= level_k(n) < 6]
    order = (sorted(hi, key=lambda n: (counts[n], level_k(n)))
             + sorted(sub1, key=level_k)
             + sorted(mid, key=lambda n: (counts[n], level_k(n))))
    with open(argv[2], "w") as f:
        f.write("\n".join(order) + "\n")
    for n in order:
        print(f"{n}\tK={level_k(n)}\tlast300={counts[n]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
