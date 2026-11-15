#!/usr/bin/env python3
"""#185 w162: a search mix with some levels' trunk lines served from their final/ (the rest stay pack donors).

  subset_mix.py MIXFILE BASE FINALS_ROOT OUT LEVEL [LEVEL ...]

MIXFILE is a radmix mixfile (`<weight> <donor.rad>`, `act <weight> <mode>`, ...). A line whose donor is the pack of one of
the LEVELS (`.../packs-mad/q27v3-635bac02-<LEVEL>-mad[-s6c57a3d1].rad`) becomes `<weight> final FINALS_ROOT/<LEVEL>-mad/final
codebook=mad`; every other line is copied as it is. OUT is a manifest: `base BASE` first, then the lines. The served bytes of
the converted lines are the pack's (final_mix.py / the k2.5 identity), so the manifest must serve the same KLD rows as the
mixfile radmix writes.
"""
import re
import sys


def main(argv):
    if len(argv) < 6:
        print(__doc__)
        return 2
    mix, base, root, out = argv[1:5]
    levels = argv[5:]
    pat = re.compile(r"^(\S+)\s+(\S*/packs-mad/q27v3-635bac02-(.+?)-mad(?:-s6c57a3d1)?\.rad)\s*$")
    n_conv = 0
    with open(mix) as f, open(out, "w") as o:
        o.write(f"base {base}\n")
        for line in f:
            m = pat.match(line)
            if m and m.group(3) in levels and not line.startswith("act "):
                o.write(f"{m.group(1)} final {root}/q27v3-635bac02-{m.group(3)}-mad/final codebook=mad\n")
                n_conv += 1
            else:
                o.write(line if line.endswith("\n") else line + "\n")
    print(f"{out}: {n_conv} trunk lines from final/ ({', '.join(levels)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
