#!/usr/bin/env python3
"""census_manifest.py [--census PATH/census.py] MANIFEST [census.py options] -- store/census.py's checks on what a
manifest serves (w162, #185), without writing the mix.

A manifest is radmix's mixfile behind a `base PATH` line (the reader is rad-exl3-mix's fork radiance, core/format/radfile.cpp:
open_manifest). This script builds the directory that manifest serves, by radmix's rules (base order; a donor's weight
replaces the base's; `drop` leaves a weight and its act key out; `add` goes after the last kept base weight of its layer,
else at the end; `act` sets the key), as census.py's own Rad object, and then runs census.py's main() on it. So every
census check (--mix, --base, --expect-*, --max-version, the donor-name rules, the act keys) runs unchanged on the manifest.

What it does not check: the bytes. Each served weight's bytes are its donor's, by construction; the byte identity of the whole
served file with radmix's output is `rad-info --image-dump` + cmp (identity.sh). The census object's `size` is 0 (the
served file's size is rad-info --bytes); the printed bpw is therefore 0 for a manifest.

Usage: python3 -I census_manifest.py [--census PATH] MANIFEST --mix MIXFILE --base BASE --fast [--expect-act a8g128 ...]
       MIXFILE is the same mixfile without the `base` line (census.py's --mix reads only the weight / act / add / drop lines).
Exit status: census.py's."""
import importlib.util, os, struct, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CENSUS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "store", "census.py"))


def load_census(path):
    spec = importlib.util.spec_from_file_location("census_base", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_manifest(path):
    """radmix's mixfile reading (mix/radmix.cpp): '#' comments; the first `base` line; act / add / drop / <weight> <donor>."""
    base, mix, act, add, drop = None, {}, {}, {}, set()
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) == 2 and p[0] == "base":
                base = p[1]
            elif len(p) == 3 and p[0] == "act":
                act[p[1]] = p[2]
            elif len(p) == 3 and p[0] == "add":
                add[p[1]] = p[2]
            elif len(p) == 2 and p[0] == "drop":
                drop.add(p[1])
            elif len(p) >= 3 and p[1] == "final":
                # a final/ line (w162 step 4, #185 B): this script does not model its entry yet. Refuse, so the gate
                # fails closed; the (B) check is rad-info --weights-dump + cmp_dumps.py (see README)
                sys.exit(f"{path}: a `final` line ({p[0]}) is not checked by census_manifest.py; use the weights-dump check")
            elif len(p) == 2:
                mix[p[0]] = p[1]
    if base is None:
        sys.exit(f"{path}: no `base PATH` line")
    return base, mix, act, add, drop


def manifest_path(manifest, p):
    return p if p.startswith("/") else os.path.join(os.path.dirname(os.path.abspath(manifest)), p)


class VirtualRad:
    """census.Rad's interface (w, meta, recipe, size, ident) for the directory a manifest serves."""

    def __init__(self, census, manifest):
        self.path = manifest
        base_p, mix, act, add, drop = parse_manifest(manifest)
        base_p = manifest_path(manifest, base_p)
        donors = {}

        def donor(p):
            p = manifest_path(manifest, p)
            if p not in donors:
                donors[p] = census.Rad(p)
            return donors[p]

        B = census.Rad(base_p)
        self.base = B
        self.src = {}                 # name -> the Rad whose entry it is (for ident)
        self.w = {}
        self.meta = dict(B.meta)
        self.recipe = B.recipe
        self.size = 0
        # the kept base weights in base order, each replaced by its donor's entry when mixed
        kept = []                     # (base index, name, source Rad)
        for i, name in enumerate(B.w):
            if name in drop:
                continue
            if name in mix:
                D = donor(mix[name])
                if name not in D.w:
                    sys.exit(f"{mix[name]} has no weight {name}")
                if D.w[name]["shape"] != B.w[name]["shape"]:
                    sys.exit(f"{name}: shape differs in {mix[name]}")
                kept.append((i, name, D))
            else:
                kept.append((i, name, B))
        # each added weight goes after the last kept base weight of its layer (else at the end), in name order
        def layer_of(n):
            if not n.startswith("blk."):
                return ""
            p = n.find(".", 4)
            return "" if p < 0 else n[:p + 1]
        after, at_end = {}, []
        for name in sorted(add):
            ly = layer_of(name)
            last = None
            for i, n in enumerate(B.w):
                if ly and n not in drop and layer_of(n) == ly:
                    last = i
            (after.setdefault(last, []) if last is not None else at_end).append(name)
        order = []
        for i, name, S in kept:
            order.append((name, S))
            for n in after.get(i, []):
                order.append((n, donor(add[n])))
        for n in at_end:
            order.append((n, donor(add[n])))
        for name, S in order:
            if name not in S.w:
                sys.exit(f"{S.path} has no weight {name}")
            self.w[name] = S.w[name]
            self.src[name] = (S, name)
        # the metadata: base's, minus the dropped weights' act keys, with the act lines set
        for name in drop:
            self.meta.pop(f"exl3.act.{name}", None)
        for name, mode in act.items():
            self.meta[f"exl3.act.{name}"] = mode

    def ident(self, name, fast=False):
        S, n = self.src[name]
        return S.ident(n, fast)


def main():
    args = sys.argv[1:]
    census_path = CENSUS
    if args[:1] == ["--census"]:
        census_path = args[1]
        del args[:2]
    if not args:
        sys.exit(__doc__)
    census = load_census(census_path)
    Orig = census.Rad

    def Rad(path):
        with open(path, "rb") as f:
            head = f.read(5)
        if head == b"base ":
            return VirtualRad(census, path)
        return Orig(path)

    census.Rad = Rad
    sys.argv = [census_path] + args
    census.main()


if __name__ == "__main__":
    main()
