#!/usr/bin/env python3
"""report.py INDEX.sqlite [--tensor NAME] -- what the tensor store holds.

  - per source (ours K1..8 / turboderp plain / turboderp SC by size / IQR / other): distinct variants, bytes, packs
  - per pack: entries, and how many of its bytes are unique to it (variants found in no other pack) -- a pack whose
    unique share is ~0 can be deleted without losing a tensor (e.g. a release that is pure reflinks of level packs)
  - turboderp: distinct EXL3 trunk variants by bitrate across all his packs (his size containers share tensors)
  - per tensor (--tensor): every variant with source, bitrate and locations
Bitrate of an EXL3 entry = bytes * 8 / (rows * cols), rounded to 1/16 (the suh/svh planes add < 0.01 bit)."""
import collections, json, re, sqlite3, sys

db = sqlite3.connect(sys.argv[1])
TENSOR = sys.argv[sys.argv.index("--tensor") + 1] if "--tensor" in sys.argv else None


def source(opts, path):
    m = re.search(r"src=(\S+?)(?:,|$)", opts or "")
    s = m.group(1) if m else ""
    if "/ours/K" in s: return "ours-K" + s.split("/ours/K")[1]
    if "td-SC_" in s: return "td-SC " + s.split("td-SC_")[1].split("bpw")[0]
    if "td-" in s: return "td-plain " + s.split("td-")[1].split("bpw")[0]
    if "iqr" in path.lower(): return "IQR"
    return "other: " + (s or path.rsplit("/", 1)[-1])


def bits(shape, nbytes):
    s = json.loads(shape); n = s[0] * s[1] if len(s) >= 2 else s[0]
    return round(nbytes * 8 / n * 16) / 16


rows = db.execute("SELECT v.vid, v.name, v.shape, v.scheme, v.options, v.bytes, l.path FROM variants v "
                  "JOIN locs l ON l.vid = v.vid").fetchall()
var = {}; locs = collections.defaultdict(set)
for vid, name, shape, scheme, opts, nb, path in rows:
    locs[vid].add(path)
    if vid not in var: var[vid] = (name, shape, scheme, opts, nb, path)

if TENSOR:
    for vid, (name, shape, scheme, opts, nb, path) in sorted(var.items(), key=lambda x: (x[1][2], x[1][4])):
        if name != TENSOR: continue
        print(f"{vid[:12]} {scheme:8s} {bits(shape, nb):6.3f} b  {source(opts, path):24s} in {len(locs[vid])} packs: "
              + ", ".join(sorted(p.rsplit('/', 1)[-1] for p in locs[vid]))[:160])
    sys.exit()

print("== by source (distinct variants)")
src = collections.defaultdict(lambda: [0, 0])
for vid, (name, shape, scheme, opts, nb, path) in var.items():
    k = source(opts, path) if scheme == "exl3" else f"({scheme})"
    src[k][0] += 1; src[k][1] += nb
for k in sorted(src): print(f"  {k:28s} {src[k][0]:5d} variants  {src[k][1] / 1e9:7.1f} GB")
tot = sum(v[4] for v in var.values())
print(f"  total {len(var)} variants, {tot / 1e9:.1f} GB of distinct tensor data")

print("\n== per pack: unique bytes (variants in no other pack)")
packs = collections.defaultdict(lambda: [0, 0, 0])
for vid, ps in locs.items():
    for p in ps:
        packs[p][0] += 1; packs[p][1] += var[vid][4]
        if len(ps) == 1: packs[p][2] += var[vid][4]
for p, (n, b, u) in sorted(packs.items(), key=lambda x: x[1][2]):
    print(f"  {u / 1e9:7.2f} / {b / 1e9:6.1f} GB unique  {n:5d} entries  {p}")

print("\n== turboderp: distinct EXL3 trunk variants by bitrate")
td = collections.defaultdict(lambda: collections.Counter())
for vid, (name, shape, scheme, opts, nb, path) in var.items():
    s = source(opts, path)
    if scheme == "exl3" and s.startswith("td-") and name.startswith("blk."):
        td[s.split()[0]][bits(shape, nb)] += 1
for fam, c in td.items():
    print(f"  {fam}: " + "  ".join(f"K{k:g}: {n}" for k, n in sorted(c.items())))
per = collections.Counter(var[v][1] for v in var if var[v][2] == "exl3" and source(var[v][3], var[v][5]).startswith("td-"))
if per: print(f"  his variants per trunk tensor: median {sorted(per.values())[len(per) // 2]}, max {max(per.values())}")
