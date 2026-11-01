#!/usr/bin/env python3
"""assemble.py INDEX.sqlite SPEC OUT.rad [--base BASE.rad] [--radmix PATH] [--dry] -- build a container from tensor-store
variants by reference.

SPEC lines: "<weight name> <variant>" ('#' comments), variant = a vid (or a unique prefix, >= 8 hex chars) from
scan.py, or "pack:<path>" to take the weight from that pack directly. Weights not named come from BASE (default: the
pack holding most of the named variants, so its header, vocab and metadata are kept). Each variant is resolved to one
of its locations (preferring BASE, then the pack used most often, so fewer files are opened), checked against the
index (name, shape), and handed to radmix as a mix file; radmix clones every weight (FICLONERANGE), so an assembled
container costs no data copy on the reflink filesystem. Afterwards the output is scanned and every requested weight
is checked to be the requested variant (content hash): OUT.rad.manifest.json records spec, base, resolution and the
output's sha256-free identity (vid per weight) -- the "release = what was measured" check."""
import collections, json, os, sqlite3, subprocess, sys, tempfile

args = sys.argv[1:]
def opt(name, default=None):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
    return default
BASE = opt("--base"); RADMIX = opt("--radmix", "/root/hproxy/radmix")
DRY = "--dry" in args
if DRY: args.remove("--dry")
IDX, SPEC, OUT = args[:3]
db = sqlite3.connect(IDX)

spec = []
for line in open(SPEC):
    line = line.split("#", 1)[0].split()
    if len(line) == 2: spec.append(tuple(line))

def locations(vid):
    return [p for (p,) in db.execute("SELECT path FROM locs WHERE vid = ?", (vid,))]

def resolve(name, v):
    if v.startswith("pack:"):
        p = os.path.realpath(v[5:])
        r = db.execute("SELECT v.vid FROM locs l JOIN variants v ON v.vid = l.vid WHERE l.path = ? AND v.name = ?",
                       (p, name)).fetchone()
        if not r: sys.exit(f"{name}: not in pack {p} (scan it first)")
        return r[0]
    rows = db.execute("SELECT vid, name FROM variants WHERE vid LIKE ?", (v + "%",)).fetchall()
    if len(rows) != 1: sys.exit(f"{name}: variant {v} matches {len(rows)} entries")
    if rows[0][1] != name: sys.exit(f"{name}: variant {v} is a {rows[0][1]}")
    return rows[0][0]

want = {name: resolve(name, v) for name, v in spec}
use = collections.Counter(p for vid in want.values() for p in locations(vid))
if not BASE: BASE = use.most_common(1)[0][0]
BASE = os.path.realpath(BASE)
base_entries = {n: vid for n, vid in db.execute(
    "SELECT v.name, v.vid FROM locs l JOIN variants v ON v.vid = l.vid WHERE l.path = ?", (BASE,))}
if not base_entries: sys.exit(f"base {BASE} is not in the index (scan it first)")
mix, plan = [], {}
for name, vid in want.items():
    if name not in base_entries: sys.exit(f"{name}: the base has no such weight")
    if base_entries[name] == vid: plan[name] = BASE; continue          # already the base's own tensor
    locs = locations(vid)
    p = max(locs, key=lambda x: (x == BASE, use[x]))
    plan[name] = p; mix.append(f"{name} {p}")
print(f"base {BASE}: {len(want)} weights requested, {len(want) - len(mix)} already in the base, {len(mix)} from "
      f"{len(set(plan.values()) - {BASE})} other packs")
if DRY:
    for l in mix: print("  ", l)
    sys.exit()
with tempfile.NamedTemporaryFile("w", suffix=".mix", delete=False) as f:
    f.write("\n".join(mix) + "\n"); mf = f.name
subprocess.run([RADMIX, OUT + ".part", BASE, mf], check=True)
os.replace(OUT + ".part", OUT); os.unlink(mf)

# verify: scan the output into a scratch index and compare vids
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scan
chk = sqlite3.connect(":memory:")
chk.executescript("""CREATE TABLE packs(path TEXT PRIMARY KEY, bytes INT, mtime INT, arch TEXT, recipe TEXT, created_by TEXT,
  n_entries INT); CREATE TABLE variants(vid TEXT PRIMARY KEY, name TEXT, shape TEXT, scheme TEXT, quantizer TEXT,
  options TEXT, bytes INT); CREATE TABLE locs(vid TEXT, path TEXT, offset INT);""")
scan.scan_pack(chk, os.path.realpath(OUT))
got = {n: vid for n, vid in chk.execute("SELECT v.name, v.vid FROM variants v JOIN locs l ON l.vid = v.vid")}
bad = [n for n, vid in want.items() if got.get(n) != vid]
json.dump({"spec": SPEC, "base": BASE, "resolution": plan, "weights": got}, open(OUT + ".manifest.json", "w"), indent=1)
print(f"{OUT}: {len(got)} weights; requested variants {'all verified' if not bad else 'MISMATCH: ' + ', '.join(bad[:5])}")
sys.exit(1 if bad else 0)
