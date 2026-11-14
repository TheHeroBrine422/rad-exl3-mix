#!/usr/bin/env python3
"""eprofsum.py DIR [BASE] [--json OUT] -- eprof.sh's in-situ per-op decode times: the last (cumulative) per-op table of
each serve-r<R>-<HOME>-<PACK>.log; per exl3i weight us/call (decode calls dominate: one short prefill call among
~MT), summed per family over the layers (= us per decode step for one call a step) and the head (output.weight);
median over rounds. Prints per pack a family x home table and home / BASE ratios (< 1 = faster). (w139, #177)"""
import glob, json, os, re, statistics as st, sys
from collections import defaultdict

argv = sys.argv[1:]; jout = None
if "--json" in argv: i = argv.index("--json"); jout = argv[i + 1]; del argv[i:i + 2]
d = argv[0]; base = argv[1] if len(argv) > 1 else "base3"
ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+\S+\s+\S+\s+(.*)$")


def last_table(path):
    lines = open(path, errors="replace").read().splitlines()
    st_ = [i for i, l in enumerate(lines) if l.startswith("per-op timing")]
    if not st_: return None
    rows = []
    for l in lines[st_[-1] + 1:]:
        m = ROW.match(l)
        if m: rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(5)))
        elif rows and not l.strip(): break
    return rows


def fam(op):
    m = re.search(r"(blk\.(\d+)\.([\w.]+?)\.weight|output\.weight)", op)
    if not m or "exl3i" not in op: return None, None
    if m.group(1) == "output.weight": return "head", -1
    return m.group(3), int(m.group(2))


res = defaultdict(lambda: defaultdict(list))          # (pack, home) -> family -> [us per step per round]
other = defaultdict(list)
for f in sorted(glob.glob(os.path.join(d, "serve-r*-*.log"))):
    r, h, p = re.match(r"serve-r(\d+)-([^-]+)-(.+)\.log", os.path.basename(f)).groups()
    t = last_table(f)
    if not t: continue
    s = defaultdict(float); tot = 0.0
    for calls, ms, us, op in t:
        tot += ms
        fa, _ = fam(op)
        if fa: s[fa] += us
    for k, v in s.items(): res[(p, h)][k].append(v)
    res[(p, h)]["_trunk"].append(sum(v for k, v in s.items() if k != "head"))
packs = sorted({k[0] for k in res}); homes = sorted({k[1] for k in res}, key=lambda x: (x != base, x))
out = {}
for p in packs:
    fams = sorted({f for h in homes for f in res[(p, h)]}, key=lambda x: (x == "_trunk", x == "head", x))
    print(f"\n### {p}\n\n| family (us / step) | " + " | ".join(homes) + " | " + " | ".join(f"{h} / {base}" for h in homes if h != base) + " |")
    print("|---" * (2 * len(homes)) + "|")
    for fa in fams:
        med = {h: st.median(res[(p, h)][fa]) if res[(p, h)][fa] else float("nan") for h in homes}
        out.setdefault(p, {})[fa] = med
        print(f"| {fa} | " + " | ".join(f"{med[h]:.1f}" for h in homes) + " | "
              + " | ".join(f"{med[h] / med[base]:.3f}" if med.get(base) else "-" for h in homes if h != base) + " |")
if jout: json.dump(out, open(jout, "w"), indent=1)
