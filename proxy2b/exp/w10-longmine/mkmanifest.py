#!/usr/bin/env python3
"""mkmanifest.py RADLS.jsonl SCOREDIR GOLDEN_DIR[,..] MIXDIR[,..] > MANIFEST.tsv -- longmine.py's manifest from the
long-trace score files (gpubox: /root/longscores/<tag>.f32). Per file:
  golden: the GOLDEN_DIR whose row count equals the file's (golden-long2 and its capped copies golden-long2-cCAP
          differ in rows); none or several -> unresolved;
  bits:   <tag>.mix or <tag minus a long- / longcap- prefix>.mix in a MIXDIR; else a uniform container whose name ends
          the tag (...-sc4.00-e4 -> RADLS's q38-27b-sc4.00-e4.rad); longverify.sh does not record the mix of a
          release container (v2-<size>), so those usually need a hand-written line;
Unresolved files go to stderr as '# UNRESOLVED <tag>: <why>' (and as comment lines in the manifest): fill them in by
hand or leave them out; longmine.py reports the arms it used."""
import glob, json, os, re, sys
import numpy as np
radls, sd, gds, mds = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4].split(",")
conts = [json.loads(l)["c"] for l in open(radls)]
rows = {}
for g in gds:
    for d in glob.glob(g):
        if os.path.exists(f"{d}/kld.json"): rows.setdefault(json.load(open(f"{d}/kld.json"))["rows"], []).append(d)
for f in sorted(glob.glob(f"{sd}/*.f32")):
    tag = os.path.basename(f)[:-4]; n = os.path.getsize(f) // 16
    why = []
    gl = rows.get(n, [])
    if len(gl) != 1: why.append(f"{len(gl)} goldens with {n} rows")
    src, base = None, re.sub(r"^(longcap-|long-)", "", tag)
    for md in mds:                                   # a mix of that name first (V2 mix names contain sc4.00-e4 too)
        for t in (tag, base):
            if src is None and os.path.exists(f"{md}/{t}.mix"): src = f"{md}/{t}.mix"
    if src is None:                                  # a uniform container: the tag ends with its name
        for c in conts:
            m = re.search(r"q38-27b-(.+)\.rad$", c)
            if m and "bf16" not in c and tag.endswith(m.group(1)): src = c
    if src is None: why.append("no container or mix for the tag")
    if why:
        print(f"# UNRESOLVED {tag}: {'; '.join(why)}"); print(f"# UNRESOLVED {tag}: {'; '.join(why)}", file=sys.stderr)
    else: print(f"{tag}\t{f}\t{gl[0]}\t{src}")
