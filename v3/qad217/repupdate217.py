#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""repupdate217.py -- (VM) regenerate the tables of the report: runs table217.py and blowup217.py on box2 and writes their output between the <!-- TABLES --> markers of
rad-local/workers/out/w217.md, plus data/qad217/table217.md / blowup217.md. The ssh helper path is the VM's."""
import subprocess, re, sys
B = "/workspace/rad-local/bin/ssh-box2-slow.sh"; R = "/workspace/rad-local/workers/out/w217.md"; D = "/workspace/rad-exl3-mix/data/qad217"
def box(cmd): return subprocess.run([B, f"cd /data/q27/w217; source code/v3/qad217/env217.sh; {cmd}"], capture_output=True, text=True).stdout
tab = box("$PY -I code/v3/qad217/table217.py scores slices/golden-j217 eps217.json --json /tmp/table217.json 2>/dev/null | grep -v '| - | - | - |.*| - | - | - | - | - |$'; cp /tmp/table217.json table217.json")
blw = box("$PY -I code/v3/qad217/blowup217.py scores slices/golden-j217 slices/j217.jsonl $MODEL --json blowup217.json 2>/dev/null")
open(f"{D}/table217.md", "w").write(tab); open(f"{D}/blowup217.md", "w").write(blw)
for f in ("table217.json", "blowup217.json"): open(f"{D}/{f}", "w").write(box(f"cat {f}"))
s = open(R).read(); body = "<!-- TABLES -->\n## Step 2 tables (auto-generated; ratio vs the unhealed release on j217, paired per document, 95 % CI, (docs better / worse))\n" + tab + "\n## Blow-up documents (auto-generated)\n" + blw + "\n<!-- /TABLES -->"
if "<!-- TABLES -->" in s: s = re.sub(r"<!-- TABLES -->.*<!-- /TABLES -->", lambda m: body, s, flags=re.S)
else: s = s.replace("## Deviations", body + "\n\n## Deviations", 1)
open(R, "w").write(s); print(len(tab), len(blw))
