#!/bin/bash
# refs183t.sh OUT.jsonl -- #183: reference models' gen-masked KL on t183 (ds8 refs/*-t183.json.rows) + exact container
# bpw (HF file sizes / 26,895,998,464 params) -> {"name", "bpw", "kld"} lines for trackplot183.py
OUT=$1
SZ=$(for r in Hero422/qwen3.8-27b-exl3-v2-rad StillDeadcode/qwen3.8-27b-iqr-v4; do curl -s -m 30 https://huggingface.co/api/models/$r/tree/main; echo; done)
timeout 120 /workspace/rad-local/bin/ssh-ds8.sh '/data/q27/venv/bin/python -I - <<PY
import numpy as np, glob, os, json
g=np.fromfile("/data/q27/w131/slices/golden-t183/gen.u8",np.uint8).astype(bool)
for f in sorted(glob.glob("/data/q27/srch183/refs/*-t183.json.rows")):
    r=np.fromfile(f,np.float32).reshape(len(g),-1)[:,0]; print(json.dumps({"name":os.path.basename(f)[:-15],"kld":float(r[g].mean())}))
PY' < /dev/null > $OUT.k 2>/dev/null || exit 1
SZ="$SZ" python3 - $OUT <<'PY'
import json, os, sys
size = {}
for l in os.environ["SZ"].splitlines():
    if l.strip().startswith("["):
        for f in json.loads(l):
            if f["path"].endswith(".rad"): size[f["path"]] = f["size"]
out = []
for l in open(sys.argv[1] + ".k"):
    r = json.loads(l); n = r["name"]
    f = f"qwen3.8-27b-exl3-v2-{n[3:]}bpw.rad" if n.startswith("v2-") else f"qwen3.8-27b-{n}-v4.rad"
    if f in size: out.append({"name": n, "bpw": size[f] * 8 / 26895998464, "kld": r["kld"]})
open(sys.argv[1], "w").write("".join(json.dumps(x) + "\n" for x in out)); print(len(out), "refs")
PY
