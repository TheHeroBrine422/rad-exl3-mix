#!/bin/bash
# test27.sh -- 2-document smoke run of mkgold27.py on the 27B (4070 box, 12 GB card: N stages, default 12, under
# VRAM_CAP_GB 7.2), plus the 0.8B on the same 2 documents as a sanity yardstick (the 27B's nll should be lower).
# GPU steps under the 4070's gpu.lock (timeout 14 min). W = work dir (default /data/yb/w18/t27).
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
M=${M:-/data/models/qwen38-27b-bf16}; M08=${M08:-/data/models/qwen35-08b}; W=${W:-/data/yb/w18/t27}
N=${N:-12}; export VRAM_CAP_GB=${VRAM_CAP_GB:-7.2} OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
rm -rf "$W"; mkdir -p "$W"
CUDA_VISIBLE_DEVICES= nice -n 19 python3 "$D/mktest.py" "$M" "$W/all.jsonl"
head -2 "$W/all.jsonl" > "$W/t.jsonl"      # the modeling source (8k tokens) and the model card (3k)
flock /data/yb/gpu.lock timeout 840 bash -c "
  set -e
  python3 $D/mkgold27.py $M $W/t.jsonl $W/g27 --scratch $W/s27 --stages $N --scratch-dtype bf16 --keep-scratch
  python3 $D/mkgold27.py $M08 $W/t.jsonl $W/g08 --scratch $W/s08 --stages 1
"
cat "$W/s27/timing.jsonl"
CUDA_VISIBLE_DEVICES= nice -n 19 python3 - "$W" <<'EOF'
import json, sys, numpy as np
W = sys.argv[1]
for g in ("g27", "g08"):
    m = json.load(open(f"{W}/{g}/kld.json")); R = m["rows"]
    rows = np.fromfile(f"{W}/{g}/rows.f32", np.float32).reshape(R, 3).astype(np.float64)
    top = np.memmap(f"{W}/{g}/top.bin", np.dtype([("id", "<i4"), ("lp", "<f4")]), "r", shape=(R, m["top"]))
    tk = np.fromfile(f"{W}/{g}/tokens.i32", np.int32)
    out = []
    for d, off in zip(m["docs"], np.cumsum([0] + [x["tokens"] for x in m["docs"]])):
        r = slice(d["row0"], d["row0"] + d["tokens"] - 1 - d["score_from"])
        nxt = tk[off + d["score_from"] + 1: off + d["tokens"]]
        out.append({"src": d["source"], "rows": r.stop - r.start, "ppl": round(float(np.exp(rows[r, 0].mean())), 3),
                    "ent": round(float(rows[r, 1].mean()), 3), "tail": float(rows[r, 2].mean()),
                    "top1_is_next": round(float((top["id"][r, 0] == nxt).mean()), 4),
                    "head_mass": round(float(np.exp(top["lp"][r].astype(np.float64)).sum(1).mean()), 6),
                    "head_plus_tail_minus_1_max": float(np.abs(np.exp(top["lp"][r].astype(np.float64)).sum(1) + rows[r, 2] - 1).max()),
                    "lp_sorted": bool((np.diff(top["lp"][r], axis=1) <= 0).all())})
    print(g, json.dumps(out))
EOF
