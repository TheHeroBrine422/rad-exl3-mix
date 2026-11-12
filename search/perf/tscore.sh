#!/bin/bash
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
# tscore.sh INSTALL HOME MIX GOLDEN OUT CARD [radiance args...] -- one radiance KLD run with hostd's score settings (tp 1,
# bf16 KV, max-model-len 16384, one sequence, host headroom 8 GiB; extra args are appended and override), every log
# line time-stamped, then one JSON line of phases on stdout (w123, #161):
#   start = process start -> first log line; init -> "load:" begins; load -> load done; prep -> "kld: scoring";
#   score -> "documents in"; exit -> process end. OUT.json + OUT.json.rows as score.sh. Caller holds the locks.
I=$1 H=$2 M=$3 G=$4 O=$5 C=$6; shift 6
t0=$(date +%s.%N)
ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$H stdbuf -oL -eL $I/bin/radiance --model $M --tp 1 --kv-cache-dtype bf16 \
  --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 "$@" --kld-ref $G --kld-out $O.json.part 2>&1 \
  | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done > $O.log
rc=${PIPESTATUS[0]}; t1=$(date +%s.%N)
[ $rc = 0 ] && mv $O.json.part.rows $O.json.rows && mv $O.json.part $O.json
python3 -I - "$O.log" "$t0" "$t1" "$rc" "$M" "$G" <<'EOF'
import json, sys
log, t0, t1, rc, m, g = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
T = {}
for l in open(log):
    t, _, s = l.partition(" "); t = float(t)
    T.setdefault("first", t)
    if "I load:" in s: T.setdefault("load0", t); T["load1"] = t
    if "kld: scoring" in s: T["kld0"] = t
    if "documents in" in s: T["kld1"] = t
    if "kld: KLD mean" in s: T["kldm"] = s.split("mean")[1].split()[0]
g_ = lambda a, b: round(T[b] - T[a], 2) if a in T and b in T else None
T["t0"], T["t1"] = t0, t1
print(json.dumps({"rc": rc, "mix": m, "golden": g, "wall": round(t1 - t0, 2), "start": g_("t0", "first"), "init": g_("first", "load0"),
                  "load": g_("load0", "load1"), "prep": g_("load1", "kld0"), "score": g_("kld0", "kld1"), "exit": g_("kld1", "t1"),
                  "kld": T.get("kldm")}))
EOF
