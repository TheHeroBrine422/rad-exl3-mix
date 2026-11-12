#!/bin/bash
# convplot3-refresh.sh [MULTIROOT] [OUT.png] -- regenerate the V3 search's convergence plot from rxrun.multi's results
# (w120, 11-12). The coordinator runs on the VM and copies every eval's small result back into MULTIROOT/results.jsonl,
# so nothing is pulled from gpubox / box2. Events (refits, slice / method changes): MULTIROOT/events.jsonl.
# For a timer as V2's convplot-refresh.sh: writes a temp file first, so a viewer never sees a half-written PNG.
set -e
R=$(cd "$(dirname "$0")/.." && pwd); M=${1:-/workspace/rad-local/multi/v3}; O=${2:-/workspace/v3-convergence.png}
[ -s $M/results.jsonl ] || { echo "no $M/results.jsonl yet"; exit 0; }
python3 -W ignore $R/search/convplot3.py $M/results.jsonl ${O%.png}.tmp.png --events $M/events.jsonl > ${O%.png}.last 2>&1
mv ${O%.png}.tmp.png $O
