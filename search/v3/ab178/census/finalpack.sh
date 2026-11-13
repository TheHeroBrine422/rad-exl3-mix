#!/bin/bash
# finalpack.sh MIXFILE BASE OUT.rad [census flags ...] -- a release / final-pick container: radmix MIXFILE over BASE, then
# store/census.py with the full byte check against the mix's donors and BASE plus the release expectations; OUT.rad
# appears only on CENSUS PASS (a failed container is kept as OUT.rad.bad for inspection, exit 1). w142 (#178, mining B2).
# Default expectations (V3 release, 11-13): A8 trunk, codebook mul1i8, the R16 exl3i head, encoding version <= 3.
# Census flags given after OUT replace the defaults (e.g. a codebook decision from #177: --expect-codebook X ...).
# Env: RADMIX (radmix binary), PY (python), CENSUS (store/census.py), RXM (rad-exl3-mix checkout). Locks are the
# caller's (the host's conv.lock + ram.lock rule for radmix).
set -euo pipefail
[ $# -ge 3 ] || { echo "finalpack.sh MIXFILE BASE OUT.rad [census flags ...]" >&2; exit 2; }
MF=$1 BASE=$2 OUT=$3; shift 3
RXM=${RXM:-$(cd "$(dirname "$0")/../../../.." && pwd)}
RADMIX=${RADMIX:-radmix} PY=${PY:-python3} CENSUS=${CENSUS:-$RXM/store/census.py}
if [ $# -gt 0 ]; then EXP=("$@")
else EXP=(--expect-act a8g128 --expect-codebook mul1i8 --expect-head 'exl3i mul1i8 K* r16' --max-version 3); fi
rm -f "$OUT.part" "$OUT.bad"
"$RADMIX" "$OUT.part" "$BASE" "$MF"
if "$PY" -I "$CENSUS" "$OUT.part" --mix "$MF" --base "$BASE" "${EXP[@]}" --json "$OUT.census.json" | tee "$OUT.census.txt"; then
  mv "$OUT.part" "$OUT"; echo "finalpack: $OUT"
else
  mv "$OUT.part" "$OUT.bad"; echo "finalpack: CENSUS FAIL, kept as $OUT.bad (see $OUT.census.txt)" >&2; exit 1
fi
