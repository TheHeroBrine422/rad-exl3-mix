#!/bin/bash
# run187.sh SNAP OUT [PAR] -- every replay job of RULES-187 (one refit point x arm), PAR at a time, nice 19, single-threaded.
# Restartable: a job whose output file exists is skipped. Run on colo from the code tree (replay.py's folder).
set -u
SNAP=$1; OUT=$2; PAR=${3:-6}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
python3 -I "$HERE/replay.py" jobs "$SNAP" > "$OUT/jobs.txt"
cat "$OUT/jobs.txt" | xargs -P "$PAR" -L 1 bash -c '
  N=$0; ARM=$1; G=$2; OUT='"$OUT"'; SNAP='"$SNAP"'; HERE='"$HERE"'
  f="$OUT/$ARM/$(printf %04d $N)-$G.json"
  [ -f "$f" ] && exit 0
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 19 python3 -I "$HERE/replay.py" job "$SNAP" "$OUT" "$N" "$ARM" "$G" \
    > "$OUT/$ARM-$(printf %04d $N)-$G.log" 2>&1 || echo "FAILED $N $ARM $G" >> "$OUT/failed.txt"
'
echo "done: $(ls "$OUT"/*/ 2>/dev/null | grep -c json) json files; failed: $(cat "$OUT/failed.txt" 2>/dev/null | wc -l)"
