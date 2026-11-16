#!/bin/bash
# readout200.sh OUTDIR (w187, #200, box2) -- the RULES200 readout over the served arms that exist (analyse198.py: per-document
# bootstrap, 2,000 reps, seed 0; the same resample for every arm). A contrast is written only when both of its arms are served.
OUT=${1:-analysis}
cd /data/q27/w187 && source env.sh
declare -A RES=([D0]=/data/q27/w170/served/L0B0T0Q0D0/result.json [A]=/data/q27/w182/served/A/result.json
                [As1]=/data/q27/w186/served/A-s1/result.json)
for t in A4M P20-4M B4 P50-4M B4P20 s2 s3 S4seed; do [ -e served/$t/result.json ] && RES[$t]=served/$t/result.json; done
ARGS=(); for t in "${!RES[@]}"; do ARGS+=(--arm "$t=${RES[$t]}"); done
PAIRS=()
add() { [ -n "${RES[$1]:-}" ] && [ -n "${RES[$2]:-}" ] && PAIRS+=(--pair "$1:$2"); }   # both served
add A D0; add As1 D0; add A4M D0; add P20-4M A4M; add P50-4M A4M; add B4 A4M; add B4 D0; add P20-4M D0
add B4P20 B4; add B4P20 D0; add P50-4M D0; add s2 D0; add s3 D0; add S4seed D0; add B4 S4seed
BEST=()
[ -n "${RES[S4seed]:-}" ] && BEST+=(--best "S4seed:D0,As1,s2,s3")
$PY code/v3/qad200/analyse198.py $OUT "${ARGS[@]}" "${PAIRS[@]}" "${BEST[@]}" > logs/readout-$(basename $OUT).out 2>&1
echo "rc $?"
