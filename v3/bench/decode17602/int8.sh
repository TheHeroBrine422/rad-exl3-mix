#!/bin/bash
# int8.sh CARD -- #176-02 int8-decode column (w159, lead note 11-14 2:30 PM CST): per uniform level, K order interleaved
# from both ends (1, 7.5, 1.125, 7, ...: drift does not line up with K), build a8-K.rad (mk17602.sh a8), one unstamped e2e
# row i8u<K>-a (ctx 0 + 8k) and one stamped row i8u<K>-s (ctx 0), both with QWEN35_EXL3_AQ_XB=1 EXL3I_MAD_GEMV=1 (int8 rows
# at decode), then delete the pack. Then the act-line held-out mixes h1 / h2 / h3 / h5 (rebuilt from recipes/m-h*.mix), rows
# i8m-h<n>-a / -b. Restartable (finished rows skipped; a pack is rebuilt only if a row of it is missing). (w159)
C=$1; W=/data/q27/w159; cd $W
EV="env=QWEN35_EXL3_AQ_XB=1,EXL3I_MAD_GEMV=1"
mapfile -t KS < <(awk '{print $1}' recipes/uniform.txt)
n=${#KS[@]}; ord=(); i=0; j=$((n - 1))
while [ $i -le $j ]; do ord+=(${KS[$i]}); [ $i -lt $j ] && ord+=(${KS[$j]}); i=$((i + 1)); j=$((j - 1)); done
for k in "${ord[@]}"; do
  [ -f STOP ] && exit 0
  [ -f res/i8u$k-a.jsonl ] && [ -f res/i8u$k-s.jsonl ] && continue
  bash tools/mk17602.sh a8 $k || continue
  printf 'i8u%s-a %s %s\ni8u%s-s %s stamp ctx=0 %s\n' $k $W/packs/a8-$k.rad $EV $k $W/packs/a8-$k.rad $EV > recipes/L-i8-$k.txt
  bash tools/run17602.sh $C recipes/L-i8-$k.txt
  rm -f packs/a8-$k.rad
done
for h in h1 h2 h3 h5; do bash tools/mk17602.sh mix $h recipes/m-$h.mix; done
{ for h in h1 h2 h3 h5; do echo "i8m-$h-a $W/packs/m-$h.rad $EV"; done; for h in h5 h3 h2 h1; do echo "i8m-$h-b $W/packs/m-$h.rad $EV"; done; } > recipes/L-i8-held.txt
bash tools/run17602.sh $C recipes/L-i8-held.txt
for h in h1 h2 h3 h5; do [ -f res/i8m-$h-a.jsonl ] && [ -f res/i8m-$h-b.jsonl ] && rm -f packs/m-$h.rad; done
echo "INT8 DONE"
