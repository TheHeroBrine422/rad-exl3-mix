#!/bin/bash
# slimloop.sh -- (w157, #174-03, box2 CPU) slim157.sh over every eighth level as quant157.sh lands them; ends when all
# 21 donors exist (or W/STOP). Log W/logs/slim157.log.
W=/data/q27/w157 D=/data/q27/emb/trellis
KS=$(awk '!/^#/ && NF >= 3 && NF < 4 {print $2}' $W/q-eighths.txt)
while [ ! -f $W/STOP ]; do
  bash $W/tools/slim157.sh $KS 2>&1 | grep -v ': no /' >> $W/logs/slim157.log
  n=0; for k in $KS; do [ -f $D/emb-t-i8-K$k.rad ] && n=$((n + 1)); done
  [ $n -eq $(echo $KS | wc -w) ] && { echo "$(date -u +%T) all $n donors" >> $W/logs/slim157.log; break; }
  sleep 60
done
