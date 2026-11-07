#!/bin/bash
# pull71.sh -- copy w71's results (out/ JSON, queue logs) from colo and Thor into data/colo, data/thor next to this
# script, then print the merged report: report71.py data/colo/out,data/thor/out data/w64cd
cd "$(dirname "$0")"; B=../../../../local/bin
for h in colo:ssh-cpu1.sh thor:ssh-thor.sh; do
  d=data/${h%%:*}; mkdir -p $d
  $B/${h#*:} 'cd /data/work/w71 && tar cf - out logs/queue.log logs/status scales 2>/dev/null' | tar xmf - -C $d
done
mkdir -p data/w64cd && $B/ssh-cpu1.sh 'cd /data/work/w64/out/cd/mul1_i8 && tar cf - .' | tar xmf - -C data/w64cd
python3 report71.py data/colo/out,data/thor/out data/w64cd
