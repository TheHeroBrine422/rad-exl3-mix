#!/bin/bash
# evprof.sh CARD MODEL TAG [extra radiance flags] -- one calib-A eval with every log line timestamped (seconds since
# start) into $RXM_WORK/evprof/TAG.log; prints the phase marks. Takes cardN.lock like evalq.sh.
: ${RXM_WORK:=/root} ${RXM_RADIANCE:=$RXM_WORK/install-card} ${RXM_KLDBIN:=$RXM_WORK/radiance-kldgold-fast}; export RXM_WORK RXM_RADIANCE RXM_KLDBIN   # paths (README: Paths)
C=$1 M=$2 T=$3; shift 3
cd ${RXM_WORK}; mkdir -p evprof
exec 9>${RXM_WORK}/card$C.lock; flock 9
export PATH=${RXM_RADIANCE}/bin:$PATH RADIANCE_HOME=${RHOME:-${RXM_RADIANCE}/share/radiance} RADIANCE_KLD_GOLDEN=golden-calA
export ROCR_VISIBLE_DEVICES=$C RADIANCE_KLD_MAXTOK=8192
rm -rf ${RXM_WORK}/evprof/run-$T
s=$(date +%s.%N)
${RXM_KLDBIN} --model $M --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 --host-headroom-mib 16384 \
  --max-num-seqs 1 --kld-record ${RXM_WORK}/evprof/run-$T --kld-corpus corpus/calA.jsonl "$@" 2>&1 \
  | while IFS= read -r l; do printf '%7.2f %s\n' "$(echo "$(date +%s.%N) - $s" | bc)" "$l"; done > evprof/$T.log
python3 -I aggscore.py golden-calA ${RXM_WORK}/evprof/run-$T/score.f32 | python3 -c "import json,sys; print('kld', json.load(sys.stdin)['kld']['mean'])"
grep -E "I load: one pass|I kld golden|I kld: [0-9]+ /|recorded pass|kld: top-1|startup report" evprof/$T.log | sed -n '1,4p;$p' | cut -c1-140
tail -n 1 evprof/$T.log | cut -c1-140
rm -rf ${RXM_WORK}/evprof/run-$T
