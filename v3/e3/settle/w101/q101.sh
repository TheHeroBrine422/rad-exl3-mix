#!/bin/bash
# q101.sh [P] -- w101 #131 queue on colo (../w101.md): the 8 tensors' preps (8 at once), then 8 x 19 (tensor, K) jobs,
# P at once (default 7; ~1 GB RSS each), nice 19, one thread each. Restartable: finished preps / rows files are skipped
# (sec101.py checks), so rerunning resumes. Stop: touch $W/STOP (checked before each job starts). W101 = work dir
# (default colo's); KORDER=rev runs the K list from the other end (Thor; its rows are copied into colo's rows/).
# Usage (colo): setsid nohup bash /data/work/w13/w101/e3/settle/w101/q101.sh > /data/work/w13/w101/sec/q.log 2>&1 < /dev/null &
W=${W101:-/data/work/w13/w101}; S=$W/e3/settle/w101; OUT=$W/sec; P=${1:-7}
export E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib W87_HOUT=/data/work/hout27/hout-bn \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p $OUT/logs; cd $S
T="3:attn_qg 19:attn_qg 39:attn_qg 63:attn_qg 2:ssm_qkv 10:ssm_qkv 30:ssm_qkv 50:ssm_qkv"
run() { [ -f $W/STOP ] && exit 0; nice -n 19 python3 sec101.py "$@" >> $OUT/logs/$2-$3.log 2>&1 || echo "FAILED $*" >> $OUT/q.err; }
export -f run; export W OUT S
echo "$(date -u +%H:%M:%S) preps" >> $OUT/q.log
for t in $T; do echo "prep ${t%%:*} ${t##*:} $OUT"; done | xargs -P 8 -L 1 bash -c 'run "$@"' _
for t in $T; do [ -f $OUT/prep/${t%%:*}-${t##*:}.pt ] || { echo "prep missing $t" >> $OUT/q.err; exit 1; }; done
echo "$(date -u +%H:%M:%S) jobs" >> $OUT/q.log
KS="2.5 4 2 3 3.5 4.5 1.5 5 2.25 2.75 3.25 3.75 4.25 4.75 1.75 1.25 1 5.25 5.5"
[ "${KORDER:-}" = rev ] && KS=$(echo $KS | tr ' ' '\n' | tac | tr '\n' ' ')   # a second box (Thor) from the other end
for K in $KS; do
  for t in $T; do echo "job ${t%%:*} ${t##*:} $K $OUT"; done
done | xargs -P $P -L 1 bash -c 'run "$@"' _
echo "$(date -u +%H:%M:%S) Q101DONE $(ls $OUT/rows/*.json | wc -l) rows files" >> $OUT/q.log
