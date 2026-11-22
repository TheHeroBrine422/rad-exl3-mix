#!/bin/bash
# gpu218.sh LOG CMD... (iceman, #218) -- one GPU step on card 0 (03:00.0) only: holds /root/card0.lock, ROCR_VISIBLE_DEVICES=0,
# refuses to start below 30 GB MemAvailable (our host-RAM cap is 25 GB; the user's server shares the box). Card 1 is never touched.
LOG=$1; shift
cd /data/w218
exec {fd}>/root/card0.lock; flock -n $fd || { echo "card 0 busy" >> $LOG; exit 75; }
AV=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
[ "$AV" -ge 30 ] || { echo "MemAvailable ${AV} GB < 30" >> $LOG; exit 76; }
export ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 TRITON_DEFAULT_FP_FUSION=1 OMP_NUM_THREADS=6 QAD_PIPELINE=/data/w218/code/pipeline
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=67108864
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') start: $*" >> $LOG
nice -n 5 "$@" >> $LOG 2>&1
rc=$?
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') rc $rc: $*" >> $LOG
exit $rc
