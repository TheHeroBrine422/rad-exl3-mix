#!/bin/bash
# runq.sh JOBS RESULTS... -- run sweep job lines (e3.py / trellis.py args) from JOBS in order, skipping each job whose
# results already exist when it would start, and any identical job already running (from any runner).
# Env limits, all checked before each start:
#   MAX     this runner's own jobs at once (default 8)
#   MAXALL  python sweep jobs on the machine at once, whoever started them (default 48)
#   MEMGB   start only while the container's cgroup memory is below this many GiB (default 100)
#   GAP     seconds after each start, so a new job's memory shows up before the next check (default 2; ~60 for
#           ffn_down, which grows to ~11 GiB)
# Also LOG, E3_CKPT, E3_HESS (passed through to the jobs).
: ${RXM_LOGS:=/workspace/logs}; export RXM_LOGS   # paths (README: Paths)
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
J=$1; shift
busy() {
  [ $(jobs -rp | wc -l) -ge ${MAX:-8} ] || [ $(pgrep -fc '^python3 -W ignore') -ge ${MAXALL:-48} ] ||
  [ $(( $(cat /sys/fs/cgroup/memory.current) >> 30 )) -ge ${MEMGB:-100} ]
}
while read -r line; do
  while busy; do sleep 20; done
  [ -z "$(echo "$line" | python3 skipdone.py "$@")" ] && continue
  pgrep -fx "python3 -W ignore $line" > /dev/null && continue
  python3 -W ignore $line > /dev/null 2>> ${LOG:-${RXM_LOGS}}/e3-runq.err &
  sleep ${GAP:-2}
done < "$J"
wait
echo "RUNQDONE $J" >> ${LOG:-${RXM_LOGS}}/e3-runq.err
