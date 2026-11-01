#!/bin/bash
# fracrun.sh -- GPU box: build exl3frac (the V3 menu's fractional quantize kernels as a side extension) and run
# fracsmoke.py on card 0. Waits for the head-slab test (headrun.sh) to finish first: the build needs RAM and the smoke
# test needs card 0. Staged dir: /data/exl3frac/{src = fork exllamav3_ext, e3 = fracsmoke/trellis/e3/viterbi + exl3frac/}.
# Log: /data/exl3frac/run.log. The prebuilt exllamav3 and the live pipeline are not touched.
set -u
W=/data/exl3frac; L=$W/run.log; RW=${RXM_WORK:-/root}
log() { echo "$(date -u +%H:%M) $*" >> $L; }
while pgrep -f "^/bin/bash /data/yaqa/headrun.sh" > /dev/null; do sleep 60; done   # the smoke test then queues on card0.lock
log "head test done; building exl3frac"
cd $W/e3 && gcc -O3 -march=native -fopenmp -shared -fPIC viterbi.c -o libviterbi.so || { log "viterbi build FAILED"; exit 1; }
export ROCM_HOME=${ROCM_HOME:-/opt/rocm/core-10.1} PYTORCH_ROCM_ARCH=gfx1201 MAX_JOBS=4 TORCH_EXTENSIONS_DIR=$W/build EXL3FRAC_SRC=$W/src
( exec 7>$RW/ram.lock; flock -s 7
  /data/venv-exl3/bin/python -c "import sys; sys.path.insert(0, 'exl3frac'); import exl3frac; m = exl3frac.load('$W/src', True); print('BUILT', m.__file__)" ) >> $W/build.log 2>&1
grep -q "^BUILT" $W/build.log || { log "exl3frac build FAILED (see build.log)"; exit 1; }
log "built; smoke test on card 0"
( exec 9>$RW/card0.lock 7>$RW/ram.lock; flock 9; flock -s 7
  ROCR_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 /data/venv-exl3/bin/python fracsmoke.py 8192 256 ) >> $W/smoke.log 2>&1
log "smoke rc=$? : $(tail -1 $W/smoke.log)"
echo FRACRUN DONE >> $L
