#!/bin/bash
# damp3.sh CARD SIGMA... -- damping across the K range, held out (user, 2026-11-02). For each SIGMA: one multi-level ourq
# pass (KS levels) over the split-Hessian layers 3 / 10 / 43 (input groups, o/out_proj, gate/up; not down: no split H),
# quantized on H_A (--hin) with --sigma-reg SIGMA for every tensor, ladder code (i8, qtf2); then d2score.py: proxy error
# on H_A (in-sample), H_B (held out) and the published H; dequantized weights deleted after scoring.
# Out /data/damp3/score.jsonl (arm "K<k>-d<sigma>"), log run.log. Rule and report: damp3report.py.
: ${RXM_WORK:=/root}
C=$1; shift
O=/data/damp3 PY=/data/venv-exl3/bin/python SRC=/data/exl3frac/src HS=/data/d2/hsplit
KS=${KS:-0.5,1,2,2.5,3,4,4.5,5.5,6,6.5,7}
mkdir -p $O; log() { echo "$(date -u +%H:%M:%S) card $C $*" | tee -a $O/run.log; }
cd $RXM_WORK; export ROCR_VISIBLE_DEVICES=$C EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build
for d in "$@"; do
  s=$(date +%s)
  ( exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
    $PY -I $O/ourq/ourq.py -K $KS -o "$O/d$d/K{K}" --layers 3-43 --only $O/only.txt --hin $HS/hA --sigma-reg $d --save-wq \
      --device 0 --codebook i8 --qtf2 --frac-ext $SRC > $O/d$d.log 2>&1
    echo "rc $?" >> $O/d$d.log
    A=""; for K in ${KS//,/ }; do A="$A K$K-d$d:ours:$O/d$d/K$K"; done
    $PY -I $O/d2score.py $O/score.jsonl 3,10,43 $HS/hA $HS/hB $RXM_WORK/challenge/hessians/calib $A > $O/score-d$d.log 2>&1 )
  n=$(grep -c "\"arm\": \"K[0-9.]*-d$d\"" $O/score.jsonl)
  log "sigma $d: $(grep '^rc' $O/d$d.log), $n scored rows, $(( ($(date +%s) - s) / 60 )) min"
  [ "$n" -gt 0 ] && rm -rf $O/d$d/K*/wq
done
