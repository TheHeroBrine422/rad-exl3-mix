#!/bin/bash
# prof2.sh CARD -- speed pass 2: ladder settings (i8, qtf2, damping sweep), layers 3-4 (attention + GDN, both with MLP),
# plain wall time (+ CPU time), the stage profile per rate, and py-spy (native) for the host side at K2.5 / K3. Out /data/prof2/
: ${RXM_WORK:=/root}; export RXM_WORK
C=$1 O=/data/prof2; mkdir -p $O; cd $RXM_WORK
export ROCR_VISIBLE_DEVICES=$C EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build
X="--codebook i8 --qtf2 --frac-ext /data/exl3frac/src --sigma-in-sweep 0.025,0.005,0.001"
exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
for K in ${KS:-3 2.5 1.5 5.5 0.5 4}; do
  QPROF_PLAIN=1 /data/venv-exl3/bin/python -I ourq/qprof3.py $K 3-4 $O/plain-K$K.json $X > $O/plain-K$K.log 2>&1
  /data/venv-exl3/bin/python -I ourq/qprof3.py $K 3-4 $O/prof-K$K.json $X > $O/prof-K$K.log 2>&1
  case " ${SPY:-2.5 3} " in *" $K "*)        # host side: py-spy, native frames, idle included (GPU waits show as such)
    QPROF_PLAIN=1 /data/venv-exl3/bin/py-spy record --native --idle --rate 40 --format raw -o $O/spy-K$K.txt -- \
      /data/venv-exl3/bin/python -I ourq/qprof3.py $K 3-4 $O/spyrun-K$K.json $X > $O/spy-K$K.log 2>&1;; esac
  echo "$(date -u +%T) K$K: $(tail -n +1 $O/plain-K$K.log | grep -a '^K' | tail -1) | $(grep -a '^K' $O/prof-K$K.log | tail -1)" >> $O/summary.txt
done
