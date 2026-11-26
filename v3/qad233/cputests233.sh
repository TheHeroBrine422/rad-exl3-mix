#!/bin/bash
# cputests233.sh CODE_DIR OLD_QAD220 OUT_DIR (box2 CPU; setsid nohup) -- the trainer's CPU tests from CODE_DIR: test220, testhf225 (Q220_OLD = OLD_QAD220:
# defaults bit for bit with that trainer), testhf231, testbranch231, testplan231, test233 (cpu), testhf233; one log each in OUT_DIR, a summary line each in OUT_DIR/summary.txt
C=$1 OLD=$2 O=$3; mkdir -p $O; PY=/data/q27/venv/bin/python
export OMP_NUM_THREADS=4 QAD_PIPELINE=$C/pipeline PYTHONDONTWRITEBYTECODE=1 HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=     # no fla on the path: the torch GDN on the CPU (w231 README)
run() { local n=$1; shift; (cd $(dirname $1) && nice -n 10 timeout 5400 env "${ENVX[@]}" $PY "$@") > $O/$n.txt 2>&1; echo "$n rc $? $(grep -cE '^PASS' $O/$n.txt) pass $(grep -cE '^FAIL' $O/$n.txt) fail $(tail -n 1 $O/$n.txt)" >> $O/summary.txt; }
ENVX=(X=1); run test220 $C/v3/qad220/test220.py
ENVX=(Q220_OLD=$OLD); run testhf225 $C/v3/qad225/testhf225.py
ENVX=(X=1); run testhf231 $C/v3/qad231/testhf231.py
ENVX=(X=1); run testbranch231 $C/v3/qad231/testbranch231.py
ENVX=(X=1); run testplan231 $C/v3/qad231/testplan231.py
ENVX=(X=1); run test233 $C/v3/qad233/test233.py cpu
ENVX=(X=1); run testhf233 $C/v3/qad233/testhf233.py
echo done >> $O/summary.txt
