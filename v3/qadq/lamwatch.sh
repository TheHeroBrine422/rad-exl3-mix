#!/bin/bash
# lamwatch.sh MACHINE (box2) -- waits for the MSE diagnostic (B gn-mse: runs/gn-mse/gradnoise.npz + done/B-gn-mse), runs gn222.py against w217's KL-only pass of the same 40 batches (runs/gn-2.50), saves the report to
# QROOT/gn222.out and writes the lambda of the rule (rms KL mean gradient / rms MSE mean gradient over layers 0-31) to QROOT/lambda.txt, which every `{lam}` queue line waits for. Start with setsid nohup.
MACHINE=$1; HERE0=$(dirname "$(readlink -f "$0")"); source $HERE0/qenv.sh $MACHINE
GN=$QROOT/runs/gn-mse/gradnoise.npz; KL=/data/q27/w217/runs/gn-2.50/gradnoise.npz
qsay "lamwatch: waiting for the gn-mse pass"
while [ ! -e $QROOT/done/B-gn-mse ] || [ ! -s $GN ]; do sleep 60; done
out=$($PY $HERE/gn222.py $GN $KL --n 40 --lo 0 --hi 31 2>&1); echo "$out" > $QROOT/gn222.out
lam=$(echo "$out" | sed -n 's/^lambda .*= \([0-9.eE+-]*\) .*/\1/p')
if $PY -c "import math,sys; v=float('$lam'); sys.exit(0 if math.isfinite(v) and v>0 else 1)" 2>/dev/null; then echo $lam > $QROOT/lambda.txt.part && mv $QROOT/lambda.txt.part $QROOT/lambda.txt; qsay "lamwatch: lambda = $lam written (gn222.out)"
else qsay "lamwatch: gn222 gave no usable lambda ('$lam'): see $QROOT/gn222.out; lambda.txt NOT written"; fi
