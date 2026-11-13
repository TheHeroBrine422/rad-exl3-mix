#!/bin/bash
# loo174.sh IN.json OUTDIR TAG... -- (w135, #174, colo) loo174.py for 4 arms (embedding term at P_0 "in" / after the
# compounding "out" x V2 prior off / on) x every calibration eval held out + none, 8 fits at a time, nice 19,
# single-threaded. Restartable (finished outputs are skipped).
set -u
IN=$1; O=$2; shift 2; mkdir -p $O; cd "$(dirname "$0")"
declare -A ARM=([in0]='{"emb_at": "in", "emb_prior_sd": 0}' [in5]='{"emb_at": "in", "emb_prior_sd": 0.5}'
                [out0]='{"emb_at": "out", "emb_prior_sd": 0}' [out5]='{"emb_at": "out", "emb_prior_sd": 0.5}')
for a in in0 in5 out0 out5; do echo "${ARM[$a]}" > $O/arm-$a.json; done
for a in in0 in5 out0 out5; do for t in none "$@"; do
  [ -f $O/$a-$t.json ] || echo "$a $t"
done; done | OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 xargs -P 8 -L 1 bash -c \
  'nice -n 19 python3 -I loo174.py '"$IN"' @'"$O"'/arm-$0.json $1 '"$O"'/$0-$1.json > '"$O"'/$0-$1.log 2>&1'
echo LOO174 DONE
