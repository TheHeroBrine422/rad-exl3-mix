#!/bin/bash
# mkv31.sh (box2; w232 part B) -- the V3.1 final-read slices cut from the published calib golden (eval/golden_pick.py, WHOLE docs, no cap):
#   golden-v31ver = verify_v31 (eval/pools/v31-verify-split.json) minus the two traces (422 docs; IDX = ver-idx.txt next to this script's args)
#   golden-v31tr  = the two traces 905 (agent-trace/rich) and 2675 (agent-trace/express)
# usage: mkv31.sh IDXFILE   (IDXFILE: the 422 doc indices, space separated).  Restartable (a finished slice has .done).
set -eu
Q=/data/q27; QR=$Q/queue; PY=$Q/venv/bin/python; G=$Q/challenge/golden/calib; CORP=$Q/challenge/corpus/calib.jsonl; GP=$QR/code/eval/golden_pick.py
IDX=$(cat $1)
mk() {   # NAME IDX...
  local N=$1; shift; local O=$QR/slices/golden-$N
  [ -e $O/.done ] && return 0
  rm -rf $O.part; nice -n 10 $PY -I $GP $G $CORP $O.part $O.jsonl.part "$@"
  touch $O.part/.done; mv $O.jsonl.part $O.jsonl; rm -rf $O; mv $O.part $O
  echo "golden-$N: $($PY -c "import json;m=json.load(open('$O/kld.json'));print(len(m['docs']),'docs',m['rows'],'rows')")"
}
mk v31ver $IDX
mk v31tr 905 2675
