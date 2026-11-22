#!/bin/bash
# early214.sh ARM SIZE -- RULES214's early stop (40Q, 25Q): when the arm's 0.5 M checkpoint has been read on the judge slice, a ratio vs the unhealed release >= 0.990
# (point estimate; paired gen-row KL) -> touch runs/ARM/SKIP (the readers skip its remaining reads) and runs/ARM/STOP (qad200 saves and exits at the next step);
# else nothing. Logs /data/q27/w214/early214.log. Restartable (idempotent).
ARM=$1 SIZE=$2
cd /data/q27/w214; source code/v3/qad214/env214.sh
R=scores/$ARM-ckpt-5e+05.json.rows
while [ ! -s $R ]; do sleep 60; done
RATIO=$($PY code/v3/qad214/cmp214.py slices/golden-j214 scores/ctl-$SIZE.json.rows $R | tail -n 1 | sed 's/.*ratio \([0-9.]*\) .*/\1/')
if python3 -c "import sys; sys.exit(0 if float('$RATIO') >= 0.990 else 1)"; then
  touch runs/$ARM/SKIP runs/$ARM/STOP; echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $ARM: 0.5 M ratio $RATIO >= 0.990: early stop (SKIP + STOP)" >> early214.log
else
  echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $ARM: 0.5 M ratio $RATIO < 0.990: continues to 1 M" >> early214.log
fi
