#!/bin/bash
# xfer10-218.sh (iceman, #218 scaling follow-up) -- when scaleup218.sh's chunk 9 and the 34-doc check gradient exist: xfer218.py over the 10
# chunks (1M rows) with the 34-doc check, splits s300 / s100 / s500 (rows-weighted) and the same with --norm-chunks. CPU only.
set -u
cd /data/w218; PY=/data/venv-exl3/bin/python; C=code/v3/qad218
while [ ! -f runs/c9/grad.json ] || [ ! -f runs/gtune34/grad.json ]; do sleep 60; done
R=runs; CH=$R/r0/g0,$R/r0/g1,$R/r0/g2,$R/r1/g0,$R/r1/g1,$R/r1/g2,$R/r2/g0,$R/r2/g1,$R/r2/g2,$R/c9
SP="s300:2:0,1;s100:0:5;s500:0,1,2,3,4:5,6,7,8,9"
OMP_NUM_THREADS=6 nice -n 10 $PY $C/xfer218.py runs/xfer-10c-t34.json --chunks $CH --check runs/gtune34 --splits "$SP" > logs/xfer-10c.log 2>&1
OMP_NUM_THREADS=6 nice -n 10 $PY $C/xfer218.py runs/xfer-10c-t34-norm.json --chunks $CH --check runs/gtune34 --splits "$SP" --norm-chunks >> logs/xfer-10c.log 2>&1
echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') xfer10 done (runs/xfer-10c-t34*.json, logs/xfer-10c.log)" >> logs/driver.log
