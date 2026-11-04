#!/bin/bash
# hdmonloop.sh HDIAG_DIR -- run hdmon.py whenever a new snapshot appears (niced, CPU, seconds); stops with the collector.
D=${1:-/data/yaqa/hdiag}; PY=${PY:-/data/venv-exl3/bin/python}; H=$(dirname $(readlink -f $0))/hdmon.py; last=""
while pgrep -f "^$PY -I collect.py" > /dev/null || [ -z "$last" ]; do
  cur=$(ls -d $D/hdiag-snap*-dy 2>/dev/null | sort -V | tail -1)
  if [ -n "$cur" ] && [ "$cur" != "$last" ]; then sleep 30; nice -n 19 $PY -I $H $D > $D/hdmon.txt 2>&1; last=$cur; fi
  sleep 120
done
nice -n 19 $PY -I $H $D > $D/hdmon.txt 2>&1
