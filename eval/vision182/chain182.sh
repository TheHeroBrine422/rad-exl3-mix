#!/bin/bash
# chain182.sh PID STAGES... -- (w156) after PID exits: install run182.sh.new (if any), then run182.sh STAGES
p=$1; shift; while kill -0 $p 2>/dev/null; do sleep 20; done
T=/data/q27/w156/tools; [ -f $T/run182.sh.new ] && mv $T/run182.sh.new $T/run182.sh
exec bash $T/run182.sh "$@"
