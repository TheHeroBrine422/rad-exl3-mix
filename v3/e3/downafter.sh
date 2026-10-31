#!/bin/bash
# downafter.sh N -- once the running sweepN jobs have drained, run sweepN.sh again (skipdone skips what is done) with
# PD down-phase jobs at once
cd "$(dirname "$0")"
while pgrep -f "^python3 -W ignore" > /dev/null; do sleep 30; done
exec ./sweep$1.sh
