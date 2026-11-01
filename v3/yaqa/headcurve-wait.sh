#!/bin/bash
# headcurve-wait.sh -- arm headcurve.sh for right after the A4 timing (overnight.sh); card 0's ladder waits for it.
touch /data/yaqa/headcurve.pending
until grep -q "^.* a4: " /data/yaqa/overnight.log 2>/dev/null; do
  pgrep -f "^/bin/bash ./overnight.sh" > /dev/null || break; sleep 30; done
cd /data/yaqa && exec ./headcurve.sh
