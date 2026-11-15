#!/bin/bash
# #185 w173 (ds8): the rollout driver. Runs rollout_level.sh over order.txt (K >= 6, then sub-1, then the middle).
#   BOX2_HOST=... BOX2_KEY=... setsid nohup nice -n 19 bash rollout.sh > rollout.out 2>&1 < /dev/null &
# Stops at the first level that would take ds8 below the floor (exit 3: the lead frees packs at the switch), and halts
# after two levels in a row that are not IDENTICAL (exit 4: that looks systematic, not per level). Any other error
# halts (exit 2); a rerun skips the levels with a result line and resumes the rest.
set -u
W=${W:-/data/q27/w173}
fails=0
while read -r L <&3; do
  [ -n "$L" ] || continue
  bash "$W/bin/rollout_level.sh" "$L" < /dev/null; rc=$?
  case $rc in
    0) fails=0 ;;
    1) fails=$((fails + 1)); if [ $fails -ge 2 ]; then echo "$(date -u +%FT%TZ) HALT: 2 levels in a row not identical"; exit 4; fi ;;
    3) echo "$(date -u +%FT%TZ) STOP at the floor, before $L"; exit 3 ;;
    *) echo "$(date -u +%FT%TZ) HALT: $L rc $rc (rerun resumes)"; exit 2 ;;
  esac
done 3< "$W/order.txt"
echo "$(date -u +%FT%TZ) ALL LEVELS IN ORDER DONE"
