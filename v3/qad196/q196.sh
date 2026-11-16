#!/bin/bash
# q196.sh CARD JOBS (w182, #196) -- one card's queue on box2. JOBS: lines "NAME|LV|NOHEAD|EMBX|FLAGS|DEP" (LV, NOHEAD, EMBX
# may be empty; DEP = a file that must exist first, or empty). A NAME of the form
#   NAME            runs run196.sh NAME CARD FLAGS (LV / NOHEAD / EMBX from the line; done when runs/NAME/ckpt-1e+06/.done)
#   @cpu:TAG        runs FLAGS (a shell command, no card) ; done when the file LV exists
#   @gpu:TAG        runs FLAGS (a shell command) under the card lock (flock -w, blocking) ; done when the file LV exists
# DEP waits first. run196.sh's busy exit (75) is retried. Restartable: rerun the same line.
CARD=$1; JOBS=$2
cd /data/q27/w182 && source env.sh
while IFS='|' read -r NAME LV NOHEAD EMBX FLAGS DEP; do
  [ -z "$NAME" ] && continue
  case "$NAME" in \#*) continue;; esac
  case "$NAME" in
    @cpu:*|@gpu:*)
      if [ -e "$LV" ]; then echo "skip $NAME (done)"; continue; fi
      while [ -n "$DEP" ] && [ ! -e "$DEP" ]; do sleep 60; done
      if [ "${NAME:0:5}" = "@gpu:" ]; then
        flock -w 7200 /data/q27/locks/card$CARD.lock -c "export ROCR_VISIBLE_DEVICES=$CARD; source env.sh; cd /data/q27/w182; $FLAGS"
      else
        bash -c "cd /data/q27/w182 && source env.sh && $FLAGS"
      fi
      echo "$(date -u +%H:%M) queue card $CARD: $NAME rc $?";;
    *)
      if [ -e runs/$NAME/ckpt-1e+06/.done ]; then echo "skip $NAME (done)"; continue; fi
      while [ -n "$DEP" ] && [ ! -e "$DEP" ]; do sleep 60; done
      while true; do
        LV="$LV" NOHEAD="$NOHEAD" EMBX="$EMBX" ./code/v3/qad196/run196.sh "$NAME" "$CARD" $FLAGS
        rc=$?
        [ $rc -eq 75 ] && { sleep 60; continue; }
        break
      done
      echo "$(date -u +%H:%M) queue card $CARD: $NAME rc $rc";;
  esac
done < "$JOBS"
