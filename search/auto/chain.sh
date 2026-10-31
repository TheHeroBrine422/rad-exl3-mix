#!/bin/bash
# chain.sh HIGH_AT FINAL_AT -- the unattended end of the V2 search (2026-10-31): at HIGH_AT (epoch s) switch the stream
# to the TP2 targets (8.0 9.0 10.0), at FINAL_AT start the ver2 final (finalsel: top 3 at 2.5-4.5, 2 elsewhere, + the
# best on ver1), wait for it, pick per target (vpick.py: paired bootstrap), then the long-context check of the preferred
# mixes (longverify.sh). No public-corpus runs. Log: $RXM_WORK/auto/chain.log. Stop: kill
# this script's process group; state/chain_stop also stops it between steps. Nothing is published.
: ${RXM_WORK:=/root}; export RXM_WORK
cd $RXM_WORK; A=$RXM_WORK/auto; L=$A/chain.log
HIGH_AT=$1 FINAL_AT=$2
log() { echo "$(date -u +%H:%M) [chain] $*" | tee -a $L >> $A/auto.log; }
stopped() { [ -f $A/state/chain_stop ] && { log "chain_stop: stopping"; exit 0; }; }
waitfit() { while pgrep -f "^python3 -W ignore kfit2.py" > /dev/null; do sleep 5; done; }
drop_pending() {   # unstarted stream candidates out of the main queue (claim lock held)
  exec 8>>$A/q.claimlock; flock 8
  python3 -I - <<'PY'
import json, os, re
W = os.environ["RXM_WORK"]
done = set(json.loads(l)["tag"] for l in open(f"{W}/evq/results.jsonl")); cl = set(os.listdir(f"{W}/evq/claim"))
L = open(f"{W}/auto/q").readlines()
keep = [l for l in L if not (l.split() and re.match(r"^(a|f2|f3)-knap-", l.split()[0]) and l.split()[0] not in done and l.split()[0] not in cl)]
open(f"{W}/auto/q", "w").writelines(keep); print(len(L) - len(keep))
PY
  flock -u 8
}
log "started: high-bpw at $(date -u -d @$HIGH_AT +%H:%M), final at $(date -u -d @$FINAL_AT +%H:%M) UTC"

# 1. high-bpw phase
while [ $(date +%s) -lt $HIGH_AT ]; do stopped; sleep 60; done
waitfit; kill $(cat $A/state/driver.pid); sleep 2
cp $A/auto.sh $A/auto.sh.pre-high
sed -i 's/^TARGETS=".*"/TARGETS="8.0 9.0 10.0"/' $A/auto.sh
n=$(drop_pending); echo 0 > $A/state/stream_lastres     # fit right away
./auto/resume.sh > /dev/null 2>&1
log "high-bpw phase: targets 8.0 9.0 10.0, $n pending low-bpw candidates dropped"

# 2. ver2 final
while [ $(date +%s) -lt $FINAL_AT ]; do stopped; sleep 60; done
waitfit
n=$(drop_pending)
export FINAL_TARGETS="1.5 1.75 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 9.0 10.0"
kill $(cat $A/state/driver.pid); sleep 2
touch $A/state/final_go
FINAL_N=2 FINAL_OURS=0 ./auto/resume.sh > /dev/null 2>&1
log "final: final_go set, $n pending stream candidates dropped (FINAL_TARGETS $FINAL_TARGETS)"
until [ -f $A/state/final.queued ]; do sleep 30; done
log "final: $(wc -l < $A/final.q) ver2 evals queued"
while kill -0 $(cat $A/state/driver.pid) 2>/dev/null; do stopped; sleep 60; done     # driver exits when the final is done
log "final done"

# 3. picks
VP_REFCHECK= python3 -I auto/vpick.py $A/final.q $A/pub.q $A/FINAL-PICKS.md >> $L 2>&1 || { log "vpick FAILED"; exit 1; }
log "picks done: auto/FINAL-PICKS.md"

# 4. long-context check (2026-10-31: no full public -- the board's private set is the test): every finalist and
#    (only the preferred mix per size; 10 min a run) and the nearest turboderp containers (run ahead of time, skipped here) on the two 250k verify traces, whole, TP2 (hproxy/longverify.sh)
while pgrep -f "^/bin/bash ./auto/evalq2.sh" > /dev/null; do sleep 30; done
{ cat $A/pub.q.pref
  for c in sc2.00-e3 plain2.50-e3 sc3.00-e4 plain3.50-e4 sc4.00-e4 sc5.00-e5 sc6.00-e6; do echo "long-ref-$c $RXM_WORK/models/final/q38-27b-$c.rad -"; done; } > $A/long.q
log "long-context check: $(grep -c . $A/long.q) runs"
while read t base mix; do
  stopped
  grep -q "\"tag\": \"$t\"" longmix.jsonl 2>/dev/null && continue
  s=$(date +%s); ./hproxy/longverify.sh $t $base $mix
  log "long $t: $(grep "\"tag\": \"$t\"" longmix.jsonl | tail -1 | cut -c1-160) ($(( ($(date +%s) - s) / 60 )) min)"
done < $A/long.q
log "CHAINDONE"
