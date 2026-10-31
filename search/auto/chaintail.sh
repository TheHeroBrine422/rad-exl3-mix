#!/bin/bash
# chaintail.sh -- the end of the morning chain (2026-10-31: card 0 is released after
# the TP2 work): wait for the ver2 final, picks (vpick.py), release containers of the preferred mix per size
# (/data/release), the TP2 serving test, then the long-context check -- the sizes that need TP2 first, then card 0
# is released (state/pause-w0, nothing of ours touches it again) and the rest run TP1 on card 1. Log: auto/chain.log.
: ${RXM_WORK:=/root}; export RXM_WORK
cd $RXM_WORK; A=$RXM_WORK/auto; L=$A/chain.log; R=/data/release; mkdir -p $R
log() { echo "$(date -u +%H:%M) [tail] $*" | tee -a $L >> $A/auto.log; }
stopped() { [ -f $A/state/chain_stop ] && { log "chain_stop: stopping"; exit 0; }; }
log "started: waiting for the ver2 final"
while kill -0 $(cat $A/state/driver.pid) 2>/dev/null; do stopped; sleep 30; done
while pgrep -f "^/bin/bash ./auto/evalq2.sh" > /dev/null; do sleep 15; done
log "final done"

# picks
VP_REFCHECK= python3 -I auto/vpick.py $A/final.q $A/pub.q $A/FINAL-PICKS.md >> $L 2>&1 || { log "vpick FAILED"; exit 1; }
log "picks done: auto/FINAL-PICKS.md ($(grep -c . $A/pub.q.pref) sizes)"

# release containers (the preferred mix per size; radmix = the evaluated container, reflinked)
: > $A/release.list
while read t base mix; do
  T=${t##*-e4-}; o=$R/qwen3.8-27b-exl3-v2-${T}bpw.rad
  ./hproxy/radmix $o $base $mix 2> $R/${T}.radmix.log && echo "$T $o $mix" >> $A/release.list || log "radmix FAILED $T"
done < $A/pub.q.pref
log "release containers: $(wc -l < $A/release.list) in $R"

# TP2 serving test on the largest release container (TP2-only size)
big=$(sort -g $A/release.list | tail -1 | awk '{print $2}')
./auto/e2e2tp2.sh $big v2tp2 >> $L 2>&1; log "TP2 serving test: $(python3 -c "
import json; r=json.load(open('/root/e2e2-v2tp2.json')); print({k: (v if not isinstance(v, (dict, list, str)) or len(str(v)) < 60 else str(v)[:60]) for k, v in r.items()})" 2>&1 | cut -c1-400)"

# which sizes fit the 262144-token bf16 KV on one card: start the server TP1 on card 1, ascending, stop at the first miss
: > $A/long.tp1; : > $A/long.tp2; fit=1
while read T o mix; do
  if [ $fit = 1 ]; then
    ( exec 8>$RXM_WORK/card1.lock; flock 8
      export PATH=/root/install-card/bin:$PATH RADIANCE_HOME=/root/install-card/share/radiance ROCR_VISIBLE_DEVICES=1
      radiance --model $o --tp 1 --kv-cache-dtype bf16 --max-model-len 262144 --max-num-seqs 1 --host-headroom-mib 8192 \
        --port 4682 > $A/fit-$T.log 2>&1 & p=$!
      for i in $(seq 1 120); do grep -q "listening on" $A/fit-$T.log && break; kill -0 $p 2>/dev/null || break; sleep 2; done
      grep -q "listening on" $A/fit-$T.log; ok=$?; kill $p 2>/dev/null; wait $p 2>/dev/null; exit $ok ) && echo "$T $o" >> $A/long.tp1 || { fit=0; echo "$T $o" >> $A/long.tp2; }
  else echo "$T $o" >> $A/long.tp2; fi
done < <(sort -g $A/release.list)
log "long check: TP1 on card 1: $(awk '{print $1}' $A/long.tp1 | tr '\n' ' ')| TP2: $(awk '{print $1}' $A/long.tp2 | tr '\n' ' ')"

run() {  # T CONTAINER (env LV_TP / LV_CARD)
  stopped; t=long-v2-$1
  grep -q "\"tag\": \"$t\"" longmix.jsonl 2>/dev/null && return
  s=$(date +%s); ./hproxy/longverify.sh $t $2 -
  log "long $t: $(grep "\"tag\": \"$t\"" longmix.jsonl | tail -1 | cut -c1-170) ($(( ($(date +%s) - s) / 60 )) min)"
}
# 2026-10-31: no long check at 8.0 / 9.0 (no reference there; almost all K7/K8/bf16)
while read T o; do case $T in 8.00|9.00|10.00) log "long check skipped: $T"; continue;; esac; run $T $o; done < $A/long.tp2
touch $A/state/pause-w0
log "CARD 0 RELEASED (TP2 work done; state/pause-w0 set)"
while read T o; do LV_TP=1 LV_CARD=1 run $T $o; done < $A/long.tp1
log "TAILDONE"
