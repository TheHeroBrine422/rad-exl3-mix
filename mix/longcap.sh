#!/bin/bash
# longcap.sh CARD -- the capped long-context check (2026-10-31): sizes whose 262144-token bf16 KV does not fit one card
# run TP1 on CARD with both 250k verify traces cut to CAP tokens. CAP starts from the list and, if the recorder
# reports a smaller KV pool ("the KV pool holds N"), is retried once at N rounded down to 4096 minus 24576 (slack the recorder needs).
# Golden per cap: golden-long2-cCAP (golden_pick --maxtok; RADIANCE_KLD_MAXTOK=CAP). The turboderp references have
# full-length per-row scores (longscores/), compared later on the same first CAP positions.
: ${RXM_WORK:=/root}; cd $RXM_WORK
C=$1; L=auto/chain.log
one() {  # T CAP -> 0 ok, else prints the pool size if that was the failure
  local T=$1 CAP=$2 t=longcap-v2-$1 G=golden-long2-c$2
  [ -d $G ] || nice python3 -I golden_pick.py --maxtok $CAP challenge/golden/calib challenge/corpus/calib.jsonl $G corpus/long2-c$CAP.jsonl 905 2675 > /tmp/$G.log 2>&1
  # max-model-len stays 262144 (the server then sizes the pool to what fits); only the documents are cut
  sed -e "s#golden-long2 corpus/long2.jsonl#$G corpus/long2-c$CAP.jsonl#" hproxy/longverify.sh > /tmp/longverify-c$CAP.sh
  RADIANCE_KLD_MAXTOK=$CAP LV_TP=1 LV_CARD=$C bash /tmp/longverify-c$CAP.sh $t /data/release/qwen3.8-27b-exl3-v2-${T}bpw.rad -
}
while read T CAP; do
  [ -f auto/state/chain_stop ] && exit 0
  t=longcap-v2-$T; grep -q "\"tag\": \"$t\", \"bytes\"" longmix.jsonl 2>/dev/null && continue
  s=$(date +%s); one $T $CAP
  if ! grep -q "\"tag\": \"$t\", \"bytes\"" longmix.jsonl; then
    # retry: 24k below what the pool holds (an exactly-full pool runs out: "pool or table exhausted")
    h=$(grep -oE "the KV pool holds [0-9]+" long-$t.log | head -1 | grep -oE "[0-9]+$")
    [ -z "$h" ] && h=$(grep -oE "backs 100% of the worst case \(1 sequences all at [0-9]+" long-$t.log | head -1 | grep -oE "[0-9]+$")
    if [ -n "$h" ]; then CAP=$(( h / 4096 * 4096 - 24576 )); one $T $CAP; fi
  fi
  echo "$(date -u +%H:%M) [longcap c$C] $t cap $CAP: $(grep "\"tag\": \"$t\"" longmix.jsonl | tail -1 | cut -c1-150) ($(( ($(date +%s) - s) / 60 )) min)" >> $L
done
