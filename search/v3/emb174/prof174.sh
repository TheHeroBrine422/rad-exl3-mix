#!/bin/bash
# prof174.sh -- (w135, #174, ds8; lead note 11-13: confirm and time once that the plugin gathers the embedding from the
# STORED form) radiance --profile-ops on two donors that differ only in the embedding: emb-e4.rad (K0.125 pack + Lloyd
# l4: embed_tq, codes resident on the card) and emb-bf16.rad (K0.125 pack + bf16: embed_lookup, by default pinned in
# host memory and read over the link in place, --embedding-placement ram). Same command as the anchor scoring (s171b,
# bf16 KV, 16k context); the KL is meaningless (K0.125 trunk), only the placement lines and per-op times matter.
# One card (card lock, ram.lock shared), after a8anchor174.sh. Logs: /data/q27/emb/prof/<name>.log
set -u
Q=/data/q27; P=$Q/emb/prof; mkdir -p $P; SB=$Q/w131/slices/golden-s171b
while pgrep -f "^bash code/a8anchor174.sh" > /dev/null; do sleep 30; done
for n in e4 bf16; do
  [ -f $P/$n.done ] && continue
  while :; do
    got=0
    for c in 0 1 2 3 4 5 6 7; do
      exec 8>$Q/locks/card$c.lock
      if flock -n 8; then
        flock -s $Q/locks/ram.lock bash -c "ROCR_VISIBLE_DEVICES=$c RADIANCE_HOME=$Q/rad10/home $Q/rad10/install/bin/radiance --model $Q/emb/emb-$n.rad --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --profile-ops --kld-ref $SB --kld-out $P/$n.json > $P/$n.log 2>&1" && touch $P/$n.done
        exec 8>&-; got=1; break
      fi
      exec 8>&-
    done
    [ $got = 1 ] && break; sleep 20
  done
  rm -f $P/$n.json $P/$n.json.rows
done
echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S CST') prof174 done" >> $P/run.log
