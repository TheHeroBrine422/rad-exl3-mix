#!/bin/bash
# a8anchor144.sh RUN... -- (w144, #174-02, ds8; RULES.md + amendment 1) the trellis embedding anchor swaps in A8, exactly
# w135's a8anchor174.sh setup: an incumbent's hostd .mix + "token_embd.weight <donor>" + "act <w> a8g128" on every trunk
# weight, radmix on V2 3.00, scored on s171b whole (bf16 KV, max-model-len 16384) -- but with this branch's build
# (rad10-perf install + /data/q27/w144/home: embed_exl3i). RUN = TAG:EMB with EMB tK (donor
# /data/q27/emb/trellis/emb-t-i8-K<k>.rad), tKm (mul1: emb-t-m1-K<k>.rad), bf16 (the K0.125 base pack: the checkpoint's
# bf16 table), i8row (emb-i8row.rad: int8 per-row RTN rows, embrtn144.py) or ctl (the incumbent unswapped: the build control). One eval at a time, on the
# first free card (polling the card locks), ram.lock shared, radmix under conv.lock; mixes deleted
# after scoring. A run whose donor does not exist yet is skipped (rerun later). Restartable: finished scores are skipped.
set -u
Q=/data/q27; W=$Q/w144/a8; LK=$Q/locks; mkdir -p $W/mix $W/scores $Q/w144/run
BASE=$Q/mixes/qwen3.8-27b-exl3-v2-3.00bpw.rad; SB=$Q/w131/slices/golden-s171b
log() { echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S CST') $*" | tee -a $W/run.log; }
donor() {  # emb -> its donor .rad (empty for ctl)
  case $1 in
    ctl) ;; bf16) ls $Q/packs/q27v3-144dd6e1-k0.125-i8-s*.rad | head -n 1 ;; i8row) echo $Q/emb/trellis/emb-i8row.rad ;;
    t*m) local k=${1#t}; echo $Q/emb/trellis/emb-t-m1-K${k%m}.rad ;; t*) echo $Q/emb/trellis/emb-t-i8-K${1#t}.rad ;;
  esac
}
mkmix() {  # tag emb -> $W/mix/<tag>-<emb>-a8.rad
  local t=$1 e=$2 n=$1-$2-a8; local mf=$W/mix/$n.mix out=$W/mix/$n.rad
  grep -q '^token_embd' $Q/mixes/$t.mix && { log "$t.mix already sets token_embd"; return 1; }
  { cat $Q/mixes/$t.mix; [ $e = ctl ] || echo "token_embd.weight $(donor $e)"
    awk '$1 ~ /^blk\./ {print "act", $1, "a8g128"}' $Q/mixes/$t.mix; } > $mf
  [ -f $out ] && return 0
  rm -f $out.part
  flock $LK/conv.lock flock -s $LK/ram.lock $Q/tools/radmix $out.part $BASE $mf >> $W/mix/$n.mixlog 2>&1 \
    && mv $out.part $out || { log "radmix $n FAILED"; return 1; }
  log "mix $n bytes $(stat -c %s $out)"
}
score() {  # name
  local n=$1 o=$W/scores/$1-s171b c=
  while :; do                        # the first free card by polling its lock, held on fd 9 until the eval ends
    for cc in 0 1 2 3 4 5 6 7; do exec 9>$LK/card$cc.lock; if flock -n 9; then c=$cc; break; fi; exec 9>&-; done
    [ -n "$c" ] && break; sleep 2
  done
  log "score $n card $c"
  flock -s $LK/ram.lock bash -c "ROCR_VISIBLE_DEVICES=$c RADIANCE_HOME=$Q/w144/home $Q/rad10-perf/install/bin/radiance --model $W/mix/$n.rad --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $SB --kld-out $o.json.part > $o.log 2>&1 && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json"
  local rc=$?
  exec 9>&-
  log "score $n rc $rc $(grep 'I kld: [0-9]* / ' $o.log | tail -n 1)"
  return $rc
}
log "start pid $$: $*"
for r in "$@"; do
  t=${r%%:*}; e=${r#*:}; n=$t-$e-a8
  [ -f $W/scores/$n-s171b.json.rows ] && continue
  [ $e = ctl ] || [ -f "$(donor $e)" ] || { log "$n: no donor yet"; continue; }
  mkmix $t $e || continue
  score $n && rm -f $W/mix/$n.rad
done
log "done"
