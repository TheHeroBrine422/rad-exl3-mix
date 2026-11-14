#!/bin/bash
# run17803.sh -- RULES-17803 on ds8 (w149, #178-03): GDN a/b B / T4 / T5 / T6 on the 5.0 / 6.0 / 6.5 incumbents, s173f.
# Per size: the incumbent's hostd .mix + "act <w> a8g128" on every blk.* weight + the bf16 embedding -> radmix once
# (conv.lock + ram.lock shared), census (full byte check vs donors and base); B scored on it; per T arm a copy, abpatch
# (w142) writes the arm's decoded a/b, census of the copy vs B (exactly the 96 a/b weights differ), score, copy deleted.
# Scores on the first free card (non-blocking tries only: never queues on a held card) + ram.lock shared.
# Restartable: finished scores are skipped; .part outputs only.
set -u
Q=/data/q27; W=$Q/w149; C=$W/code; mkdir -p $W/mix $W/scores $W/census
BASE=$Q/mixes/qwen3.8-27b-exl3-v2-3.00bpw.rad; G=$Q/w131/slices/golden-s173f; EMB=$Q/emb/emb-bf16.rad
PY="/data/q27/venv/bin/python -I"; export RXQ_RXM=$C
declare -A TAG=([5.0]=v3s171-5-pair-00449 [6.0]=v3s171-6-explore-00372 [6.5]=v3s171-6.5-pair-00588)
log() { echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S %Z') $*" >> $W/run.log; }

card() {  # run "$2" under the first free card lock (+ ram.lock shared), ROCR_VISIBLE_DEVICES set; polls, never queues
  while :; do
    for c in 0 1 2 3 4 5 6 7; do
      exec 8>$Q/locks/card$c.lock
      if flock -n 8; then
        log "card $c: $1"
        flock -s $Q/locks/ram.lock env ROCR_VISIBLE_DEVICES=$c bash -c "$2"; local rc=$?
        exec 8>&-; return $rc
      fi
      exec 8>&-
    done
    sleep 30
  done
}

mkmix() {  # size -> $W/mix/<size>-B.rad, census-checked
  local t=${TAG[$1]} mf=$W/mix/$1-B.mix out=$W/mix/$1-B.rad
  [ -f $out ] && return 0
  { cat $Q/mixes/$t.mix; awk '$1 ~ /^blk\./ {print "act", $1, "a8g128"}' $Q/mixes/$t.mix; echo "token_embd.weight $EMB"; } > $mf
  rm -f $out.part
  flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock $Q/tools/radmix $out.part $BASE $mf > $W/mix/$1-B.mixlog 2>&1 \
    || { log "radmix $1 FAILED"; rm -f $out.part; return 1; }
  $PY $C/store/census.py $out.part --mix $mf --base $BASE --expect-act a8g128 --expect-codebook mul1i8 \
      --expect-emb 'plain - bf16' --max-version 3 > $W/census/$1-B.txt 2>&1 \
    || { log "census $1-B FAILED: $(tail -n 1 $W/census/$1-B.txt)"; rm -f $out.part; return 1; }
  mv $out.part $out
  log "mix $1 $t bytes $(stat -c %s $out) acts $(grep -c '^act ' $mf) $(tail -n 1 $W/census/$1-B.txt)"
}

score() {  # size arm container
  local o=$W/scores/$1-$2-s173f
  card "score $1 $2" "RADIANCE_HOME=$Q/rad10/home $Q/rad10/install/bin/radiance --model $3 --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $o.json.part > $o.log 2>&1 && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json"
  log "score $1 $2 rc $? $(grep 'documents in' $o.log)"
}

arm() {  # size arm
  local s=$1 a=$2 r=$W/mix/$1-$2.rad
  [ -f $W/scores/$s-$a-s173f.json.rows ] && return 0
  mkmix $s || return 1
  if [ $a = B ]; then score $s B $W/mix/$s-B.rad; return; fi
  rm -f $r; cp --reflink=auto $W/mix/$s-B.rad $r || { log "copy $s $a FAILED"; rm -f $r; return 1; }
  $PY $C/abpatch.py $r $Q/w142/tq/ab-$a.safetensors >> $W/run.log 2>&1 || { log "abpatch $s $a FAILED"; rm -f $r; return 1; }
  # census: the copy, checked against B's mix and base, differs in exactly the 96 a/b weights (from base), nothing else
  # (store/census.py runs its --base comparison only with --mix: --base alone checks nothing)
  $PY $C/store/census.py $r --mix $W/mix/$s-B.mix --base $BASE --json $W/census/$s-$a.json > $W/census/$s-$a.txt 2>&1
  $PY -c "import json,re,sys; m=json.load(open(sys.argv[1]))['mismatches']; ok=len(m)==96 and all(re.fullmatch(r'blk\.\d+\.ssm_[ab]\.weight: not named by the mix and differs from base',x) for x in m); print('ARMCENSUS', 'PASS' if ok else 'FAIL', len(m)); sys.exit(0 if ok else 1)" \
      $W/census/$s-$a.json >> $W/census/$s-$a.txt 2>&1 || { log "census $s $a FAILED"; rm -f $r; return 1; }
  score $s $a $r
  rm -f $r
}

log "start pid $$"
for s in 5.0 6.0 6.5; do
  ( for a in B T4 T5 T6; do arm $s $a; done
    n=$(ls $W/scores/$s-*-s173f.json.rows 2>/dev/null | wc -l)
    [ "$n" = 4 ] && rm -f $W/mix/$s-B.rad && log "size $s done, B mix deleted" ) &
  sleep 5
done
wait
log "done"
