#!/bin/bash
# tp2night.sh -- gpubox, night of 2026-11-05 (user: both cards tonight, card 1 back tomorrow). Stages, each logged to
# $N/night.log, each skipped when done (rerun to continue):
#   0. the swap test's seed arm (s1) is moved to card 0 later: killed if it starts; wait for swap27 RUN DONE
#   0b. speedcal on both cards at once (each labelled with its live power1_cap; user: card 0 at 250 W, card 1 300 W)
#   1. 27B bf16 reference goldens, stock radiance 1.17 TP2 --kld-record: xcorp 8k (5.0M tok), own traces 32k (2.5M)
#   2. d1108 arm D: serve the bf16 27B (TP2), generate its own continuations of sgprep's first 640 train prompts
#   3. Σdy² collection over them (collect.py, as #33-24: fp32, diag + extras, gen-masked, 512 sequences)
#   3b. more goldens on TP2, ~1M-token chunks (corpora/chunks: xcorp 8k b2/b3, traces 32k b2/b3), until $N/STOP2
#   (3b runs all 16 chunks, ~4 h, unless STOP2; nothing after it needs TP2; user)
#   4. single-card-capable, card 1 helping while it is ours (4-layer chunk queue, swap27k.sh): the swap test's seed arm
#      s1 (K2.5) -> base4 -> bn4 (K4 on V2 4.00, H27.md amendment 3) -> paired comparisons
# Stop the two-card part at any time: tools/stop2.sh (touches STOP2, ends the collector after its current sequence or
# the current record chunk); the chain then goes straight to stage 4 on card 0. Rerun this script to resume.
set -u
N=/data/night1105 R=/data/install-117 H=$R/share/radiance CK=/data/workspace/models/qwen38-27b-bf16 PY=/data/venv-exl3/bin/python
mkdir -p $N; L=$N/night.log; log() { echo "$(date -u +%T) $*" | tee -a $L; }
S=/data/swap27; export ROCR_VISIBLE_DEVICES=0,1
LOCKS="exec 5>/root/card0.lock 6>/root/card1.lock 7>/root/ram.lock; flock 5; flock 6; flock 7"
# ---- 0
until grep -q "RUN DONE" <(tail -n 2 $S/result.txt); do
  p=$(pgrep -f "[o]urq.py -K 2.5 -o $S/q-s1/"); [ -n "$p" ] && { kill $p; log "s1 quant stopped (moved to stage 4)"; }
  sleep 30; done
log "swap test lanes done: $(grep -c "pair base" $S/result.txt) pairs logged"
# ---- 0b
if [ ! -f $N/speedcal.done ]; then
  ( eval "$LOCKS"
    for c in 0 1; do cap=$(( $(cat /sys/class/drm/card$c/device/hwmon/hwmon*/power1_cap) / 1000000 ))
      ( while true; do echo "$(date +%s) $(cat /sys/class/drm/card$c/device/hwmon/hwmon*/power1_average)" >> $N/power-c$c.log; sleep 1; done ) & PW[$c]=$!
      ROCR_VISIBLE_DEVICES=$c WATTS=${cap}W BIN=/data/bench bash /data/benchsrc/v3/bench/speedcal.sh $N/speedcal-c$c.jsonl > $N/speedcal-c$c.log 2>&1 &
      SC[$c]=$!; done
    wait ${SC[0]} ${SC[1]}; kill ${PW[0]} ${PW[1]}
    touch $N/speedcal.done )
  log "speedcal: c0 $(head -c 120 $N/speedcal-c0.jsonl | grep -o 'watts_cap[^,}]*') c1 $(head -c 120 $N/speedcal-c1.jsonl | grep -o 'watts_cap[^,}]*'), $(wc -l < $N/speedcal-c0.jsonl) rows each"
fi
# ---- 1
stop2() { [ -f $N/STOP2 ]; }
rec() { name=$1 corpus=$2 len=$3; [ -f $N/gold/$name/.done ] && return 0; stop2 && return 1
  rm -rf $N/gold/$name
  ( eval "$LOCKS"; s=$(date +%s); RADIANCE_HOME=$H $R/bin/radiance --model ${MODEL:-$CK} --tp 2 --kv-cache-dtype bf16 \
      --max-model-len $len --host-headroom-mib 6144 --kld-record $N/gold/$name --kld-corpus $corpus --kld-top 512 \
      > $N/rec-$name.log 2>&1 & echo $! > $N/tp2.pid; wait $!; rc=$?
    [ $rc = 0 ] && touch $N/gold/$name/.done; cp ${corpus}.meta.jsonl $N/gold/$name/corpus.meta.jsonl 2>/dev/null
    echo "rec $name rc=$rc $(( ($(date +%s) - s) / 60 )) min" ) | tee -a $L
  [ -f $N/gold/$name/.done ]; }
mkdir -p $N/gold
if ! stop2 && ! rec xc8k-b1 $N/corpora/xc8k-b1.jsonl 8448; then
  log "HF checkpoint did not load as is: converting to .rad"; mkdir -p /data/models-rad
  [ -f /data/models-rad/qwen38-27b-bf16.rad ] || ( exec 4>/root/conv.lock; flock 4; RADIANCE_HOME=$H $R/bin/rad-convert $CK -o /data/models-rad/qwen38-27b-bf16.rad > $N/convert.log 2>&1 )
  export MODEL=/data/models-rad/qwen38-27b-bf16.rad; rec xc8k-b1 $N/corpora/xc8k-b1.jsonl 8448 || log "GOLDEN xc8k FAIL"
fi
stop2 || rec tr32k-b1 $N/corpora/tr32k-b1.jsonl 33024 || log "GOLDEN tr32k FAIL"
# ---- 2
G=$N/gen-D
if [ ! -f $G/.done ] && ! stop2; then
  ( eval "$LOCKS"
    RADIANCE_HOME=$H $R/bin/radiance --model ${MODEL:-$CK} --tp 2 --kv-cache-dtype bf16 --max-model-len 4096 \
      --max-num-seqs 16 --host 127.0.0.1 --port 8099 > $N/serve.log 2>&1 & SV=$!; echo $SV > $N/tp2.pid
    for i in $(seq 1 120); do curl -sf http://127.0.0.1:8099/v1/models > /dev/null && break; sleep 15; done
    cd /tmp && XCORP_TOKENIZER=$CK $PY -I $N/tools/sgserve.py $N/prep/prompts-train.npy $G --url http://127.0.0.1:8099 \
      --limit 640 --conc 16 >> $N/sgserve.log 2>&1 && touch $G/.done
    kill $SV; wait $SV 2>/dev/null )
  log "gen D: $(tail -n 2 $N/sgserve.log | head -1)"
fi
# ---- 3
W=$N/w; MD=$N/mD
if [ -f $G/.done ] && [ ! -f $N/hdiag-D/.done ] && ! stop2; then
  [ -f $MD/build.json ] || $PY -I $N/tools/sgbuild.py $N/prep $G $MD selfgen-train >> $L 2>&1
  mkdir -p $W/challenge/golden; ln -sfn $MD $W/challenge/golden/calib
  ( eval "$LOCKS"; cd /data/yaqa && RXM_WORK=$W RXM_CKPT=$CK YAQA_POOLS=$MD/pools.json YAQA_GENMASK=1 YAQA_XOFF=bf16 \
      YAQA_FP32=1 YAQA_DIAG=1 YAQA_DIAG_EXTRA=1 YAQA_SEED=0 YAQA_PREC=fp32 TRITON_CACHE_AUTOTUNING=1 \
      YAQA_SNAP=128,256,384,512 YAQA_SAVE_EVERY=25 $PY -I collect.py $N/hdiag-D 512 2048 ) >> $N/collect-D.log 2>&1
  grep -q "^DONE" <(tail -n 3 $N/collect-D.log) && { touch $N/hdiag-D/.done; log "collect D: $(grep '^DONE' $N/collect-D.log | tail -1)"; } || log "COLLECT D FAIL (rerun to resume)"
fi
# ---- 3b
for c in xc8k-b2-c0 tr32k-b2-c0 xc8k-b2-c1 xc8k-b2-c2 tr32k-b2-c1 xc8k-b2-c3 xc8k-b2-c4 tr32k-b2-c2 \
         xc8k-b3-c0 tr32k-b3-c0 xc8k-b3-c1 xc8k-b3-c2 tr32k-b3-c1 xc8k-b3-c3 xc8k-b3-c4 tr32k-b3-c2; do
  stop2 && break
  len=$([[ $c == tr32k* ]] && echo 33024 || echo 8448)
  rec $c $N/corpora/chunks/$c.jsonl $len || { stop2 || log "GOLDEN $c FAIL"; }
done
stop2 && log "STOP2: two-card part ended by the user"
# ---- 4
log "stage 4: single-card-capable queue (card 1 helps unless STOP2)"
K4=/data/release/qwen3.8-27b-exl3-v2-4.00bpw.rad SK=/data/swap27/tools/swap27k.sh
cd $S
bash $SK s1 2.5 /data/release/qwen3.8-27b-exl3-v2-3.00bpw.rad --seed-offset 1 >> $S/run4.out 2>&1
[ -f $K4 ] || log "no V2 4.00 container: K4 arms skipped"
[ -f $K4 ] && bash $SK base4 4 $K4 >> $S/run4.out 2>&1
[ -f $K4 ] && bash $SK bn4 4 $K4 --hout $S/hout-bn --sigma-reg-out 0.01 >> $S/run4.out 2>&1
for p in "base s1" "base4 bn4"; do set -- $p; [ -f $S/$1.json.rows ] && [ -f $S/$2.json.rows ] && \
  log "pair $1 -> $2: $($PY -I $S/tools/cmp115.py $S/golden-search27 $S/$1.json.rows $S/$2.json.rows 2>&1 | tail -n 1)"; done
log "NIGHT DONE"
