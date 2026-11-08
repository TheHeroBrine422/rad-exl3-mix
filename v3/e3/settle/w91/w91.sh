#!/bin/bash
# w91.sh -- ../w91.md item 1 (rules written before data): 27B fixed-code oracle, channel 3994's writer rows in bf16.
# gpubox; card 0 under /root/card0.lock + /root/ram.lock (shared); packs under /root/conv.lock + /root/ram.lock
# EXCLUSIVE (lead 11-07). Idempotent: rerun after any failure. Stop: touch $A/STOP (checked between steps).
#   w91.sh slice      golden-search27b rebuilt (w84's mk27b.sh route) into $A; G0: kld.json == the repo's copies
#   w91.sh census     census91.py on the control level (card 0) -> $A/census.jsonl; pick91.py -> S, keys8, arm D
#   w91.sh p0|p1|D    mkpatch91 (CPU) -> packs C / A (rad-convert --reuse <control mix> --changed) -> radmix into the
#                     control mix (the dense writers only) -> census, G1 hygiene (rxq.pairs mixes) + bytes91 -> scores
#                     on golden-search27b-{x,c} (+ the control mix itself for p0: G2 decode gate)
#   w91.sh verdict    verdict91.py
: ${RXM_WORK:=/root}; export RXM_WORK
A=${A:-/data/w91}; CODE=${CODE:-$A/code}; W=$CODE/v3/e3/settle/w91; W84=$CODE/v3/e3/settle/w84
MODEL=/data/workspace/models/qwen38-27b-bf16 HESS=/root/challenge/hessians/calib
T=/data/swap27/tools I=/data/install-116 H=/data/swap27/home PY=/data/venv-exl3/bin/python
export PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1 OMP_NUM_THREADS=8
mkdir -p $A; cd $A
log() { echo "$(date -u +%H:%M:%S) ${STEP:-w91}: $*" | tee -a $A/result.txt; }
die() { log "$*: stop"; exit 1; }
[ -f $A/STOP ] && die "STOP file"
memok() { local m=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  [ $m -ge ${MINMEM_GB:-6} ] || die "MemAvailable $m GB < ${MINMEM_GB:-6} GB before $1 (rerun later)"; }
RX=(env PYTHONPATH=$CODE/pipeline RXQ_RXM=$CODE $PY)
TRUNK() { for l in $(seq 0 63); do
    if [ $((l % 4)) = 3 ]; then for t in attn_qg attn_k attn_v attn_output; do echo blk.$l.$t.weight; done
    else for t in ssm_qkv ssm_z ssm_out; do echo blk.$l.$t.weight; done; fi
    echo blk.$l.ffn_gate_up.weight; echo blk.$l.ffn_down.weight; done; }
radname() { echo $1 | sed -E 's/^model\.language_model\.layers\.([0-9]+)\.(linear_attn\.out_proj|self_attn\.o_proj|mlp\.down_proj)$/\1 \2/' |
  awk '{f = $2 == "linear_attn.out_proj" ? "ssm_out" : ($2 == "self_attn.o_proj" ? "attn_output" : "ffn_down"); print "blk." $1 "." f ".weight"}'; }
score_g() { # score_g GOLDEN OUTBASE MIX -> OUTBASE.json(.rows); w84's score_g (card 0, bf16 KV)
  local g=$1 out=$2 mix=$3; [ -f $out.json.rows ] && return 0
  [ -f $mix ] || { log "score $(basename $out): no mix $mix"; return 1; }
  ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; s=$(date +%s)
    ROCR_VISIBLE_DEVICES=0 RADIANCE_HOME=$H $I/bin/radiance --model $mix --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $g --kld-out $out.json.part > $out.log 2>&1 \
      && mv $out.json.part $out.json; [ -f $out.json.part.rows ] && mv $out.json.part.rows $out.json.rows
    echo "$(date -u +%H:%M:%S) score $(basename $out) on $(basename $g) rc=$? $(( $(date +%s) - s ))s" ) >> $A/result.txt
  [ -f $out.json.rows ] || { log "score $(basename $out) FAIL ($out.log)"; return 1; }; }

case $1 in
slice) STEP=slice
  T84=$T W84=$W84 A=$A bash $W/mk27b91.sh >> $A/slice.log 2>&1 || die "mk27b91 FAIL ($A/slice.log)"
  for p in c x; do cmp -s $A/golden-search27b-$p/kld.json $CODE/v3/e3/settle/data/w84/golden-search27b-$p-kld.json \
      && log "G0 golden-search27b-$p: kld.json identical to w84's" || die "G0 golden-search27b-$p differs from w84's"; done
  touch $A/slice.pass; log "SLICE DONE" ;;
census) STEP=census
  if ! grep -q "CENSUS91 DONE" $A/census.log 2>/dev/null; then
    memok census
    ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; s=$(date +%s)
      ROCR_VISIBLE_DEVICES=0 nice -n 5 $T/memguard.sh 3000 "${RX[@]}" $W/census91.py /data/v3ladder2/K2.5 $MODEL $HESS $A/census.jsonl \
        --device cuda:0 >> $A/census.log 2>&1; echo "census rc=$? $(( $(date +%s) - s ))s" >> $A/result.txt )
    grep -q "CENSUS91 DONE" $A/census.log || die "census91 FAIL ($A/census.log)"
  fi
  $PY -I $W/pick91.py $A/census.jsonl | tee -a $A/result.txt; log "CENSUS DONE" ;;
p0|p1|D) STEP=$1; P=$1
  [ -f $A/slice.pass ] || die "slice gate G0 not passed"
  [ -f $A/census.jsonl.pick.json ] || die "no census pick"
  case $P in
    p0|D) LEVEL=/data/v3ladder2/K2.5 CMIX=/data/swap27/mix-base.rad CREC=/data/swap27/base.recipe ;;
    p1)   LEVEL=/data/swap27/q-s1/K2.5 CMIX=/data/swap27/mix-s1.rad CREC=/data/swap27/s1.recipe ;;
  esac
  KS=$($PY -c "import json; p = json.load(open('$A/census.jsonl.pick.json')); print(','.join(p['keysD'] if '$P' == 'D' else p['keys8']))")
  ROWS=$($PY -c "import json; print(','.join(map(str, json.load(open('$A/census.jsonl.pick.json'))['S'])))")
  [ -n "$KS" ] || die "no keys for $P (arm D not triggered?)"
  O=$A/$P; mkdir -p $O
  # 1. the two checkpoint dirs (CPU)
  if [ ! -f $O/patch.json ]; then memok mkpatch
    nice -n 10 "${RX[@]}" $W/mkpatch91.py $LEVEL $MODEL $KS $ROWS $O > $O/mkpatch.log 2>&1 || die "mkpatch91 FAIL ($O/mkpatch.log)"
    log "patch dirs: $(echo $KS | tr ',' '\n' | wc -l) writers, rows $ROWS"; fi
  NAMES=$(for k in $(echo $KS | tr ',' ' '); do radname $k; done)
  # 2. packs: the control recipe, per weight, minus the dense writers (they pass through as bf16)
  SRC=$(awk 'NR == 1 {for (i = 3; i <= NF; i++) printf "%s ", $i}' $CREC)
  TRUNK | grep -vxF "$NAMES" | sed "s|\$| exl3i $SRC|" > $O/pack.recipe
  [ $(wc -l < $O/pack.recipe) = $((336 - $(echo "$NAMES" | wc -l))) ] || die "pack recipe: $(wc -l < $O/pack.recipe) rules"
  for X in C A; do PK=$O/pack-$X.rad MX=$O/mix-$X.rad
    [ -f $MX ] && [ -f $O/mix-$X.ok ] && continue
    if [ ! -f $PK ]; then memok "pack $X"
      ( exec 9>$RXM_WORK/conv.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock 7; s=$(date +%s); rm -f $PK.part
        RADIANCE_HOME=$H RADIANCE_HOST_DEVICES=1 nice $T/memguard.sh 2500 $I/bin/rad-convert $O/ck-$X --recipe $O/pack.recipe \
          --set quantization_config.quant_method=exl3 --reuse $CMIX --changed $O/keys.txt -o $PK.part > $O/conv-$X.log 2>&1 && mv $PK.part $PK
        echo "$(date -u +%H:%M:%S) pack $P-$X rc=$? $(( $(date +%s) - s ))s" >> $A/result.txt )
      [ -f $PK ] || { rm -f $PK.part; die "pack $X FAIL ($O/conv-$X.log)"; }
    fi
    echo "$NAMES" | sed "s|\$| $PK|" > $O/$X.mix
    $T/radmix $MX.part $CMIX $O/$X.mix > $O/radmix-$X.log 2>&1 && mv $MX.part $MX || die "radmix $X FAIL"
    $PY -I $T/census.py $MX --mix $O/$X.mix --base $CMIX > $O/census-$X.txt 2>&1 || die "census $X FAIL ($O/census-$X.txt)"
    touch $O/mix-$X.ok; rm -f $PK; log "mix $X ready ($(du -h --apparent-size $MX | cut -f1))"
  done
  # 3. G1 hygiene
  if [ ! -f $O/g1.pass ]; then
    ARM=$KS
    r1=$(nice -n 10 "${RX[@]}" -m rxq.pairs mixes $O/mix-C.rad $CMIX --arm $ARM | tail -n 1); log "G1 C vs control mix: $r1"
    r2=$(nice -n 10 "${RX[@]}" -m rxq.pairs mixes $O/mix-A.rad $O/mix-C.rad --arm $ARM | tail -n 1); log "G1 A vs C: $r2"
    r3=$(nice -n 10 $PY -I $W/bytes91.py $T/census.py $O/mix-A.rad $O/mix-C.rad $O 2>&1 | tail -n 1); log "G1 bytes: $r3"
    [[ "$r1" == "PAIR OK"* && "$r2" == "PAIR OK"* && "$r3" == "BYTES91 PASS"* ]] && touch $O/g1.pass || die "G1 FAIL"
  fi
  # 4. scores (card 0)
  mkdir -p $A/b27
  for part in x c; do G=$A/golden-search27b-$part
    score_g $G $A/b27/$P-C-$part $O/mix-C.rad || die "score C-$part"
    score_g $G $A/b27/$P-A-$part $O/mix-A.rad || die "score A-$part"
    [ $P = p0 ] && { score_g $G $A/b27/base-$part $CMIX || die "score base-$part"; }
  done
  log "W91 $P DONE" ;;
verdict) STEP=verdict
  $PY -I $W/verdict91.py $A > $A/verdict.txt 2>&1 || die "verdict91 FAIL ($A/verdict.txt)"
  log "$(tail -n 1 $A/verdict.txt)" ;;
*) echo "w91.sh slice|census|p0|p1|D|verdict"; exit 2 ;;
esac
