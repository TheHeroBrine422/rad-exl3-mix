#!/bin/bash
# q.sh -- #46-03 step 1 on gpubox card 0 (w72; rules v3/e3/a4/README.md amendment 1): i4 K2 / K3 / K1 trunks (ourq
# --codebook i4, ladder settings), packed exl3i mul1i4 and mixed into V2 3.00; per K a baseline with every trunk weight
# a8g128 and one arm per (family x 16-layer band) at a4g64 (then a4g128), scored on golden-search27 with the real A4
# kernel (radiance 1.17 + qc-exl3-v3 42960d5 plugins in /data/a4cost/home). Two lanes: Q quantizes K2 -> K3 -> K1,
# S packs / mixes / scores each K once its 64 layers exist. Every card step holds /root/card0.lock then /root/ram.lock
# (shared) for one ~10-min step (a 16-layer quant chunk or a batch of 4 scores: q2's chunks hold it ~12 min, and the
# lock goes to whichever waiter wakes first, so equal-length holds share the card ~evenly with q2 / asym27 / the slab
# head; $A/turn.lock, so only one of this queue's lanes waits for the card at a time). Lock order: quant and pack
# conv -> turn -> card0 -> ram (conv: never beside anyone's rad-convert; pack: ram exclusive), score turn -> card0 -> ram
# shared; no other queue holds card0 while waiting for conv, so no cycle. Idempotent: rerun
# (setsid nohup bash /data/a4cost/tools/q.sh > /data/a4cost/q.out 2>&1 < /dev/null &) resumes; finished steps are
# skipped by their output files. Log /data/a4cost/q.log ("FAIL" lines, "A4COST VERDICT", "A4COST DONE").
A=/data/a4cost; T=$A/tools; PY=/data/venv-exl3/bin/python; I=/data/install-117; H=$A/home
V2=/data/release/qwen3.8-27b-exl3-v2-3.00bpw.rad; G=/data/swap27/golden-search27; SRC=/data/exl3frac/src
L=$A/q.log; R=$A/results.jsonl; mkdir -p $A/arms $A/scores $A/logs
export PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1
log() { echo "$(date -u +%T) $*" | tee -a $L; }
[ -f $A/STOP ] && { log "STOP file: not starting"; exit 0; }
exec 3>$A/q.pid.lock; flock -n 3 || { echo "q.sh already running"; exit 0; }

TRUNK() { for l in $(seq 0 63); do
    if [ $((l % 4)) = 3 ]; then for t in attn_qg attn_k attn_v attn_output; do echo blk.$l.$t.weight; done
    else for t in ssm_qkv ssm_z ssm_out; do echo blk.$l.$t.weight; done; fi
    echo blk.$l.ffn_gate_up.weight; echo blk.$l.ffn_down.weight; done; }
# the weights of a family in layers a..b (shared-input groups whole: attn_in = qg + k + v, gdn_in = qkv + z)
FAM() { f=$1 a=$2 b=$3; for l in $(seq $a $b); do att=$(( l % 4 == 3 ))
    case $f in
      gate_up) echo blk.$l.ffn_gate_up.weight ;;
      down) echo blk.$l.ffn_down.weight ;;
      attn_in) [ $att = 1 ] && for t in attn_qg attn_k attn_v; do echo blk.$l.$t.weight; done ;;
      attn_out) [ $att = 1 ] && echo blk.$l.attn_output.weight ;;
      gdn_in) [ $att = 0 ] && for t in ssm_qkv ssm_z; do echo blk.$l.$t.weight; done ;;
      ssm_out) [ $att = 0 ] && echo blk.$l.ssm_out.weight ;;
    esac; done; }

# ---- lane Q: quantize one level, 4-layer chunks, each under the card lock
quant() { K=$1 O=$A/q-i4/K$1
  [ $(ls $O/q-L*.safetensors 2>/dev/null | wc -l) = 64 ] && return 0
  log "quant i4 K$K: start"; s=$(date +%s)
  qc=$QC
  for a in $(seq 0 $qc 63); do b=$((a + qc - 1))
    [ $(for l in $(seq $a $b); do ls $O/q-L$(printf %02d $l).safetensors; done 2>/dev/null | wc -l) = $qc ] && continue
    [ -f $A/STOP ] && { log "quant K$K: STOP"; return 1; }
    # conv.lock too: a 27B quant (~24 GB) beside anyone's rad-convert runs gpubox out of RAM (memguard kills the pack)
    ( exec 9>/root/conv.lock; flock 9; exec 4>$A/turn.lock; flock 4; exec 5>/root/card0.lock; flock 5; exec 7>/root/ram.lock; flock -s 7; cd /root
      ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build \
        exec $PY -I ourq/ourq.py -K $K -o "$A/q-i4/K{K}" --device 0 --layers $a-$b --frac-ext $SRC --scaled-hessian \
        --sigma-reg 0.025 --sigma-reg-in 0.005 --codebook i4 >> $A/logs/quant-K$K.log 2>&1 )
  done
  n=$(ls $O/q-L*.safetensors 2>/dev/null | wc -l)
  [ "$n" = 64 ] && log "quant i4 K$K: done in $(( ($(date +%s) - s) / 60 )) min" || { log "quant i4 K$K: FAIL ($n layers, $A/logs/quant-K$K.log)"; return 1; }
}

# ---- lane S: pack + base mix (all trunk a8g128) for one level; census: 336 exl3i K mul1i4 a8g128, the rest = V2's bytes
basemix() { K=$1 O=$A/q-i4/K$1 P=$A/pack-K$1.rad M=$A/mix-K$1-a8.rad
  [ -f $M ] && [ -f $A/census-K$K.ok ] && return 0
  [ $(ls $O/q-L*.safetensors 2>/dev/null | wc -l) = 64 ] || return 1
  TRUNK | sed 's/^blk\.[0-9]*\./blk.*./' | sort -u | sed "s|\$| exl3i src=$O codebook=mul1i4 rate=$K|" > $A/K$K.recipe
  if [ ! -f $P ]; then
    # one queue turn with the card AND ram.lock exclusive: rad-convert beside a 27B quant (~24 GB) runs gpubox out of
    # RAM (11-07 12:01 UTC: memguard killed the K2 pack); the card idles for the pack's ~9 min
    ( exec 9>/root/conv.lock; flock 9; exec 4>$A/turn.lock; flock 4; exec 5>/root/card0.lock; flock 5; exec 7>/root/ram.lock; flock 7
      cd /workspace; s=$(date +%s); rm -f $P.part
      RADIANCE_HOME=$H RADIANCE_HOST_DEVICES=1 nice /data/swap27/tools/memguard.sh 2500 $I/bin/rad-convert models/qwen38-27b-bf16 \
        --recipe $A/K$K.recipe --set quantization_config.quant_method=exl3 -o $P.part > $A/logs/conv-K$K.log 2>&1 && mv $P.part $P
      rc=$?; echo "$(date -u +%T) pack K$K rc=$rc $(( $(date +%s) - s ))s" ) | tee -a $L
    [ -f $P ] || { rm -f $P.part; log "pack K$K FAIL ($A/logs/conv-K$K.log)"; return 1; }
  fi
  { TRUNK | sed "s|\$| $P|"; TRUNK | sed 's/^/act /; s/$/ a8g128/'; } > $A/K$K-a8.mix
  $T/radmix $M.part $V2 $A/K$K-a8.mix > $A/logs/mix-K$K.log 2>&1 && mv $M.part $M || { log "radmix K$K FAIL"; return 1; }
  $PY -I /data/swap27/tools/census.py $M --mix $A/K$K-a8.mix --base $V2 > $A/census-K$K.txt 2>&1 || { log "census K$K FAIL ($A/census-K$K.txt)"; return 1; }
  grep -qE "exl3i  *$K(\.0+)?  *mul1i4 *a8g128  *336 " $A/census-K$K.txt || { log "K$K: not 336 exl3i K$K mul1i4 a8g128 trunk weights"; return 1; }
  touch $A/census-K$K.ok; rm -f $P
  log "K$K: base mix ready, $($PY -c "import os; print(f'{os.path.getsize(\"$M\") * 8 / 26895998464:.4f} bpw')")"
}

# score containers, all inside ONE card-lock hold (then ram shared): NAME=MODEL ... -> $A/scores/NAME.json(.rows)
score1() { name=$1 M=$2 s=$(date +%s)
  ROCR_VISIBLE_DEVICES=0 RADIANCE_HOME=$H $I/bin/radiance --model $M --tp 1 --kv-cache-dtype bf16 \
    --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $A/scores/$name.json.part > $A/logs/score-$name.log 2>&1
  rc=$?; [ $rc = 0 ] && [ -f $A/scores/$name.json.part.rows ] && mv $A/scores/$name.json.part.rows $A/scores/$name.json.rows && mv $A/scores/$name.json.part $A/scores/$name.json
  echo "$(date -u +%T) score $name rc=$rc $(( $(date +%s) - s ))s" >> $L; }
score() { todo=(); for nm in "$@"; do [ -f $A/scores/${nm%%=*}.json.rows ] || todo+=("$nm"); done
  [ ${#todo[@]} = 0 ] && return 0
  ( exec 4>$A/turn.lock; flock 4; exec 5>/root/card0.lock; flock 5; exec 7>/root/ram.lock; flock -s 7
    for nm in "${todo[@]}"; do [ -f $A/STOP ] && break; score1 ${nm%%=*} ${nm#*=}; done )
  for nm in "${todo[@]}"; do [ -f $A/scores/${nm%%=*}.json.rows ] || { log "score ${nm%%=*} FAIL ($A/logs/score-${nm%%=*}.log)"; return 1; }; done; }

# one arm's container: NAME K MODE FAMILIES BANDS (comma lists; BANDS a-b; MODE f16 = every trunk weight f16) -> mix +
# census (the named weights = the level's base mix, act keys as set, every other tensor = V2's bytes)
prep() { name=$1 K=$2 mode=$3 fams=$4 bands=$5 M=$A/arms/$1.rad B=$A/mix-K$2-a8.rad
  [ -f $A/scores/$name.json.rows ] && return 0
  [ -f $M ] && grep -q "^CENSUS PASS" $A/arms/$name.census 2>/dev/null && return 0
  [ -f $A/census-K$K.ok ] || return 1
  for f in ${fams//,/ }; do for bd in ${bands//,/ }; do FAM $f ${bd%-*} ${bd#*-}; done; done | sort -u > $A/arms/$name.set
  [ -s $A/arms/$name.set ] || { log "arm $name: empty weight set"; return 1; }
  { TRUNK | sed "s|\$| $B|"
    TRUNK | while read w; do grep -qxF $w $A/arms/$name.set && echo "act $w $mode" || echo "act $w a8g128"; done; } > $A/arms/$name.mix
  $T/radmix $M.part $V2 $A/arms/$name.mix > $A/logs/mix-$name.log 2>&1 && mv $M.part $M || { log "radmix $name FAIL"; return 1; }
  $PY -I /data/swap27/tools/census.py $M --mix $A/arms/$name.mix --base $V2 > $A/arms/$name.census 2>&1 || { log "census $name FAIL"; rm -f $M; return 1; }
}
# arms in batches of NB (one card-lock hold each): lines "NAME K MODE FAMILIES BANDS" on stdin; readout vs K<K>-a8
batch() { mapfile -t todo < <(while read n k m f b; do [ -f $A/scores/$n.json.rows ] || echo "$n $k $m $f $b"; done)
  i=0; while [ $i -lt ${#todo[@]} ]; do [ -f $A/STOP ] && return 1
    chunk=("${todo[@]:$i:$NB}"); i=$((i + NB)); sc=()
    for x in "${chunk[@]}"; do set -- $x; prep "$@" && sc+=("$1=$A/arms/$1.rad"); done
    score "${sc[@]}"
    for x in "${chunk[@]}"; do set -- $x; [ -f $A/scores/$1.json.rows ] && ! grep -q "\"tag\": \"$1\"" $R 2>/dev/null && readout $1 K$2-a8; rm -f $A/arms/$1.rad; done
  done; }
readout() { name=$1 base=$2
  out=$($PY -I $T/cmp.py $G $A/scores/$base.json.rows $A/scores/$name.json.rows --tag $name --base $base 2>&1 | tail -1)
  echo "$out" >> $R; log "pair $base -> $name: $out"; }

laneS() { for K in $KS; do
    until [ $(ls $A/q-i4/K$K/q-L*.safetensors 2>/dev/null | wc -l) = 64 ] || [ -f $A/STOP ]; do
      kill -0 $QPID 2>/dev/null || break; sleep 120; done
    t=0; until basemix $K; do t=$((t + 1)); [ $t -ge 3 ] || [ -f $A/STOP ] && break; log "K$K: base mix retry $t in 10 min"; sleep 600; done
    [ -f $A/census-K$K.ok ] || continue
    if [ $K = 2 ]; then     # with the K2 baseline, once: V2 3.00 itself (unit tie to the board) and the K2 trunk at f16
      prep K2-f16 2 f16 gate_up,down,attn_in,gdn_in,attn_out,ssm_out 0-63
      score K$K-a8=$A/mix-K$K-a8.rad v2-300=$V2 K2-f16=$A/arms/K2-f16.rad
      [ -f $A/scores/K$K-a8.json.rows ] || continue
      for x in v2-300 K2-f16; do [ -f $A/scores/$x.json.rows ] && ! grep -q "\"tag\": \"$x\"" $R 2>/dev/null && readout $x K2-a8; done; rm -f $A/arms/K2-f16.rad
    else score K$K-a8=$A/mix-K$K-a8.rad || continue; fi
    { for f in gate_up down attn_in gdn_in attn_out ssm_out; do for bd in 0-15 16-31 32-47 48-63; do
        echo "K$K-$f-$bd-g64 $K a4g64 $f $bd"; done; done
      echo "K$K-gu+dn-all-g64 $K a4g64 gate_up,down 0-15,16-31,32-47,48-63"; } | batch
  done
  [ -f $A/STOP ] || { $PY -I $T/analyze.py $A > $A/analysis-g64.txt 2>&1 && log "(g64 phase) $(grep '^A4COST VERDICT' $A/analysis-g64.txt | tail -1)" \
    || log "analyze FAIL ($A/analysis-g64.txt)"; }
  for K in $KS; do [ -f $A/scores/K$K-a8.json.rows ] || continue
    { for f in gate_up down attn_in gdn_in attn_out ssm_out; do for bd in 0-15 16-31 32-47 48-63; do
        echo "K$K-$f-$bd-g128 $K a4g128 $f $bd"; done; done
      echo "K$K-gu+dn-all-g128 $K a4g128 gate_up,down 0-15,16-31,32-47,48-63"; } | batch; done; }

KS=${KS:-"2 3 1"} QC=${QC:-16} NB=${NB:-4}   # levels in order; layers a quant hold; scores a hold
log "A4COST start (pid $$; levels $KS)"
# GPU smoke of this build once (test_a4 + exl3i_gpu_test on card 0; w34 / w53 ran the same commit)
if [ ! -f $A/gputest.ok ]; then
  ( exec 4>$A/turn.lock; flock 4; exec 5>/root/card0.lock; flock 5; exec 7>/root/ram.lock; flock -s 7; cd $A/b/exl3i
    ROCR_VISIBLE_DEVICES=0 timeout 900 ./test_a4 radiance_home/kernels/exl3i.so > $A/logs/test_a4.log 2>&1 && \
    ROCR_VISIBLE_DEVICES=0 timeout 600 ./exl3i_gpu_test radiance_home/kernels/exl3i.so > $A/logs/gpu_test.log 2>&1 ) \
    && touch $A/gputest.ok && log "gpu tests PASS ($(grep -ciE 'pass' $A/logs/test_a4.log) pass lines)" \
    || { log "gpu tests FAIL ($A/logs/test_a4.log, gpu_test.log)"; exit 1; }
fi
( for K in $KS; do quant $K || break; done ) & QPID=$!
laneS
wait $QPID
n=$(ls $A/scores/*.json.rows 2>/dev/null | wc -l)
$PY -I $T/analyze.py $A > $A/analysis.txt 2>&1 && log "$(grep '^A4COST VERDICT' $A/analysis.txt | tail -1)" || log "analyze FAIL ($A/analysis.txt)"
[ -f $A/STOP ] && log "A4COST stopped (STOP; $n scores)" || log "A4COST DONE ($n scores)"
