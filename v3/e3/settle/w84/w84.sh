#!/bin/bash
# w84.sh -- #117 27B KLD pairs (rule: v3/e3/settle/w84.md, written before any data): the residual-block permutation as
# a block-order LDLQ of the 272 residual readers (route B, blkorder84.py) on top of a plain K2.5 control level; pack,
# mix into V2 3.00, hygiene, score on golden-search27, pair. gpubox, card 0 under /root/card0.lock + /root/ram.lock
# (shared); packs under /root/conv.lock + /root/ram.lock EXCLUSIVE (lead 11-07). Idempotent: rerun after any failure.
#   w84.sh chk          gate G2 (signchk84.py: route A == route B, the sign trap), once; FAIL stops every arm
#   w84.sh ARM          ARM = p0 (control /data/v3ladder2/K2.5, seed 0, "base") | p1 (/data/swap27/q-s1/K2.5, "s1")
#     noop   route B identity order on 9 readers (L0 ssm_qkv / ssm_z / gate / up, L3 q / k / v / gate / up): rxq.pairs
#            same CONTROL -> noop.pass; a FAIL makes the control a fresh identity-order requant of all 272 readers (<ARM>n)
#     arm    route B consensus order (order82.json) on the 272 readers -> q-perm
#     asm    rxq.pairs splice CONTROL + q-perm -> level-perm; rxq.pairs levels: PAIR OK
#     pack   swap27k's packmixk (exl3i, mul1i8, rate 2.5) into V2 3.00 -> mix-<ARM>perm.rad
#     hyg    rxq.pairs mixes mix-<ARM>perm.rad CONTROL_MIX --arm readers (stored swap27 mix; refused -> fresh control
#            <ARM>ctl packed from the control level)
#     score  the arm; the control mix re-scored (<ARM>re): bit-identical rows to the stored ones -> stored rows are the
#            control, else the fresh rows
#     pair   tools/paired.py control rows -> arm rows -> <ARM>/pair.json (golden-search27: report only, amendment 1)
#     score27b  the arm and its control on golden-search27b's parts (-x xcorp, -c calib; mk27b.sh) -> $A/b27/<tag>-{x,c}
#   w84.sh verdict      score27b for p0 / p1 where missing, then verdict84.py (amendment 1's two-part rule)
#   w84.sh vcheck       item 3 (lead): bn / bns1 / bns2 / s2 (swap27 mixes) on golden-verify27 -> $A/vchk, vcheck84.py
# Env: A (/data/w84), CODE ($A/code), STEPS, GUARD_MB (3000), MINMEM_GB (6). Stop: touch $A/<ARM>/STOP (between steps).
: ${RXM_WORK:=/root}; export RXM_WORK
A=${A:-/data/w84}; CODE=${CODE:-$A/code}; W=$CODE/v3/e3/settle/w84
GUARD_MB=${GUARD_MB:-3000} MINMEM_GB=${MINMEM_GB:-6}
MODEL=${MODEL:-/data/workspace/models/qwen38-27b-bf16} HESS=${HESS:-/root/challenge/hessians/calib}
ORDER=$W/order82.json
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=/data/exl3frac/build PYTORCH_HIP_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
source <(sed -n '/^: \${RXM_WORK/,/^quant() /p' /data/swap27/tools/swap27.sh | sed '$d')
S=$A V2=/data/release/qwen3.8-27b-exl3-v2-3.00bpw.rad K=2.5; G=/data/swap27/golden-search27
# swap27k's packmixk with the lead's 11-07 pack rule (conv.lock + ram.lock EXCLUSIVE, stale .part removed), as cdhout27.sh
packmixk() { arm=$1 src=$2 P=$S/pack-$1.rad M=$S/mix-$1.rad
  [ -f $M ] && [ -f $S/census-$arm.ok ] && return 0
  [ $(ls $src/q-L*.safetensors 2>/dev/null | wc -l) = 64 ] || { log "$arm: $src has not 64 layers"; return 1; }
  TRUNK | sed 's/^blk\.[0-9]*\./blk.*./' | sort -u | sed "s|\$| exl3i src=$src codebook=mul1i8 rate=$K|" > $S/$arm.recipe
  if [ ! -f $P ]; then
    ( exec 9>$RXM_WORK/conv.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock 7; cd /workspace; s=$(date +%s); rm -f $P.part
      RADIANCE_HOME=$H RADIANCE_HOST_DEVICES=1 nice $T/memguard.sh 2500 $I/bin/rad-convert models/qwen38-27b-bf16 \
        --recipe $S/$arm.recipe --set quantization_config.quant_method=exl3 -o $P.part > $S/conv-$arm.log 2>&1 && mv $P.part $P
      echo "pack $arm rc=$? $(( $(date +%s) - s ))s" ) | tee -a $S/result.txt
    [ -f $P ] || { rm -f $P.part; log "pack $arm FAIL ($S/conv-$arm.log)"; return 1; }
  fi
  TRUNK | sed "s|\$| $P|" > $S/$arm.mix
  $T/radmix $M.part $V2 $S/$arm.mix > $S/mix-$arm.log 2>&1 && mv $M.part $M || { log "radmix $arm FAIL"; return 1; }
  $PY -I $T/census.py $M --mix $S/$arm.mix --base $V2 > $S/census-$arm.txt 2>&1 || { log "census $arm FAIL"; return 1; }
  grep -qE "exl3i  *$K(\.0+)?  *mul1i8 *f16  *336 " $S/census-$arm.txt || { log "$arm: not 336 exl3i K$K trunk weights"; return 1; }
  touch $S/census-$arm.ok; rm -f $P; log "$arm: mix ready"; }
mkdir -p $A; cd $A
ARM=$1; O=$A/$ARM; mkdir -p $O
L=$O/w84.log; log() { echo "$(date -u +%H:%M:%S) $ARM: $*" | tee -a $L >> $A/result.txt; tail -n 1 $L; }
want() { [[ " ${STEPS:-noop arm asm pack hyg score pair score27b} " == *" $1 "* ]] && [ ! -f $O/STOP ]; }
die() { log "$*: stop"; exit 1; }
memok() { local m=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  [ $m -ge $MINMEM_GB ] || die "MemAvailable $m GB < $MINMEM_GB GB before $1 (rerun later)"; }
card() { ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; cd $RXM_WORK
    $T/memguard.sh $GUARD_MB nice -n 5 "$@" ); }
host() { nice -n 10 "$@"; }
score_g() { # score_g GOLDEN OUTBASE MIX: swap27's score (card 0, bf16 KV) on another golden -> OUTBASE.json(.rows)
  local g=$1 out=$2 mix=$3; [ -f $out.json.rows ] && return 0
  [ -f $mix ] || { log "score $(basename $out): no mix $mix"; return 1; }
  ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; s=$(date +%s)
    ROCR_VISIBLE_DEVICES=0 RADIANCE_HOME=$H $I/bin/radiance --model $mix --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $g --kld-out $out.json.part > $out.log 2>&1 \
      && mv $out.json.part $out.json; [ -f $out.json.part.rows ] && mv $out.json.part.rows $out.json.rows
    echo "$(date -u +%H:%M:%S) score $(basename $out) on $(basename $g) rc=$? $(( $(date +%s) - s ))s" ) >> $A/result.txt
  [ -f $out.json.rows ] || { log "score $(basename $out) FAIL ($out.log)"; return 1; }; }
RX=(env PYTHONPATH=$CODE/pipeline RXQ_RXM=$CODE RXQ_EXL3_SRC=${SRC:-/data/exl3frac/src} $PY)
READERS=ssm_qkv,ssm_z,attn_qg,attn_k,attn_v,ffn_gate,ffn_up
for f in pipeline/rxq/pairs.py pipeline/config/v3.toml store/census.py v3/e3/exl3frac2/exl3frac2.py \
         v3/e3/settle/w84/blkorder84.py v3/e3/settle/w84/w84q.py v3/e3/settle/w84/signchk84.py v3/e3/settle/w84/order82.json; do
  [ -f $CODE/$f ] || die "$CODE/$f missing"; done

# ---------------------------------------------------------------------------------------------------- G2 (once)
if [ "$ARM" = chk ]; then
  [ -f $O/signchk.pass ] && { log "G2 already PASS"; exit 0; }
  memok signchk; s=$(date +%s)
  card "${RX[@]}" $W/signchk84.py $ORDER $MODEL $HESS $O 0 > $O/signchk84.log 2>&1
  r=$(grep "^SIGNCHK" $O/signchk84.log | tail -n 1); log "G2 route A == route B ($(( $(date +%s) - s )) s): ${r:-no result ($O/signchk84.log)}"
  [[ "$r" == "SIGNCHK PASS"* ]] && touch $O/signchk.pass && exit 0
  touch $O/signchk.fail; exit 1
fi
if [ "$ARM" = verdict ]; then
  for a in p0 p1; do STEPS=score27b bash $0 $a || die "score27b $a FAIL"; done
  $PY -I $W/verdict84.py $A p0 p1 > $A/verdict.txt 2>&1 || die "verdict84.py FAIL ($A/verdict.txt)"
  log "$(tail -n 1 $A/verdict.txt)"; exit 0
fi
if [ "$ARM" = vcheck ]; then
  mkdir -p $A/vchk; GV=/data/verify27/golden-verify27
  for t in bn bns1 bns2 s2; do score_g $GV $A/vchk/$t /data/swap27/mix-$t.rad || die "vcheck score $t FAIL"; done
  ALL=attn_qg,attn_k,attn_v,attn_output,ssm_qkv,ssm_z,ssm_out,ffn_gate,ffn_up,ffn_down
  for p in "bn base" "bns1 s1" "bns2 s2"; do set -- $p
    log "vcheck hygiene $2 -> $1: $(host "${RX[@]}" -m rxq.pairs mixes /data/swap27/mix-$1.rad /data/swap27/mix-$2.rad --arm $ALL | tail -n 1)"; done
  $PY -I $W/vcheck84.py $A > $A/vchk/vcheck.txt 2>&1 || die "vcheck84.py FAIL"
  log "$(tail -n 1 $A/vchk/vcheck.txt)"; exit 0
fi
case $ARM in
  p0) BASE=/data/v3ladder2/K2.5 CTAG=base ;;
  p1) BASE=/data/swap27/q-s1/K2.5 CTAG=s1 ;;
  *) echo "w84.sh chk|p0|p1|verdict|vcheck"; exit 2 ;;
esac
[ -f $A/chk/signchk.pass ] || die "gate G2 has not passed ($A/chk): no arm before it"
[ $(ls $BASE/q-L*.safetensors 2>/dev/null | wc -l) = 64 ] || die "$BASE: not 64 layer shards"
X=$("${RX[@]}" -m rxq levelargs $BASE) || die "rxq levelargs: $BASE has settings rxq's config cannot express"
[[ " $X " == *" --hout "* ]] && die "$BASE has H_out: this pair is on plain levels"
X="$X --set quant.tile_cd.enable=false"
KEYS=$("${RX[@]}" -c "
from rxq import qwen35
print(','.join(k for l in range(64) for _, ks in qwen35.layer_plan(l) for k in ks if qwen35.family(k) in '$READERS'.split(',')))")
NK=$(echo $KEYS | tr ',' '\n' | wc -l); [ $NK = 272 ] || die "reader keys: $NK, expected 272"
KEYS9=$("${RX[@]}" -c "
L = 'model.language_model.layers'
print(','.join([f'{L}.0.linear_attn.in_proj_qkv', f'{L}.0.linear_attn.in_proj_z', f'{L}.0.mlp.gate_proj', f'{L}.0.mlp.up_proj',
                f'{L}.3.self_attn.q_proj', f'{L}.3.self_attn.k_proj', f'{L}.3.self_attn.v_proj', f'{L}.3.mlp.gate_proj', f'{L}.3.mlp.up_proj']))")
log "start: control $BASE ($X), steps: ${STEPS:-all}"

quant() { # quant DIR ORDER KEYS N
  local d=$1 ord=$2 keys=$3 n=$4
  [ $(ls $d/t-*.safetensors 2>/dev/null | wc -l) = $n ] && return 0
  memok "quant $(basename $d)"; s=$(date +%s)
  card "${RX[@]}" $W/w84q.py $ord --keys $keys $X --ckpt $MODEL --hess $HESS -o $d --device 0 --no-prof >> $d.log 2>&1
  [ $(ls $d/t-*.safetensors 2>/dev/null | wc -l) = $n ] || die "quant $d FAIL ($d.log; a rerun resumes per tensor)"
  log "$(basename $d): $n tensors, $(( $(date +%s) - s )) s"; }
if want noop && [ ! -f $O/noop.pass ] && [ ! -f $O/noop.fail ]; then
  quant $O/q-noop ident $KEYS9 9
  r=$(host "${RX[@]}" -m rxq.pairs same $BASE $O/q-noop | tail -n 1); log "noop gate (identity order, 9 readers): $r"
  [[ "$r" == "PAIR OK"* ]] && touch $O/noop.pass || touch $O/noop.fail
fi
NOOPFAIL=$([ -f $O/noop.fail ] && echo 1)
if want arm; then
  quant $O/q-perm $ORDER $KEYS 272
  [ -n "$NOOPFAIL" ] && quant $O/q-noopfull ident $KEYS 272
  log "readers' proxy_err perm / control (one-sided, after the refit): $($PY -I $W/proxy84.py $BASE/proxy.jsonl $O/q-perm/log.jsonl 2>&1 | tail -n 1)"
fi
if want asm; then
  [ -f $O/level-perm/spliced.json ] || host "${RX[@]}" -m rxq.pairs splice $BASE $O/q-perm $O/level-perm --arm $READERS | tee -a $L || die "splice FAIL"
  r=$(host "${RX[@]}" -m rxq.pairs levels $BASE $O/level-perm --arm $READERS | tail -n 1); log "level hygiene: $r"
  [[ "$r" == "PAIR OK"* ]] || die "level-perm differs from the control outside the readers"
  if [ -n "$NOOPFAIL" ] && [ ! -f $O/level-noop/spliced.json ]; then
    host "${RX[@]}" -m rxq.pairs splice $BASE $O/q-noopfull $O/level-noop --arm $READERS | tee -a $L || die "splice (noop) FAIL"; fi
fi
CTLTAG=$CTAG CTLMIX=/data/swap27/mix-$CTAG.rad CR=/data/swap27/$CTAG.json.rows
[ -n "$NOOPFAIL" ] && CTLTAG=${ARM}n CTLMIX=$A/mix-${ARM}n.rad CR=$A/${ARM}n.json.rows
[ -z "$NOOPFAIL" ] && [ -f $O/ctl.fresh ] && CTLTAG=${ARM}ctl CTLMIX=$A/mix-${ARM}ctl.rad CR=$A/${ARM}ctl.json.rows
if want pack; then
  packmixk ${ARM}perm $O/level-perm || die "pack ${ARM}perm FAIL"
  [ "$CTLTAG" = ${ARM}n ] && { packmixk ${ARM}n $O/level-noop || die "pack ${ARM}n FAIL"; }
fi
if want hyg && [ ! -f $O/hyg.pass ]; then
  r=$(host "${RX[@]}" -m rxq.pairs mixes $A/mix-${ARM}perm.rad $CTLMIX --arm $READERS | tail -n 1); log "mix hygiene vs $CTLTAG: $r"
  if [[ "$r" != "PAIR OK"* ]] && [ "$CTLTAG" = $CTAG ]; then
    touch $O/ctl.fresh; CTLTAG=${ARM}ctl CTLMIX=$A/mix-${ARM}ctl.rad CR=$A/${ARM}ctl.json.rows
    packmixk $CTLTAG $BASE || die "pack $CTLTAG FAIL"
    r=$(host "${RX[@]}" -m rxq.pairs mixes $A/mix-${ARM}perm.rad $CTLMIX --arm $READERS | tail -n 1); log "mix hygiene vs $CTLTAG: $r"
  fi
  [[ "$r" == "PAIR OK"* ]] && touch $O/hyg.pass || die "the arm's mix differs from $CTLTAG outside the readers: not scored"
fi
if want score; then
  [ -f $O/hyg.pass ] || die "no hygiene pass: not scored"
  C=0 score ${ARM}perm || die "score ${ARM}perm FAIL"
  if [ "$CTLTAG" = $CTAG ]; then        # the stored control rows: re-score the stored control mix, use them if identical
    if [ ! -f $A/${ARM}re.json.rows ]; then
      ln -sf $CTLMIX $A/mix-${ARM}re.rad; C=0 score ${ARM}re || die "score ${ARM}re FAIL"; fi
    if cmp -s $A/${ARM}re.json.rows $CR; then log "control re-score: rows bit-identical to $CR (stored rows used)"
    else log "control re-score: rows differ from $CR -> the fresh rows are the control"; CR=$A/${ARM}re.json.rows; echo $CR > $O/ctl.rows; fi
  else C=0 score $CTLTAG || die "score $CTLTAG FAIL"; fi
fi
[ -f $O/ctl.rows ] && CR=$(cat $O/ctl.rows)
if want pair && [ -f $A/${ARM}perm.json.rows ]; then
  [ -f $CR ] || die "no control rows $CR"
  $PY -I $T/paired.py $G $CR $A/${ARM}perm.json.rows | tail -n 1 > $O/pair.json || die "paired.py FAIL"
  echo "{\"arm\": \"$ARM\", \"control\": \"$CR\", \"perm\": \"$A/${ARM}perm.json.rows\", \"level\": \"$BASE\", \"noop\": \"$([ -n "$NOOPFAIL" ] && echo fail || echo pass)\", \"hygiene\": \"pass\"}" > $O/pair-meta.json
  log "pair $(basename ${CR%.json.rows}) -> ${ARM}perm: $(cat $O/pair.json)"
fi
if want score27b; then
  [ -f $O/hyg.pass ] || die "no hygiene pass: not scored"
  echo $CTLTAG > $O/ctl.tag; mkdir -p $A/b27
  for part in x c; do
    score_g $A/golden-search27b-$part $A/b27/${ARM}perm-$part $A/mix-${ARM}perm.rad || die "score27b ${ARM}perm-$part FAIL"
    score_g $A/golden-search27b-$part $A/b27/$CTLTAG-$part $CTLMIX || die "score27b $CTLTAG-$part FAIL"
  done
  log "score27b: ${ARM}perm and $CTLTAG scored on golden-search27b-x / -c"
fi
log "W84 $ARM DONE (${STEPS:-all})"
