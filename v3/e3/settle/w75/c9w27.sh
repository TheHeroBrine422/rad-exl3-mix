#!/bin/bash
# c9w27.sh -- w75 #112-02 (rule: ../w75.md): C9 svh multipliers on the 27B writers (16 o_proj, 48 out_proj), fixed
# codes, on the swj K2.5 levels /data/swj27/lv/{baseswj,s1swj}; vs the same levels without, two seed pairs.
# gpubox, card 0 only: every GPU step holds /root/card0.lock then /root/ram.lock (shared); packs hold /root/conv.lock then
# /root/ram.lock EXCLUSIVE (lead 11-07) (locks as
# cd27.sh / asym27.sh / a4cost; it interleaves with them through the locks). Idempotent: rerun after any failure
#   setsid nohup bash /data/c9w27/code/c9w27.sh >> /data/c9w27/run.out 2>&1 < /dev/null &
# Per (seed s, arm v):  mult   c9e27.py (card): multipliers from the level's decoded writers, W, calib H_in
#                       apply  swjapply.py (host): level with svh *= c, everything else byte for byte
#                       pack   swap27k's packmixk: rad-convert (install-116 / swap27 home, the controls' build) -> mix
#                              into V2 3.00 -> census (conv.lock)
#                       gate   radiff.py: the arm's mix == the control mix except the changed writers' svh planes
#                       score  golden-search27 + golden-verify27 (card), the control mix re-scored the same way
#                       pairs  tools/paired.py; v75.py -> $A/report.md, verdict lines -> $A/verdict.txt
# Arm a: attn_o -1, gdn_out 3 (0.8B pl α*). Arm b: attn_o -1, gdn_out = $A/alpha27.txt (#112-01's α_27, relayed from
# the 4070); "none" = no arm b. The driver waits up to WAIT_B_H hours (default 8) for alpha27.txt after arm a.
# Log $A/result.txt ("C9W27 DONE" last). Stop: touch $A/STOP (between steps).
: ${RXM_WORK:=/root}; export RXM_WORK
A=${A:-/data/c9w27}; CODE=$A/code
V2IN=/data/release/qwen3.8-27b-exl3-v2-3.00bpw.rad
GUARD_MB=${GUARD_MB:-3000} MINMEM_GB=${MINMEM_GB:-5} WAIT_B_H=${WAIT_B_H:-8}
export ROCR_VISIBLE_DEVICES=0 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       PYTORCH_HIP_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
# swap27's variables (S T I H PY TRUNK ...) and swap27k's packmixk, sourced read-only as cd27.sh does; then S = $A
source <(sed -n '/^: \${RXM_WORK/,/^quant() /p' /data/swap27/tools/swap27.sh | sed '$d')
source <(sed -n '/^packmixk() /,/^STOPF=/p' /data/swap27/tools/swap27k.sh | sed '$d')
S=$A V2=$V2IN K=2.5
# packmixk ARM SRC: swap27k.sh's packmixk with the lead's 11-07 rule for every pack on iceman: conv.lock + /root/ram.lock
# EXCLUSIVE (flock 7, not -s: nothing else using RAM runs beside a rad-convert; 12:50 UTC the s1c9a pack was killed by
# memguard while a bn4s1 eval ran beside it); a stale .part is removed before the convert and after a failed one.
# Otherwise swap27k's text unchanged (recipe, radmix, census and its 336-weight check).
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
  touch $S/census-$arm.ok; rm -f $P; log "$arm: mix ready, $($PY -c "import os; print(f'{os.path.getsize(\"$M\") * 8 / 26895998464:.4f} bpw')")"; }
GSR=/data/swap27/golden-search27 GVF=/data/verify27/golden-verify27
mkdir -p $A/vf; cd $A
exec 3>$A/.driver.lock; flock -n 3 || { echo "c9w27.sh already running"; exit 0; }
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $A/result.txt; }
stopped() { [ -f $A/STOP ]; }
memwait() { local n=0 g=${2:-$MINMEM_GB}
  while [ $(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo) -lt $g ]; do
    stopped && return 1; [ $n = 0 ] && log "waiting for MemAvailable >= $g GB before $1"; n=$((n + 1))
    [ $n -gt 180 ] && { log "$1: MemAvailable < $g GB for 3 h: FAIL (rerun later)"; return 1; }; sleep 60; done; }
# packtry TAG SRC: packmixk, retried (at most 2 times, 10 min apart, after MemAvailable >= PACKMEM_GB) when memguard
# killed the pack for host memory (rc 99, 11-07 12:50 UTC); any other failure stops
packtry() { local n
  for n in 1 2 3; do
    [ -f $A/mix-$1.rad ] && [ -f $A/census-$1.ok ] && return 0; stopped && return 1
    memwait "pack $1" ${PACKMEM_GB:-8} || return 1
    packmixk $1 $2 && return 0
    tail -n 3 $A/conv-$1.log 2>/dev/null | grep -q "memguard: MemAvailable" || return 1
    [ $n = 3 ] && break; log "$1: the pack was stopped by memguard (host memory low): retry $n of 2 in 10 min"; sleep 600
  done; return 1; }
card() { ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; cd $RXM_WORK
    $T/memguard.sh $GUARD_MB nice -n 5 "$@" ); }
host() { nice -n 10 "$@"; }
# score TAG MIX GOLDEN OUTDIR: swap27.sh's score (same binary, home, flags), any golden / out dir
score() { local tag=$1 M=$2 G=$3 O=$4; [ -f $O/$tag.json ] && [ -f $O/$tag.json.rows ] && return 0; stopped && return 1
  memwait "score $tag" || return 1
  ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; s=$(date +%s)
    ROCR_VISIBLE_DEVICES=0 RADIANCE_HOME=$H $I/bin/radiance --model $M --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $O/$tag.json.part > $O/score-$tag.log 2>&1
    rc=$?; [ $rc = 0 ] && mv $O/$tag.json.part $O/$tag.json; [ -f $O/$tag.json.part.rows ] && mv $O/$tag.json.part.rows $O/$tag.json.rows
    echo "$(date -u +%H:%M:%S) score $tag on $(basename $G) rc=$rc $(( $(date +%s) - s ))s" ) | tee -a $A/result.txt
  [ -f $O/$tag.json ] && [ -f $O/$tag.json.rows ] || { log "score $tag on $(basename $G) FAIL ($O/score-$tag.log)"; return 1; }; }
# pairl A B GOLDEN DIR: paired.py line (once)
pairl() { grep -qF "pair $1 -> $2 ($(basename $3)):" $A/result.txt 2>/dev/null && return 0
  log "pair $1 -> $2 ($(basename $3)): $($PY -I $T/paired.py $3 $4/$1.json.rows $4/$2.json.rows | tail -n 1)"; }
alpha_of() { case $1 in a) echo "attn_o=-1,gdn_out=3" ;;
  b) local x; [ -f $A/alpha27.txt ] && x=$(tr -d ' \n' < $A/alpha27.txt) && [ "$x" != none ] && [ -n "$x" ] && echo "attn_o=-1,gdn_out=$x" ;; esac; }

arm() { local s=$1 v=$2 al; al=$(alpha_of $v); [ -n "$al" ] || return 0
  local O=$A/$s-$v LV=/data/swj27/lv/${s}swj CM=/data/swj27/mix-${s}swj.rad tag=${s}c9$v
  mkdir -p $O; [ -f $O/alpha.txt ] || echo "$al" > $O/alpha.txt
  [ "$(cat $O/alpha.txt)" = "$al" ] || { log "$tag: alpha changed ($(cat $O/alpha.txt) -> $al): FAIL (clear $O by hand)"; return 1; }
  [ -f $LV/.done ] && [ -f $CM ] || { log "$tag: level $LV or control mix $CM missing: FAIL"; return 1; }
  if [ ! -f $O/scales/.done ]; then stopped && return 1; memwait "mult $tag" || return 1; local t0=$(date +%s)
    card env DEV=cuda $PY -I $CODE/c9e27.py $LV $O/scales --alpha $al >> $O/c9e27.log 2>&1
    [ -f $O/scales/.done ] || { log "$tag: c9e27 FAIL ($O/c9e27.log)"; return 1; }
    log "$tag: multipliers ($al) in $(( $(date +%s) - t0 )) s: $($PY -I -c "
import json; L = [json.loads(l) for l in open('$O/scales/c9.jsonl')]
for f in ('gdn_out', 'attn_o'):
    x = [o for o in L if o['fam'] == f]
    if x: print(f, len(x), 'e mean %.4f max %.3f, c mean %.4f max %.3f min %.3f, rows c>2: %d' % (sum(o['e_mean'] for o in x) / len(x), max(o['e_max'] for o in x),
        sum(o['c_mean'] for o in x) / len(x), max(o['c_max'] for o in x), min(o['c_min'] for o in x), sum(o['n_gt2'] for o in x)), end='; ')")"; fi
  if [ ! -f $O/level/.done ]; then stopped && return 1
    host $PY -I $CODE/swjapply.py $LV $O/scales $O/level >> $O/apply.log 2>&1 || { log "$tag: swjapply FAIL ($O/apply.log)"; return 1; }
    log "$tag: $(tail -n 1 $O/apply.log)"; fi
  stopped && return 1
  packtry $tag $O/level || { log "$tag: pack FAIL"; return 1; }
  if [ ! -f $O/radiff.pass ]; then
    host $PY -I $CODE/radiff.py $A/mix-$tag.rad $CM $O/scales/scales > $O/radiff.log 2>&1; local rc=$?
    log "$tag: identity gate: $(tail -n 1 $O/radiff.log)"
    [ $rc = 0 ] && touch $O/radiff.pass || { log "$tag: identity gate FAIL: not scored ($O/radiff.log)"; return 1; }; fi
  [ -L $A/mix-ctl-${s}swj.rad ] || ln -s $CM $A/mix-ctl-${s}swj.rad
  score ctl-${s}swj $CM $GSR $A && score $tag $A/mix-$tag.rad $GSR $A || return 1
  pairl ctl-${s}swj $tag $GSR $A
  score ctl-${s}swj $CM $GVF $A/vf && score $tag $A/mix-$tag.rad $GVF $A/vf || return 1
  pairl ctl-${s}swj $tag $GVF $A/vf
  if [ $v = b ] && [ -f $A/vf/${s}c9a.json.rows ]; then pairl ${s}c9a $tag $GSR $A; pairl ${s}c9a $tag $GVF $A/vf; fi
  log "$tag done"; }
verdict() { $PY -I $CODE/v75.py $A > $A/report.md.part 2>> $A/v75.err && mv $A/report.md.part $A/report.md || { log "v75 FAIL ($A/v75.err)"; return 1; }
  touch $A/verdict.txt   # verdict lines live here (a rule-level "FAIL" in result.txt would read as a job failure)
  grep '^C9W27 VERDICT' $A/report.md | while read -r l; do grep -qxF "$(date -u +%F) $l" $A/verdict.txt || grep -qF " $l" $A/verdict.txt \
    || { echo "$(date -u +%F) $l" >> $A/verdict.txt; log "verdict line written to $A/verdict.txt"; }; done; }

log "start (pid $$): arms a$([ -n "$(alpha_of b)" ] && echo " b ($(alpha_of b))")"
ok=1; for s in s1 base; do arm $s a || ok=0; done
[ $ok = 1 ] && verdict
if [ $ok = 1 ] && [ ! -f $A/alpha27.txt ]; then
  log "arm a done; waiting up to $WAIT_B_H h for $A/alpha27.txt (#112-01's alpha_27, relayed from the 4070)"
  n=0; while [ ! -f $A/alpha27.txt ] && [ $n -lt $(( WAIT_B_H * 6 )) ] && ! stopped; do sleep 600; n=$((n + 1)); done; fi
if [ $ok = 1 ] && [ -f $A/alpha27.txt ]; then
  for s in s1 base; do arm $s b || ok=0; done
  verdict && [ $ok = 1 ] && log "C9W27 DONE" && exit 0; fi
stopped && log "stopped (STOP)"
log "C9W27 INCOMPLETE (ok=$ok, alpha27.txt $([ -f $A/alpha27.txt ] && echo present || echo missing)); rerun to resume"
