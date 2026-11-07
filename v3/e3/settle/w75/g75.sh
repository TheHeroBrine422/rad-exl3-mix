#!/bin/bash
# g75.sh -- w75 #112-01 (gdn_out C9 α beyond 3) and #106-01 (o_proj head act order on the real quantizer), 0.8B, 4070
# (../w75.md). Idempotent (rerun = continue: every product is skipped when its file exists); every GPU unit holds
# /data/yb/gpu.lock (< 15 min each). Runs a snapshot: /data/yb/w75/snap/. Reads w64's products (/data/yb/w64) and
# d1112's shards read-only.
# Steps (G75_STEPS, default "i1s i1x i1c p06 report"):
#   i1s  #112-01 screen, seed 0, fit: gdn_out g in {3,4,5,6,8} (others at w64's α*), stacks st (u) / pl (p), K3 / K2
#   i1x  the pick (r75.py pick -> alpha75.json); a fit arm at α*_new if off the grid; the clamp check (c <= 2) if needed
#   i1c  confirm α*_new != 3 on seeds 0-2, search + verify, paired with w64's c9 arm (g 3)
#        then #112-02 arm B's α -> alpha27.txt ("X" or "none"; relayed to the gpubox queue)
#   p06  #106-01: permuted checkpoint + Hessians, bf16 gate, per (K, seed) no-op gate + perm quant of the 6 o_proj,
#        decode, un-permute, score; plumbing check; rel06.jsonl
#   report  r75.py -> O/report.md
# Stop: touch /data/yb/w75/STOP (checked between units).
P=/workspace/rad-exl3-mix/proxy2b
SNAP=$(dirname "$(readlink -f "$0")")
C9W=$SNAP/c9w75.py R75=$SNAP/r75.py DEQ=$SNAP/deq.py PERM=$SNAP/perm06.py BCHK=$SNAP/bf16chk.py ASM=$SNAP/assemble.py
M=/data/models/qwen35-08b F=/data/g08-fit1 GS=/data/g08-search1 GV=/data/g08-verify1 H=/data/h08/calib
O=${G75_OUT:-/data/yb/w75} W64=/data/yb/w64 D12=/data/yb/d1112
C18=/data/yb/c18/hdiag-snap1024-dy SRC=/data/src/exllamav3/exllamav3/exllamav3_ext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
STEPS=${G75_STEPS:-i1s i1x i1c p06 report}
mkdir -p $O/stats; cd /tmp
log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
stopped() { [ -f $O/STOP ]; }
want() { [[ " $STEPS " == *" $1 "* ]] && ! stopped; }
lock() { exec 9>/data/yb/gpu.lock; flock 9; echo "$(date -u +%T) lock $1" >> $O/lock.log; }
unlock() { echo "$(date -u +%T) unlock" >> $O/lock.log; flock -u 9; }
sfx() { [ $1 = 0 ] && echo "" || echo "_s$1"; }
nq() { ls $1/*.safetensors 2>/dev/null | wc -l; }
cpu() { CUDA_VISIBLE_DEVICES= DEV=cpu OMP_NUM_THREADS=4 nice -n 19 "$@"; }
# ev TAG WQ GOLDEN [MODEL]: evalq score file O/TAG.f32 (skipped if present); call under lock
ev() { local tag=$1 wq=$2 g=$3 mdl=${4:-$M} r; [ -f $O/$tag.f32 ] && return 0
  r=$(python3 $P/evalq.py $mdl $g --wq $wq --out $O/$tag.f32 2>> $O/eval.err | tail -n 1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')
  [ -f $O/$tag.f32 ] || { log "EVAL FAIL $tag"; return 1; }; log "$tag: $r"; }
# pair A B GOLDEN [FA FB]: pcmp line (once); "w64:TAG" names default to W64/TAG.f32
pair() { local a=$1 b=$2 g=$3 fa=${4:-} fb=${5:-}
  [ -n "$fa" ] || { [[ $a == w64:* ]] && fa=$W64/${a#w64:}.f32 || fa=$O/$a.f32; }; [ -n "$fb" ] || fb=$O/$b.f32
  grep -qF "pair $a -> $b: \"ratio\"" $O/result.txt 2>/dev/null && return 0
  [ -f $fa ] && [ -f $fb ] || { log "PAIR FAIL $a -> $b (missing score)"; return 1; }
  log "pair $a -> $b: $(python3 $P/pcmp.py $g $fa $fb | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"; }
evs() { stopped && return 1; lock "eval $1"; ev $1 $2 $GS && ev vf-$1 $2 $GV; local rc=$?; unlock; return $rc; }

# ---- #112-01 ---------------------------------------------------------------------------------------------------------
oth() { [ $1 = u3 ] && echo "attn_o=0,down=0" || echo "attn_o=-1,down=0"; }
dnof() { [ ${1:0:1} = u ] && echo $W64/dn-k${1:1:1}$2; }       # dnof STK X -> the asym stack's dn dir (empty for pl)
# c9 TAG BASEMIX ALPHA DN [CMAX]: c9w75.py -> O/mix-TAG (CPU, no lock); the c9w.jsonl kept in O/stats/TAG.jsonl
c9() { [ -f $O/mix-$1/c9w.jsonl ] && return 0; rm -rf $O/mix-$1.part
  cpu python3 $C9W $2 $O/mix-$1 --alpha $3 ${4:+--dn $4} ${5:+--cmax $5} --device cpu >> $O/c9w.log 2>&1 \
    || { log "C9W FAIL $1"; return 1; }; cp $O/mix-$1/c9w.jsonl $O/stats/$1.jsonl; }
# fitarm STK G [CMAX]: seed-0 fit arm, paired with w64's fit base and (G != 3) with the g3 arm
fitarm() { local sk=$1 g=$2 cm=${3:-} t=fit-$1-g$2${3:+-cm$3}
  if [ ! -f $O/$t.f32 ]; then stopped && return 1
    c9 $t $W64/mix-$sk "$(oth $sk),gdn_out=$g" "$(dnof $sk "")" $cm || return 1
    lock "screen $t"; ev $t $O/mix-$t $F; local rc=$?; unlock; [ $rc = 0 ] || return 1; fi
  pair w64:fit-$sk $t $F || return 1
  if [ -n "$cm" ]; then pair fit-$sk-g$g $t $F || return 1; elif [ $g != 3 ]; then pair fit-$sk-g3 $t $F || return 1; fi
  rm -rf $O/mix-$t; }
i1screen() { local sk g
  for sk in u3 u2 p3 p2; do for g in 3 4 5 6 8; do fitarm $sk $g || return 1; done; done
  log "#112-01 screen done"; }
i1extra() { [ -f $O/alpha75.json ] || { python3 $R75 pick $O > $O/alpha75.json.part && mv $O/alpha75.json.part $O/alpha75.json \
    || { log "PICK FAIL"; return 1; }; log "alpha*_new: $(cat $O/alpha75.json)"; }
  local sk a
  for sk in u3 u2 p3 p2; do a=$(python3 -c "import json; print(f\"{json.load(open('$O/alpha75.json'))['$sk']:g}\")")
    fitarm $sk $a || return 1                                    # a no-op for grid values (scored by the screen)
    if python3 $R75 needclamp $O fit-$sk-g$a; then fitarm $sk $a 2 || return 1; log "clamp check $sk g$a: done"; fi; done
  log "#112-01 extras done"; }
i1confirm() { local sk a s x t
  for sk in u3 u2 p3 p2; do a=$(python3 -c "import json; print(f\"{json.load(open('$O/alpha75.json'))['$sk']:g}\")")
    [ "$a" = 3 ] && continue
    for s in 0 1 2; do x=$(sfx $s); t=$sk${x}g$a
      [ -f $O/vf-$t.f32 ] || { c9 $t $W64/mix-$sk$x "$(oth $sk),gdn_out=$a" "$(dnof $sk "$x")" || return 1; evs $t $O/mix-$t || return 1; }
      pair w64:$sk${x}c9 $t $GS && pair w64:vf-$sk${x}c9 vf-$t $GV && pair w64:$sk$x $t $GS && pair w64:vf-$sk$x vf-$t $GV || return 1
    done; done
  log "#112-01 confirm done"; }
a27() { [ -f $O/alpha27.txt ] && return 0
  local v; v=$(python3 $R75 a27 $O) || { log "A27 FAIL"; return 1; }
  echo "$v" > $O/alpha27.txt.part && mv $O/alpha27.txt.part $O/alpha27.txt; log "#112-02 arm B gdn_out alpha_27: $v"; }

# ---- #106-01 ---------------------------------------------------------------------------------------------------------
PC=$O/m08perm PW=$O/p08perm OK=$O/okeys.txt
# oq DIR CKPT WORK K S: q2b.sh 08 K i8 DIR --hout C18 --sigma-reg-out 0.01 --seed-offset S --only the 6 o_proj, with
# the checkpoint and Hessian dir given (q2b.sh's exact command line otherwise; call under lock)
oq() { local d=$1 ck=$2 wk=$3 k=$4 s=$5
  [ -f $wk/template.json ] || python3 -c "import json; c=json.load(open('$ck/config.json')); c['quantization_config']={'quant_method': 'exl3'}; json.dump(c, open('$wk/template.json', 'w'))"
  mkdir -p /tmp/q2b-run-w75 && ( cd /tmp/q2b-run-w75 && PYTHONPATH=/data/src/exllamav3 RXM_WORK=$wk RXM_CKPT=$ck RXM_TEMPLATE=$wk/template.json \
    TORCH_EXTENSIONS_DIR=/data/torch_ext PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 /workspace/rad-exl3-mix/quantize/ourq.py -K $k -o $d --device 0 --layers 0-23 --codebook i8 --qtf2 \
    --frac-ext $SRC --scaled-hessian --sigma-reg 0.025 --sigma-reg-in 0.005 --save-wq \
    --hout $C18 --sigma-reg-out 0.01 --seed-offset $s --only $OK ) >> $d.log 2>&1; }
# mixo TAG BASEMIX DEC: BASEMIX's 150 weights with DEC's 6 o_proj in place -> O/mix-TAG
mixo() { local d=$O/mix-$1; [ $(nq $d) = 150 ] && [ -f $d/.ok ] && return 0; rm -rf $d; mkdir -p $d
  for f in $2/*.safetensors; do ln -s $(readlink -f $f) $d/; done
  for f in $3/*.safetensors; do ln -sfn $(readlink -f $f) $d/; done
  local nw=0; for f in $3/*.safetensors; do [ "$(readlink -f $d/$(basename $f))" = "$(readlink -f $f)" ] && nw=$((nw + 1)); done
  [ $(nq $d) = 150 ] && [ $nw = 6 ] || { log "MIX FAIL $1 ($(nq $d) files, $nw o_proj)"; return 1; }; touch $d/.ok; }
dec6() { [ $(nq $2) = 6 ] && return 0; rm -rf $2.part
  cpu python3 $DEQ wq $1 $2.part $OK > $2.log 2>&1 && [ $(nq $2.part) = 6 ] && mv $2.part $2 || { log "DEQ FAIL $2"; return 1; }; }
p06cell() { local k=$1 s=$2 x=$(sfx $2); local dn=$O/q-on$k$x dp=$O/q-op$k$x
  stopped && return 1
  if [ ! -f $O/noop-$k$x.pass ] && [ ! -f $O/noop-$k$x.fail ]; then
    [ $(ls $dn/q-L*.safetensors 2>/dev/null | wc -l) = 6 ] || { lock "noop k$k$x"; oq $dn $M /data/p08 $k $s; unlock; }
    [ $(ls $dn/q-L*.safetensors 2>/dev/null | wc -l) = 6 ] || { log "QUANT FAIL noop k$k$x"; return 1; }
    local r; r=$(cpu python3 $ASM same $D12/q-k${k}ste$x $dn 2>&1 | tail -n 1); local rc=$?
    [[ "$r" == *PASS* ]] && touch $O/noop-$k$x.pass || { touch $O/noop-$k$x.fail; r="${r//FAIL/differs: control = the no-op requant}"; }
    log "noop gate k$k$x: $r"; fi
  [ $(ls $dp/q-L*.safetensors 2>/dev/null | wc -l) = 6 ] || { stopped && return 1; lock "perm k$k$x"; oq $dp $PC $PW $k $s; unlock; }
  [ $(ls $dp/q-L*.safetensors 2>/dev/null | wc -l) = 6 ] || { log "QUANT FAIL perm k$k$x"; return 1; }
  dec6 $dp $O/dec-op$k$x || return 1
  [ $(nq $O/dec-opn$k$x) = 6 ] || cpu python3 $PERM unperm $PC/perm06.json $O/dec-op$k$x $O/dec-opn$k$x >> $O/perm06.log 2>&1 \
    || { log "UNPERM FAIL k$k$x"; return 1; }
  mixo perm$k$x $W64/mix-p$k$x $O/dec-opn$k$x || return 1
  evs perm$k$x $O/mix-perm$k$x || return 1
  if [ -f $O/noop-$k$x.fail ]; then dec6 $dn $O/dec-on$k$x && mixo pn$k$x $W64/mix-p$k$x $O/dec-on$k$x && evs pn$k$x $O/mix-pn$k$x || return 1; fi
  true; }
p06() {
  if [ ! -f $PC/perm06.json ]; then mkdir -p $PW
    cpu python3 $PERM make $M $H $PC $PW > $O/perm06.log 2>&1 || { log "PERM06 MAKE FAIL"; return 1; }
    log "perm06: $(grep -c '^L' $O/perm06.log) layers; $(tail -n 1 $O/perm06.log)"; fi
  printf 'model.language_model.layers.%d.self_attn.o_proj\n' 3 7 11 15 19 23 > $OK
  if [ ! -f $O/.bf16gate ]; then [ -f $O/.bf16fail ] && return 1; stopped && return 1
    lock "bf16 gate"; python3 $BCHK $M $PC $F > $O/bf16chk.log 2>&1; local rc=$?; unlock
    log "$(grep 'BF16 GATE' $O/bf16chk.log || echo "BF16 GATE FAIL (crash rc $rc: $O/bf16chk.log)")"
    [ $rc = 0 ] && touch $O/.bf16gate || { touch $O/.bf16fail; return 1; }; fi
  local k s x
  for k in 3 2; do for s in 0 1 2; do x=$(sfx $s); p06cell $k $s || return 1
    local c=w64:p$k$x vc=w64:vf-p$k$x; [ -f $O/noop-$k$x.fail ] && { c=pn$k$x; vc=vf-pn$k$x; }
    pair $c perm$k$x $GS && pair $vc vf-perm$k$x $GV || return 1; done; done
  if [ ! -f $O/plumb-perm3.f32 ]; then [ $(nq $O/mix-perm3st) = 150 ] || cpu python3 $PERM permwq $PC/perm06.json $O/mix-perm3 $O/mix-perm3st >> $O/perm06.log 2>&1
    stopped && return 1; lock "plumbing"; ev plumb-perm3 $O/mix-perm3st $GS $PC; local rc=$?; unlock; [ $rc = 0 ] || return 1
    log "plumbing check (perm3 on the permuted checkpoint, stored order): $(python3 $P/pcmp.py $GS $O/perm3.f32 $O/plumb-perm3.f32 | grep -o '"ratio": [0-9.]*')"; fi
  [ -f $O/rel06.jsonl ] || cpu python3 $R75 rel $O >> $O/perm06.log 2>&1 || { log "REL06 FAIL"; return 1; }
  log "#106-01 done"; }
report() { python3 $R75 report $O > $O/report.md.part 2>> $O/report.err && mv $O/report.md.part $O/report.md; }

if want i1s; then i1screen || log "#112-01 screen incomplete"; report; fi
if want i1x; then i1extra || log "#112-01 extras incomplete"; report; fi
if want i1c; then if i1confirm; then a27; else log "#112-01 confirm incomplete"; fi; report; fi
if want p06; then p06 || log "#106-01 incomplete"; fi
want report && report
stopped && { log "stopped (STOP)"; exit 0; }
python3 $R75 done $O > $O/done.txt; if [ $? = 0 ]; then log "G75 DONE"; else log "G75 INCOMPLETE ($(cat $O/done.txt); rerun to retry)"; fi
