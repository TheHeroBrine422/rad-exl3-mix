#!/bin/bash
# v27.sh [STAGE...] -- #104 amendment 2 (v3/yaqa/SWJ27.md): swj's K2.5 / K4 gains re-scored on a fresh calib-verify
# slice (golden-verify27), gpubox card 0. Stages (default: all, in order; each idempotent, a rerun resumes):
#   golden  golden-verify27 cut from the published calib golden (calib_slice.py + golden_pick.py, as golden-search27;
#           no card): verify pool, docs in excl/kld.json left out, 250,000 tokens asked, cap 8192, seed 20261107.
#           The document list must equal the frozen golden-verify27.docs (else FAIL: nothing is scored).
#   mix     head-matched K4 controls: base4 / bn4's trunk (from their swap27 mixes, which swap27k.sh put into V2 3.00:
#           its V2=$3 is overwritten by the sourced swap27.sh) re-mixed into V2 4.00, like base4swj / bn4swj (radmix
#           clones, no pack) -> mix-base4v4.rad, mix-bn4v4.rad; census must show the 336 K4 trunk weights and the
#           same head / embedding as base4swj.
#   score   card 0 (card0.lock, then ram.lock shared, as swap27.sh's score): 10 arms on golden-verify27 (gating pairs
#           first), then base4v4 / bn4v4 on golden-search27 (corrects amendment 1's K4 pairs for the head; report only).
#   report  v27report.py -> result.txt (per arm mean / p99 KLD / top-1 on the scored rows, pairs, amendment 2 verdict).
# Everything under /data/verify27; log result.txt. Kill any time; a rerun resumes per arm.
# Dry run (no card): V27_DIR=<scratch copy> V27_INSTALL=<dir with a fake bin/radiance> RXM_WORK=<own lock dir>.
: ${RXM_WORK:=/root}; export RXM_WORK
D=${V27_DIR:-/data/verify27} T=/data/swap27/tools I=${V27_INSTALL:-/data/install-116} H=/data/swap27/home PY=/data/venv-exl3/bin/python
CAL=/data/challenge/golden/calib CORP=/data/challenge/corpus/calib.jsonl POOLS=/data/challenge/pools-v3.json
GV=$D/golden-verify27 GS=/data/swap27/golden-search27 V4=/data/release/qwen3.8-27b-exl3-v2-4.00bpw.rad
TOK=250000 CAP=8192 SEED=20261107 C=0
export PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1
cd $D || exit 1
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $D/result.txt; }

golden() {
  [ -f $GV/kld.json ] && return 0
  docs=$(EXCLUDE_GOLDEN=$D/excl POOLS=$POOLS python3 -I $T/calib_slice.py $CORP $TOK $CAP $SEED verify 2> $D/slice.log)
  [ "$docs" = "$(cat $D/golden-verify27.docs)" ] || { log "golden: slice differs from golden-verify27.docs ($D/slice.log) FAIL"; return 1; }
  rm -rf $GV.part
  ( exec 7>$RXM_WORK/ram.lock; flock -s 7
    nice -n 10 $T/memguard.sh 2500 $PY -I $T/golden_pick.py $CAL $CORP $GV.part $D/golden-verify27.jsonl $docs ) > $D/pick.log 2>&1 \
    && mv $GV.part $GV || { log "golden: golden_pick FAIL ($D/pick.log)"; return 1; }
  log "golden: $(cat $D/slice.log); $($PY -I -c "import json,numpy as np; m=json.load(open('$GV/kld.json')); print(len(m['docs']), 'docs,', m['rows'], 'rows,', int(np.fromfile('$GV/gen.u8', np.uint8).sum()), 'scored')")"
}

# NAME SRC: SRC's 336 trunk weights (from /data/swap27/mix-SRC.rad) into V2 4.00 -> mix-NAME.rad
remix() { a=$1 src=/data/swap27/mix-$2.rad M=$D/mix-$1.rad
  [ -f $M ] && [ -f $D/census-$a.ok ] && return 0
  [ -f $src ] || { log "mix $a: $src missing FAIL"; return 1; }
  sed "s|/data/swap27/pack-$2.rad\$|$src|" /data/swap27/$2.mix > $D/$a.mix
  [ $(grep -c " $src\$" $D/$a.mix) = 336 ] || { log "mix $a: $D/$a.mix has not 336 donor lines FAIL"; return 1; }
  $T/radmix $M.part $V4 $D/$a.mix > $D/mix-$a.log 2>&1 && mv $M.part $M || { log "mix $a: radmix FAIL ($D/mix-$a.log)"; return 1; }
  nice $PY -I $T/census.py $M --mix $D/$a.mix --base $V4 > $D/census-$a.txt 2>&1 || { log "mix $a: census FAIL ($D/census-$a.txt)"; return 1; }
  grep -qE "exl3i  *4(\.0+)?  *mul1i8 *f16  *336 " $D/census-$a.txt || { log "mix $a: not 336 exl3i K4 trunk weights FAIL"; return 1; }
  [ "$(sed -n 2,99p $D/census-$a.txt | grep -v exl3i)" = "$(sed -n 2,99p /data/swj27/census-base4swj.txt | grep -v exl3i)" ] \
    || { log "mix $a: head / embedding / plain weights differ from base4swj's FAIL"; return 1; }
  touch $D/census-$a.ok; log "mix $a: ready, $($PY -c "import os; print(f'{os.path.getsize(\"$M\") * 8 / 26895998464:.4f} bpw')")"
}

mixof() { for f in $D/mix-$1.rad /data/swj27/mix-$1.rad /data/swap27/mix-$1.rad; do [ -f $f ] && { echo $f; return; }; done; }

# ARM GOLDEN OUTDIR: as swap27.sh's score (same build, home, flags), against GOLDEN -> OUTDIR/ARM.json(.rows)
score() { arm=$1 G=$2 O=$3 M=$(mixof $1); mkdir -p $O
  [ -f $O/$arm.json ] && [ -f $O/$arm.json.rows ] && return 0
  [ -n "$M" ] || { log "score $arm: no mix-$arm.rad FAIL"; return 1; }
  ( exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; s=$(date +%s)
    ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$H $I/bin/radiance --model $M --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $O/$arm.json.part > $O/score-$arm.log 2>&1
    rc=$?; r=$(( $(stat -c %s $G/rows.f32 2>/dev/null) / 12 * 16 ))
    [ $rc = 0 ] && [ "$(stat -c %s $O/$arm.json.part.rows 2>/dev/null)" = "$r" ] && mv $O/$arm.json.part.rows $O/$arm.json.rows && mv $O/$arm.json.part $O/$arm.json
    echo "$(date -u +%H:%M:%S) score $arm on $(basename $G) rc=$rc $(( $(date +%s) - s ))s" ) | tee -a $D/result.txt
  [ -f $O/$arm.json ] || { log "score $arm on $(basename $G) FAIL ($O/score-$arm.log)"; return 1; }
}

ARMS_V="base4v4 base4swj bn4v4 bn4swj base4 bn4 base baseswj s1 s1swj"   # gating pairs first
st_score() { f=0
  for a in $ARMS_V; do score $a $GV $D/verify || f=1; done
  for a in base4v4 bn4v4; do score $a $GS $D/search || f=1; done
  return $f; }

st_report() { $PY -I $D/tools/v27report.py $D > $D/report.txt 2> $D/report.err && cat $D/report.txt >> $D/result.txt \
  || { log "report FAIL ($D/report.err)"; return 1; }; }

[ $# = 0 ] && set -- golden mix score report
for s in "$@"; do
  case $s in
    golden) golden || exit 1 ;;
    mix) remix base4v4 base4 && remix bn4v4 bn4 || exit 1 ;;
    score) st_score || exit 1 ;;
    report) st_report || exit 1 ;;
    *) echo "v27.sh [golden|mix|score|report]..."; exit 2 ;;
  esac
done
log "V27 $* DONE"
