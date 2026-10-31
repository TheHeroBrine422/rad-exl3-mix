#!/bin/bash
# pr362/run.sh -- effect of radiance PR #362 (prefill softmax rescaled per row, not per wave) on our KLD measurements.
# Builds libr4d.so (the r4d kernels live there, not in kernels.rkb) with the PR's change into its own home (the frozen install and dev-home are never touched; the
# build tree's file is restored and libr4d.so rebuilt back to the original md5), then scores, on golden-ver1:
#   6.00 unpacked (SEQS=1)        fixed vs control (control = y2-base, already scored with the frozen home)
#   3.00 packed (SEQS=8, 4096-token steps, prefill span 2) and unpacked, each fixed and control
# Paired differences (mix/paired.py) in RESULTS. Card 1.
: ${RXM_WORK:=/root}
W=${RXM_WORK}/pr362 SRC=${RXM_WORK}/radiance-hero F=libr4d/r4d_attn_prefill_h256_gqa6.hip
H0=${RXM_WORK}/install-card/share/radiance H1=$W/home-362 C=1
ORIG=$(md5sum < $H0/kernels/libr4d.so | cut -d" " -f1)
mkdir -p $W; cd $W; R=$W/RESULTS
log() { echo "$(date -u +%H:%M) $*" | tee -a $R; }

if [ ! -f $H1/kernels/libr4d.so ]; then
  cp $SRC/$F $W/orig.hip
  python3 -I - $SRC/$F <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
old = """            if (wave_any(smax > m_ref + PGROW)) {
                const float rmax  = fmaxf(smax, swap16(smax));
                const float mnew  = fmaxf(fmaxf(m_ref, rmax), -3.0e38f);
"""
new = """            // PR #362: the branch stays wave-uniform, but each query row moves its scale only on its own growth
            const float rmax = fmaxf(smax, swap16(smax));
            const bool row_grow = live && rmax > m_ref + PGROW;
            if (wave_any(row_grow)) {
                const float mnew = row_grow ? fmaxf(fmaxf(m_ref, rmax), -3.0e38f) : m_ref;
"""
assert s.count(old) == 1; open(p, "w").write(s.replace(old, new))
EOF
  nice cmake --build $SRC/build-card -j6 > $W/build-fixed.log 2>&1; rc=$?
  if [ $rc = 0 ]; then mkdir -p $H1 && cp -a $H0/. $H1/ && cp $SRC/build-card/radiance_home/kernels/libr4d.so $H1/kernels/libr4d.so; fi
  cp $W/orig.hip $SRC/$F; touch $SRC/$F
  nice cmake --build $SRC/build-card -j6 > $W/build-restore.log 2>&1
  m=$(md5sum < $SRC/build-card/radiance_home/kernels/libr4d.so | cut -d" " -f1)
  f=$(md5sum < $H1/kernels/libr4d.so 2>/dev/null | cut -d" " -f1)
  log "build rc=$rc; fixed libr4d.so $f; build tree restored: $m ($( [ $m = $ORIG ] && echo same as frozen || echo DIFFERS))"
  [ $rc = 0 ] && [ -n "$f" ] && [ "$f" != $ORIG ] || { log "fixed libr4d.so missing or identical to frozen: abort"; rm -rf $H1; exit 1; }
fi

ev() {  # TAG MODEL HOME [packed]
  local t=$1 m=$2 h=$3 pk=$4
  [ -f $W/$t/score.f32 ] && return
  ( exec 9>${RXM_WORK}/card$C.lock 7>${RXM_WORK}/ram.lock; flock 9; flock -s 7
    cd ${RXM_WORK}
    if [ -n "$pk" ]; then
      w=$(SEQS=8 CARD=$C RHOME=$h BIN=${RXM_WORK}/radiance-kldgold-fast ./gold2.sh $m golden-ver1 corpus/ver1.jsonl $W/$t bf16 16384 --max-num-batched-tokens 4096 --prefill-span 2)
    else
      w=$(SEQS=1 CARD=$C RHOME=$h BIN=${RXM_WORK}/radiance-kldgold-fast ./gold2.sh $m golden-ver1 corpus/ver1.jsonl $W/$t bf16 16384)
    fi
    log "$t $w kld $(python3 -c "import json; print(json.load(open('$W/$t.json'))['kld']['mean'])" 2>/dev/null)" )
}
M6=/data/release/qwen3.8-27b-exl3-v2-6.00bpw.rad M3=/data/release/qwen3.8-27b-exl3-v2-3.00bpw.rad
ev fix-6.00 $M6 $H1
ev ctl-3.00-pk $M3 $H0 pk; ev fix-3.00-pk $M3 $H1 pk
ev ctl-3.00 $M3 $H0;       ev fix-3.00 $M3 $H1
pr() { log "$1: $(python3 -I ${RXM_WORK}/hproxy/paired.py ${RXM_WORK}/golden-ver1 $2 $3)"; }
pr "6.00 fixed - control" ${RXM_WORK}/evq/scores/y2-base.f32 $W/fix-6.00/score.f32
pr "3.00 packed: fixed - control" $W/ctl-3.00-pk/score.f32 $W/fix-3.00-pk/score.f32
pr "3.00 unpacked: fixed - control" $W/ctl-3.00/score.f32 $W/fix-3.00/score.f32
pr "3.00 control: packed - unpacked" $W/ctl-3.00/score.f32 $W/ctl-3.00-pk/score.f32
pr "3.00 fixed: packed - unpacked" $W/fix-3.00/score.f32 $W/fix-3.00-pk/score.f32
log "PR362 DONE"
