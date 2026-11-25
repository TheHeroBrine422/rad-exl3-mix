#!/bin/bash
# chain229.sh (box2; setsid nohup, log /data/q27/xgap/chain.log) -- lead 11-25 (user: "test xgap"): the first xgap screen. Restartable:
# every step is skipped when its output exists.
#   1. downloads in the background (refs for the baseline): IQ8R-V4, the FP8 container (official weights), our EXL3-V3-7.00 (host-emb)
#   2. bf16 goldens, TP4, all four card locks held: xg-judge (gap judge, xgap holdout, docs <= 64k) -> QROOT/slices/golden-jgap;
#      xg-scr1 (screen training docs, non-holdout, <= 8.6k) -> QROOT/ext/gold/xg-scr1 (+ ext/genmask/xg-scr1.u8); gen masks by genmask181
#   3. cont files: cont-xg1.json (1,552,745 rows of xg-scr1, per-cell quotas), cont-xgc.json (half xg1, half cont-calib's calib rows)
#   4. jobs: baseline reads on the gap judge (JG-*: unhealed 2.50, 10M end, C228-ctl, C228-opmix), arms C229-xg / C229-xgc from
#      I25-step-10M's 10M state + their JC / JG / JS / JT / JL reads
#   5. TP2 reference reads (cards 2+3 locked): FP8, IQ8R-V4, EXL3-V3-7.00 on the gap judge (JG) and jc228 (JC)
set -u
Q=/data/q27; QROOT=$Q/queue; W=$Q/xgap; E=$QROOT/ext; L=$W/chain.log; S=$W/screen1; R=$W/refs
log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> $L; }
source $QROOT/code/v3/qadq/qenv.sh box2
mkdir -p $R; log "chain229 start (pid $$)"
dl() { [ -e $R/$3.ok ] && return; hf download $1 $2 --local-dir $R >> $W/dl.log 2>&1 && touch $R/$3.ok && log "downloaded $3"; }
( dl StillDeadcode/qwen3.8-27b-iqr-v4 qwen3.8-27b-iq8r-v4.rad iq8r
  dl StillDeadcode/qwen3.8-27b-fp8 qwen3.8-27b-fp8.rad fp8
  dl Hero422/qwen3.8-27b-exl3-v3-rad host-emb/qwen3.8-27b-exl3-v3-7.00bpw.rad e700 ) &
DLP=$!
for f in xg-judge xg-scr1; do [ -s $S/$f.jsonl ] && [ -s $S/$f.meta.jsonl ] || { log "missing $S/$f: stop"; exit 1; }; done

rec() {   # NAME CORPUS DEST MML SEQS
  local N=$1 C=$2 G=$3 M=$4 SQ=$5
  [ -e $G/.done ] && return 0
  local AV=$(df -BG --output=avail $QROOT | tail -1 | tr -dc 0-9); [ "$AV" -lt 130 ] && { log "only $AV GB free: not recording $N"; return 1; }
  rm -rf $G.part; mkdir -p $(dirname $G); local Y=$W/rec-$N.yaml
  cat > $Y <<YEOF
memory:
  host-headroom-mib: 8192
models:
  $N:
    model: $MODEL
    tp: 4
    kv-cache-dtype: bf16
    max-model-len: $M
    max-num-seqs: $SQ
    kld-record: $G.part
    kld-corpus: $C
    kld-top: 512
YEOF
  local s=$(date +%s)
  LD_LIBRARY_PATH="$RLIBS" RADIANCE_HOME=$RH ROCR_VISIBLE_DEVICES=0,1,2,3 $RB --config $Y > $W/rec-$N.log 2>&1; local rc=$?
  log "record $N rc $rc in $(( ($(date +%s) - s) / 60 )) min"
  [ $rc = 0 ] && [ -e $G.part/kld.json ] || return 1
  $PY $QROOT/code/v3/qad181/genmask181.py make $G.part $G.part/gen.u8 $MODEL >> $L 2>&1 || { log "genmask $N failed"; return 1; }
  cp ${C%.jsonl}.meta.jsonl $G.part/corpus.meta.jsonl; touch $G.part/.done; mv $G.part $G
  log "$N golden: $($PY -c "import json,numpy as np;m=json.load(open('$G/kld.json'));g=np.fromfile('$G/gen.u8',np.uint8);print(len(m['docs']),'docs',m['rows'],'rows',int(g.sum()),'gen')")"
}
if [ ! -e $QROOT/slices/golden-jgap/.done ] || [ ! -e $E/gold/xg-scr1/.done ]; then
  ( exec 5>$(CARDLOCK 0) 6>$(CARDLOCK 1) 7>$(CARDLOCK 2) 8>$(CARDLOCK 3) 9>$RAMLOCK; flock 5; flock 6; flock 7; flock 8; flock -s 9
    log "four card locks held: recording"
    rec xg-judge $S/xg-judge.jsonl $QROOT/slices/golden-jgap 66560 2
    rec xg-scr1 $S/xg-scr1.jsonl $E/gold/xg-scr1 9216 4 )
fi
[ -e $QROOT/slices/golden-jgap/.done ] && [ -e $E/gold/xg-scr1/.done ] || { log "goldens missing: stop"; exit 1; }
[ -e $E/genmask/xg-scr1.u8 ] || { cp $E/gold/xg-scr1/gen.u8 $E/genmask/xg-scr1.u8.part && mv $E/genmask/xg-scr1.u8.part $E/genmask/xg-scr1.u8; }

C=$QROOT/cont228
if [ ! -e $C/cont-xgc.json ]; then
$PY - <<PYEOF >> $L 2>&1 || { log "cont files failed: stop"; exit 1; }
import json, collections, sys
import numpy as np
sys.path.insert(0, "$QROOT/code/v3/qad181"); import data181 as D
T = 1552745; G = "$E/gold/xg-scr1"
g = D.Gold(G, genfile="$E/genmask/xg-scr1.u8"); meta = [json.loads(l) for l in open(G + "/corpus.meta.jsonl")]
W = {"long-docs": 330, "agent-sessions": 250, "multilingual-text": 330, "multilingual-chat": 220, "long-chat": 220, "code-langs": 220, "tools-other": 110}
quota = {c: T * w / sum(W.values()) for c, w in W.items()}
got = collections.Counter(); items = []
for i in np.random.default_rng(0).permutation(len(meta)):
    c = meta[i]["cell"]; r = g.nrows(int(i), 8192)
    if r < 16 or got[c] >= quota[c]: continue
    items.append(["xg-scr1", int(i), int(r)]); got[c] += r
print("chain229: xg1 rows by cell", dict(got), "total", sum(got.values()), "quota", {c: int(q) for c, q in quota.items()})
json.dump({"goldens": ["xg-scr1"], "seed": 0, "items": items}, open("$C/cont-xg1.json", "w"))
half, s, a = T // 2, 0, []
for x in items:
    if s >= half: break
    a.append(x); s += x[2]
cal = json.load(open("$C/cont-calib.json"))["items"]; b, t = [], 0
for x in cal:
    if t >= T - s: break
    b.append(x); t += x[2]
json.dump({"goldens": ["xg-scr1"], "seed": 0, "items": a + b}, open("$C/cont-xgc.json", "w"))
print("chain229: xgc rows xgap", s, "calib", t)
PYEOF
log "cont files written"
fi

if ! grep -q "^T C229-xg " $QROOT/jobs.txt; then
  A="--lr 3e-4 --lr-steps 3e-4:1.5e6,1e-4:1.11e7 --plan-ext {ext}/ext-plan.json --ext-root {ext} --compile-scales --compile-ew --a8-int8"
  F="--fork-from $QROOT/runs-ice/I25-step-10M/state.pt"
  ( exec 9>$QROOT/queue.lock; flock 9; cp $QROOT/jobs.txt $QROOT/jobs.txt.pre-chain229; cat >> $QROOT/jobs.txt <<JEOF
# lead 11-25 (chain229, user "test xgap"): gap judge baseline (JG = golden-jgap, xgap holdout) + screen arms from the 10M state (1.55M rows each)
U JG-UN2.50 2.50
R JG-I25-step-10M 2.50 $QROOT/runs-ice/I25-step-10M/ckpt-1e+07
R JG-C228-ctl 2.50 $QROOT/runs-ice/C228-ctl/ckpt-1e+07
R JG-C228-opmix 2.50 C228-opmix/ckpt-1e+07
T C229-xg 2.50 1.15e7 ckpt-1e+07 $F --cont $C/cont-xg1.json $A
T C229-xgc 2.50 1.15e7 ckpt-1e+07 $F --cont $C/cont-xgc.json $A
R JC-C229-xg 2.50 C229-xg/ckpt-1e+07
R JG-C229-xg 2.50 C229-xg/ckpt-1e+07
R JS-C229-xg 2.50 C229-xg/ckpt-1e+07
R JL-C229-xg 2.50 C229-xg/ckpt-1e+07
R JC-C229-xgc 2.50 C229-xgc/ckpt-1e+07
R JG-C229-xgc 2.50 C229-xgc/ckpt-1e+07
R JS-C229-xgc 2.50 C229-xgc/ckpt-1e+07
R JL-C229-xgc 2.50 C229-xgc/ckpt-1e+07
JEOF
  ); log "jobs queued"
fi

wait $DLP
ref() {   # NAME FILE JUDGE GOLDEN MML [ram]
  local N=$1 F=$2 J=$3 G=$4 M=$5 O=$QROOT/scores/$3-REF-$1
  [ -s $O.json ] && return 0
  [ -e $F ] || { log "ref $N: no file"; return 1; }
  local Y=$O.yaml EP=""; [ "${6:-}" = ram ] && EP="    embedding-placement: ram"
  cat > $Y <<YEOF
memory:
  host-headroom-mib: 8192
models:
  $J-REF-$N:
    model: $F
    tp: 2
    tp-wire: exact
    kv-cache-dtype: bf16
    max-model-len: $M
    max-num-seqs: 1
$EP
    kld-ref: $G
    kld-out: $O.json.part
YEOF
  LD_LIBRARY_PATH="$RLIBS" RADIANCE_HOME=$RH ROCR_VISIBLE_DEVICES=2,3 $RB --config $Y > $O.log 2>&1; local rc=$?
  [ $rc = 0 ] && [ -s $O.json.part ] && { mv $O.json.part.rows $O.json.rows; mv $O.json.part $O.json; }
  log "ref read $J-REF-$N rc $rc"
}
( exec 7>$(CARDLOCK 2) 8>$(CARDLOCK 3) 9>$RAMLOCK; flock 7; flock 8; flock -s 9
  for J in JG JC; do
    [ $J = JG ] && { G=$QROOT/slices/golden-jgap; M=67584; } || { G=$QROOT/slices/golden-jc228; M=16448; }
    ref FP8 $R/qwen3.8-27b-fp8.rad $J $G $M
    ref IQ8R $R/qwen3.8-27b-iq8r-v4.rad $J $G $M
    ref E700 $R/host-emb/qwen3.8-27b-exl3-v3-7.00bpw.rad $J $G $M ram
  done )
log "chain229 done"
