#!/bin/bash
# qscore.sh MACHINE NAME RAD CARD [GOLDEN] -- the judge of the global queue: plain radiance --kld-ref of one container on one card (w221's score221.sh with qenv.sh's paths; default golden j217).
# A8 as released, --embedding-placement ram, bf16 KV. Output QROOT/scores/NAME.json(.rows) (written .part, renamed). The card lock is held by qcard.sh (QCARD_LOCKED=1), else taken here.
set -u
MACHINE=$1 NAME=$2 RAD=$3 CARD=$4
HERE0=$(dirname "$(readlink -f "$0")"); source $HERE0/qenv.sh $MACHINE
G=${5:-$J217}; O=$QROOT/scores; mkdir -p $O; MML=16448
case $NAME in JL-*) G=${5:-$JLONG}; MML=262144;;              # lead 11-22: JL-* names read the long-context judge (qad-pool long docs, whole, up to 247k tokens; positions >= 8192 scored)
  JC-*) G=${5:-$QROOT/slices/golden-jc228};;
  JT-*) G=${5:-$QROOT/slices/golden-jteach};;                   # lead 11-24: held-out teacher-written text (59 onpol-b1 docs no arm trained on)
  JU-*) G=${5:-$QROOT/slices/golden-jteachc};;
  JS-*) G=${5:-$QROOT/slices/golden-jstud};;                    # lead 11-24: held-out STUDENT-written traces (59 cuts no arm used)
  JV-*) G=${5:-$QROOT/slices/golden-jstudc};;
  JG-*) G=${5:-$QROOT/slices/golden-jgap}; MML=67584;;
  JV31-*) G=${5:-$QROOT/slices/golden-v31ver}; MML=101376;;       # lead 11-26 (w232): V3.1 FINAL-READ set, 422 whole non-trace verify docs (max 100,000 tokens). Read ONCE per final decision, by the lead.
  JT31-*) G=${5:-$QROOT/slices/golden-v31tr}; MML=262144;;         # w232: the two whole verify traces (905 rich, 2675 express; 250k / 253k tokens). Final read only.
  JX31-*) G=${5:-$QROOT/slices/golden-v31x};;                      # w232: the xcorp part of the final read (golden-v31x: not built yet). Final read only.
  JX-*) G=${5:-$QROOT/slices/golden-jt2}; MML=67584;;           # lead 11-25: teacher set-2 judge (w230; chat at xhigh / medium / low / thinking-off)           # lead 11-25: the gap judge (xgap-v1 holdout, docs <= 64k)
  esac               # lead 11-24: held-out student-written calib cuts (20)              # lead 11-24: same for onpol-calib-b1 (64 docs)                 # lead 11-24: JC-* names read the continuation judge jc228 (642 unused calib docs, 737,597 rows at cap 8192; v3/qadq/contpool.py)
[ -s $O/$NAME.json ] && { echo "$NAME done"; exit 0; }
if [ -z "${QCARD_LOCKED:-}" ]; then eval "exec {fd}>$(CARDLOCK $CARD)"; flock $fd; fi
eval "exec {rf}>$RAMLOCK"; flock -s $rf
Y=$O/$NAME.yaml
cat > $Y <<YEOF
memory:
  host-headroom-mib: 8192
models:
  $NAME:
    model: $(readlink -f $RAD)
    tp: 1
    kv-cache-dtype: bf16
    max-model-len: $MML
    max-num-seqs: 1
    embedding-placement: ram
    kld-ref: $G
    kld-out: $O/$NAME.json.part
YEOF
LD_LIBRARY_PATH="$RLIBS" RADIANCE_HOME=$RH env $(GPUENV $CARD) $RB --config $Y > $O/$NAME.log 2>&1
rc=$?
if [ $rc -eq 0 ] && [ -s $O/$NAME.json.part ]; then mv $O/$NAME.json.part.rows $O/$NAME.json.rows; mv $O/$NAME.json.part $O/$NAME.json; fi
echo "$NAME rc=$rc"; exit $rc
