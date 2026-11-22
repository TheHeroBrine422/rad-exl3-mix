#!/bin/bash
# score220.sh NAME RAD CARD [GOLDEN] -- w220 judge: plain radiance --kld-ref of one container on one box2 card (score217.sh with W = w220; default golden j217). A8 as released,
# --embedding-placement ram. Output /data/q27/w220/scores/NAME.json(.rows) (.part, renamed). Card lock, then ram.lock shared.
set -u
NAME=$1 RAD=$2 CARD=$3 G=${4:-/data/q27/w217/slices/golden-j217}
W=/data/q27/w220; O=$W/scores; mkdir -p $O
[ -s $O/$NAME.json ] && { echo "$NAME done"; exit 0; }
RB=${RADIANCE_BIN:-/data/q27/w206/install/bin/radiance}; RH=${RADIANCE_HOME:-/data/q27/w206/install/share/radiance}
LIBS=${LIBS:-/opt/rocm-10.1/lib:/data/q27/ffmpeg-libs}
exec {fd}> /data/q27/locks/card$CARD.lock; flock $fd
exec {rf}> /data/q27/locks/ram.lock; flock -s $rf
Y=$O/$NAME.yaml
cat > $Y <<YEOF
memory:
  host-headroom-mib: 8192
models:
  $NAME:
    model: $(readlink -f $RAD)
    tp: 1
    kv-cache-dtype: bf16
    max-model-len: 16448
    max-num-seqs: 1
    embedding-placement: ram
    kld-ref: $G
    kld-out: $O/$NAME.json.part
YEOF
LD_LIBRARY_PATH="$LIBS" RADIANCE_HOME=$RH ROCR_VISIBLE_DEVICES=$CARD $RB --config $Y > $O/$NAME.log 2>&1
rc=$?
if [ $rc -eq 0 ] && [ -s $O/$NAME.json.part ]; then mv $O/$NAME.json.part.rows $O/$NAME.json.rows; mv $O/$NAME.json.part $O/$NAME.json; fi
echo "$NAME rc=$rc"; exit $rc
