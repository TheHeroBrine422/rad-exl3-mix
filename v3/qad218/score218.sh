#!/bin/bash
# score218.sh NAME RAD (iceman, #218) -- the served read: plain radiance --kld-ref of one container on card 0 (03:00.0) on w214's judge slice
# j214 (rebuilt on iceman with golden_pick.py from judge214.args: kld.json / gen.u8 / tokens.i32 md5 = box2's), A8 as released, host-emb
# (embedding-placement ram), bf16 KV; score214.sh's config. Build: radbuild218.sh. Output /data/w218/scores/NAME.json(.rows).
set -u
NAME=$1 RAD=$2; W=/data/w218; O=$W/scores; G=$W/slices/golden-j214; mkdir -p $O
[ -s $O/$NAME.json ] && { echo "$NAME done"; exit 0; }
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
RADIANCE_HOME=$W/rad/home ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 $W/rad/install/bin/radiance --config $Y > $O/$NAME.log 2>&1
rc=$?
if [ $rc -eq 0 ] && [ -s $O/$NAME.json.part ]; then mv $O/$NAME.json.part.rows $O/$NAME.json.rows; mv $O/$NAME.json.part $O/$NAME.json; fi
echo "$NAME rc=$rc"; exit $rc
