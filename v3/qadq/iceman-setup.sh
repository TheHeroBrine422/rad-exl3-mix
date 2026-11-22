#!/bin/bash
# iceman-setup.sh (iceman, /data/queue; run after iceman-pull.sh and dl.sh) -- w214's student pieces and the path-fixed chunk list, same inputs as box2's /data/q27/w214:
#   lv/<size>/ = rad2level214.py of the HF release's host-emb container (rel/host-emb/*.rad), --check 4; gold -> ../gold; rel -> ../rel; chunks214.iceman.txt = chunks214.txt with iceman's gold paths.
# Restartable (lv/<size>.ok). Log /data/queue/logs/setup.log.
set -u
Q=/data/queue; IN=$Q/w214; PY=/data/venv-exl3/bin/python; MODEL=/data/workspace/models/qwen38-27b-bf16
cd $Q; exec >> $Q/logs/setup.log 2>&1
ln -sfn ../gold $IN/gold; ln -sfn ../rel $IN/rel; mkdir -p $IN/lv
sed "s#/data/q27/w214/gold/#$Q/gold/#" $IN/chunks214.txt > $IN/chunks214.iceman.txt
export PYTHONPATH=$Q/code/pipeline PYTHONDONTWRITEBYTECODE=1
for S in 2.50 3.00 4.00; do
  [ -e $IN/lv/$S.ok ] && continue
  while [ ! -e $Q/rel/host-emb/qwen3.8-27b-exl3-v3-${S}bpw.rad.ok ]; do sleep 20; done
  echo "$(date -u +%T) rad2level $S"; rm -rf $IN/lv/$S.part
  nice -n 10 $PY $Q/code/v3/qad214/rad2level214.py $Q/rel/host-emb/qwen3.8-27b-exl3-v3-${S}bpw.rad $IN/lv/$S.part --model $MODEL --check 4 \
    && { rm -rf $IN/lv/$S; mv $IN/lv/$S.part $IN/lv/$S; touch $IN/lv/$S.ok; echo "$(date -u +%T) done $S"; } || echo "FAIL $S"
done
echo SETUP DONE
