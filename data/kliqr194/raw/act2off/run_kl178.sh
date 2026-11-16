#!/bin/bash
# run_kl178.sh -- w178 (#194-02): t183 reads of the three KL-IQR entries with two-pass activations OFF
# (RADIANCE_ACT_TWO=0), plus a same-day control read of Nano with the container's own setting (act_two=1). Then mean KL and
# the per-document bootstrap (kl_mean.py / kl_boot.py, w177's tools). Restartable: score178.sh skips finished reads.
W=/data/q27/w178; T=$W/tools; G=/data/q27/w164/slices/golden-t183; MD=/data/q27/w177/models; S=$T/score178.sh
L() { echo "$(date -u +%FT%TZ) $*" >> $W/logs/run_kl.log; }
L "run_kl178 start"
$S nano-t183-a2off $MD/nano/qwen3.8-27b-kl-nano.rad $G RADIANCE_ACT_TWO=0 &
$S tiny-t183-a2off $MD/tiny/qwen3.8-27b-kl-tiny.rad $G RADIANCE_ACT_TWO=0 &
$S small-t183-a2off $MD/small/qwen3.8-27b-kl-small.rad $G RADIANCE_ACT_TWO=0 &
$S nano-t183-ctl $MD/nano/qwen3.8-27b-kl-nano.rad $G &
wait
L "reads done"
mkdir -p $W/out
for f in $W/rows/*-t183-*.json.rows; do /data/q27/venv/bin/python -I $T/kl_mean.py $f $G; done > $W/out/kl-t183.jsonl
for f in $W/rows/*-t183-*.json.rows; do /data/q27/venv/bin/python -I $T/kl_boot.py $f $G; done > $W/out/boot-t183.jsonl
L "kl and boot written"
