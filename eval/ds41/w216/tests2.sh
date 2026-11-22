#!/bin/bash
# w216 follow-up on box2: exl3v3 tiles and engine_e2e_test on card3 (under its lock), the exl3i plugin's symbol gate.
W=/data/q27/w216; SRC=$W/src; B=$SRC/build; L=$W/logs
n=$(strings -a $W/build-exl3i/radiance_home/kernels/exl3i.so | grep -iE 'ggml|exl3r4d|exl3ref|radiance-exl3|weight_sidecar' | grep -vc ggml_rms_norm); echo "gate exl3i.so: $n"
exec {fd}> /data/q27/locks/card3.lock; flock -w 1200 $fd || { echo "card3 busy"; exit 0; }
export HIP_VISIBLE_DEVICES=3
timeout 900 $W/build-exl3v3/exl3v3_tool tiles --tiles 4096 --host 512 > $L/t-exl3v3-tiles.log 2>&1; echo "exl3v3 tiles RC=$? $(tail -n 1 $L/t-exl3v3-tiles.log | cut -c1-220)"
ctest --test-dir $B -R '^engine_e2e_test$' --timeout 1800 --output-on-failure > $L/t-e2e.log 2>&1; echo "engine_e2e_test RC=$? $(grep -E 'tests passed|tests failed' $L/t-e2e.log)"
echo T2_DONE
