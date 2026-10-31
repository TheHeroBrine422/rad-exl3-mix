#!/bin/bash
# knob.sh TAG "SED-EXPR"... -- apply sed edits to a scratch copy of exl3r4d.hip, build into dev-home, report vgpr + profile;
# the working tree's exl3r4d.hip is restored afterwards
H=/workspace/forks/radiance-hero/plugin/exl3r4d/exl3r4d.hip
T=$1; shift
cp $H /tmp/knob-orig.hip
for e in "$@"; do sed -i "$e" $H; done
/workspace/kldtool/bin/devsync.sh 2>&1 | grep -E "rror|dev-home" | head -5
/workspace/kldtool/card-scripts/vgpr.sh
cp /tmp/knob-orig.hip $H
/workspace/kldtool/card-scripts/devprof.sh $T 2>&1 | tail -11
