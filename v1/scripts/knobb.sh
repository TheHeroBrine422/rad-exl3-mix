#!/bin/bash
# knobb.sh "SED-EXPR"... -- scratch edits to exl3r4d.hip, build into dev-home, ntbench gate_up + down on card 1; restores the tree
H=/workspace/forks/radiance-hero/plugin/exl3r4d/exl3r4d.hip
cp $H /tmp/knob-orig.hip
for e in "$@"; do sed -i "$e" $H; done
/workspace/kldtool/bin/devsync.sh 2>&1 | tail -1
cp /tmp/knob-orig.hip $H
/workspace/kldtool/card-scripts/vgpr.sh | tail -2
ssh-gpu.sh 'cd /root; export ROCR_VISIBLE_DEVICES=1; ./ntbench /root/dev-home/kernels/exl3_r4d.so 2048 34816 5120 4 20 1; ./ntbench /root/dev-home/kernels/exl3_r4d.so 2048 5120 17408 4 20 0'
