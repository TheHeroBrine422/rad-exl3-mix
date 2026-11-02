#!/bin/bash
# testdec.sh [CARD] -- build and run test_decode.hip (exl3i fractional / sub-1 decode vs the oracle). Log /data/qtf2/testdec.log
C=${1:-1}; cd /root/ourq/exl3i/kern
exec 7>/root/ram.lock; flock -s 7
/opt/rocm/core-10.1/bin/hipcc -O3 --offload-arch=gfx1201 -ffp-contract=off -I/root/radiance-hero/plugin/exl3r4d -I/root/radiance-hero/abi test_decode.hip ../ref/exl3i_ref.c -o /tmp/test_decode > /data/qtf2/testdec.log 2>&1 || { echo BUILD FAIL >> /data/qtf2/testdec.log; exit 1; }
exec 9>/root/card$C.lock; flock 9
ROCR_VISIBLE_DEVICES=$C /tmp/test_decode >> /data/qtf2/testdec.log 2>&1
