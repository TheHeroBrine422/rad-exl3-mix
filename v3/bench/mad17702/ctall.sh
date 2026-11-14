#!/bin/bash
# ctall.sh CARD BUILD -- every device test of BUILD's exl3i on box2 card CARD (gt.sh each, under the card lock):
# exl3i_gpu_test, test_twopass (FUSE default, on, off), test_ks (mul1i8, mul1, mad), test_r16, test_aq (+ libr4d's rows),
# test_a4; summary to res/BUILD-ctall.txt. (w150, #177-02)
C=$1; B=$2; W=/data/q27/w150; T=$W/tools/gt.sh; R4=/data/q27/rad10-perf/install/share/radiance/kernels/libr4d.so
o=$W/res/$B-ctall.txt; : > $o
r() { echo "$* -> $("$@" | tail -n 1)" >> $o; }
r $T $C $B exl3i_gpu_test
TAG=def r $T $C $B test_twopass
TAG=fuse1 EXL3I_A8_FUSE=1 r $T $C $B test_twopass
TAG=fuse0 EXL3I_A8_FUSE=0 r $T $C $B test_twopass
TAG=cb r $T $C $B test_ks --codebooks mul1i8,mul1,mad
r $T $C $B test_r16
r $T $C $B test_aq $R4
r $T $C $B test_a4
echo done >> $o
