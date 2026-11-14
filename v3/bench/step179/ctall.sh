#!/bin/bash
# ctall.sh CARD TC BUILD -- every exl3i device test of build b<TC><BUILD> (w150's ctall list): exl3i_gpu_test, test_twopass
# (FUSE default / on / off), test_ks (mul1i8, mul1, mad), test_r16, test_aq (+ libr4d's rows: the pre stages and the
# residual add), test_a4 -> res/gt/<TC><BUILD>-ctall.txt. (w152)
C=$1 TC=$2 B=$3; W=/data/q27/w152; T="bash $W/tools/gt.sh"
case $TC in 10) R4=/data/q27/rad10-perf/install/share/radiance/kernels/libr4d.so ;; 714) R4=/data/q27/rad/install/share/radiance/kernels/libr4d.so ;; esac
o=$W/res/gt/$TC$B-ctall.txt; mkdir -p $W/res/gt; : > $o
r() { echo "$* -> $($@ | tail -n 1)" >> $o; }
r $T $C $TC $B test_aq $R4
r $T $C $TC $B exl3i_gpu_test
TAG=def r $T $C $TC $B test_twopass
TAG=fuse1 EXL3I_A8_FUSE=1 r $T $C $TC $B test_twopass
TAG=fuse0 EXL3I_A8_FUSE=0 r $T $C $TC $B test_twopass
TAG=cb r $T $C $TC $B test_ks --codebooks mul1i8,mul1,mad
r $T $C $TC $B test_r16
r $T $C $TC $B test_a4
echo done >> $o
