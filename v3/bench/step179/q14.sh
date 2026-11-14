#!/bin/bash
# q14.sh -- card 1, after q13: the decode-path KLD's reference without the keyed counters' slot limit: EXL3I_MAD_SPLIT on
# b10mord (9f0a059 with every launch ordered: race-free, the per-stream counters, so every 8-row piece takes KS); with
# fdd480c the pieces' distinct y offsets use up the 64 counter slots and later pieces run unsplit (another f32 order). (w152)
W=/data/q27/w152; cd $W; Q=/data/q27
while pgrep -f "^bash tools/q13[.]sh" > /dev/null; do sleep 20; done
for K in 2.625 4.125; do k=${K/./}; MP=$Q/w150/packs/q27v3-144dd6e1-k$K-mad-a8.rad
  bash tools/kldenv.sh 1 k$k-gemv-ord=$W/b10mord/home=$MP=EXL3I_MAD_SPLIT=1,EXL3I_A8_MIN_ROWS=1000000 >> logs/kldr.log 2>&1
done
echo "$(date -u +%T) q14 DONE" >> logs/q.log
