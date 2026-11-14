#!/bin/bash
# q6.sh -- card 1, after q5: IQR-V4 iq2r (drafter off) stamped as V3 was (stamp.sh, 320-token decodes at ctx 0 / 8k,
# one pass in 8) for the op-by-op comparison without --profile-ops' syncs. (w152)
W=/data/q27/w152; cd $W; Q=/data/q27; M=$Q/w138/hf/iq2r/qwen3.8-27b-iq2r-v4.rad
while pgrep -f "^bash tools/q[45][.]sh" > /dev/null; do sleep 20; done
XARGS="--num-speculative-tokens 0" bash tools/stampx.sh 1 $W/b10base/home $M res/st/iqr-iq2r-d0
XARGS="--num-speculative-tokens 0" bash tools/stampx.sh 1 $W/b10base/home $M res/st/iqr-iq2r-d8 $Q/w137/prompts/ctx-8k.txt
echo "$(date -u +%T) q6 DONE" >> logs/q.log
