#!/bin/bash
# w213: speculation on vs off (greedy), the short prompt and a long one; sequential server starts
cd /data/q27/w213
S=scripts/w213/smoke-port.sh
EXTRA_YAML="num-speculative-tokens: 0" $S smoke-nospec
L="Write a detailed 350-word explanation of how a lighthouse lens (Fresnel lens) works, then list five famous lighthouses with one sentence each."
PROMPT="$L" MAXTOK=600 $S smoke-spec-long
PROMPT="$L" MAXTOK=600 EXTRA_YAML="num-speculative-tokens: 0" $S smoke-nospec-long
echo SPECPAIR_DONE
