#!/bin/bash
# nfval.sh TAG MIX AQ [GOLDEN CORPUS] -- validate a knap3 solution: splice its EXL3 stand-in (radmix onto sc4.00-e4;
# the mix names every tensor and the embedding), then score it with the E1b build, activation fake-quant per its .aq
# (EXL3_AQ_MASK / EXL3_AQ_GROUPS). One line -> $RXM_WORK/e1b/results.jsonl (via e1brun.sh). Removes the spliced container.
: ${RXM_WORK:=/root} ${RXM_MODELS:=$RXM_WORK/models/final} ${RXM_MIX:=/data/mix}; export RXM_WORK RXM_MODELS RXM_MIX   # paths (README: Paths)
T=$1 MIX=$2 AQ=$3 G=${4:-golden-calA} CO=${5:-corpus/calA.jsonl}
cd ${RXM_WORK}; m=${RXM_MIX}/$T.rad
./hproxy/radmix $m ${RXM_MODELS}/q38-27b-sc4.00-e4.rad $MIX 2> ${RXM_WORK}/nf/val/$T.radmix.log || { echo "{\"tag\": \"$T\", \"fail\": \"radmix\"}" >> ${RXM_WORK}/e1b/results.jsonl; rm -f $m; exit 1; }
cd ${RXM_WORK}/e1b && ./e1brun.sh $T $m $G $CO $(grep -E '^EXL3_AQ_(MASK|GROUPS)=' $AQ)
rm -f $m
