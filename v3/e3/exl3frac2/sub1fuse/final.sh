#!/bin/bash
# final.sh W -- #142 final suite on one card (dad1: W=/data/w108 holding ref/ (HEAD's exl3frac2), new/ (with
# sub1fuse/qtf2.cuh), tools/ (fusecheck.py, speed.py, build.py, env.sh)). Holds /root/card0.lock for the GPU parts.
# Identity: new vs ref, i8 + mul1 (fusecheck); speed: ref vs new, i8 at batches 32-256, mul1 at 128 (speed.py).
set -e
W=${1:-/data/w108}; cd $W; . tools/env.sh; export QTF2_DEFS=
for t in ref new; do for v in i8 mul1; do python3 -I tools/build.py $W/$t $v 2>&1 | grep -E "error|built"; done; done
flock /root/card0.lock bash -c "
for t in ref new; do for v in i8 mul1; do timeout 900 python3 -I tools/fusecheck.py run $W/\$t \$SRC \$v out/final-\$t-\$v.pt > logs/final-\$t-\$v.log 2>&1; done; done
for t in ref new; do timeout 900 python3 -I tools/speed.py $W/\$t \$SRC i8 32,64,128,256 | grep tps; timeout 900 python3 -I tools/speed.py $W/\$t \$SRC mul1 128 | grep tps; done > out/final-speed.jsonl"
for v in i8 mul1; do python3 -I tools/fusecheck.py cmp out/final-ref-$v.pt out/final-new-$v.pt > out/final-cmp-$v.jsonl || true; tail -1 out/final-cmp-$v.jsonl; done
