#!/bin/bash
# mk157.sh NAME REV [TCS] -- (w157, #174-03; w152's mk.sh) git archive of the w157 worktree's REV -> box2
# /data/q27/w157/src-NAME, all five plugins built (plugb.sh, OVL=1: the tree's rad_block_dflash2.h, which this branch
# changes) per toolchain in TCS (default "10 714") into b10NAME / b714NAME; the qwen35_exl3 and exl3dense host tests'
# output in b<TC>NAME/test*.log. CPU only.
N=$1 REV=$2 TCS=${3:-10 714}; WT=/workspace/wt/w157-radiance; B2=/workspace/rad-local/bin/ssh-box2.sh
P=/workspace/rad-local/workers/w117/box2-put.sh
git -C $WT archive --format=tar.gz $REV | $P /data/q27/w157/arc-$N.tgz
git -C $WT rev-parse $REV | $P /data/q27/w157/arc-$N.commit
timeout 3600 $B2 "cd /data/q27/w157 && rm -rf src-$N && mkdir src-$N && tar xzf arc-$N.tgz -C src-$N && cp arc-$N.commit src-$N/COMMIT && \
  for tc in $TCS; do rm -rf b\${tc}$N; OVL=1 J=12 bash tools/plugb.sh \$tc /data/q27/w157/src-$N /data/q27/w157/b\${tc}$N < /dev/null; \
    tail -n 2 b\${tc}$N/plugb.log; b\${tc}$N/pb/qwen35_exl3/qwen35_exl3_test > b\${tc}$N/test.log 2>&1; \
    echo \"TC \$tc qwen35_exl3_test rc=\$?\"; grep -E 'Assert|FAIL|ALL TESTS' b\${tc}$N/test.log | tail -3; \
    grep -E 'embed_exl3i|tests passed|Passed|Failed' b\${tc}$N/pb/exl3dense.log | tail -4; done"
