#!/bin/bash
# mk.sh NAME REV [TCS] -- git archive of the w152 worktree's REV -> box2 /data/q27/w152/src-NAME, all five plugins built
# (plugb.sh) per toolchain in TCS (default "10 714") into b10NAME / b714NAME, the qwen35_exl3 host test's output in
# b<TC>NAME/test.log. Env SEDQ: a sed expression applied to plugin/qwen35_exl3/exl3_fuse.h (a negative control). (w152, #179)
N=$1 REV=$2 TCS=${3:-10 714}; WT=/workspace/wt/w152-radiance; B2=/workspace/rad-local/bin/ssh-box2.sh
P=/workspace/rad-local/workers/w117/box2-put.sh
git -C $WT archive --format=tar.gz $REV | $P /data/q27/w152/arc-$N.tgz
timeout 1800 $B2 "cd /data/q27/w152 && rm -rf src-$N && mkdir src-$N && tar xzf arc-$N.tgz -C src-$N && \
  { [ -z '${SEDQ:-}' ] || sed -i '${SEDQ:-}' src-$N/plugin/qwen35_exl3/exl3_fuse.h; } && \
  for tc in $TCS; do rm -rf b\${tc}$N; bash tools/plugb.sh \$tc /data/q27/w152/src-$N /data/q27/w152/b\${tc}$N < /dev/null; \
    tail -n 2 b\${tc}$N/plugb.log; b\${tc}$N/pb/qwen35_exl3/qwen35_exl3_test > b\${tc}$N/test.log 2>&1; \
    echo \"TC \$tc test rc=\$?\"; grep -E 'Assert|FAIL|ALL TESTS' b\${tc}$N/test.log | tail -3; done"
