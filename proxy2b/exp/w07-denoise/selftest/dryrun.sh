#!/bin/bash
# dryrun.sh -- the three w07 run.sh scripts end to end on the VM with the GPU programs stubbed (q2b.sh, evalq.py,
# moments.py, mkdenoise.py, hgen.py, mkgenh.py: stubs/), the real summ.py, and every path moved under a scratch dir:
# checks the shell glue (order, idempotence on a second run, gates, file names, picks, verdicts), not the science.
set -u
R=$(cd $(dirname $0)/../../../.. && pwd); T=${1:-/tmp/w07dry}; rm -rf $T; mkdir -p $T/repo/proxy2b/exp $T/repo/quantize $T/repo/eval/pools
for d in w07-denoise w07-seqsib w07-genh; do cp -r $R/proxy2b/exp/$d $T/repo/proxy2b/exp/; done
ST=$R/proxy2b/exp/w07-denoise/selftest/stubs; export STUB=$ST/common_stub.py
cp $ST/q2b.sh $ST/evalq.py $T/repo/proxy2b/; for f in moments mkdenoise; do cp $ST/$f.py $T/repo/proxy2b/exp/w07-denoise/; done
for f in hgen mkgenh; do cp $ST/$f.py $T/repo/proxy2b/exp/w07-genh/; done
echo 'ap.add_argument("--wpre")' > $T/repo/quantize/ourq.py; echo '{"pool": []}' > $T/repo/eval/pools/pools-v3.json
python3 $ST/mkdata.py $T                                   # mock /data/yb base quants + scores, goldens, calib H
for d in w07-denoise w07-seqsib w07-genh; do
  sed -e "s#/workspace/rad-exl3-mix#$T/repo#g" -e "s#/data/yb#$T/yb#g" -e "s#/data/g08-#$T/g08-#g" -e "s#/data/h08/calib#$T/h08#g" \
      -e "s#/data/models/qwen35-08b#$T/m08#g" -e "s#/data/challenge/calib-meta#$T/meta#g" -e "s#cd /tmp\$#cd $T#" \
      $T/repo/proxy2b/exp/$d/run.sh > $T/repo/proxy2b/exp/$d/run-dry.sh
  for pass in 1 2; do
    bash $T/repo/proxy2b/exp/$d/run-dry.sh > $T/$d-$pass.log 2>&1; rc=$?
    echo "== $d pass $pass: exit $rc, $(grep -c . $T/$d-$pass.log) log lines, last: $(tail -1 $T/$d-$pass.log)"
  done
  grep -E "gate|pick|=>|DONE" $T/yb/$d/result.txt | sed 's/^/   /' | cut -c1-230
done
