#!/bin/bash
# inputs-digest.sh MACHINE OUT -- digests of the queue's inputs, to diff between box2 and iceman (w222): data / student / judge files by md5 (sizes + md5 of small files for the 40 GB calib golden and the
# bf16 checkpoint), written to OUT. Idle priority.
MACHINE=$1 OUT=$2; HERE0=$(dirname "$(readlink -f "$0")")
case $MACHINE in box2) GOLD=/data/q27/w214/gold; IN=/data/q27/w214; J=/data/q27/w217/slices; CTL=/data/q27/w217/scores; CAL=/data/q27/challenge/golden/calib; MOD=/data/q27/models/qwen38-27b-bf16;;
  iceman) GOLD=/data/queue/gold; IN=/data/queue/w214; J=/data/queue/slices; CTL=/data/queue/ctl/scores; CAL=/data/challenge/golden/calib; MOD=/data/workspace/models/qwen38-27b-bf16;; esac
{
cd $IN; for f in pools214.json judge214.args; do md5sum $f; done; find genmask -type f | sort | xargs md5sum
cd $GOLD; for d in xc8k-b2-c0 xc8k-b2-c1 tr32k-b2-c0; do (cd $d; find . -type f | sort | xargs nice -n 19 ionice -c3 md5sum | sed "s#\./#$d/#"); done
cd $J; find . -type f | sort | xargs nice -n 19 md5sum
cd $CTL; md5sum ctl-*.json.rows ctl-*.json
echo "--- calib golden: sizes"; cd $CAL; find . -type f -printf '%s %P\n' | sort -k2 | md5sum; find . -type f -size -20M | sort | xargs md5sum | md5sum
echo "--- bf16 model"; cd $MOD; ls -l *.safetensors | awk '{print $5, $NF}' | md5sum; md5sum model.safetensors.index.json config.json
echo "--- lv"; cd $IN/lv; for s in 2.50 3.00 4.00; do echo "lv $s"; (cd $s; find . -type f | sort | xargs nice -n 19 ionice -c3 md5sum | md5sum; find . -type f | sort | xargs ls -l | awk '{print $5}' | md5sum); done
} > $OUT 2>&1
echo DIGEST DONE >> $OUT
