#!/bin/bash
# run.sh -- w08-searchreplay (ranked F7 / F8), CPU only, no GPU anywhere.
#   F8: on the VM from data/v2 (seconds)              -> f8replay.out
#   F7: on gpubox, read-only (~1 min, niced; needs V2's /root/knap/mixes, /root/evq/results.jsonl, /root/auto/fit-s*.txt
#       and frontier-s*.txt, which exist only there)  -> f7replay.out, f7replay.json
X=$(cd "$(dirname "$0")" && pwd); R=$(cd $X/../../.. && pwd); S=$R/local/bin/ssh-gpubox.sh
[ -s $X/f8replay.out ] || python3 $X/f8replay.py $R/data/v2/results.jsonl $R/data/v2/fit-final.txt --sim 2000 > $X/f8replay.out
if [ ! -s $X/f7replay.json ]; then
  $S 'ls /root/knap/mixes | wc -l; ls /root/auto/frontier-s*.txt | wc -l' < /dev/null || { echo "gpubox unreachable: F7 not run"; exit 1; }
  cat $X/f7replay.py | $S 'nice -n 19 python3 - /root --out /tmp/w08-f7.json' > $X/f7replay.out && $S 'cat /tmp/w08-f7.json' < /dev/null > $X/f7replay.json
fi
tail -12 $X/f7replay.out
