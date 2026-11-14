#!/bin/bash
# kldalt157.sh TAG OUT ARM... -- (w157, lead note 11-14 3:20 AM CST: core rebuild) KLD of W/mix/<ARM>.rad on golden-s173e
# with OUT's own install + home (radbuild-perf.sh: /data/q27/rad10-v3b = qc-exl3-v3 3bc8ad2, /data/q27/rad10-emb157 = this
# branch) instead of rad10-perf + W/b10e1/home, card 1 (lock exclusive, ram.lock shared) -> W/res/kld-<ARM>-<TAG>.json.rows;
# the rows' sha256 beside the b10e1 build's (gpu157.sh kld) in the log: equal rows = the rebuilt core serves as before.
set -u
Q=/data/q27 W=/data/q27/w157 C=1 L=/data/q27/locks G=$W/in/golden-s173e R=$W/res
T=$1 O=$2; shift 2
say() { echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S CST') $*" >> $W/logs/gpu157.log; }
for a in "$@"; do
  o=$R/kld-$a-$T; [ -f $o.json.rows ] && continue
  ( flock 5; flock -s 4; bash $Q/tools/score.sh $O/install $O/home $W/mix/$a.rad $G $o $C > /dev/null ) 5>$L/card$C.lock 4>$L/ram.lock
  say "kldalt $a $T: $(grep -E 'I kld: [0-9]+ / ' $o.log | tail -n 1 | cut -c1-60) rows $(sha256sum < $o.json.rows 2>/dev/null | cut -c1-16) vs b10e1 $(sha256sum < $R/kld-$a.json.rows 2>/dev/null | cut -c1-16)"
done
