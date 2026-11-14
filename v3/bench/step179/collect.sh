#!/bin/bash
# collect.sh -- the #179 summaries on box2 -> /data/q27/w152/summary/ (small text: per-op stamp tables, ABBA clocks, e2e,
# KLD row checks, gdec checks, device tests), for rad-exl3-mix data/step179/. CPU only. (w152)
W=/data/q27/w152; S=$W/summary; T=$W/tools; cd $W/res; rm -rf $S; mkdir -p $S
st() { python3 -I $T/stsum.py --top 40 "$@"; }
( cd st
  st base-v3-k2.5-r16k2.5-a8-d0.csv step2-v3-k2.5-r16k2.5-a8-d0.csv iqr-iq2r-d0.csv > $S/stamp-k2.5-mul1i8-d0.txt
  st base-v3-k2.5-r16k2.5-a8-d8.csv step2-v3-k2.5-r16k2.5-a8-d8.csv iqr-iq2r-d8.csv > $S/stamp-k2.5-mul1i8-d8.txt
  st base-v3-k4.125-r16k4-a8-d0.csv step2-v3-k4.125-r16k4-a8-d0.csv > $S/stamp-k4.125-mul1i8-d0.txt
  st base-v3-k4.125-r16k4-a8-d8.csv step2-v3-k4.125-r16k4-a8-d8.csv > $S/stamp-k4.125-mul1i8-d8.txt
  P=q27v3-635bac02-k2.5-mad-a8
  st mfix-$P-d0.csv fstep-$P-d0.csv fres-$P-d0.csv fres2-$P-d0.csv iqr-iq2r-d0.csv > $S/stamp-k2.5-mad-d0.txt
  st mfix-$P-d8.csv fstep-$P-d8.csv fres2-$P-d8.csv iqr-iq2r-d8.csv > $S/stamp-k2.5-mad-d8.txt
  st mbase-$P-d0.csv mstep-$P-d0.csv > $S/stamp-k2.5-mad-d0-RACE-garbage.txt
  st ab-1-base-v3-k2.5-r16k2.5-a8.csv ab-2-step2-v3-k2.5-r16k2.5-a8.csv ab-3-step2-v3-k2.5-r16k2.5-a8.csv ab-4-base-v3-k2.5-r16k2.5-a8.csv > $S/abba-k2.5-mul1i8-stamps.txt
  for f in ab-*.clock.json; do cp $f $S/; done
  python3 -I $T/gemvpos.py base-v3-k2.5-r16k2.5-a8-d0.csv step2-v3-k2.5-r16k2.5-a8-d0.csv > $S/gemvpos-k2.5-base-step2.txt
)
for f in prof/*-p8.log; do python3 -I $T/dprof2.py --prefill --ops $f > $S/prefill-profile-$(basename $f .log).txt; done
python3 -I $T/e2sum.py e2 > $S/e2e.txt; cp e2/*.jsonl $S/ 2>/dev/null; mkdir -p $S/e2; mv $S/*.jsonl $S/e2/
( cd kld; for f in *.json; do echo "$f $(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["all"]["kld"]["mean"])' $f)"; done ) > $S/kld-means.txt
( cd kld; for pr in "v3-k2.5-r16k2.5-a8-base-def v3-k2.5-r16k2.5-a8-step2-def" "v3-k2.5-r16k2.5-a8-base-c64 v3-k2.5-r16k2.5-a8-step2-c64" \
    "v3-k4.125-r16k4-a8-base-def v3-k4.125-r16k4-a8-step2-def" "v3-k4.125-r16k4-a8-base-c64 v3-k4.125-r16k4-a8-step2-c64" \
    "v3-k4.125-r16k4-a8-base-714-def v3-k4.125-r16k4-a8-step2-714-def" \
    "q27v3-635bac02-k2.5-mad-a8-mbase-def q27v3-635bac02-k2.5-mad-a8-mstep-def" "q27v3-635bac02-k2.5-mad-a8-mbase-c64 q27v3-635bac02-k2.5-mad-a8-mstep-c64" \
    "q27v3-144dd6e1-k4.125-mad-a8-mbase-def q27v3-144dd6e1-k4.125-mad-a8-mstep-def" "q27v3-144dd6e1-k4.125-mad-a8-mbase-c64 q27v3-144dd6e1-k4.125-mad-a8-mstep-c64" \
    "q27v3-635bac02-k2.5-mad-a8-mbase-def q27v3-635bac02-k2.5-mad-a8-mstep3-def" "q27v3-635bac02-k2.5-mad-a8-mbase-714-def q27v3-635bac02-k2.5-mad-a8-mstep3-714-def" \
    "q27v3-635bac02-k2.5-mad-a8-mbase-def q27v3-635bac02-k2.5-mad-a8-fres2-def" "q27v3-635bac02-k2.5-mad-a8-mbase-c64 q27v3-635bac02-k2.5-mad-a8-fres2-c64" \
    "v3-k2.5-r16k2.5-a8-base-def v3-k2.5-r16k2.5-a8-fres2-def" "v3-k2.5-r16k2.5-a8-base-c64 v3-k2.5-r16k2.5-a8-fres2-c64"; do
    set -- $pr; [ -f $1.json ] && [ -f $2.json ] && echo "$1 vs $2: $(python3 -I $T/rowcmp.py $1.json $2.json)"; done ) > $S/kld-rows.txt
( cd gdec; for p in v3-k2.5-r16k2.5-a8 v3-k4.125-r16k4-a8; do for r in base2 step step2; do echo "$p base vs $r: $(python3 -I $T/gdec.py cmp $p-base.jsonl $p-$r.jsonl | tail -1)"; done; done
  for p in q27v3-635bac02-k2.5-mad-a8 q27v3-144dd6e1-k4.125-mad-a8; do for r in mfix fstep fres fres2; do [ -f $p-$r.jsonl ] && echo "$p mord vs $r: $(python3 -I $T/gdec.py cmp $p-mord.jsonl $p-$r.jsonl | tail -1)"; done
    for r in mbase mbase2 mstep; do echo "$p $r: $(grep -c 'no finite logit' $p-$r.log) non-finite-logit warnings, $(grep -c 'finish=length' $p-$r.log) of 8 requests finished"; done; done ) > $S/gdec.txt
cat gt/*ctall.txt > $S/device-tests.txt; grep -h 'q12' $W/logs/q.log >> $S/device-tests.txt
cp $W/logs/q.log $S/queue-log.txt; du -sh $S
