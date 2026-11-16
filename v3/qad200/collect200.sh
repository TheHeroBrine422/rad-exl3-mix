#!/bin/bash
# collect200.sh OUT.tgz ARM... (w187, #200, box2): the small raw files of each training arm (train / plan / fixed / mon / run
# logs, the checkpoint's mon.json, the served result, census, mix, eval log, score json, gzipped score rows) into one tarball
# for data/qad200/arms/ on the VM. Checkpoints and resume states stay on box2.
OUT=$1; shift
cd /data/q27/w187 && rm -rf collect && mkdir -p collect
for t in "$@"; do
  d=collect/arms/$t; mkdir -p $d/served $d/ckpt
  for f in train.jsonl plan.json fixed.jsonl mon.jsonl run.log; do [ -e runs/$t/$f ] && cp runs/$t/$f $d/; done
  for ck in runs/$t/ckpt-*/ soups/$t/; do [ -d "$ck" ] && for f in mon.json soup.json; do [ -e "$ck$f" ] && cp "$ck$f" $d/ckpt/$(basename $ck)-$f; done; done
  for f in result.json census.txt mix.txt eval.log score-x181g.json score-fe38.json; do [ -e served/$t/$f ] && cp served/$t/$f $d/served/; done
  for f in served/$t/score-*.json.rows; do [ -e "$f" ] && gzip -c "$f" > $d/served/$(basename $f).gz; done
done
[ -d analysis-a123 ] && mkdir -p collect/analysis-a123 && cp analysis-a123/* collect/analysis-a123/
[ -d analysis ] && mkdir -p collect/analysis && cp analysis/* collect/analysis/
[ -d analysis-final ] && mkdir -p collect/analysis-final && cp analysis-final/* collect/analysis-final/
tar czf $OUT -C collect .
echo "wrote $OUT"
