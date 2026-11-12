cd /data/w110; PY=/data/venv-exl3/bin/python; O=/data/w110/export; mkdir -p $O
cp verdict.txt verdict.json result.txt $O/; cp mix/census-*.txt mix/hyg-*.txt $O/ 2>/dev/null; cp c9/*/e3rows.jsonl $O/ 2>/dev/null
for k in k05 k1 k15; do cp c9/$k-e3x/e3rows.jsonl $O/e3rows-$k.jsonl; cp c9/$k-cur/c9.jsonl $O/c9cur-$k.jsonl; done; rm -f $O/e3rows.jsonl
for t in k1-os k1-es k1-cn k1-on k1-en; do echo "## $t vs k1-cs"; $PY code/eval/docstat/docstat.py golden-search27d-x golden-search27d-c scores/k1-cs-x.json.rows scores/k1-cs-c.json.rows scores/$t-x.json.rows scores/$t-c.json.rows; done > $O/docstat-M1.txt 2>&1
for t in sp-oo-ss sp-ee-ss sp-cc-nn sp-oo-nn sp-ee-nn sp-oc-ss sp-co-ss sp-ec-ss sp-ce-ss sp-cc-ns sp-cc-sn; do echo "## $t vs sp-cc-ss"; $PY code/eval/docstat/docstat.py golden-search27d-x golden-search27d-c scores/sp-cc-ss-x.json.rows scores/sp-cc-ss-c.json.rows scores/$t-x.json.rows scores/$t-c.json.rows; done > $O/docstat-M2.txt 2>&1
$PY - <<'PY'
import json, sys
sys.path.insert(0, "code/v3/e3/settle/w84"); from pair84 import doc_sums
import glob, os
out = {}
for f in sorted(glob.glob("scores/*-[xc].json.rows")):
    tag, p = os.path.basename(f)[:-12].rsplit("-", 1)
    s, n, t, _ = doc_sums(f"golden-search27d-{p}", f)
    out.setdefault(tag, {})[p] = {"kl_sum": [round(float(x), 6) for x in s], "top1_sum": [int(x) for x in t]}
out["_positions"] = {p: [int(x) for x in doc_sums(f"golden-search27d-{p}", f"scores/k1-cs-{p}.json.rows")[1]] for p in "xc"}
json.dump(out, open("export/docsums.json", "w"))
print("docsums", len(out) - 1, "mixes")
PY
for k in 0.5 1 1.5; do d=lv/q27v3-159591b2-k$k-i8; echo "k$k: $(cat $d/DONE) $(grep -c . $d/logs/jobs.jsonl) job lines, $(du -sh $d/level | cut -f1) level"; done > $O/levels.txt
cp lv/run.log $O/rxrun-run.log; cp lv/swj-times.jsonl $O/; grep -a "stage .*927611\|fused check\|plan\|stages" lv/swj-*.log > $O/swj-pass.log
PYTHONPATH=code/pipeline nice -n 10 $PY -m rxrun.seedsd lv > $O/seedsd.txt 2>&1; tail -15 $O/seedsd.txt
ls $O | wc -l; du -sh $O
