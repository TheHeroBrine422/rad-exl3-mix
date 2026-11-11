#!/bin/bash
# gpubox.sh -- #142 real-job check on gpubox card 0: one K0.125 i8 layer (L10: 4 group jobs, today's settings) through
# `python -m rxq job` (as rad-quant-at-home / w92's layer92.sh run them), once with today's qtf2 (code_ref = HEAD's tree)
# and once with sub1fuse/qtf2.cuh (code_new), same specs, same inputs. Then: tensors identical? per-tensor / job seconds.
# Layout under A=/data/w108: exl3/ (fork b0052f7c), code_ref/, code_new/ (git archive of pipeline + v3/e3 kernels +
# settle/w92), build_{ref,new}/ (TORCH_EXTENSIONS_DIR), jobs/, out/{ref,new}/, logs/. Builds run without the card lock;
# the jobs hold /root/card0.lock then /root/ram.lock (shared), under swap27's memguard. Restartable: finished jobs
# (job.json present) are skipped.
set -u
A=${A:-/data/w108}; PY=/data/venv-exl3/bin/python; T=/data/swap27/tools; GUARD_MB=${GUARD_MB:-3000}
MODEL=/data/workspace/models/qwen38-27b-bf16 HESS=/root/challenge/hessians/calib HOUT=/data/swap27/hout-bn
SRC=$A/exl3/exllamav3/exllamav3_ext LAYER=${LAYER:-10}
export PYTORCH_ROCM_ARCH=gfx1201 ROCM_HOME=/opt/rocm/core-10.1 ROCM_PATH=/opt/rocm/core-10.1 MAX_JOBS=4 ROCR_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
mkdir -p $A/logs $A/out
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $A/logs/gpubox.log; }
envof() { echo "PYTHONPATH=$A/code_$1/pipeline:$A/exl3:/data/exllamav3 RXQ_RXM=$A/code_$1 RXQ_EXL3_SRC=$SRC TORCH_EXTENSIONS_DIR=$A/build_$1 RXQ_REFIT_CPU_FALLBACK=1"; }
for t in ref new; do
  log "build $t"
  env $(envof $t) nice -n 10 $PY -c "
import sys; sys.path[:0] = ['$A/code_$t/v3/e3/exl3frac', '$A/code_$t/v3/e3/exl3cb', '$A/code_$t/v3/e3/exl3frac2']
import exl3frac, exl3cb, exl3frac2
exl3frac.load('$SRC'); exl3cb.load('$SRC', 'i8'); exl3frac2.load('$SRC', 'i8'); print('built $t')" > $A/logs/build-$t.log 2>&1 || { log "build $t FAILED"; exit 1; }
done
if [ ! -f $A/jobs/jobs.jsonl ]; then
  rm -rf $A/jobs.part
  env $(envof ref) $PY $A/code_ref/v3/e3/settle/w92/mkjobs92.py 0.125-i8 0 $MODEL $HESS $HOUT $A/jobs.part --layers $LAYER-$LAYER \
    > $A/logs/mkjobs.log 2>&1 && mv $A/jobs.part $A/jobs || { log "mkjobs FAILED"; exit 1; }
fi
log "$(tail -1 $A/logs/mkjobs.log)"
run() {   # holds the card: every job of the layer, ref then new
  for t in ref new; do for s in $A/jobs/L*.json; do n=$(basename $s .json); o=$A/out/$t/$n
    [ -f $o/job.json ] && continue
    rm -rf $o; mkdir -p $o; st=$(date +%s%N)
    $T/memguard.sh $GUARD_MB nice -n 5 env $(envof $t) $PY -m rxq job $s -o $o --device 0 >> $o/rxq.log 2>&1 || { echo "JOBFAIL $t $n"; return 1; }
    echo "$t $n $(( ($(date +%s%N) - st) / 1000000 ))" >> $A/out/wall-ms.txt
  done; done
}
log "waiting for card0.lock"
( exec 5>/root/card0.lock; flock 5; exec 7>/root/ram.lock; flock -s 7; log "card held"; run ) || { log "a job failed"; exit 1; }
log "jobs done"
$PY - <<EOF | tee -a $A/logs/gpubox.log
import glob, json, os, torch
from safetensors.torch import load_file
A = "$A"; same = True; rows = []
for d in sorted(glob.glob(f"{A}/out/ref/L*")):
    n = os.path.basename(d); jr, jn = json.load(open(f"{d}/job.json")), json.load(open(f"{A}/out/new/{n}/job.json"))
    for f in sorted(glob.glob(f"{d}/t-*.safetensors")):
        a, b = load_file(f), load_file(f.replace("/out/ref/", "/out/new/"))
        eq = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a); same &= eq
        print(f"{n} {os.path.basename(f)[2:-12]}: identical {eq}")
    sr = {r["key"]: r["s"] for r in map(json.loads, open(f"{d}/log.jsonl"))}
    sn = {r["key"]: r["s"] for r in map(json.loads, open(f"{A}/out/new/{n}/log.jsonl"))}
    for k in sr: rows.append((n, k.split(".")[-2], sr[k], sn[k]))
    print(f"{n}: job s ref {jr['seconds']:.1f} new {jn['seconds']:.1f} (x{jr['seconds'] / jn['seconds']:.2f}); settings {jr['settings_stamp']} / {jn['settings_stamp']}; code {jr['code_stamp']} / {jn['code_stamp']}")
for r in rows: print(f"  {r[0]} {r[1]}: tensor s ref {r[2]:.1f} new {r[3]:.1f} x{r[2] / r[3]:.2f}")
tr, tn = sum(r[2] for r in rows), sum(r[3] for r in rows)
print(f"GPUBOX {'IDENTICAL' if same else 'DIFFERENT'}: tensor s ref {tr:.1f} new {tn:.1f} x{tr / tn:.2f}")
EOF
