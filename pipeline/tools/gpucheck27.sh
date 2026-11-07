#!/bin/bash
# gpucheck27.sh UNIT -- w76's GPU checks for the lead (gpubox, card 0 under /root/card0.lock + /root/ram.lock shared;
# CPU parts after the card is released). Not run by w76 (no GPU allowed). Code: a private tree $CODE staged from the VM
# (pipeline/README.md "GPU checks"); outputs $O (default /data/w76chk); results appended to $O/result.txt.
#   cd   tile CD on the real quantizer, 27B q_proj of layers 3 / 19 / 43:
#        (1) K2.5, no H_out: tests/tilecd_gate_gpu.py -- at every CD call rxq/tilecd.py == the research module
#            v3/e3/tilecd/tilecd.py (Wq, indices, rel), and cd(M = I) == cd(M = None) bit for bit       (~6 min)
#        (2) K4 with H_out (/data/swap27/hout-bn, sigma_out 0.01): rxq quant CD on vs off -> two-sided proxy_err ratio
#            and tilecd.jsonl ratios (report only; the KLD answer is tools/CDHOUT27.md's pair)              (~10 min)
#   i4   the qtf2 i4 path (w53's qtf2-i4.patch, applied 11-07):
#        (1) tests/qtf2i4check.py: qtf2 i4 == exl3cb i4 at every rate both have (indices and values), every 1/8 rate
#            and sub-1 on the int4 grid, pack / unpack exact                                                (~5 min + build)
#        (2) ourq.py (patched) vs rxq at K0.5 i4 + qtf2, sigma_in 0.05: layer 39 k_proj / v_proj (group members j > 0)
#            and o_proj -> tests/check_quant.py bit-identity                                                 (~3 min)
#        (3) ourq K2 i4 with --qtf2 vs without (exl3cb's integer kernel): identical (qtf2's MASK-0 route)    (~3 min)
#   c9   C9 inside the quant job (user 11-07: adopted, make it cheap): rxq quant K2.5 i8 of 4 writers (out_proj 10 / 11,
#        o_proj 3 / 39) with post.c9 on -> tests/check_c9job.py vs c9e27.py (C9E27, default /data/c9w27/code, read-only)
#        on the card (bit identity, same device) and on the CPU after the card (report max |dc| / c); prints the
#        per-writer in-job C9 time (log.jsonl rxq.c9_s)                                                     (~5 min)
#   all  cd, i4, then c9 (one hold of the card each)
# Stage (from the VM; a private tree, nothing live is touched):
#   cd /workspace/rad-exl3-mix; I=local/bin/ssh-gpubox.sh; D=$(mktemp -d); mkdir -p $D/v3/e3 $D/quantize $D/store
#   cp -r pipeline $D/ && cp -r v3/e3/exl3frac2 v3/e3/exl3frac v3/e3/exl3cb v3/e3/tilecd $D/v3/e3/ && cp quantize/ourq.py $D/quantize/
#   find $D -name __pycache__ -prune -exec rm -rf {} +
#   tar -C $D -cf - . | $I 'mkdir -p /data/w76chk/code && tar -C /data/w76chk/code -xf -' && rm -rf $D
# Run: $I 'cd /data/w76chk && nohup bash /data/w76chk/code/pipeline/tools/gpucheck27.sh all > run.out 2>&1 < /dev/null &'
# Results: /data/w76chk/result.txt (copy it and the *.txt / *.log into rad-exl3-mix data/w76-2026-11-07/gpu/).
set -u
U=${1:?cd|i4|c9|all}; CODE=${CODE:-/data/w76chk/code} O=${O:-/data/w76chk}; mkdir -p $O
: ${RXM_WORK:=/root}
PY=/data/venv-exl3/bin/python SRC=${SRC:-/data/exl3frac/src}
MODEL=${MODEL:-/data/workspace/models/qwen38-27b-bf16} HESS=${HESS:-/root/challenge/hessians/calib} HOUT=${HOUT:-/data/swap27/hout-bn}
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=/data/exl3frac/build PYTORCH_HIP_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 \
       PYTHONPATH=$CODE/pipeline RXQ_RXM=$CODE RXQ_EXL3_SRC=$SRC RXM_CKPT=$MODEL RXM_WORK=$RXM_WORK
P=model.language_model.layers
log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
card() { ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; t=$(date +%s); "$@"; rc=$?
           echo "$(date -u +%T) card held $(( $(date +%s) - t )) s (rc $rc)" >> $O/result.txt; exit $rc ); }
cd $O

cd_unit() {
  QK=$P.3.self_attn.q_proj,$P.19.self_attn.q_proj,$P.43.self_attn.q_proj
  card bash -c "
    nice -n 5 $PY $CODE/pipeline/tests/tilecd_gate_gpu.py --keys $QK -K 2.5 --ckpt $MODEL --hess $HESS -o $O/cd-gate > $O/cd-gate.log 2>&1
    for cdon in false true; do
      nice -n 5 $PY -m rxq quant --keys $QK -K 4 --codebook i8 --ckpt $MODEL --hess $HESS --hout $HOUT --no-prof \
        --set quant.tile_cd.enable=\$cdon -o $O/cd-hout-\$cdon > $O/cd-hout-\$cdon.log 2>&1
    done"
  log "cd (1): $(tail -n 1 $O/cd-gate.log)"
  log "cd (2): $($PY -c "
import json, math
p = lambda f: {j['key']: j['proxy_err'] for j in map(json.loads, open(f))}
a, b = p('$O/cd-hout-false/log.jsonl'), p('$O/cd-hout-true/log.jsonl'); x = {k: b[k] / a[k] for k in b if k in a}
t = [json.loads(l) for l in open('$O/cd-hout-true/tilecd.jsonl')]
print('two-sided proxy cd / off', {k.split('.')[3]: round(v, 4) for k, v in x.items()}, 'geo %.4f;' % math.exp(sum(map(math.log, x.values())) / len(x)),
      'tilecd ratios', [round(r['ratio'], 4) for r in t], 'hout', [r.get('hout') for r in t], 'CD s', [round(sum(r['sec'].values())) for r in t])" 2>&1)"; }

i4_unit() {
  printf '%s\n' $P.39.self_attn.k_proj $P.39.self_attn.v_proj $P.39.self_attn.o_proj > $O/i4-keys.txt
  IK=$(paste -sd, $O/i4-keys.txt)
  OQ=(nice -n 5 $PY -I $CODE/quantize/ourq.py --device 0 --frac-ext $SRC --scaled-hessian --sigma-reg 0.025 --codebook i4 --only $O/i4-keys.txt)
  card bash -c "
    nice -n 5 $PY -I $CODE/pipeline/tests/qtf2i4check.py $CODE/v3/e3/exl3frac2 $O/i4-dumps > $O/i4-kern.txt 2>&1
    ${OQ[*]} -K 0.5 -o $O/i4-ourq-K0.5 --qtf2 --sigma-reg-in 0.05 > $O/i4-ourq-K0.5.log 2>&1
    nice -n 5 $PY -m rxq quant --keys $IK -K 0.5 --codebook i4 --ckpt $MODEL --hess $HESS --no-prof --set quant.tile_cd.enable=false \
      -o $O/i4-rxq-K0.5 > $O/i4-rxq-K0.5.log 2>&1
    ${OQ[*]} -K 2 -o $O/i4-ourq-K2-qtf2 --qtf2 --sigma-reg-in 0.005 > $O/i4-ourq-K2-qtf2.log 2>&1
    ${OQ[*]} -K 2 -o $O/i4-ourq-K2-cb --sigma-reg-in 0.005 > $O/i4-ourq-K2-cb.log 2>&1"
  log "i4 (1): $(tail -n 1 $O/i4-kern.txt)"
  $PY $CODE/pipeline/tests/check_quant.py $O/i4-ourq-K0.5 $O/i4-rxq-K0.5 > $O/i4-e2e.txt 2>&1; log "i4 (2): $(tail -n 1 $O/i4-e2e.txt)"
  r=PASS; for f in $(cd $O/i4-ourq-K2-qtf2 && ls q-L*.safetensors); do
    $PY $CODE/pipeline/tests/stdiff.py $O/i4-ourq-K2-qtf2/$f $O/i4-ourq-K2-cb/$f > $O/i4-K2-$f.txt 2>&1
    grep -q DIFF $O/i4-K2-$f.txt && r=FAIL; done
  log "i4 (3): K2 qtf2 vs exl3cb tensors $r (metadata stamps differ by design: --qtf2 is in the stamp)"; }

c9_unit() {
  CK=$P.3.self_attn.o_proj,$P.10.linear_attn.out_proj,$P.11.linear_attn.out_proj,$P.39.self_attn.o_proj
  C9E27=${C9E27:-/data/c9w27/code/c9e27.py}
  card bash -c "
    nice -n 5 $PY -m rxq quant --keys $CK -K 2.5 --codebook i8 --ckpt $MODEL --hess $HESS --no-prof -o $O/c9-job > $O/c9-job.log 2>&1
    DEV=cuda nice -n 5 $PY $CODE/pipeline/tests/check_c9job.py $O/c9-job $C9E27 $MODEL $HESS $O/c9-gpu > $O/c9-gpu.txt 2>&1"
  log "c9 (card): $(tail -n 1 $O/c9-gpu.txt); $(grep -o 'in-job C9 [0-9.]* s' $O/c9-gpu.txt | paste -sd, -)"
  DEV=cpu CUDA_VISIBLE_DEVICES= nice -n 10 $PY $CODE/pipeline/tests/check_c9job.py $O/c9-job $C9E27 $MODEL $HESS $O/c9-cpu > $O/c9-cpu.txt 2>&1
  log "c9 (CPU ref, report only): $(tail -n 1 $O/c9-cpu.txt); max |dc|/c $(grep -o 'max |dc|/c [0-9.e+-]*' $O/c9-cpu.txt | awk '{print $3}' | paste -sd, -)"; }

case $U in
  cd) cd_unit ;; i4) i4_unit ;; c9) c9_unit ;; all) cd_unit; i4_unit; c9_unit ;;
  *) echo "unit: cd | i4 | c9 | all"; exit 2 ;;
esac
log "GPUCHECK27 $U DONE"
