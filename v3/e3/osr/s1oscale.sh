#!/bin/bash
# s1oscale.sh K BASE_TAG V2SIZE CARD -- pre-requant checklist #1: gate/up output scales OFF (upstream's test, ratio > 1.05)
# vs ON, paired. Both arms quantized fresh by today's ourq (mul1 codebook, K), only the gate+up of layers with a flagged
# projection; swapped as whole ffn_gate_up tensors into the V2 release V2SIZE; scored on golden-s2k (all 583 search
# docs, 2k cap). Rule (fixed 2026-11-01): OFF wins by more than its paired SE -> the requant uses the per-tensor rule.
# e.g.  s1oscale.sh 3 sc3.00-e4 3.00 1 ; s1oscale.sh 5 sc5.00-e5 5.00 1.   Out /data/oscale/K<K>/, result in result.txt
: ${RXM_WORK:=/root}; export RXM_WORK
K=$1 B=$2 SZ=$3 C=$4 D=/data/oscale/K$1 S=/data/oscale PY=/data/venv-exl3/bin/python
G=$RXM_WORK/golden-s2k CO=$RXM_WORK/corpus/s2k.jsonl
mkdir -p $D; log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $D/run.log; }
cd $RXM_WORK
# keys: gate + up of every layer with a flagged projection (whole Hessian groups)
python3 - $S/off-1.05.txt $D/only.txt $D/layers.txt <<'EOF'
import re, sys
off = [l.strip() for l in open(sys.argv[1]) if l.strip()]
L = sorted({int(re.search(r"layers\.(\d+)\.", k).group(1)) for k in off})
open(sys.argv[2], "w").write("".join(f"model.language_model.layers.{l}.mlp.{p}_proj\n" for l in L for p in ("gate", "up")))
open(sys.argv[3], "w").write(" ".join(map(str, L)) + "\n")
EOF
LAYERS=$(cat $D/layers.txt); log "K$K: $(wc -l < $D/only.txt) tensors in $(echo $LAYERS | wc -w) layers, $(wc -l < $S/off-1.05.txt) scales off"
exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
export ROCR_VISIBLE_DEVICES=$C
for arm in on off; do
  [ -f $D/$arm.rad ] && continue
  X=""; [ $arm = off ] && X="--out-scales-off $S/off-1.05.txt"
  $PY -I ourq/ourq.py -K $K -o $D/$arm --only $D/only.txt --device 0 $X > $D/$arm.log 2>&1 || { log "$arm: ourq FAIL"; exit 1; }
  GL=""; for l in $LAYERS; do GL="$GL blk.$l.ffn_gate_up.weight"; done
  flock -u 7
  ourq/ourconv.sh $D/$arm.rad $B $D/$arm $GL >> $D/run.log 2>&1 || { log "$arm: ourconv FAIL"; exit 1; }
  flock -s 7
  for l in $LAYERS; do echo "blk.$l.ffn_gate_up.weight $D/$arm.rad"; done > $D/$arm.mix
  ./hproxy/radmix $D/v2-$SZ-$arm.rad /data/release/qwen3.8-27b-exl3-v2-${SZ}bpw.rad $D/$arm.mix 2> $D/$arm.radmix.log \
    || { log "$arm: radmix FAIL"; exit 1; }
  log "$arm: container $(stat -c %s $D/v2-$SZ-$arm.rad) bytes, out_scales False on $(grep -c '"out_scales": false' $D/$arm/proxy.jsonl) tensors"
done
for arm in on off; do
  [ -f $D/score-$arm.f32 ] && continue
  w=$(SEQS=8 CARD=$C BIN=$RXM_WORK/radiance-kldgold-fast ./gold2.sh $D/v2-$SZ-$arm.rad $G $CO $D/run-$arm bf16 16384 \
      --max-num-batched-tokens 4096 --prefill-span 2)
  cp $D/run-$arm/score.f32 $D/score-$arm.f32 && log "$arm eval: $w $(python3 -c "import json; print(json.load(open('$D/run-$arm.json'))['kld']['mean'])")"
done
python3 hproxy/paired.py $G $D/score-on.f32 $D/score-off.f32 | tee $D/result.txt
log "K$K done (result.txt: B = off, A = on; negative = off better)"
