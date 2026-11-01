#!/bin/bash
# gate2e.sh -- after gate 2c: gradient-weighted input Hessians (always) and, if gate 2c passed, the sample-count test.
# Collection on blocks 22/23 (fp32, full tokens, backward truncated at 22; the pilot's windows/labels where they exist):
#   gate 2c PASSED (two-sided beat one-sided at K2 and K3, z <= -2): 400 sequences, snapshots at 100/200/400, even-half
#     sums (split-half check), H_in too. Quants: two-sided at 100/200/400 (gate 2c's better sigma_reg_out), one-sided
#     with the 400-sequence gradient-weighted H_in.
#   FAILED: 200 sequences, snapshot at 100, H_in. Quants: one-sided with the gradient-weighted H_in at 100 and 200.
# K2 and K3 each; swapped into V2 6.00, golden-ver1, paired vs one-sided (y2-1s-K*). Log: /data/yaqa/gate2e.log.
: ${RXM_WORK:=/root}; export RXM_WORK
Y=/data/yaqa F=/data/yaqa/full64 QD=/data/yaqa/qe
BASE=/data/release/qwen3.8-27b-exl3-v2-6.00bpw.rad
PY=/data/venv-exl3/bin/python
log() { echo "$(date -u +%H:%M) $*" | tee -a $Y/gate2e.log; }
until grep -q "gate2c done" $Y/gate2c.log 2>/dev/null; do
  pgrep -f "^/bin/bash /data/yaqa/gate2c.sh" > /dev/null || { log "gate2c not running and not done: stop"; exit 1; }; sleep 60; done
read VERDICT SO < <($PY -I - $Y/gate2c.log <<'EOF'
import json, re, sys
best = {}; ok = {}
for l in open(sys.argv[1]):
    m = re.search(r"K(\d) (fullr?)-K\d vs one-sided: (\{.*\})", l)
    if not m: continue
    d = json.loads(m.group(3)); K, v = m.group(1), m.group(2)
    if K not in best or d["kld_b"] < best[K][0]: best[K] = (d["kld_b"], v, d)
for K, (_, v, d) in best.items(): ok[K] = d["diff"] < 0 and d["z"] <= -2
so = "0.3" if best.get("2", (0, "fullr"))[1] == "fullr" else "0.025"
print("PASS" if len(ok) == 2 and all(ok.values()) else "FAIL", so)
EOF
)
log "gate 2c verdict: $VERDICT (sigma_reg_out $SO)"
cd ${RXM_WORK}
export YAQA_FP32=1 YAQA_START=0 YAQA_FULL=22,23 YAQA_LABELS=$Y/rows32 YAQA_HIN=1 TRITON_CACHE_AUTOTUNING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ $VERDICT = PASS ]; then N=400; export YAQA_SNAP=100,200,400 YAQA_HALVES=1; else N=200; export YAQA_SNAP=100,200; fi
if ! grep -q "^DONE" $Y/full64.log 2>/dev/null; then
  ( exec 8>${RXM_WORK}/card0.lock 9>${RXM_WORK}/card1.lock 7>${RXM_WORK}/ram.lock; flock 8; flock 9; flock -s 7
    cd $Y && $PY -I collect.py $F $N 2048 32 ) > $Y/full64.log 2>&1
  grep -q "^DONE" $Y/full64.log || { log "collection FAIL"; exit 1; }
fi
log "collected $N sequences ($(grep '^DONE' $Y/full64.log))"
[ $VERDICT = PASS ] && $PY -I $Y/hhalf.py $F >> $Y/gate2e.log 2>&1
export EXL3_QT_OPTIMIZED=1
quant() {  # CARD then "VARIANT K EXTRA-ARGS(comma-joined)"
  local c=$1; shift
  while [ $# -gt 0 ]; do
    local v=$1 K=$2 x=${3//,/ }; shift 3
    [ -f $QD/$v/q-L23.safetensors ] && continue
    ( exec 9>${RXM_WORK}/card$c.lock; flock 9
      $PY -I ourq/ourq.py -K $K -o $QD/$v --layers 22-23 --device 0 $x 2>&1 \
        | grep -v -e Warning -e "cpu = _conv" ) > $QD/$v.log 2>&1 && grep -q '^done' $QD/$v.log || { log "$v quantize FAIL"; continue; }
    log "$v quantized"
  done
}
mkdir -p $QD
if [ $VERDICT = PASS ]; then
  V="h100-K2 h200-K2 h400-K2 h100-K3 h200-K3 h400-K3 hin400-K2 hin400-K3"
  ROCR_VISIBLE_DEVICES=0 quant 0 h100-K2 2 --hout,$F/hout-snap100,--sigma-reg-out,$SO h400-K2 2 --hout,$F/hout-snap400,--sigma-reg-out,$SO \
    h200-K3 3 --hout,$F/hout-snap200,--sigma-reg-out,$SO hin400-K3 3 --hin,$F/hin-snap400 &
  ROCR_VISIBLE_DEVICES=1 quant 1 h200-K2 2 --hout,$F/hout-snap200,--sigma-reg-out,$SO h100-K3 3 --hout,$F/hout-snap100,--sigma-reg-out,$SO \
    h400-K3 3 --hout,$F/hout-snap400,--sigma-reg-out,$SO hin400-K2 2 --hin,$F/hin-snap400 &
else
  V="hin100-K2 hin200-K2 hin100-K3 hin200-K3"
  ROCR_VISIBLE_DEVICES=0 quant 0 hin200-K2 2 --hin,$F/hin-snap200 hin100-K3 3 --hin,$F/hin-snap100 &
  ROCR_VISIBLE_DEVICES=1 quant 1 hin200-K3 3 --hin,$F/hin-snap200 hin100-K2 2 --hin,$F/hin-snap100 &
fi
wait
T22="ffn_down ffn_gate_up ssm_out ssm_qkv ssm_z"; T23="attn_k attn_output attn_qg attn_v ffn_down ffn_gate_up"
G=""; for t in $T22; do G="$G blk.22.$t.weight"; done; for t in $T23; do G="$G blk.23.$t.weight"; done
Q=$Y/gate2e.q; : > $Q
for v in $V; do
  [ -f $QD/$v/q-L23.safetensors ] || continue
  ./ourq/ourconv.sh $QD/$v.rad sc4.00-e4 $QD/$v $G >> $Y/gate2e.log 2>&1 || { log "$v conv FAIL"; continue; }
  : > $QD/$v.mix; for g in $G; do echo "$g $QD/$v.rad" >> $QD/$v.mix; done
  echo "y6-$v $BASE $QD/$v.mix golden-ver1 corpus/ver1.jsonl" >> $Q
  log "$v converted, queued"
done
echo END >> $Q
./auto/evalq2.sh 0 $Q >> $Y/gate2e-eval0.log 2>&1 & ./auto/evalq2.sh 1 $Q >> $Y/gate2e-eval1.log 2>&1 & wait
S=${RXM_WORK}/evq/scores
for v in $V; do
  K=${v##*-K}; [ -f $S/y6-$v.f32 ] || continue
  log "$v vs one-sided: $($PY -I ${RXM_WORK}/hproxy/paired.py ${RXM_WORK}/golden-ver1 $S/y2-1s-K$K.f32 $S/y6-$v.f32)"
done
rm -f $QD/*.rad                     # scratch: scores are kept (13.5 GB each)
log "gate2e done"
