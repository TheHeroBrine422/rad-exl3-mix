#!/bin/bash
# w106.sh [STEP ...] -- #140 (../w106.md): C9 at low K on the 0.8B, gpubox card 0. Default steps:
#   hin gold  K0.5:0 K1:0 K2:0 K0.5:1 K1:1 K2:1 K1.5:0 K1.5:1  report
#   hin     calib input Hessians of the 0.8B (proxy2b/hcollect.py, every calib document) -> $A/h08
#   gold    g08-fit1 / g08-search1 rebuilt (proxy2b/mkgold.py, the old kld.json's documents) + gate G2
#   K:S     one (K, seed) cell: rxq quant (24 layers, today's settings, block order off) -> swj -> C9 arms (CPU, gate
#           G1) -> evalq of every distinct arm on g08-search1
#   report  report106.py -> $A/report.md
# GPU units hold /root/card0.lock then /root/ram.lock shared (behind other card-0 jobs), under swap27's memguard.
# Every step skips finished work (outputs written as .part, renamed; .done after the data). Stop: touch $A/STOP.
set -u
: ${RXM_WORK:=/root}
A=${A:-/data/w106}; CODE=$A/code; EXL3=$A/exl3; W=$A/w106; P=$CODE/proxy2b PY=/data/venv-exl3/bin/python
M=$A/model META=/data/challenge/golden/calib C18=$A/in/w03/ref/c18/hdiag-snap1024-dy GS=$A/g08-search1
# h08 and g08-fit1: w103 rebuilt both on card 0 with the same recipe (setup08.sh / mkgold.py) minutes before this run;
# reused read-only (amendment 1 of ../w106.md). If w103's hcollect dies, hin() builds $A/h08 itself.
H08=/data/w103/h08/calib GF=/data/w103/g08-fit1 T=/data/swap27/tools GUARD_MB=${GUARD_MB:-3000}
STEPS=${*:-hin gold K0.5:0 K1:0 K2:0 K0.5:1 K1:1 K2:1 K1.5:0 K1.5:1 report}
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=$A/build OMP_NUM_THREADS=4 MAX_JOBS=4 PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1 \
       VRAM_CAP_GB=${VRAM_CAP_GB:-30} POOLS=$CODE/eval/pools/pools-v3.json TOKENIZERS_PARALLELISM=false
ENVQ=(PYTHONPATH=$CODE/pipeline:$EXL3:/data/exllamav3 RXQ_RXM=$CODE RXQ_EXL3_SRC=/data/exl3frac/src RXQ_REFIT_CPU_FALLBACK=1)
mkdir -p $A/q $A/mix $A/ev $A/logs; cd /tmp
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $A/result.txt; }
die() { log "W106 STOP: $*"; exit 1; }
stopped() { [ -f $A/STOP ] && { log "STOP file: exiting"; exit 0; }; return 0; }
nq() { ls $1/*.safetensors 2>/dev/null | wc -l; }
# card CMD...: one GPU unit (card0.lock, then ram.lock shared), memguard'ed, niced
card() { ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
  $T/memguard.sh $GUARD_MB nice -n 5 "$@" ); }

hin() {
  if [ ! -f $H08/.done ]; then log "hin: waiting for w103's h08 ($H08/.done)"
    while [ ! -f $H08/.done ] && pgrep -f "[h]collect.py /data/w103/model" > /dev/null; do sleep 60; done; fi
  if [ -f $H08/.done ]; then [ $(nq $H08) -ge 96 ] || die "h08 $H08: $(nq $H08) files"; log "hin: using $H08 ($(nq $H08) files)"; return 0; fi
  H08=$A/h08; [ -f $H08/.done ] && return 0; stopped; local s=$(date +%s)
  log "hin: w103's hcollect is gone without .done: building $H08"
  rm -rf $H08.part
  card $PY $P/hcollect.py $M $META $H08.part --layers 0-23 > $A/logs/hin.log 2>&1 || die "hcollect (logs/hin.log)"
  [ $(nq $H08.part) = 96 ] || die "hcollect: $(nq $H08.part) groups, want 96"
  mv $H08.part $H08 && touch $H08/.done; log "hin: 96 groups, $(( $(date +%s) - s )) s (incl. lock wait)"
}

gold() {
  for g in search1:$A/in/w09/g08-search1; do
    local n=${g%%:*} old=${g#*:} d=$A/g08-${g%%:*}
    [ -f $d/.done ] && continue; stopped; local s=$(date +%s)
    local args; args=$($PY -c "import json; print(' '.join(f\"{d['orig']}:{d['tokens']}:{d['score_from']}\" for d in json.load(open('$old/kld.json'))['docs']))")
    rm -rf $d; card $PY $P/mkgold.py $M $META $d $args > $A/logs/gold-$n.log 2>&1 || die "mkgold $n"
    $PY - $old/kld.json $d/kld.json <<'PY' >> $A/result.txt || die "G2 $n"
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
ka = [(d["orig"], d["tokens"], d["score_from"]) for d in a["docs"]]; kb = [(d["orig"], d["tokens"], d["score_from"]) for d in b["docs"]]
ok = ka == kb and a["rows"] == b["rows"]
print(f"G2 {sys.argv[2]}: {'PASS' if ok else 'FAIL'} ({len(kb)} docs, {b['rows']} rows vs {a['rows']}; kernels {b['kernels']})")
sys.exit(0 if ok else 1)
PY
    touch $d/.done; log "gold $n: $(( $(date +%s) - s )) s"
  done
}

ev() {  # ev TAG WQDIR
  local f=$A/ev/$1.f32; [ -f $f ] && return 0; stopped
  card $PY $P/evalq.py $M $GS --wq $2 --out $f.part > $A/logs/ev-$1.log 2>&1 && mv $f.part $f || die "evalq $1"
  log "ev $1: $(tail -n 1 $A/logs/ev-$1.log | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"
}

cell() {  # cell K S
  local K=$1 S=$2 tag=K$1s$2; local qd=$A/q/$tag sw=$A/q/${tag}_swj
  [ -f $A/mix/$tag.done ] && return 0; stopped
  if [ ! -f $qd/.done ]; then local s=$(date +%s)
    card env "${ENVQ[@]}" $PY -m rxq quant --layers 0-23 -K $K --codebook i8 --ckpt $M --hess $H08 --hout $C18 --seed-offset $S \
      -o $qd --save-wq --no-prof --set quant.block_order.enable=false >> $A/logs/q-$tag.log 2>&1 || die "quant $tag"
    [ $(ls $qd/t-*.safetensors | wc -l) = 150 ] && [ $(nq $qd/wq) = 150 ] && [ $(ls $qd/c9-*.safetensors | wc -l) = 24 ] || die "quant $tag incomplete"
    touch $qd/.done; log "quant $tag: $(( $(date +%s) - s )) s"; fi
  if [ ! -f $sw/.done ]; then local s=$(date +%s); rm -rf $sw $A/q/${tag}_swu
    card $PY $P/exp/w02-fixedscale/swiglufit.py $M $qd $GF $sw $A/q/${tag}_swu > $A/logs/swj-$tag.log 2>&1 || die "swj $tag"
    [ -f $sw/.done ] && [ $(nq $sw/wq) = 150 ] || die "swj $tag incomplete"; rm -rf $A/q/${tag}_swu
    log "swj $tag: $(( $(date +%s) - s )) s"; fi
  if [ ! -f $A/mix/$tag-c9.json ]; then
    ( exec 7>$RXM_WORK/ram.lock; flock -s 7; CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= nice -n 19 \
      env "${ENVQ[@]}" $PY $W/c9arms106.py $qd $sw/wq $A/mix $tag $K --model $M --hess $H08 ) > $A/logs/c9-$tag.log 2>&1
    local rc=$?; log "c9 $tag: $(tail -n 1 $A/logs/c9-$tag.log)"; [ $rc = 0 ] || die "C9 arms $tag (G1 or error)"; fi
  for arm in off cur ak e3; do
    [ -f $A/mix/$tag-$arm.same ] && continue
    ev $tag-$arm $A/mix/$tag-$arm
  done
  touch $A/mix/$tag.done; log "cell $tag done"
}

report() { $PY -I $W/report106.py $A > $A/report.md.part 2>> $A/logs/report.err && mv $A/report.md.part $A/report.md && log "report written"; }

[ -f $GF/kld.json ] || die "no $GF"
log "w106.sh start: $STEPS (code $(cat $CODE/COMMIT), fork $(cat $EXL3/COMMIT))"
for st in $STEPS; do
  case $st in
    hin) hin ;; gold) gold ;; report) report ;;
    K*:*) k=${st%%:*}; cell ${k#K} ${st#*:}; report ;;
    *) die "unknown step $st" ;;
  esac
done
log "W106 DONE ($STEPS)"
