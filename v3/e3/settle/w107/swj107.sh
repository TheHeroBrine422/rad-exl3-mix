#!/bin/bash
# swj107.sh -- #141-04 (../w107.md) = #132-01 (w100.md item 6, rule unchanged), the port of w100/swj4070.sh from the 4070
# to gpubox card 0 on today's 0.8B rxq stack (as w106's #140 cells): per K in {0.5, 1} x seed offset {0, 1}:
#   base = `rxq quant` (24 layers, i8, v3.toml stamp 87a7e4ab, block order off, H_out c18 Σdy², h08, --save-wq) -> wq
#   swj  = proxy2b/exp/w02-fixedscale/swiglufit.py on g08-fit1 from it (codes fixed; gate / up / down output scales)
# Reuses w106's finished cells read-only (/data/w106/q/K<K>s<S>{,_swj}/.done: the same commands); else makes them in
# $W/q with the same commands and inputs. Then evalq on w106's g08-search1 (base and swj), pcmp pair -> $W/result.txt.
# Waits for w106's card steps: until its result.txt says "W106 DONE", or w106.sh is not running and its result.txt has
# been quiet for 60 min (a stopped w106 is not waited for forever). Every GPU unit: /root/card0.lock then /root/ram.lock
# shared, memguard'ed. Idempotent (finished quants / scores skipped). Stop: touch $W/STOP.
# Usage (gpubox): setsid nohup bash /data/w107/code/v3/e3/settle/w107/swj107.sh > /data/w107/swj.out 2>&1 < /dev/null &
set -u
W=/data/w107/swj; CODE=/data/w107/code; EXL3=/data/w107/exl3; P=$CODE/proxy2b PY=/data/venv-exl3/bin/python
A=/data/w106; M=$A/model C18=$A/in/w03/ref/c18/hdiag-snap1024-dy GS=$A/g08-search1 GF=/data/w103/g08-fit1
T=/data/swap27/tools
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=/data/w107/build OMP_NUM_THREADS=4 MAX_JOBS=4 PATH=/opt/rocm/core-10.1/bin:$PATH \
       ROCM_PATH=/opt/rocm/core-10.1 VRAM_CAP_GB=30 POOLS=$CODE/eval/pools/pools-v3.json TOKENIZERS_PARALLELISM=false
ENVQ=(PYTHONPATH=$CODE/pipeline:$EXL3:/data/exllamav3 RXQ_RXM=$CODE RXQ_EXL3_SRC=/data/exl3frac/src RXQ_REFIT_CPU_FALLBACK=1)
mkdir -p $W/q $W/ev $W/logs; cd /tmp
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $W/result.txt; }
stopped() { [ -f $W/STOP ] && { log "STOP file: exiting"; exit 0; }; return 0; }
nq() { ls $1/*.safetensors 2>/dev/null | wc -l; }
card() { ( exec 5>/root/card0.lock; flock 5; exec 7>/root/ram.lock; flock -s 7; $T/memguard.sh 3000 nice -n 5 "$@" ); }

w106_busy() { grep -q "W106 DONE" $A/result.txt 2>/dev/null && return 1
  pgrep -f "[w]106/w106.sh" > /dev/null && return 0
  [ -n "$(find $A/result.txt -mmin -60 2>/dev/null)" ]; }
while w106_busy; do stopped; sleep 300; done
H08=/data/w103/h08/calib; [ -f $H08/.done ] || H08=$A/h08
[ -f $H08/.done ] || { log "no finished h08 ($H08)"; exit 1; }
log "swj107 start: w106 idle (code $(cat $CODE/COMMIT), exl3 $(cat $EXL3/COMMIT), h08 $H08)"
[ -f $GS/.done ] && [ -f $GF/.done -o -f $GF/kld.json ] || { log "inputs missing (g08-search1 / g08-fit1)"; exit 1; }

ev() {  # ev TAG WQDIR
  local f=$W/ev/$1.f32; [ -f $f ] && return 0; stopped
  card $PY $P/evalq.py $M $GS --wq $2 --out $f.part > $W/logs/ev-$1.log 2>&1 && mv $f.part $f || { log "evalq $1 FAILED"; return 1; }
  log "ev $1: $(tail -n 1 $W/logs/ev-$1.log | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"
}

for K in 0.5 1; do for S in 0 1; do
  stopped; tag=K${K}s$S; qd=$A/q/$tag; sw=$A/q/${tag}_swj
  [ -f $qd/.done ] || qd=$W/q/$tag
  [ -f $sw/.done ] || sw=$W/q/${tag}_swj
  if [ ! -f $qd/.done ]; then
    card env "${ENVQ[@]}" $PY -m rxq quant --layers 0-23 -K $K --codebook i8 --ckpt $M --hess $H08 --hout $C18 --seed-offset $S \
      -o $qd --save-wq --no-prof --set quant.block_order.enable=false >> $W/logs/q-$tag.log 2>&1 \
      && [ $(nq $qd/wq) = 150 ] && touch $qd/.done || { log "quant $tag FAILED"; continue; }
    log "quant $tag: made here"; fi
  if [ ! -f $sw/.done ]; then rm -rf $sw $W/q/${tag}_swu
    card env PYTHONPATH=$P $PY $P/exp/w02-fixedscale/swiglufit.py $M $qd $GF $sw $W/q/${tag}_swu > $W/logs/swj-$tag.log 2>&1 \
      && [ -f $sw/.done ] && [ $(nq $sw/wq) = 150 ] || { log "swj $tag FAILED"; continue; }
    rm -rf $W/q/${tag}_swu; log "swj $tag: made here"; fi
  log "$tag: base $qd, swj $sw"
  [ $(nq $qd/wq) = 150 ] && [ $(nq $sw/wq) = 150 ] || { log "$tag: wq incomplete"; continue; }
  ev $tag-base $qd/wq && ev $tag-swj $sw/wq || continue
  grep -q "^.* pair $tag " $W/result.txt || log "pair $tag base -> swj: $($PY $P/pcmp.py $GS $W/ev/$tag-base.f32 $W/ev/$tag-swj.f32 | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"
done; done
log "SWJ107 DONE"
