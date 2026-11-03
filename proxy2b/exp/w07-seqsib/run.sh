#!/bin/bash
# run.sh -- w07-seqsib (ranked.md C6): intra-layer sequential inputs on the 0.8B, K3 / K2. For each base quant (seed
# offsets 0 / 1 / 2), the o_proj / out_proj / down_proj input moments with that quant's own q/k/v (GDN qkv/z) and gate/up
# in place (../w07-denoise/moments.py --mode sib), then those 72 tensors requantized with the same seeds on
#   sym:  G = H~ (the quantized-sibling input covariance; lam = rho = 1, plain sequential GPTQ)       -- --hin only
#   asym: G = H~, target W F with lam 1 / rho 0.5 (partial GPTAQ-style correction toward the clean output) -- + --wpre
# and every other tensor kept from the base (so arm and base differ only in those 72). Noise floor: the same 72
# requantized on the calib H at seed offsets 7 / 8 / 9 (controls rs7..rs9, base seed 0's siblings).
# 4070, one GPU, idempotent and resumable. asym needs ../w07-denoise/wpre.patch merged (sym and the controls do not).
P=/workspace/rad-exl3-mix/proxy2b X=$P/exp/w07-seqsib D7=$P/exp/w07-denoise M=/data/models/qwen35-08b
S=/data/g08-search1 F=/data/g08-fit1 Y=/data/yb O=/data/yb/w07-seqsib H=/data/h08/calib META=/data/challenge/calib-meta
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
export POOLS=/workspace/rad-exl3-mix/eval/pools/pools-v3.json
mkdir -p $O; log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
sfx() { [ $1 = 0 ] && echo "" || echo "_s$1"; }
ev() { [ -f $O/$1$4.f32 ] && return
  log "$1$4: $(python3 $P/evalq.py $M $3 --wq $2 --out $O/$1$4.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
python3 -c "
for l in range(24):
    p = f'model.language_model.layers.{l}'
    print(f'{p}.self_attn.o_proj' if l % 4 == 3 else f'{p}.linear_attn.out_proj'); print(f'{p}.mlp.down_proj')" > $O/only.txt
qo() { # qo TAG K BASEWQ ourq-args..: requantize the 72 tensors of only.txt, then TAG's full wq = BASEWQ with them replaced
  t=$1 k=$2 bw=$3; shift 3
  [ $(ls $O/q-$t/wq/*.safetensors 2>/dev/null | wc -l) -ge 48 ] || $P/q2b.sh 08 $k i8 $O/q-$t --only $O/only.txt "$@" > $O/q-$t.log 2>&1
  [ $(ls $O/q-$t/wq/*.safetensors 2>/dev/null | wc -l) = 48 ] || { log "QUANT FAIL $t (see q-$t.log)"; exit 1; }
  d=$O/q-$t/wqall; mkdir -p $d
  for f in $bw/*.safetensors; do ln -sfn $f $d/; done
  for f in $O/q-$t/wq/*.safetensors; do ln -sfn $f $d/; done
  [ $(ls $d/*.safetensors | wc -l) = 150 ] || { log "ASSEMBLE FAIL $t"; exit 1; }; }
mom() { [ -f $1/model.language_model.layers.23.mlp.down_proj.input.safetensors ] && return
  for r in 0-11 12-23; do
    python3 $D7/moments.py $M $META $1 --layers $r --tokens 1000000 "${@:2}" >> $1.log 2>&1 || { log "MOMENTS FAIL $1 $r"; exit 1; }
  done; log "moments $(basename $1): $(grep '^MOMENTS DONE' $1.log | tail -1)"; }
mk() { [ -f $2/stats.jsonl ] && [ $(wc -l < $2/stats.jsonl) = 48 ] || python3 $D7/mkdenoise.py $H $1 $2 "${@:3}" --ckpt $M > $2.log 2>&1 || { log "MKDENOISE FAIL $2"; exit 1; }; }
cd /tmp
# 1. moments per base seed (~5 min each), metric / target dirs
for k in 3 2; do for s in 0 1 2; do
  b=k${k}base$(sfx $s); mom $O/mom-$b --mode sib --wq $Y/q-$b/wq --seed 0
  mk $O/mom-$b $O/dn-sym-$b --lam 1 --rho 1; mk $O/mom-$b $O/dn-asym-$b --lam 1 --rho 0.5
done; done
# 0. no-op gate (~1 min): the 48 tensors requantized on mkdenoise.py's lam = rho = 0 metric (= calib's H) with base seed 0
#    must equal k3base's own; else the base quants are stale (ourq / exllamav3 changed since) and nothing here counts
if [ ! -f $O/noop.done ]; then
  mk $O/mom-k3base $O/dn-noop --lam 0 --rho 0
  qo noop 3 $Y/q-k3base/wq --hin $O/dn-noop/hin
  python3 - $Y/q-k3base/wq $O/q-noop/wq <<'PY' | tee -a $O/result.txt
import glob, os, sys, torch
from safetensors.torch import load_file
a, b = sys.argv[1:3]; fs = sorted(os.path.basename(f) for f in glob.glob(f"{b}/*.safetensors"))
bad = [f for f in fs if not torch.equal(load_file(f"{a}/{f}")["weight"], load_file(f"{b}/{f}")["weight"])]
print(f"noop gate: {len(fs)} requantized tensors, {len(bad)} differ from k3base {bad[:2]} -> {'PASS' if len(fs) == 48 and not bad else 'FAIL'}")
sys.exit(1 if bad or len(fs) != 48 else 0)
PY
  [ ${PIPESTATUS[0]} = 0 ] || { log "NOOP GATE FAIL"; exit 1; }
  touch $O/noop.done
fi
for k in 3 2; do
  # 2. fit-slice screen at seed 0: sym vs asym; noise floor controls (search slice)
  ev k${k}base $Y/q-k${k}base/wq $F .fit
  qo k${k}sym $k $Y/q-k${k}base/wq --hin $O/dn-sym-k${k}base/hin; ev k${k}sym $O/q-k${k}sym/wqall $F .fit
  qo k${k}asym $k $Y/q-k${k}base/wq --hin $O/dn-asym-k${k}base/hin --wpre $O/dn-asym-k${k}base/wpre; ev k${k}asym $O/q-k${k}asym/wqall $F .fit
  for c in 7 8 9; do qo k${k}rs$c $k $Y/q-k${k}base/wq --seed-offset $c; ev k${k}rs$c $O/q-k${k}rs$c/wqall $S ""; done
  best=$(python3 $D7/summ.py pick $O $k sym,asym 2>> $O/result.txt); [ -n "$best" ] || { log "PICK FAIL K$k"; exit 1; }
  log "K$k: fit-slice pick $best"
  # 3. the pick at seeds 0 / 1 / 2 on search, each on its own seed's siblings and moments
  for s in 0 1 2; do
    b=k${k}base$(sfx $s); t=k${k}$best$(sfx $s); dn=$O/dn-$best-$b
    qo $t $k $Y/q-$b/wq --hin $dn/hin $([ -d $dn/wpre ] && echo --wpre $dn/wpre) --seed-offset $s
    ev $t $O/q-$t/wqall $S ""
    [ -f $Y/$b.f32 ] || ev $b $Y/q-$b/wq $S ""
  done
  python3 $D7/summ.py verdict $O $S $Y $k $best ctrl 0.98 rs7,rs8,rs9 | tee -a $O/result.txt
done
log SEQSIB DONE
