#!/bin/bash
# run.sh -- w07-denoise (ranked.md C1): activation-denoising LDLQ target and metric (arXiv 2610.07522) on the 0.8B, K3 / K2.
# Two noise models, three (lam, rho) each, picked per K on the fit slice (seed 0), confirmed on search with 3 seeds.
#   quant: the measured upstream error of the uniform-K base quant at seed offset 3 (the paper's "parallel-quantized
#          upstream error" proxy; seed 3 is never an evaluated seed), one collection per K
#   gauss: per-channel Gaussian noise (gamma x channel max) injected at every group input and propagated (the paper's
#          model), gamma set per K so that the layer-input noise matches the quant model's (median over layers of
#          tr N / tr H; a gamma 0.01 pilot sets the scale, variance ~ gamma^2): the paper's gamma 0.05 is 10-20x the K3
#          error in variance on this model (README), one collection per K
# 4070, one GPU, idempotent and resumable (rerun after any failure). Needs wpre.patch merged into quantize/ourq.py.
P=/workspace/rad-exl3-mix/proxy2b X=$P/exp/w07-denoise M=/data/models/qwen35-08b S=/data/g08-search1 F=/data/g08-fit1
Y=/data/yb O=/data/yb/w07-denoise H=/data/h08/calib META=/data/challenge/calib-meta
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
export POOLS=/workspace/rad-exl3-mix/eval/pools/pools-v3.json
mkdir -p $O; log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
declare -A ARM=([g1]="gauss 1 0.9" [g2]="gauss 1 0.5" [g3]="gauss 2 1.8" [q1]="quant 1 1" [q2]="quant 1 0.75" [q3]="quant 1 0.5")
sfx() { [ $1 = 0 ] && echo "" || echo "_s$1"; }
ev() { # ev TAG WQDIR GOLDEN SUFFIX: one eval, skipped when its scores exist
  [ -f $O/$1$4.f32 ] && return
  log "$1$4: $(python3 $P/evalq.py $M $3 --wq $2 --out $O/$1$4.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
q() { tag=$1; k=$2; shift 2   # quantize (whole trunk) unless its wq exist
  [ $(ls $O/q-$tag/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || $P/q2b.sh 08 $k i8 $O/q-$tag "$@" > $O/q-$tag.log 2>&1
  [ $(ls $O/q-$tag/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || { log "QUANT FAIL $tag (see q-$tag.log)"; exit 1; }; }
mom() { # mom OUT ARGS..: moments in two layer passes (all 24 layers' sums do not fit beside the model)
  [ -f $1/model.language_model.layers.23.mlp.down_proj.input.safetensors ] && return
  for r in 0-11 12-23; do
    python3 $X/moments.py $M $META $1 --layers $r "${@:2}" >> $1.log 2>&1 || { log "MOMENTS FAIL $1 $r (see $1.log)"; exit 1; }
  done; log "moments $(basename $1): $(grep '^MOMENTS DONE' $1.log | tail -1)"; }
grep -q -- '"--wpre"' /workspace/rad-exl3-mix/quantize/ourq.py || { log "ourq.py has no --wpre: merge $X/wpre.patch first"; exit 1; }
cd /tmp
# 1. moments (1M qad tokens, ~5-10 min each): the base quants' upstream error per K; a gamma 0.01 Gaussian pilot (200k
#    tokens); the level-matched Gaussian per K
rin() { python3 -c "
import glob, statistics, sys; from safetensors import safe_open
r = []
for f in glob.glob('$1/model.language_model.layers.*.input.safetensors'):
    if not f.endswith(tuple(f'layers.{l}.input.safetensors' for l in range(1, 24))): continue
    with safe_open(f, 'pt') as h: r.append(float(h.get_tensor('N').trace() / h.get_tensor('H').trace()))
print(statistics.median(r))"; }
for k in 3 2; do mom $O/mom-quant$k --mode quant --wq $Y/q-k${k}base_s3/wq --seed 0 --tokens 1000000; done
mom $O/mom-pilot --mode noise --gamma 0.01 --seed 0 --tokens 200000 --sigma $O/sigma.pt --sigma-tokens 1000000
for k in 3 2; do
  [ -f $O/gamma$k.txt ] || python3 -c "print(round(0.01 * ($(rin $O/mom-quant$k) / $(rin $O/mom-pilot)) ** 0.5, 5))" > $O/gamma$k.txt
  mom $O/mom-gauss$k --mode noise --gamma $(cat $O/gamma$k.txt) --seed 0 --tokens 1000000 --sigma $O/sigma.pt
  log "K$k: layer-input tr N / tr H: quant $(rin $O/mom-quant$k), pilot (gamma 0.01) $(rin $O/mom-pilot), gauss (gamma $(cat $O/gamma$k.txt)) $(rin $O/mom-gauss$k)"
done
# 0. no-op gate (~3 min): mkdenoise.py at lam = rho = 0 (G = calib's H, rebuilt) as --hin, identity transforms as --wpre
#    -> every wq tensor identical to the existing K3 base quant (same seeds); else nothing here counts
if [ ! -f $O/noop.done ]; then
  [ -f $O/dn-noop/stats.jsonl ] || python3 $X/mkdenoise.py $H $O/mom-quant3 $O/dn-noop --lam 0 --rho 0 > $O/dn-noop.log 2>&1
  python3 -c "
import glob, os, torch; from safetensors.torch import save_file
o = '$O/dn-noop'; os.makedirs(o + '/wpre', exist_ok=True)
for f in glob.glob(o + '/hin/*.safetensors'):
    k = int(__import__('safetensors').safe_open(f, 'pt').metadata()['k']); save_file({'Ft': torch.eye(k)}, o + '/wpre/' + os.path.basename(f))"
  q noop 3 --hin $O/dn-noop/hin --wpre $O/dn-noop/wpre
  python3 - $Y/q-k3base/wq $O/q-noop/wq <<'PY' | tee -a $O/result.txt
import glob, os, sys, torch
from safetensors.torch import load_file
a, b = sys.argv[1:3]; fs = sorted(os.path.basename(f) for f in glob.glob(f"{a}/*.safetensors"))
bad = [f for f in fs if not torch.equal(load_file(f"{a}/{f}")["weight"], load_file(f"{b}/{f}")["weight"])]
print(f"noop gate: {len(fs)} wq tensors, {len(bad)} differ from k3base {bad[:2]} -> {'PASS' if fs and not bad else 'FAIL'}"); sys.exit(1 if bad or not fs else 0)
PY
  [ ${PIPESTATUS[0]} = 0 ] || { log "NOOP GATE FAIL: the hin / wpre path changes the quant (or k3base is stale: ourq / exllamav3 changed since)"; exit 1; }
  touch $O/noop.done
fi
# 2. metric / target dirs, then the fit-slice screen at seed 0 (base: the existing quant, evaluated on fit)
for k in 3 2; do
  ev k${k}base $Y/q-k${k}base/wq $F .fit
  for a in g1 g2 g3 q1 q2 q3; do
    read src lam rho <<< "${ARM[$a]}"; mo=$O/mom-$src$k; D=$O/dn-${a}k$k
    if ! { [ -f $D/stats.jsonl ] && [ $(wc -l < $D/stats.jsonl) = 96 ]; }; then
    python3 $X/mkdenoise.py $H $mo $D --lam $lam --rho $rho --ckpt $M > $D.log 2>&1 || { log "MKDENOISE FAIL $D"; exit 1; }
    log "dn-${a}k$k: $(python3 -c "
import json, statistics as st; r = [json.loads(l) for l in open('$D/stats.jsonl')]; wf = [v for x in r for k_, v in x.items() if k_.startswith('wf_')]
print(f'clip median {st.median(x[\"clip\"] for x in r):.1e} max {max(x[\"clip\"] for x in r):.1e}; hsub median {st.median(x[\"hsub_vs_calib\"] for x in r):.3f}'
      + (f'; |WF|/|W| median {st.median(wf):.4f} range {min(wf):.4f}-{max(wf):.4f}' if wf else '; F = I'))")"
    fi
    q k${k}$a $k --hin $D/hin $([ -d $D/wpre ] && echo --wpre $D/wpre)
    ev k${k}$a $O/q-k${k}$a/wq $F .fit
  done
done
# 3. pick per K (lowest fit KL at seed 0), confirm on search with seeds 0 / 1 / 2 against base's seeds
for k in 3 2; do
  best=$(python3 $X/summ.py pick $O $k g1,g2,g3,q1,q2,q3 2>> $O/result.txt); [ -n "$best" ] || { log "PICK FAIL K$k"; exit 1; }
  log "K$k: fit-slice pick $best (${ARM[$best]})"
  D=$O/dn-${best}k$k
  for s in 0 1 2; do
    t=k${k}$best$(sfx $s); [ $s = 0 ] || q $t $k --hin $D/hin $([ -d $D/wpre ] && echo --wpre $D/wpre) --seed-offset $s
    ev $t $O/q-$t/wq $S ""
    [ -f $Y/k${k}base$(sfx $s).f32 ] || ev k${k}base$(sfx $s) $Y/q-k${k}base$(sfx $s)/wq $S ""
  done
done
for k in 3 2; do
  python3 $X/summ.py verdict $O $S $Y $k $(python3 $X/summ.py pick $O $k g1,g2,g3,q1,q2,q3 2>/dev/null) seed $([ $k = 2 ] && echo 0.97 || echo 0.985) | tee -a $O/result.txt
done
log DENOISE DONE
