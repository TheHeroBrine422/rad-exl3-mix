#!/bin/bash
# run.sh -- w07-genh (ranked.md C13): gen-weighted input Hessians on the 0.8B, K3 / K2. H(w0) = calib H with the unscored
# positions (gen.u8 = 0) of the non-verify documents outside the two eval goldens re-weighted to w0 (hgen.py, mkgenh.py);
# w0 = 0 (g0) or 0.3 (g3), picked per K on the fit slice (seed 0), confirmed on search with 3 seeds. Base settings
# otherwise (q2b.sh); --hin replaces every group's H. 4070, one GPU, idempotent and resumable; no patch needed.
P=/workspace/rad-exl3-mix/proxy2b X=$P/exp/w07-genh D7=$P/exp/w07-denoise M=/data/models/qwen35-08b
S=/data/g08-search1 F=/data/g08-fit1 Y=/data/yb O=/data/yb/w07-genh H=/data/h08/calib META=/data/challenge/calib-meta
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
export POOLS=/workspace/rad-exl3-mix/eval/pools/pools-v3.json
mkdir -p $O; log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
declare -A W0=([g0]=0 [g3]=0.3)
sfx() { [ $1 = 0 ] && echo "" || echo "_s$1"; }
ev() { [ -f $O/$1$4.f32 ] && return
  log "$1$4: $(python3 $P/evalq.py $M $3 --wq $2 --out $O/$1$4.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
q() { tag=$1; k=$2; shift 2
  [ $(ls $O/q-$tag/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || $P/q2b.sh 08 $k i8 $O/q-$tag "$@" > $O/q-$tag.log 2>&1
  [ $(ls $O/q-$tag/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || { log "QUANT FAIL $tag (see q-$tag.log)"; exit 1; }; }
cd /tmp
# 0. collector gate (~1 min): every calib position of every document, layer 0, must reproduce calib's layer-0 Hessians
if [ ! -f $O/chk.done ]; then
  python3 $X/hgen.py $M $META $O/chk --weight all --pools fit,search,qad,verify --layers 0-0 > $O/chk.log 2>&1 || { log "HGEN CHECK FAIL (chk.log)"; exit 1; }
  python3 - $H $O/chk <<'PY' | tee -a $O/result.txt
import glob, os, sys, torch
from safetensors import safe_open
a, b = sys.argv[1:3]; ok, fs = True, sorted(glob.glob(f"{b}/*.input.safetensors"))
for f in fs:                     # the layer-input group (norm of the embedding) is kernel-independent: 1e-6; the others pass
    g = os.path.basename(f)[:-12]                                   # the GDN kernels, whose version may have moved: 1e-3
    with safe_open(f, "pt") as h: x, rx = h.get_tensor("hessian_upper").double(), h.metadata()["rows"]
    with safe_open(f"{a}/{g}.safetensors", "pt") as h: y, ry = h.get_tensor("hessian_upper").double(), h.metadata()["rows"]
    r = float((x - y).norm() / y.norm()); lim = 1e-6 if g.endswith("layers.0.input") else 1e-3; ok &= rx == ry and r <= lim
    print(f"hgen gate: {g}: rows {rx} vs {ry}, rel diff {r:.1e} (limit {lim:.0e})")
ok &= len(fs) == 4
print(f"hgen gate -> {'PASS' if ok else 'FAIL'}"); sys.exit(0 if ok else 1)
PY
  [ ${PIPESTATUS[0]} = 0 ] || { log "HGEN GATE FAIL: positions or arithmetic differ from hcollect.py's"; exit 1; }
  touch $O/chk.done
fi
# 1. the unscored-position sums (~5 min), the weighted Hessians
[ -f $O/ng/model.language_model.layers.23.mlp.down_proj.input.safetensors ] || \
  python3 $X/hgen.py $M $META $O/ng --weight ng --pools fit,search,qad --exclude $F,$S > $O/ng.log 2>&1 || { log "HGEN FAIL (ng.log)"; exit 1; }
[ $(ls $O/gh/w0.3/*.safetensors 2>/dev/null | wc -l) = 96 ] || python3 $X/mkgenh.py $H $O/ng $O/gh 0,0.3 | tee -a $O/result.txt
for k in 3 2; do
  # 2. fit-slice screen at seed 0
  ev k${k}base $Y/q-k${k}base/wq $F .fit
  for a in g0 g3; do q k${k}$a $k --hin $O/gh/w${W0[$a]}; ev k${k}$a $O/q-k${k}$a/wq $F .fit; done
  best=$(python3 $D7/summ.py pick $O $k g0,g3 2>> $O/result.txt); [ -n "$best" ] || { log "PICK FAIL K$k"; exit 1; }
  log "K$k: fit-slice pick $best (w0 ${W0[$best]})"
  # 3. 3 seeds on search
  for s in 0 1 2; do
    t=k${k}$best$(sfx $s); [ $s = 0 ] || q $t $k --hin $O/gh/w${W0[$best]} --seed-offset $s
    ev $t $O/q-$t/wq $S ""
    [ -f $Y/k${k}base$(sfx $s).f32 ] || ev k${k}base$(sfx $s) $Y/q-k${k}base$(sfx $s)/wq $S ""
  done
  python3 $D7/summ.py verdict $O $S $Y $k $best mean 0.97 | tee -a $O/result.txt
done
log GENH DONE
