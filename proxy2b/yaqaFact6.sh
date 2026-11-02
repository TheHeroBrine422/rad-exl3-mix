#!/bin/bash
# yaqaFact6.sh -- #33-16 (YAQA08.md amendment 9): seed replicates of base / bdiag, and the identity H_out.
P=/workspace/rad-exl3-mix/proxy2b M=/data/models/qwen35-08b O=/data/yb G=/data/g08-search1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
log() { echo "$(date -u +%T) $*" | tee -a $O/result-fact6.txt; }
q() { tag=$1; k=$2; shift 2
  [ -f $O/$tag.f32 ] && return
  [ $(ls $O/q-$tag/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || $P/q2b.sh 08 $k i8 $O/q-$tag "$@" > $O/q-$tag.log 2>&1
  log "$tag: $(python3 $P/evalq.py $M $G --wq $O/q-$tag/wq --out $O/$tag.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
pair() { log "pair $1 -> $2: $(python3 $P/pcmp.py $G $O/$1.f32 $O/$2.f32 | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"; }
cd /tmp
[ -f $O/ident/.done ] || python3 - <<'PY'
import glob, os, torch
from safetensors import safe_open
from safetensors.torch import save_file
src, dst = "/data/yb/dv8192/hout", "/data/yb/ident/hout"; os.makedirs(dst, exist_ok=True)
for f in glob.glob(f"{src}/*.safetensors"):
    with safe_open(f, "pt") as h: d = h.get_tensor("hout_diag")
    save_file({"hout_diag": torch.ones_like(d)}, f"{dst}/{os.path.basename(f)}", metadata={"sketch": "identity"})
open("/data/yb/ident/.done", "w").close()
PY
for k in 3 2; do
  q k${k}bident $k --hout $O/ident/hout --sigma-reg-out 0.01; pair k${k}base k${k}bident; pair k${k}bdiag k${k}bident
done
for s in 1 2; do for k in 3 2; do
  q k${k}base_s$s $k --seed-offset $s
  q k${k}bdiag_s$s $k --hout $O/dv8192/hout --sigma-reg-out 0.01 --seed-offset $s
  pair k${k}base_s$s k${k}bdiag_s$s; pair k${k}base k${k}base_s$s
done; done
log FACT6 DONE
