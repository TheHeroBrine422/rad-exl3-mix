#!/bin/bash
# yaqaFact4.sh -- #33-13 0.8B part (YAQA08.md amendment 6): dose of the diagonal H_out, and the hout_diag file path.
P=/workspace/rad-exl3-mix/proxy2b M=/data/models/qwen35-08b O=/data/yb G=/data/g08-search1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date -u +%T) $*" | tee -a $O/result-fact4.txt; }
q() { tag=$1; k=$2; shift 2
  [ -f $O/$tag.f32 ] && return
  [ $(ls $O/q-$tag/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || $P/q2b.sh 08 $k i8 $O/q-$tag "$@" > $O/q-$tag.log 2>&1
  log "$tag: $(python3 $P/evalq.py $M $G --wq $O/q-$tag/wq --out $O/$tag.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
pair() { log "pair $1 -> $2: $(python3 $P/pcmp.py $G $O/$1.f32 $O/$2.f32 | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"; }
for n in 1024 2048 8192; do [ -f $O/dv$n/.done ] || python3 - $n <<'PY'
import glob, os, sys, torch
from safetensors import safe_open
from safetensors.torch import save_file
n = sys.argv[1]; src, dst = f"/data/yb/snap{n}/hout", f"/data/yb/dv{n}/hout"; os.makedirs(dst, exist_ok=True)
for f in glob.glob(f"{src}/*.safetensors"):
    with safe_open(f, "pt") as h: d = h.get_tensor("hout").diagonal().contiguous(); meta = h.metadata()
    save_file({"hout_diag": d}, f"{dst}/{os.path.basename(f)}", metadata=meta)
open(f"/data/yb/dv{n}/.done", "w").close()
PY
done
q k3bdv8192 3 --hout $O/dv8192/hout --sigma-reg-out 0.01; pair k3bdiag k3bdv8192
for n in 1024 2048; do for k in 3 2; do
  q k${k}bdiag$n $k --hout $O/dv$n/hout --sigma-reg-out 0.01; pair k${k}base k${k}bdiag$n; pair k${k}bdiag k${k}bdiag$n
done; done
log FACT4 DONE
