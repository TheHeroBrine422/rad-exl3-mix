#!/bin/bash
# yaqaFact.sh -- #33-10, factorial of the YAQA arms on the 0.8B at K3 (YAQA08.md amendment 3). ~10-15 min on the 4070.
P=/workspace/rad-exl3-mix/proxy2b M=/data/models/qwen35-08b O=/data/yb G=/data/g08-search1 S=/data/yb/snap8192
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date -u +%T) $*" | tee -a $O/result-fact.txt; }
q() { tag=$1; k=$2; shift 2
  [ -f $O/$tag.f32 ] && return
  [ $(ls $O/q-$tag/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || $P/q2b.sh 08 $k i8 $O/q-$tag "$@" > $O/q-$tag.log 2>&1
  log "$tag: $(python3 $P/evalq.py $M $G --wq $O/q-$tag/wq --out $O/$tag.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
P01="--sigma-reg 0.01 --sigma-reg-in 0.01 --sigma-reg-out 0.01"
# diag(H_out) as a full matrix with zero off-diagonal
[ -f $O/snap8192diag/.done ] || python3 - <<'PY'
import glob, os, torch
from safetensors import safe_open
from safetensors.torch import save_file
src, dst = "/data/yb/snap8192/hout", "/data/yb/snap8192diag/hout"; os.makedirs(dst, exist_ok=True)
for f in glob.glob(f"{src}/*.safetensors"):
    with safe_open(f, "pt") as h: H = h.get_tensor("hout"); meta = h.metadata()
    save_file({"hout": torch.diag(H.diagonal()).contiguous()}, f"{dst}/{os.path.basename(f)}", metadata=meta)
open("/data/yb/snap8192diag/.done", "w").close()
PY
NOSH=1 q k3hin 3 $P01 --hin $S/hin
NOSH=1 q k3hout 3 $P01 --hout $S/hout
NOSH=1 q k3hdiag 3 $P01 --hout $O/snap8192diag/hout
for a in hin hout hdiag yb8192; do for ref in ldlq base; do
  log "pair k3$ref -> k3$a: $(python3 $P/pcmp.py $G $O/k3$ref.f32 $O/k3$a.f32 | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"; done; done
log FACT DONE
