#!/bin/bash
# cpucheck.sh -- the pre-GPU predictors in cpucheck.txt: refitq.py --dry and swiglufit.py --dry on the 4070 box's data,
# CPU only (CUDA_VISIBLE_DEVICES= nice -n 19, 4 threads, < 10 min each; the scripts go over stdin, nothing is written
# there). B1: per-family change of the one-sided and h-weighted proxy errors; C10: SwiGLU-output residual, in-sample
# and fitted on even / scored on odd documents (56 fit-pool docs capped at 1024 tokens; GDN on the torch path).
H=$(cd $(dirname $0) && pwd) S=/workspace/rad-exl3-mix/local/bin/ssh-4070.sh
bundle() { python3 - $H/fsutil.py $H/$1 "$2" <<'PY'
import sys
fs, sc, pre = open(sys.argv[1]).read(), open(sys.argv[2]).read(), sys.argv[3]
print(f"import os, sys, types\n{pre}\nfs = types.ModuleType('fsutil'); exec(compile({fs!r}, 'fsutil.py', 'exec'), fs.__dict__); sys.modules['fsutil'] = fs")
print("import torch; torch.set_num_threads(4)")
print(f"exec(compile({sc!r}, 'x.py', 'exec'), {{'__name__': '__main__', '__file__': '/tmp/x.py'}})")
PY
}
run() { $S "cd /tmp && CUDA_VISIBLE_DEVICES= timeout 590 nice -n 19 python3 - $*"; }
for c in "q-k3bdiag --rounds 6 --undamped --hout /data/yb/dv8192/hout" "q-k3bdiag --rounds 6 --undamped" \
         "q-k2bdiag --rounds 6 --undamped --hout /data/yb/dv8192/hout"; do
  set -- $c; s=$1; shift; echo "== B1 $s $*"
  bundle refitq.py "" | run /data/yb/$s /tmp/none $* --dry --device cpu 2>&1
done
for s in k3base k2base; do echo "== C10 $s"
  bundle swiglufit.py "os.environ['P2B_BLOCK'] = 'causal_conv1d,fla'" | run /data/models/qwen35-08b /data/yb/q-$s /data/g08-fit1 \
    /tmp/x /tmp/y --device cpu --dry --maxdocs 56 --cap 1024 --grid 0.8:1.2:0.01 2>&1 | grep -E "^L|done"
done
