#!/bin/bash
# setup4070.sh -- #100c (r100c.md) on the 4070, CPU only: (1) the mirror the hash arms run from, /data/yb/w48/code:
# quantize/ourq.py = the repo's + ourq-hash.patch, v3/e3/{exl3frac,exl3cb,exl3frac2,exl3hash} copies (so their JIT
# builds never share /data/torch_ext with the lead's runs); (2) the prebuild of exl3frac + the hash extensions into
# /data/yb/w48/torch_ext (niced, 4 jobs, no card). Rerun after a code change (rsync / ninja skip what's current).
set -e
R=/workspace/rad-exl3-mix; W=/data/yb/w48; C=$W/code; HERE=$(dirname "$(readlink -f "$0")")
SRC=/data/src/exllamav3/exllamav3/exllamav3_ext
mkdir -p $C/quantize $C/v3/e3 $W/torch_ext /tmp/w48-build
cp $R/quantize/ourq.py $C/quantize/ourq.py.part && patch -s $C/quantize/ourq.py.part $HERE/ourq-hash.patch && mv $C/quantize/ourq.py.part $C/quantize/ourq.py
for d in exl3frac exl3cb exl3frac2 exl3hash; do
  rsync -a --delete --exclude __pycache__ --exclude build --exclude data $R/v3/e3/$d/ $C/v3/e3/$d/; done
cd /tmp/w48-build
CUDA_VISIBLE_DEVICES= TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 OMP_NUM_THREADS=4 PYTHONPATH=/data/src/exllamav3 \
  TORCH_EXTENSIONS_DIR=$W/torch_ext nice -n 19 python3 -c "
import sys; sys.path[:0] = ['$C/v3/e3/exl3frac', '$C/v3/e3/exl3hash']
import exl3frac, exl3hash
exl3frac.load('$SRC'); print('exl3frac built', flush=True)
print('hash builds in', exl3hash.build('$SRC'), flush=True)"
