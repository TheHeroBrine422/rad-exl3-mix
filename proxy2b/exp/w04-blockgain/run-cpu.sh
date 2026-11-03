#!/bin/bash
# run-cpu.sh -- reproduce the CPU analysis (data/) from the VM: stats on the 4070's data (CPU only, 4 threads, niced,
# script on stdin, nothing written there; ~1 min each), predict.py here, downperm.py on the 4070 (~4.5 min).
cd "$(dirname "$0")"; S=../../../local/bin/ssh-4070.sh; mkdir -p data
for K in 3 2; do
  [ -s data/stats08-k$K.npz ] && continue
  W=""; for t in base base_s1 base_s2 bdiag bdiag_s1 bdiag_s2 bident; do W="$W --wq k$K$t=/data/yb/q-k$K$t/wq"; done
  $S "cd /tmp && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 timeout 600 nice -n 19 python3 - /data/models/qwen35-08b /data/yb/dv8192/hout - --hin /data/h08/calib $W" < stats.py > data/stats08-k$K.npz
done
[ -s data/proxy-k3base.jsonl ] || $S 'cat /data/yb/q-k3base/proxy.jsonl' > data/proxy-k3base.jsonl
nice python3 predict.py data/pred08.json data/stats08-k3.npz data/stats08-k2.npz --proxy data/proxy-k3base.jsonl \
  --perms-out data/perms08.npz > data/pred08.txt 2>&1
if [ ! -s data/downperm08.json ]; then             # blk.py + the permutations + downperm.py as one stdin script
  ( cat blk.py; echo "PERMS_B64 = '$(base64 -w0 data/perms08.npz)'"; cat downperm.py ) > /tmp/w04-bundle.py
  $S 'cd /tmp && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 timeout 600 nice -n 19 python3 - /data/models/qwen35-08b /data/h08/calib - -' \
    < /tmp/w04-bundle.py > data/downperm08.json 2> data/downperm08.log
fi
cat data/pred08.txt
