#!/bin/bash
# smoke1.sh -- d1109 implementation checks (run under the GPU lock, < 15 min): Fisher speed / memory (bf16, fp32),
# the identity gate (realq --gd-off == q2b.sh on layers 0-1, with H_out), REAL-Q GD on layers 0-1 (timing, losses).
P=/workspace/rad-exl3-mix/proxy2b X=$P/exp/d1109 M=/data/models/qwen35-08b SM=/data/yb/d1109/smoke
HO="--hout /data/yb/c18/hdiag-snap1024-dy --sigma-reg-out 0.01"
export VRAM_CAP_GB=7.2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $SM; cd /tmp
STEP=${1:-all}
if [[ $STEP == all || $STEP == fisher ]]; then
  python3 $X/fisher.py $M $SM/f8 8 2>&1 | grep -v Warn | tail -3
  python3 $X/fisher.py $M $SM/f8fp32 8 --fp32 2>&1 | grep -v Warn | tail -3
fi
if [[ $STEP == all || $STEP == gate ]]; then
  rm -rf $SM/q-plain $SM/q-off
  $P/q2b.sh 08 3 i8 $SM/q-plain --layers 0-1 $HO > $SM/q-plain.log 2>&1; tail -2 $SM/q-plain.log
  FISHER=$SM/f8/fisher.pt $X/rq.sh 3 i8 $SM/q-off --gd-off --ncal 32 -- --layers 0-1 $HO > $SM/q-off.log 2>&1; tail -3 $SM/q-off.log
  python3 - $SM <<'EOF'
import sys, glob, os, torch
from safetensors.torch import load_file
sm = sys.argv[1]; bad = 0; n = 0
for f in sorted(glob.glob(f"{sm}/q-plain/wq/*.safetensors")) + sorted(glob.glob(f"{sm}/q-plain/q-L*.safetensors")):
    g = f.replace("/q-plain/", "/q-off/"); a, b = load_file(f), load_file(g)
    for k in a:
        n += 1
        if not torch.equal(a[k], b[k]): bad += 1; print("DIFF", os.path.basename(f), k, float((a[k].float() - b[k].float()).abs().max()))
print(f"GATE {'PASS' if bad == 0 and n > 0 else 'FAIL'}: {n} tensors compared, {bad} differ")
EOF
fi
if [[ $STEP == all || $STEP == gd ]]; then
  rm -rf $SM/q-gd
  FISHER=$SM/f8/fisher.pt $X/rq.sh 3 i8 $SM/q-gd --ncal ${NCAL:-64} ${GDARGS} -- --layers 0-1 $HO > $SM/q-gd.log 2>&1; tail -4 $SM/q-gd.log
  python3 -c "
import json
for l in open('$SM/q-gd/realq.jsonl'):
    r = json.loads(l); print(r['key'][28:], 'deq %.1e' % (r['deq_err'] or 0), r['steps'], r['btot'], 'lr', r['lr'], 'lr_r %.2e' % (r['lr_r'] or 0), 'g0 %.2e' % (r['g_rms0'] or 0), 'dW/W %.4f' % (r['delta_rms'] / max(r['w_rms'] or 1e-30, 1e-30)), 'gd', r['gd_sec'], 's', r['sec'], 'mem', r['max_mem_gib'], 'loss', [round(x, 6) for x in r['losses'][:3]], '..', [round(x, 6) for x in r['losses'][-2:]])
"
fi
