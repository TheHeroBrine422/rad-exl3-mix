#!/bin/bash
# cardzero.sh -- card 0 for the rest of the night, one job at a time, card0.lock held throughout (no interleaving).
#  1. end-to-end test on the new integer K8 head (quantize, small plan, headsim); stops on failure
#  2. A4 timing rerun with the rebuilt ntbench2 (decision input for the int8 kernel)
#  3. the slab-head curve from integer donors K3-K8 (33 budgets, one headsim pass)
#  4. starts card 0's ladder share. Card 1's ladder is paused during head quants (host RAM).
: ${RXM_WORK:=/root}; export RXM_WORK
Y=/data/yaqa H=/data/yaqa/headslab2 PY=/data/venv-exl3/bin/python B=/data/bench DUMP=${RXM_WORK}/e1b/dumphead
CK=/data/workspace/models/qwen38-27b-bf16 SO=/root/nfk/home/kernels/exl3_r4d.so
log() { echo "$(date -u +%H:%M) $*" | tee -a $H/card0.log; }
touch /data/yaqa/headcurve.pending; trap 'rm -f /data/yaqa/headcurve.pending' EXIT
cd ${RXM_WORK}
exec 9>${RXM_WORK}/card0.lock; flock 9; log "card 0 held"
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/exl3frac/build
# ---- card 1's ladder pauses during head quants: a head quant needs ~22 GB host RAM and OOMs next to a ladder
# process (3 oom kills, 2026-11-01). ourq.py resumes at the unfinished layer, so a pause costs <= one layer (~3 min).
L1="2.75 3.5 3 4 4.5 2 5.5"
ladder1_stop() {
  pkill -f "^/bin/bash /data/v3levels/levels.sh 1 "; sleep 1
  pkill -f "^/data/venv-exl3/bin/python -I ourq/ourq.py -K [0-9.]* -o /data/v3levels/K"
  while pgrep -f "^/data/venv-exl3/bin/python -I ourq/ourq.py -K [0-9.]* -o /data/v3levels/K" > /dev/null; do sleep 2; done
  sleep 5; log "card 1 ladder paused"
}
ladder1_start() {
  (setsid nohup /data/v3levels/levels.sh 1 "$L1" > /dev/null 2>&1 < /dev/null 9>&- &); log "card 1 ladder resumed"   # 9>&-: never hand it card 0's lock
}
# ---- head quant helper (RAM lock exclusive: ~22 GB)
hq() {
  [ -f $H/K$1/wq/lm_head.rowerr.npz ] && return 0
  ( exec 7>${RXM_WORK}/ram.lock; flock 7
    $PY -I ourq/ourq.py -K $1 -o $H/K$1 --only-head --device 0 --save-wq --head-perm $H/perm.npy \
      --frac-ext /data/exl3frac/src ) > $H/K$1.log 2>&1
  [ -f $H/K$1/wq/lm_head.rowerr.npz ] && log "K$1 head quantized: $(tail -c 300 $H/K$1/proxy.jsonl | grep -o '"proxy_err": [0-9.e-]*')" \
    || { log "K$1 head FAIL: $(grep -v -e Warning -e '^\s*$' $H/K$1.log | tail -n 3 | tr '\n' ' ')"; return 1; }
}
sim() {  # PLAN OUT
  ( exec 7>${RXM_WORK}/ram.lock; flock -s 7; $PY -I $Y/headsim.py $DUMP $H $H/perm.npy $1 $CK $2 ) > $2.log 2>&1
}
# ---- 1. end to end on the new K8 head (integer donors only: a fractional head needs > 26 GB host RAM and was
#      OOM-killed twice on the 30 GB box, 2026-11-01): quantize, a small plan, headsim
export HS_TRAIN=qad HS_FLOOR=3
ladder1_stop; hq 8; rc=$?; ladder1_start
[ $rc = 0 ] || { log "E2E FAIL at the K8 quant: stop (card 0 left idle for debugging)"; trap - EXIT; exit 1; }
HS_KS=3,4,5,6,7,8 HS_BUDGETS=4.75 $PY -I $Y/headslab.py plan $Y/hs/headsens.npz $H/perm.npy $H $H/e2e-plan.json >> $H/card0.log 2>&1 \
  || { log "E2E FAIL at the plan: stop"; trap - EXIT; exit 1; }
sim $H/e2e-plan.json $H/e2e.json
if [ -s $H/e2e.json ]; then log "E2E OK: $(python3 -c "
import json; p = json.load(open('$H/e2e.json'))['plans']
print({n: (round(v['avg_bits'], 3), f\"{v['kl']:.5f}\", None if 'rel' not in v else f\"{v['rel']:+.1%}\") for n, v in p.items()})")"
else log "E2E FAIL at headsim (see e2e.json.log): stop"; trap - EXIT; exit 1; fi
# ---- 2. A4 rerun (decision input)
a=$($B/ntbench2 $SO 2048 34816 5120 4 5 1 | tail -1); b=$(NFQ_MASK=8 $B/ntbench2 $SO 2048 34816 5120 4 5 1 | tail -1)
log "sanity f16:  $a"; log "sanity nfq8: $b"
ha=$(echo "$a" | grep -o 'out-hash [0-9a-f]*'); hb=$(echo "$b" | grep -o 'out-hash [0-9a-f]*')
if [ -n "$ha" ] && [ -n "$hb" ] && [ "$ha" != "$hb" ]; then
  mv -f $B/a4time.jsonl $B/a4time-invalid.jsonl 2>/dev/null
  NB=$B/ntbench2 $B/a4time.sh $B/a4time.jsonl >> $B/a4time.log 2>&1
  python3 $B/a4report.py $B/a4time.jsonl $B/speedcal-stock.jsonl > $B/a4report.txt 2>&1
  log "A4 rerun: $(grep -c '"ms"' $B/a4time.jsonl) points, report $B/a4report.txt"
else log "A4 rerun SKIPPED: the int8 arm is still not taken (same hash)"; fi
# ---- 3. the curve: integer donors K3-K8, 3-slab plans at every 1/8 bit from K3.5 to K7.5 (33), one headsim pass
bud() { python3 -c "import sys; a, b = map(float, sys.argv[1:3]); print(','.join(f'{a + i / 8:g}' for i in range(int((b - a) * 8) + 1)))" $1 $2; }
HS_KS=3,4,5,6,7,8 HS_BUDGETS=$(bud 3.5 7.5) $PY -I $Y/headslab.py plan $Y/hs/headsens.npz $H/perm.npy $H $H/curve-plan.json >> $H/card0.log 2>&1
sim $H/curve-plan.json $H/curve.json
[ -s $H/curve.json ] && log "curve priced ($(python3 -c "import json; print(len(json.load(open('$H/curve.json'))['plans']))") plans)" \
  || log "curve headsim FAIL (see curve.json.log)"
log "card0 done: releasing card 0 to the ladder"
rm -f /data/yaqa/headcurve.pending; flock -u 9
(setsid nohup /data/v3levels/levels.sh 0 "3.25 3.75 2.5 4.25 2.25 4.75 5 6" > /dev/null 2>&1 < /dev/null 9>&- &)
