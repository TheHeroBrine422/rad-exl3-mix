#!/bin/bash
# run151h.sh STEP ... -- (ds8) #177-03 R16 head donors with the mad codebook: w122's recipe (run122.sh worker: R16 perm,
# H_out hd-R16 blocks group, sigma_out 0.03, scaled Hessian, sigma 0.025, qtf2, --save-wq) run by the STAGED tree's ourq
# (rad-exl3-mix 0ce451c: --codebook mad, A / B as v3.toml) instead of w119's copy. Restartable: outputs .part / claims by
# mkdir, finished units skipped (the rowerr file is written last). Locks: card<c>.lock + ram.lock shared, MemAvailable >= 30 GiB.
#   run151h.sh gate CARD          identity: K4 i8 with the new ourq -> q/w151chk-i8/K4; tensor bytes of q-head, wq and rowerr
#                                 must equal q/R16s003/K4 (w122's; the q-head metadata stamp differs: ourq.py changed) -> W/gate151.ok
#   run151h.sh worker CARD QUEUE  donors from QUEUE lines "NAME K SIGMA_OUT [CODEBOOK]" (needs gate151.ok)
#   run151h.sh build QUEUE CARDS  one worker a card (never HIP 7), waits for all
#   run151h.sh score CARD         run122.sh score CARD fit + search (every unscored R16s003* donor: the mad ones are R16s003mad)
#   run151h.sh drop               wq/lm_head.safetensors of mad donors scored on both sets (rowerr and q-head stay)
# Stop: touch /data/q27/heads/STOP. Process name `bash /data/q27/mad/tools/run151h.sh`. (w151)
set -u
Q=/data/q27 W=/data/q27/heads LK=/data/q27/locks; source $Q/tools/env-ds8.sh
W1=/data/q27/w119; [ "$(readlink $W1)" = $W/w119 ] || { echo "$W1 is not w122's symlink to $W/w119"; exit 1; }
C=$Q/stage/code; SRC=$Q/stage/exl3/exllamav3/exllamav3_ext; CK=$Q/models/qwen38-27b-bf16
export PYTHONPATH=$Q/stage/exl3:$Q/extbase TORCH_EXTENSIONS_DIR=$Q/mad/hbuild
export RXM_WORK=$Q RXM_CKPT=$CK RXM_TEMPLATE=$W1/in/template-config.json HS_CKPT=$CK
mkdir -p $W/logs $Q/mad/hbuild
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $W/run151h.log; }
waitmem() { while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt ${1:-30} ]; do sleep 20; done; }
quant() {   # quant CARD NAME K SIGMA CB
  local card=$1 name=$2 k=$3 sig=$4 cb=$5 d=$W/q/$2/K$3
  mkdir -p $W/q/$name; mkdir $d.claim 2>/dev/null || return 0
  [ -f $W/STOP ] && { rmdir $d.claim; log "STOP"; exit 0; }
  waitmem 30; log "quant $name K$k (R16, sigma_out $sig, $cb, staged ourq) card $card start"
  ( exec 5>$LK/card$card.lock; flock 5; exec 7>$LK/ram.lock; flock -s 7; cd $W1
    ROCR_VISIBLE_DEVICES=$card nice -n 5 $PY $C/quantize/ourq.py -K $k -o $d --only-head --device 0 --save-wq \
      --head-perm $W1/perms/R16.npy --qtf2 --frac-ext $SRC --scaled-hessian --sigma-reg 0.025 --codebook $cb \
      --hout $W1/perms/hd-R16 --hout-blocks group --sigma-reg-out $sig < /dev/null ) > $d.log 2>&1
  if [ -f $d/wq/lm_head.rowerr.npz ]; then log "quant $name K$k OK: $(tail -n 1 $d/proxy.jsonl 2>/dev/null | cut -c1-120)"
  else log "quant $name K$k FAIL ($d.log)"; rmdir $d.claim; fi
}
step=$1; shift
case $step in
gate)
  [ -f $W/gate151.ok ] && { log "gate151: passed earlier"; exit 0; }
  quant $1 w151chk-i8 4 0.03 i8
  $PY - $W/q/w151chk-i8/K4 $W/q/R16s003/K4 <<'EOF' && { touch $W/gate151.ok; log "GATE151 PASS"; } || { touch $W/gate151.fail; log "GATE151 FAIL"; exit 1; }
import sys, numpy as np, torch
from safetensors import safe_open
a, b = sys.argv[1:3]; bad = n = 0
for f in ("q-head.safetensors", "wq/lm_head.safetensors"):
    with safe_open(f"{a}/{f}", "pt") as x, safe_open(f"{b}/{f}", "pt") as y:
        ka, kb = sorted(x.keys()), sorted(y.keys()); assert ka == kb, (f, ka, kb)
        for k in ka:
            n += 1; ta, tb = x.get_tensor(k), y.get_tensor(k)
            if ta.dtype != tb.dtype or ta.shape != tb.shape or not ta.contiguous().view(-1).view(torch.uint8).equal(tb.contiguous().view(-1).view(torch.uint8)): print("DIFF", f, k); bad += 1
za, zb = np.load(f"{a}/wq/lm_head.rowerr.npz"), np.load(f"{b}/wq/lm_head.rowerr.npz")
for k in zb.files:
    n += 1
    if not np.array_equal(za[k], zb[k]): print("DIFF rowerr", k); bad += 1
print(f"gate151: {n} tensors / arrays, {bad} differ"); sys.exit(1 if bad else 0)
EOF
  ;;
worker)
  [ -f $W/gate151.ok ] || { log "no gate151.ok: run the gate first"; exit 1; }
  card=$1 qf=$2; [ $card = 7 ] && { log "HIP 7 is w153's"; exit 1; }
  while read -r name k sig cb; do
    [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
    [ -f $W/q/$name/K$k/wq/lm_head.rowerr.npz ] && continue
    quant $card $name $k $sig ${cb:-mad}
  done < $qf
  log "worker card $card: queue $qf done"
  ;;
build)
  qf=$1; shift; for c in $@; do [ $c = 7 ] && continue; bash $0 worker $c $qf > $W/logs/w151-c$c.out 2>&1 & done; wait
  log "build $qf: $(awk '{print $1 "/K" $2}' $qf | while read u; do [ -f $W/q/$u/wq/lm_head.rowerr.npz ] && echo; done | wc -l) / $(grep -c . $qf) built"
  ;;
score)
  bash $W/code/v3/yaqa/w122/run122.sh score $1 fit; bash $W/code/v3/yaqa/w122/run122.sh score $1 search
  ;;
drop)
  k=0; for d in $W/q/R16s003mad/K*; do [ -d $d ] || continue; n=R16s003mad-$(basename $d); f=$d/wq/lm_head.safetensors
    [ -f $f ] && [ -f $W/scores/fit/$n.npz ] && [ -f $W/scores/search/$n.npz ] && [ -f $d/q-head.safetensors ] && { rm -f $f; k=$((k + 1)); }; done
  log "wq dropped for $k mad donor(s); $(df -h /data | awk 'NR == 2 {print $4}') free"
  ;;
esac
