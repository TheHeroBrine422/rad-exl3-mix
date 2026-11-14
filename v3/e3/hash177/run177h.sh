#!/bin/bash
# run177h.sh STEP ... -- #177 head bench on ds8 (RULES-177h.md, w139): restartable, finished units skipped. Locks as
# run122 ($LK/card<c>.lock per card, $LK/ram.lock shared). Stop: touch $W/STOP (checked between units).
#   run177h.sh setup                 own tree $W/code = w119's donor tree + ourq-hash.patch + exl3hashm; maps (mkmap.py)
#                                    (needs $W/in/{ourq-hash.patch,exl3hashm/} copied from the repo)
#   run177h.sh worker "CARDS" QUEUE head quants from QUEUE lines "NAME K CODEBOOK MAP [SEED]" (MAP: - | hg | hc02;
#                                    SEED: ourq --seed-offset, default 0), R16 order;
#                                    each unit takes the first free card of CARDS (polls every 20 s; BLOCK=1: waits on
#                                    the first card's lock, queued like the other users' blocking flocks)
#   run177h.sh gate                  RULES gate: $W/q/gate-i8/K4 vs w122's q/R16s003/K4, every tensor
#   run177h.sh score "CARDS" SET     score119 on w119's dump SET for every donor of ours not yet scored (first free card)
set -u
Q=/data/q27 W=/data/q27/w139h LK=${LK:-/data/q27/locks}
W1=/data/q27/w119      # w119's donor tree: ds8 = a symlink to w122's copy (/data/q27/heads/w119), box2 = w119's own dir
if [ -f $Q/tools/env-ds8.sh ]; then source $Q/tools/env-ds8.sh; HD=/data/q27/heads
  [ "$(readlink $W1)" = $HD/w119 ] || { echo "$W1 is not w122's symlink"; exit 1; }
else source $Q/tools/env-box2.sh; HD=$W/hd; fi    # box2: w122's control scores copied to $W/hd/scores/{fit,search}
export PYTHONPATH=$W1/exl3:$Q/extbase TORCH_EXTENSIONS_DIR=$W/build EXL3HASH_A=0xd0474b EXL3HASH_B=0x2545F491
C=$W/code; CK=$Q/models/qwen38-27b-bf16; SRC=$W1/exl3/exllamav3/exllamav3_ext
export RXM_WORK=$Q RXM_CKPT=$CK RXM_TEMPLATE=$W1/in/template-config.json HS_CKPT=$CK
mkdir -p $W/logs
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $W/run177h.log; }
stop() { [ -f $W/STOP ] && { log "STOP"; exit 0; }; }
waitmem() { while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt ${1:-30} ]; do sleep 20; done; }
step=$1; shift
case $step in
setup)
  if [ ! -f $C/.ok ]; then
    rm -rf $C; mkdir -p $C
    (cd $W1/code && tar c --exclude build --exclude __pycache__ .) | (cd $C && tar x)
    (cd $C && patch -p1 < $W/in/ourq-hash.patch) || { log "PATCH FAIL"; exit 1; }
    mkdir -p $C/v3/e3/hash177 && cp -a $W/in/exl3hashm $C/v3/e3/hash177/ && touch $C/.ok
  fi
  mkdir -p $W/maps
  $PY $C/v3/e3/hash177/exl3hashm/mkmap.py hg $EXL3HASH_A $EXL3HASH_B $W/maps/hg.npy
  $PY $C/v3/e3/hash177/exl3hashm/mkmap.py hc:0.02 $EXL3HASH_A $EXL3HASH_B $W/maps/hc02.npy
  log "setup done"
  ;;
worker)
  cards=$1 qf=$2
  while read -r name k cb map seed; do
    [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
    d=$W/q/$name/K$k
    [ -f $d/wq/lm_head.rowerr.npz ] && continue
    mkdir -p $W/q/$name; mkdir $d.claim 2>/dev/null || continue
    stop; waitmem 30
    mp=""; [ "$map" != "-" ] && mp=$W/maps/$map.npy
    ( if [ "${BLOCK:-0}" = 1 ]; then c=${cards%% *}; exec 5>$LK/card$c.lock; flock 5      # BLOCK=1: queue on the first card
      else while :; do for c in $cards; do exec 5>$LK/card$c.lock; flock -n 5 && break 2; exec 5>&-; done; sleep 20; done; fi
      card=$c; log "quant $name K$k ($cb, map ${map}) card $card start"
      exec 7>$LK/ram.lock; flock -s 7; cd $C
      EXL3HASH_MAP=$mp ROCR_VISIBLE_DEVICES=$card nice -n 5 $PY $C/quantize/ourq.py -K $k -o $d --only-head --device 0 --save-wq --seed-offset ${seed:-0} \
        --head-perm $W1/perms/R16.npy --qtf2 --frac-ext $SRC --scaled-hessian --sigma-reg 0.025 --codebook $cb \
        --hout $W1/perms/hd-R16 --hout-blocks group --sigma-reg-out 0.03 < /dev/null
      if [ -f $d/wq/lm_head.rowerr.npz ]; then        # score on the same card before releasing it (fit-gen, search-gen)
        for set in fit search; do mkdir -p $W/scores/$set
          for f in ref.npz R16s003-K2.5.npz R16s003-K4.npz R16s003-K6.npz; do [ -f $W/scores/$set/$f ] || cp $HD/scores/$set/$f $W/scores/$set/; done
          [ -f $W/scores/$set/$name-K$k.npz ] || ROCR_VISIBLE_DEVICES=$card $PY -I $C/v3/yaqa/w119/score119.py $W1/dump/$set $CK $W/scores/$set \
            "$name-K$k=exl3:wq=$d/wq/lm_head.safetensors,perm=$W1/perms/R16.npy" >> $W/logs/score-$set.log 2>&1
        done
      fi ) > $d.log 2>&1
    if [ -f $d/wq/lm_head.rowerr.npz ]; then log "quant $name K$k OK: $(tail -1 $d/proxy.jsonl 2>/dev/null | cut -c1-140)"
    else log "quant $name K$k FAIL ($d.log)"; rmdir $d.claim; fi
  done < $qf
  log "worker cards $cards: queue $qf done"
  ;;
gate)
  $PY - $W/q/gate-i8/K4 $HD/q/R16s003/K4 <<'EOF' | tee -a $W/run177h.log
import sys, torch
from safetensors import safe_open
a, b = sys.argv[1:3]; n = bad = 0
for f in ("q-head.safetensors", "wq/lm_head.safetensors"):
    with safe_open(f"{a}/{f}", "pt") as x, safe_open(f"{b}/{f}", "pt") as y:
        if sorted(x.keys()) != sorted(y.keys()): print("KEYS DIFFER", f); bad += 1; continue
        for k in x.keys():
            n += 1; u, v = x.get_tensor(k).contiguous(), y.get_tensor(k).contiguous()
            if u.dtype != v.dtype or u.shape != v.shape or not u.view(-1).view(torch.uint8).equal(v.view(-1).view(torch.uint8)): print("DIFF", f, k); bad += 1
print(f"GATE tensors {n}, differ {bad} ->", "SAME" if n and not bad else "FAIL")
EOF
  ;;
score)
  card=$1 set=$2; sp=(); mkdir -p $W/scores/$set     # card: a list, the first free one is used
  for f in ref.npz R16s003-K2.5.npz R16s003-K4.npz R16s003-K6.npz; do [ -f $W/scores/$set/$f ] || cp $HD/scores/$set/$f $W/scores/$set/; done
  for d in $W/q/*/K*; do
    [ -f $d/wq/lm_head.rowerr.npz ] || continue
    n=$(basename $(dirname $d))-$(basename $d)
    [ -f $W/scores/$set/$n.npz ] || sp+=("$n=exl3:wq=$d/wq/lm_head.safetensors,perm=$W1/perms/R16.npy")
  done
  [ ${#sp[@]} -eq 0 ] && { log "score $set: nothing new"; exit 0; }
  ( if [ "${BLOCK:-0}" = 1 ]; then c=${card%% *}; exec 5>$LK/card$c.lock; flock 5
    else while :; do for c in $card; do exec 5>$LK/card$c.lock; flock -n 5 && break 2; exec 5>&-; done; sleep 20; done; fi
    exec 7>$LK/ram.lock; flock -s 7
    ROCR_VISIBLE_DEVICES=$c $PY -I $C/v3/yaqa/w119/score119.py $W1/dump/$set $CK $W/scores/$set "${sp[@]}" ) >> $W/logs/score-$set.log 2>&1
  log "score $set rc $? (${#sp[@]} options)"
  ;;
esac
