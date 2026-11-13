#!/bin/bash
# quant144.sh QUEUE -- (w144, #174-02, box2 / ds8) trellis token tables with our quantizer: ourq.py's head path on the token
# table (prep144.py: the table as lm_head.weight, identity input Hessian = plain row error, as w135 priced it), the
# head donors' settings without H_out (--qtf2 --frac-ext <staged exllamav3> --scaled-hessian --sigma-reg 0.025
# --codebook CB --save-wq), one level a job, QUEUE lines "NAME K CB ORDER" (CB i8 | mul1, ORDER natural | r16).
# Output /data/q27/emb/trellis/NAME/K<k>/ (+ src/ for the importer, embsrc.py; row error -> W/rel.jsonl).
# The first free card (HIP index = ROCR index) by polling its lock, ram.lock shared, MemAvailable >= 30 GB; one job at a time; a
# level is claimed by mkdir (O/NAME/K<k>.claim, also visible to the other host's worker only through its own disk:
# split the queue between hosts). Host by `hostname`-free test: env-ds8.sh present -> ds8 (cards 0-7), else box2 (0-3).
# Restartable: a level with src/q-embed.safetensors is skipped; a killed one reruns (ourq resumes nothing).
# Stop: touch W/STOP.
set -u
Q=/data/q27 W=/data/q27/w144 O=/data/q27/emb/trellis LK=/data/q27/locks
if [ -f $Q/tools/env-ds8.sh ]; then source $Q/tools/env-ds8.sh; CARDS="0 1 2 3 4 5 6 7"; else source $Q/tools/env-box2.sh; CARDS="0 1 2 3"; fi
export RXM_WORK=$W/qin/work RXM_CKPT=$W/qin/ckpt RXM_TEMPLATE=$Q/w119/in/template-config.json
log() { echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S CST') $*" | tee -a $W/logs/quant144.log; }
waitmem() { while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt 30 ]; do sleep 20; done; }
mkdir -p $O $W/logs
rm -rf $O/*/K*.claim                             # one worker a host: claims left by a killed run
while read -r name k cb ord; do
  [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
  [ -f $W/STOP ] && { log "STOP"; exit 0; }
  d=$O/$name/K$k; [ -f $d/src/q-embed.safetensors ] && continue
  mkdir -p $O/$name; mkdir $d.claim 2>/dev/null || continue
  perm=(); [ "$ord" = r16 ] && perm=(--head-perm $Q/w119/perms/R16.npy)
  waitmem
  while :; do                          # the first free card by polling its lock, held on fd 9 until the end
    got=
    for cc in $CARDS; do exec 9>$LK/card$cc.lock; if flock -n 9; then got=$cc; break; fi; exec 9>&-; done
    [ -n "$got" ] && break; sleep 2
  done
  rm -rf $d; mkdir -p $d; t=$(date +%s)
  log "quant $name K$k ($cb, $ord) card $got start"
  ( exec 7>$LK/ram.lock; flock -s 7
    ROCR_VISIBLE_DEVICES=$got nice -n 5 $PY $Q/stage/code/quantize/ourq.py -K $k -o $d --only-head --device 0 --save-wq \
      --qtf2 --frac-ext $RXQ_EXL3_SRC --scaled-hessian --sigma-reg 0.025 --codebook $cb "${perm[@]}" < /dev/null ) > $d.log 2>&1
  exec 9>&-
  if [ -f $d/wq/lm_head.rowerr.npz ] && $PY -I $W/embsrc.py $d >> $W/rel.jsonl 2>> $d.log; then
    log "quant $name K$k OK $(( $(date +%s) - t )) s: $(tail -n 1 $W/rel.jsonl)"
  else log "quant $name K$k FAIL ($d.log)"; rmdir $d.claim; fi
done < $1
log "queue $1 done"
