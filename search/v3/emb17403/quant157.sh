#!/bin/bash
# quant157.sh QUEUE -- (w157, #174-03, box2 card 1) the eighth-step trellis token tables with w144's tool and settings
# (quant144.sh: ourq.py's head path on the token table, prep144.py's inputs in /data/q27/w144/qin = the table as
# lm_head.weight + an identity input Hessian; --qtf2 --frac-ext --scaled-hessian --sigma-reg 0.025 --codebook CB
# --save-wq), run from the code tree w144's levels were made with (stage/code-old-c015001: its K2.5 rerun reproduced
# w144's tensors exactly, RULES amendment 1). Each level's stamp.json must list the rerun's sources and w144's arguments
# but K (amendment 2: the stamp hashes paths and K, so it differs per level); else renamed .badstamp, the queue stops.
# QUEUE lines "NAME K CB [OUTROOT]" (OUTROOT default /data/q27/emb/trellis; a K2.5 rerun into W/rerun is the determinism
# check against w144's K2.5). Card 1 only (its lock held per level, ram.lock shared), MemAvailable >= 30 GB.
# Output OUTROOT/NAME/K<k>/ (+ src/ for the importer: embsrc.py; row error -> W/rel.jsonl); the bf16 wq/ copy is deleted
# once rel is written (2.4 GB a level). Restartable: a level with src/q-embed.safetensors is skipped. Stop: touch W/STOP.
set -u
Q=/data/q27 W=/data/q27/w157 LK=/data/q27/locks C=1 CODE=/data/q27/stage/code-old-c015001 REF=/data/q27/w157/rerun/t-i8/K2.5.badstamp/stamp.json
source $Q/tools/env-box2.sh
export RXM_WORK=$Q/w144/qin/work RXM_CKPT=$Q/w144/qin/ckpt RXM_TEMPLATE=$Q/w119/in/template-config.json
log() { echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S CST') $*" | tee -a $W/logs/quant157.log; }
waitmem() { while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt 30 ]; do sleep 20; done; }
mkdir -p $W/logs
while read -r name k cb root; do
  [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
  [ -f $W/STOP ] && { log "STOP"; exit 0; }
  root=${root:-$Q/emb/trellis}; d=$root/$name/K$k
  [ -f $d/src/q-embed.safetensors ] && continue
  mkdir -p $root/$name; waitmem
  exec 9>$LK/card$C.lock; flock 9
  rm -rf $d; mkdir -p $d; t=$(date +%s)
  log "quant $name K$k ($cb) card $C start"
  ( exec 7>$LK/ram.lock; flock -s 7
    ROCR_VISIBLE_DEVICES=$C nice -n 5 $PY $CODE/quantize/ourq.py -K $k -o $d --only-head --device 0 --save-wq \
      --qtf2 --frac-ext $RXQ_EXL3_SRC --scaled-hessian --sigma-reg 0.025 --codebook $cb < /dev/null ) > $d.log 2>&1
  exec 9>&-
  # the stamp hashes paths, contents and every argument (K too): check its inputs instead (RULES amendment 2): the
  # source list of the verified K2.5 rerun ($REF) and w144's K2.5 arguments but K
  if ! $PY -I -c "
import json, sys
n, r, w = (json.load(open(f))[0] for f in sys.argv[1:4])
ok = n['sources'] == r['sources'] and {k: v for k, v in n['args'].items() if k != 'K'} == {k: v for k, v in w['args'].items() if k != 'K'}
sys.exit(0 if ok else 1)" $d/stamp.json $REF $Q/emb/trellis/t-i8/K2.5/stamp.json; then
    log "quant $name K$k: sources or arguments differ from the K2.5 rerun / w144 -- stopping"; mv $d $d.badstamp; exit 1; fi
  if [ -f $d/wq/lm_head.rowerr.npz ] && $PY -I $W/tools/embsrc.py $d >> $W/rel.jsonl 2>> $d.log; then
    rm -rf $d/wq; log "quant $name K$k OK $(( $(date +%s) - t )) s: $(tail -n 1 $W/rel.jsonl)"
  else log "quant $name K$k FAIL ($d.log)"; fi
done < $1
log "queue $1 done"
