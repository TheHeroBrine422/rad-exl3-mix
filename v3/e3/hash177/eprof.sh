#!/bin/bash
# eprof.sh TAG ROUNDS "HOMES" "PACKS" -- in-situ per-op decode times of hbuild.sh's plugin variants (lead 11-13: decode
# tables from real e2e runs, not isolated GEMVs): radiance server (rad10-perf install, bf16 KV, spec off) with
# --profile-ops, one request (w124's dstep.sh, LEVELS 1, short prompt, MT tokens, default 512), then the server's per-op
# table (serve log, written at shutdown) -> eprofsum.py. Packs with mul1i8 codes under a hash home give garbage logits,
# valid timing. Homes interleaved per (round, pack), one box2 card under its lock (CARDS, default "0 1 3 2") + ram.lock
# shared; rocm-smi before / after. Profiled times run ~10 % high (syncs per op): read ratios between homes.
# Out /data/q27/w139/eprof/TAG/. (w139, #177)
T=$1; N=$2; HS=$3; PS=$4
Q=/data/q27; H=$Q/w139; L=$Q/locks; O=$H/eprof/$T; RAD=$Q/rad10-perf/install/bin/radiance; W=$Q/w124
mkdir -p $O; exec >> $O/eprof.log 2>&1
say() { echo "$(date -u +%T) $*"; }
while :; do
  for c in ${CARDS:-0 1 3 2}; do
    exec 5>$L/card$c.lock
    if flock -n 5; then CARD=$c; break 2; fi
    exec 5>&-
  done
  sleep 30
done
exec 4>$L/ram.lock; flock -s 4
G=$((3 - CARD)); PORT=$((8149 + CARD))
smi() { rocm-smi -d $G --showuse --showpower --showtemp --showclocks --showmemuse 2>/dev/null | grep -E "GPU\[" > $O/smi-$1.txt; }
say "card $CARD (rocm-smi $G) locked; homes $HS; packs $PS; rounds $N"
ulimit -n 65536
for r in $(seq 1 $N); do
  for p in $PS; do
    pn=$(basename $p .rad)
    for h in $HS; do
      t=r$r-$h-$pn; [ -f $O/$t.done ] && continue
      smi $t-before
      ROCR_VISIBLE_DEVICES=$CARD RADIANCE_HOME=$H/home-$h $RAD --model $p --tp 1 --kv-cache-dtype bf16 --max-model-len 12288 \
        --max-num-seqs 8 --num-speculative-tokens 0 --profile-ops --host 127.0.0.1 --port $PORT > $O/serve-$t.log 2>&1 &
      SP=$!; i=0
      while [ $i -lt 900 ] && ! curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; do sleep 3; i=$((i + 3)); kill -0 $SP 2>/dev/null || break; done
      if curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; then
        say "serve $t up"
        P=$PORT PROMPT=$W/prompts/short.txt LEVELS="1" MT=${MT:-512} bash $W/tools/dstep.sh > $O/dstep-$t.txt 2>&1
        kill -INT $SP 2>/dev/null; i=0; while kill -0 $SP 2>/dev/null && [ $i -lt 60 ]; do sleep 1; i=$((i + 1)); done
        grep -q "per-op timing" $O/serve-$t.log && touch $O/$t.done || say "serve $t: no per-op table"
      else say "serve $t: no health"; fi
      kill $SP 2>/dev/null; wait $SP 2>/dev/null
      smi $t-after
    done
  done
done
say "ALL DONE (card $CARD released)"
