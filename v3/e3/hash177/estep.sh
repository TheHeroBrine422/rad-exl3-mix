#!/bin/bash
# estep.sh TAG ROUNDS "HOMES" "PACKS" -- whole-model timing of hbuild.sh's plugin variants (homes /data/q27/w139/home-NAME =
# rad10-perf's home with that exl3i.so) on existing V3 packs (mul1i8 codes: under a hash build the logits are garbage, the
# timing is valid): radiance server (rad10-perf install, bf16 KV, spec decode off), decode step at C 1 and C 8 (w124's
# dstep.sh: ignore_eos, steady-state window), e2e prefill at 8k (3 requests, unique first line, max_tokens 1). Homes
# interleaved per (round, pack). One box2 card under its lock (CARDS, default "3 2 1") + ram.lock shared; rocm-smi state
# before / after each server. Out /data/q27/w139/estep/TAG/. (w139, #177)
T=$1; N=$2; HS=$3; PS=$4
Q=/data/q27; H=$Q/w139; L=$Q/locks; O=$H/estep/$T; RAD=$Q/rad10-perf/install/bin/radiance; W=$Q/w124
mkdir -p $O; exec >> $O/estep.log 2>&1
say() { echo "$(date -u +%T) $*"; }
while :; do
  for c in ${CARDS:-3 2 1}; do
    exec 5>$L/card$c.lock
    if flock -n 5; then CARD=$c; break 2; fi
    exec 5>&-
  done
  sleep 30
done
exec 4>$L/ram.lock; flock -s 4
G=$((3 - CARD)); PORT=$((8139 + CARD))
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
        --max-num-seqs 8 --num-speculative-tokens 0 --host 127.0.0.1 --port $PORT > $O/serve-$t.log 2>&1 &
      SP=$!; i=0
      while [ $i -lt 900 ] && ! curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; do sleep 3; i=$((i + 3)); kill -0 $SP 2>/dev/null || break; done
      if curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; then
        grep -m1 -i "exl3i" $O/serve-$t.log > /dev/null; say "serve $t up"
        P=$PORT PROMPT=$W/prompts/short.txt LEVELS="1 8" MT=${MT:-1200} bash $W/tools/dstep.sh > $O/dstep-$t.txt 2>&1
        bash $W/tools/ask.sh $PORT ctx8k prefill-only > $O/prefill-$t.txt 2>&1
        touch $O/$t.done
      else say "serve $t: no health"; fi
      kill $SP 2>/dev/null; wait $SP 2>/dev/null
      smi $t-after
    done
  done
done
say "ALL DONE (card $CARD released)"
