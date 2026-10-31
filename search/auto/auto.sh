#!/bin/bash
# auto.sh -- the overnight pipeline driver (the GPU box, $RXM_WORK). Idempotent: every piece of state is a file, so
# stop.sh + resume.sh (or a plain restart) picks up where it was. Log: $RXM_WORK/auto/auto.log; summary: $RXM_WORK/auto/SUMMARY.md
#
#  card 1: our quantization, K order 8 4 6 2 7 3 5 1 (ourchain2.sh, QORDER) -> $RXM_OURS/ours-K*.rad + hproxy rows
#  card 0: eval worker (evalq2.sh) on $RXM_WORK/auto/q (+ q.prio, served first); card 1 joins once quantization is done
#  rounds: fit (kfit2.py, all fit-pool data, bootstrap) -> DP mixes at every available target (mean + Thompson) ->
#          evals on the round's fit slice (calA / fit2 / fit3 in rotation) -> repeat. Every 2nd round the mean mixes
#          also go to the verify slice golden-ver1 (w- tags, never fitted).
#  priority jobs: embedding swaps (once the donors exist), each ours-K's smoke + band swaps as it lands, the anchors of
#          a fit slice before its first round, the bf16 floor (TP2) once both cards are free.
#  final: MANUAL -- only once state/final_go exists: the best measured mixes per target (finalsel.py) + reference
#         containers on golden-ver2 (v2- tags), END, summary.
: ${RXM_WORK:=/root} ${RXM_MODELS:=$RXM_WORK/models/final} ${RXM_OURS:=/data/ours}; export RXM_WORK RXM_MODELS RXM_OURS   # paths (README: Paths)
cd ${RXM_WORK}; A=${RXM_WORK}/auto; mkdir -p $A/mixes $A/state
echo $$ > $A/state/driver.pid
[ -f $A/state/deadline ] || echo $(( $(date +%s) + ${HOURS:-24} * 3600 )) > $A/state/deadline
DEADLINE=$(cat $A/state/deadline); FINAL_H=${FINAL_H:-2.5}
FINAL_AT=$(python3 -c "print(int($DEADLINE - $FINAL_H * 3600))")
Q=$A/q; touch $Q $Q.prio
log() { echo "$(date +%H:%M) [auto] $*" | tee -a $A/auto.log >> ${RXM_WORK}/ourchain.log; }
TARGETS="1.5 1.6 1.75 1.8 2.0 2.25 2.5 2.75 3.0 3.25 3.5 3.75 4.0 4.25 4.5 4.75 5.0 5.25 5.5 5.75 6.0 6.5 7.0 7.5"
declare -A GOLD=([a]="golden-calA corpus/calA.jsonl" [f2]="golden-fit2 corpus/fit2.jsonl" [f3]="golden-fit3 corpus/fit3.jsonl")
SLICES=(a f2 f3)

quantdone() { grep -q "OURCHAINDONE card 1" ${RXM_WORK}/ourchain.log; }
alive() { [ -f $1 ] && kill -0 $(cat $1) 2>/dev/null; }
start_workers() {
  alive $A/state/quant.pid || quantdone || {
    ( QORDER="8 4 6 2 7 3 5 1" NOHEADS=1 exec ./ourq/ourchain2.sh 1 ) > $A/quant.out 2>&1 < /dev/null &
    echo $! > $A/state/quant.pid; log "quantization started on card 1 (pid $!)"; }
  [ -f $A/state/pause-w0 ] || alive $A/state/w0.pid || { nohup ./auto/evalq2.sh 0 $Q > $A/w0.out 2>&1 < /dev/null & echo $! > $A/state/w0.pid; log "card 0 worker started"; }
  if quantdone && ! alive $A/state/w1.pid; then
    nohup ./auto/evalq2.sh 1 $Q > $A/w1.out 2>&1 < /dev/null & echo $! > $A/state/w1.pid; log "card 1 worker started"; fi
}
scored_or_failed() { [ -f evq/scores/$1.f32 ] || grep -q "\"tag\": \"$1\", .*fail" evq/results.jsonl; }
pending() { awk '$1!="END" && $1 !~ /^#/ {print $1}' "$@" | while read t; do scored_or_failed $t || echo $t; done; }
excl() { cat $A/state/exclude 2>/dev/null | tr "\n" "," | sed 's/,$//'; }

watch_prio() {   # adds priority jobs whose inputs exist, once each
  if [ ! -f $A/state/emb ] && ls ${RXM_MODELS}/q38-27b-sc4.00-e{1,2,7,8}.rad > /dev/null 2>&1 && ! pgrep -f "^/bin/bash ./embdonors" > /dev/null; then
    python3 -I auto/calmix.py emb >> $Q.prio && touch $A/state/emb && log "embedding swaps queued (1/2/7/8 bit, bf16)"; fi
  for K in 8 4 6 2 7 3 5 1; do
    [ -f $A/state/ours-K$K ] && continue
    grep -q "K$K container + proxy done" ${RXM_WORK}/ourchain.log || continue
    python3 -I auto/calmix.py ours $K >> $Q.prio && touch $A/state/ours-K$K && log "ours-K$K landed: smoke + 3 band swaps queued"
  done
  for K in 8 4 6 2 7 3 5 1; do   # smoke check: a failed or absurd whole-model eval takes that K out of the pool
    [ -f $A/state/smoke-K$K ] && continue
    t=a-ref-ours-K$K; scored_or_failed $t || continue
    k=$(grep "\"tag\": \"$t\"" evq/results.jsonl | tail -1 | python3 -c "import json,sys; r=json.loads(sys.stdin.read()); print(r.get('kld', -1))")
    if [ "$k" = "-1" ] || { [ $K -ge 2 ] && python3 -c "import sys; sys.exit(0 if $k > 1.0 else 1)"; }; then
      echo ours-K$K >> $A/state/exclude; log "ours-K$K SMOKE FAIL (kld $k): excluded from the pool"
    else log "ours-K$K smoke ok: whole-model kld $k"; fi
    touch $A/state/smoke-K$K
  done
  if quantdone && [ ! -f $A/state/bf16floor ]; then
    echo "a-ref-bf16 ${RXM_MODELS}/q38-27b-bf16.rad - ${GOLD[a]}" >> $Q.prio; touch $A/state/bf16floor
    log "bf16 floor eval (TP2) queued"; fi
}
anchors() {   # reference containers on a fit slice before its first round
  local s=$1; [ $s = a ] || [ -f $A/state/anchors-$s ] && return
  for c in sc2.00-e3 sc3.00-e4 sc4.00-e4 sc5.00-e5 sc6.00-e6; do
    echo "$s-ref-$c ${RXM_MODELS}/q38-27b-$c.rad - ${GOLD[$s]}" >> $Q.prio; done
  touch $A/state/anchors-$s; log "anchors for slice $s queued"
}
fitround() {   # TAG TARGETS NTHOMPSON -> mixes in ${RXM_WORK}/knap/mixes, fit report in $A/fit-TAG.txt
  rm -f knap/mixes/knap-$1-*.mix      # never pick up stale mixes of the same tag
  ( cd knap && KFIT_TAG=$1- KFIT_BOOT=${BOOT:-12} KFIT_THOMPSON=$3 KFIT_FOLDS=${FOLDS:-5} KFIT_VERBOSE=1 KFIT_EXCLUDE=$(excl) \
      nice python3 -W ignore kfit2.py ${RXM_WORK}/knap $2 > $A/frontier-$1.txt 2> $A/fit-$1.txt ) \
  || { [ -f knap/kfit2.py.fallback ] && log "fit $1 FAILED ($(tail -1 $A/fit-$1.txt | cut -c1-120)); rerun with kfit2.py.fallback, no tilts" \
       && rm -f knap/mixes/knap-$1-*.mix && ( cd knap && KFIT_TILT=0 KFIT_TAG=$1- KFIT_BOOT=${BOOT:-12} KFIT_THOMPSON=$3 KFIT_FOLDS=${FOLDS:-5} KFIT_VERBOSE=1 \
       KFIT_EXCLUDE=$(excl) nice python3 -W ignore kfit2.py.fallback ${RXM_WORK}/knap $2 > $A/frontier-$1.txt 2> $A/fit-$1.txt ); }
}

r=$(cat $A/state/round 2>/dev/null || echo 4)
while true; do
  start_workers; watch_prio
  now=$(date +%s)
  # state/final_after N: go to the final once round N is done (re-read every loop; the time rule stays a safety net)
  # past round final_after: no more rounds. The final (ver2) is MANUAL (2026-10-30): it starts only once
  # state/final_go exists. To run more rounds instead, raise state/final_after (both re-read every minute).
  if [ ! -f $A/state/stream ] && [ -f $A/state/final_after ] && [ $r -gt $(cat $A/state/final_after) ]; then
    if [ -f $A/state/final_go ]; then log "final_go: final"; break; fi
    [ -f $A/state/final_wait_logged ] || { log "round $(cat $A/state/final_after) done; waiting: touch state/final_go for the ver2 final, or raise state/final_after for more rounds"; ./auto/summary.sh > /dev/null 2>&1; touch $A/state/final_wait_logged; }
    sleep 60; continue
  fi
  [ -f $A/state/stream ] && [ -f $A/state/stream_max ] && [ $(cat $A/state/streamfit 2>/dev/null || echo 1) -gt $(cat $A/state/stream_max) ] || rm -f $A/state/final_wait_logged
  # STREAMING MODE (state/stream, 2026-10-30): fit back to back; after each fit the queue's unstarted candidates are
  # replaced by this fit's (streamq.py). No rounds, no waiting on priority jobs; the final stays manual (final_go).
  if [ -f $A/state/stream ]; then
    if [ -f $A/state/final_go ]; then log "final_go: final"; break; fi
    n=$(cat $A/state/streamfit 2>/dev/null || echo 1); s=${SLICES[$((n % 3))]}
    if [ -f $A/state/stream_max ] && [ $n -gt $(cat $A/state/stream_max) ]; then   # fit budget used up: wait
      [ -f $A/state/final_wait_logged ] || { log "stream: $(cat $A/state/stream_max) fits done; waiting: touch state/final_go for the ver2 final, or raise state/stream_max"; ./auto/summary.sh > /dev/null 2>&1; touch $A/state/final_wait_logged; }
      sleep 60; continue
    fi
    # refit only after STREAM_MIN (default 6) new results since the last fit's start: a fit takes ~2-3 min, an eval
    # ~2 min a card, so back-to-back fits would churn the queue on 1-2 new measurements
    nres=$(wc -l < evq/results.jsonl)
    if [ $nres -lt $(( $(cat $A/state/stream_lastres 2>/dev/null || echo 0) + ${STREAM_MIN:-6} )) ]; then sleep 30; continue; fi
    echo $nres > $A/state/stream_lastres
    T=$(echo $TARGETS | tr " " ",")
    log "stream fit s$n: fitting (slice $s)"
    # ours-only options from 2026-10-30 22:55 (r11 A/B: tie at 2-8 bpw); turboderp measurements stay in the fit
    KFIT_TILT=${STREAM_TILT:-2} KFIT_LS=${STREAM_LS:-1} KFIT_OURS_ONLY=${STREAM_OURS_ONLY:-1} KFIT_TD_MIN=${STREAM_TD_MIN:-5} KFIT_SEED=$((1000 + n)) BOOT=${SBOOT:-4} FOLDS=${SFOLDS:-3} fitround s$n $T 1
    head -3 $A/fit-s$n.txt | tail -2 | cut -c1-300 >> $A/auto.log
    exec 8>>$Q.claimlock; flock 8
    # uncertainty swaps only while fewer than 6 priority jobs are pending (they would otherwise crowd out the mixes)
    # per fit (~6 evals): at most UNCSWAPS (1) new swaps, and only while < 2 priority jobs pend -- so a fit's slots go
    # ~1 swap + ~4 mean mixes + ~1 Thompson (streamq interleaves 1 per 4 means), not swaps crowding out the rest
    [ $(pending $Q.prio | wc -l) -lt 2 ] && python3 -I auto/uncswaps.py s$n ${UNCSWAPS:-1} >> $Q.prio 2>> $A/auto.log
    log "$(python3 -I auto/streamq.py s$n $s $n 2>&1)"
    # held-out ver1 check every VER1_EVERY s (default 2 h): the current best measured mix per target (finalsel.py's
    # pick), on q.prio so it actually runs; mixes already measured on ver1 are skipped. Monitoring only, never fitted.
    if [ $(( $(date +%s) - $(stat -c %Y $A/state/ver1_last 2>/dev/null || echo 0) )) -ge ${VER1_EVERY:-7200} ]; then
      touch $A/state/ver1_last
      python3 -I auto/finalsel.py 1 0 2>/dev/null | grep -E -- "-(2\.00|2\.50|3\.00|3\.50|4\.00|4\.50|5\.00|6\.00|7\.00) " \
        | sed "s/^v2-/w-/; s/golden-ver2 corpus\/ver2.jsonl/golden-ver1 corpus\/ver1.jsonl/" \
        | while read t rest; do [ -f evq/scores/$t.f32 ] || grep -q "^$t " $Q.prio || echo "$t $rest"; done > $A/ver1.q
      cat $A/ver1.q >> $Q.prio; log "ver1 check: $(wc -l < $A/ver1.q) best-measured mixes queued on golden-ver1"
    fi
    flock -u 8; exec 8>&-
    echo $((n + 1)) > $A/state/streamfit
    continue
  fi
  # a round in flight? wait for it
  if [ -f $A/state/round-$r.queued ]; then
    if [ -z "$(pending $A/round-$r.q)" ]; then
      log "round $r done"; ./auto/summary.sh > /dev/null 2>&1
      r=$((r + 1)); echo $r > $A/state/round
    else sleep 60; fi
    continue
  fi
  # wait for the priority jobs that change the pool (embedding, ours calibration, anchors) before fitting
  if [ -n "$(pending $Q.prio)" ]; then sleep 60; continue; fi
  # state/wait_quant: hold the next fit until quantization is done (its priority jobs then come first, above)
  if [ -f $A/state/wait_quant ] && ! quantdone; then sleep 60; continue; fi
  s=${SLICES[$((r % 3))]}
  if [ $s != a ] && [ ! -f $A/state/anchors-$s ]; then anchors $s; continue; fi
  T=""; for t in $TARGETS; do
    if python3 -c "import sys; sys.exit(0 if $t > 7.5 else 1)"; then quantdone || continue; fi
    if python3 -c "import sys; sys.exit(0 if $t < 1.6 else 1)"; then [ -f ${RXM_OURS}/ours-K1.rad ] && [ -f ${RXM_OURS}/ours-K2.rad ] || continue; fi
    T="$T,$t"; done; T=${T#,}
  NT=1; quantdone && NT=2
  log "round $r: fitting (slice $s, targets $T, Thompson $NT)"
  fitround r$r $T $NT
  head -3 $A/fit-r$r.txt | tail -2 | cut -c1-300 >> $A/auto.log
  : > $A/round-$r.q
  for m in knap/mixes/knap-r$r-*.mix; do
    n=$(basename $m .mix); echo "$s-$n ${RXM_MODELS}/q38-27b-sc4.00-e4.rad ${RXM_WORK}/$m ${GOLD[$s]}" >> $A/round-$r.q; done
  if [ $((r % 2)) = 0 ] && quantdone; then   # verify-slice check of this round's mean mixes (both cards free)
    for m in knap/mixes/knap-r$r-sc4.00-e4-*.mix; do
      n=$(basename $m .mix); echo "w-$n ${RXM_MODELS}/q38-27b-sc4.00-e4.rad ${RXM_WORK}/$m golden-ver1 corpus/ver1.jsonl" >> $A/round-$r.q; done
  fi
  # state/oursonly-rN: also solve round N with our quants only (A/B vs the full option set, same slice)
  if [ -f $A/state/oursonly-r$r ]; then
    log "round $r: ours-only A/B fit"
    KFIT_OURS_ONLY=1 fitround r${r}oo $(cat $A/state/oursonly-r$r) 0
    for m in knap/mixes/knap-r${r}oo-*.mix; do
      n=$(basename $m .mix); echo "$s-$n ${RXM_MODELS}/q38-27b-sc4.00-e4.rad ${RXM_WORK}/$m ${GOLD[$s]}" >> $A/round-$r.q; done
  fi
  cat $A/round-$r.q >> $Q; touch $A/state/round-$r.queued
  # targeted swaps for the options this round's mixes use without direct data (run at priority, fitted next round)
  python3 -I auto/uncswaps.py $r ${UNCSWAPS:-10} >> $Q.prio 2>> $A/auto.log
  log "round $r: $(wc -l < $A/round-$r.q) evals queued"
done

# ------------------------------------------------------------------ final: last fit, held-out verification on ver2
if [ ! -f $A/state/final.queued ]; then
  # 2026-10-30: the final confirms the best MEASURED mixes per target (finalsel.py: slice-normalised KLD on the fit
  # slices; FINAL_N best + the best all-ours one when FINAL_OURS=0), not a last fit's mean mixes (they lose to
  # Thompson draws 70-90% of the time). Plus the reference containers.
  log "final: ver2 verification of the best measured mixes (finalsel N=${FINAL_N:-2}, ours-only=${FINAL_OURS:-0})"
  : > $A/final.q
  for c in sc1.40-e3 sc1.60-e3 sc1.80-e3 sc2.00-e3 sc2.20-e3 plain2.50-e3 sc3.00-e4 plain3.00-e4 plain3.50-e4 sc4.00-e4 sc5.00-e5 sc6.00-e6; do
    echo "v2-ref-$c ${RXM_MODELS}/q38-27b-$c.rad - golden-ver2 corpus/ver2.jsonl" >> $A/final.q; done
  python3 -I auto/finalsel.py ${FINAL_N:-2} ${FINAL_OURS:-0} >> $A/final.q 2> $A/finalsel.txt
  cat $A/final.q >> $Q; touch $A/state/final.queued; log "final: $(wc -l < $A/final.q) evals queued on ver2"
fi
until [ -z "$(pending $A/final.q)" ]; do start_workers; sleep 60; done
echo END >> $Q
./auto/summary.sh > /dev/null 2>&1
log "pipeline finished"
