#!/bin/bash
# run155.sh STAGE... -- (w155, #180-02, ds8 HIP 7) DFlash2 candidates rescored against OUR main head (exl3i_rerank) vs
# the int8 copy (w153's dflash.rerank_head) vs none, on w153's packs: P1-mx-rr (K4.125 mul1i8 a8g128 trunk, R16s003 K4
# mul1i8 head, bf16 embedding, MXFP4 drafter + the int8 copy) and P3-mx (the same without the copy). Home W/home
# (build155.sh). Restartable: every unit skips when its output exists. Log W/logs/run155.log. Stages:
#   gtest  (card + ram shared) exl3i's device tests: test_rerank (full: the real-size heads at M 1 / 7 / 16), test_r16,
#          exl3i_gpu_test
#   kld    (card + ram shared) golden-search27c-c-cut13 (w153's): P3 spec 0, P3 drafter on (main-head rescoring), P1
#          drafter on (copy), P3 drafter on with QWEN35_EXL3_RERANK=0 (none); rows hash vs w153's bb6b5a8a8dd77831
#   kld7   (card + ram shared) the three drafter configurations again at --num-speculative-tokens 7 (KLD mode's auto
#          depth is 0, so `kld` declares no drafter): the drafter, its taps and the rescoring op declared beside the trunk
#   serve  (card + ram shared) greedy servers: P3 spec 0 (text gen: the reference), then three rounds of
#          {main P3, copy P1, none P3 + QWEN35_EXL3_RERANK=0}, interleaved: text gen (gen8, 128 tokens, top-20) in
#          round 1, bench.py ctx 0 gen 256 + ctx 8k gen 32 (warm 1 + 4) every round, /metrics, VRAM / KV from the log
#   note   (card + ram shared) each configuration's plugin notes (GET /graph: which head the drafter rescores against)
#          and its rescoring ops
#   serve2 (card + ram shared) rounds 4-6 of the serve stage's bench (main, copy, none interleaved) on the final build
#   acc    (card + ram shared) acceptance on fixed prompts: gen-prompts (16) x 256 greedy tokens at C 1 on a fresh server
#          per configuration (spec 0, main, copy, none): /metrics' draft tokens / accepted; dcmp + tiemargin vs spec 0
#   cmp    dcmp + tiemargin of the text gens (each config vs spec 0; main vs copy)
W=/data/q27/w155; Q=/data/q27; I=$Q/rad10-perf/install; RAD=$I/bin/radiance; HD=$W/home; CARD=${CARD:-7}
P=$Q/w153/packs; T=$Q/w153/tools; PR=$Q/w153/in/prompts; PB=$Q/w153/in/p137/prompts
G=$Q/w124/gold/golden-search27c-c-cut13; L=$Q/locks; R=$W/res
mkdir -p $R $W/logs; ulimit -n 65536
exec >> $W/logs/run155.log 2>&1
say() { echo "$(date -u +%T) $*"; }
say "start $* (pid $$)"

kld() {   # kld TAG PACK ENV [radiance args]
  local t=$1 m=$2 e=$3; shift 3
  [ -f $R/kld-$t.json ] && return 0
  say "kld $t"
  env $e ROCR_VISIBLE_DEVICES=$CARD RADIANCE_HOME=$HD timeout 1800 $RAD --model $m --tp 1 --kv-cache-dtype bf16 \
    --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $R/kld-$t.json.part "$@" \
    > $R/kld-$t.log 2>&1 && mv $R/kld-$t.json.part.rows $R/kld-$t.json.rows && mv $R/kld-$t.json.part $R/kld-$t.json
  say "kld $t rc $? $(python3 -c "import json; j=json.load(open('$R/kld-$t.json'))['all']; print(j['kld']['mean'], j['top1_agreement'])" 2>/dev/null) rows $(sha256sum < $R/kld-$t.json.rows 2>/dev/null | cut -c1-16)"
}

PORT=$((8155 + CARD)); SP=
up() {   # up TAG MODEL ENV [radiance args]
  local t=$1 m=$2 e=$3 i=0; shift 3
  env $e ROCR_VISIBLE_DEVICES=$CARD RADIANCE_HOME=$HD $RAD --model $m --tp 1 --kv-cache-dtype bf16 --max-model-len 40960 \
    --max-num-seqs 8 --host 127.0.0.1 --port $PORT "$@" > $R/serve-$t.log 2>&1 &
  SP=$!
  while [ $i -lt 1800 ] && ! curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; do sleep 3; i=$((i + 3)); kill -0 $SP 2>/dev/null || break; done
  curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null
}
down() { kill -INT $SP 2>/dev/null; for j in $(seq 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill $SP 2>/dev/null; wait $SP 2>/dev/null; SP=; }
srv() {   # srv TAG MODEL ENV WHAT [radiance args]; WHAT: t (text gen) b (bench)
  local t=$1 m=$2 e=$3 wh=$4; shift 4
  [ -f $R/$t.done ] && return 0
  say "serve $t"
  if up $t $m "$e" "$@"; then
    rocm-smi --showmeminfo vram 2>/dev/null > $R/vram-$t.txt
    case $wh in *t*) python3 -I $T/gen.py $PORT $PR/gen8.jsonl $R/txt-$t.jsonl 1 128 > /dev/null 2>&1 ;; esac
    case $wh in *b*) rm -f $R/bench-$t.jsonl
      python3 -I $T/bench.py --port $PORT --prompts $PB --out $R/bench-$t.jsonl --tag $t --ctx 0 --warm 1 --runs 4 --gen 256 --server-log $R/serve-$t.log > $R/bench-$t.txt 2>&1
      python3 -I $T/bench.py --port $PORT --prompts $PB --out $R/bench-$t.jsonl --tag $t --ctx 8k --warm 1 --runs 4 --gen 32 --server-log $R/serve-$t.log >> $R/bench-$t.txt 2>&1 ;; esac
    curl -s -m 10 http://127.0.0.1:$PORT/metrics > $R/metrics-$t.txt
    touch $R/$t.done
    say "serve $t: $(grep -c '"error"' $R/txt-$t.jsonl 2>/dev/null) text errors; $(grep -m1 'DFlash2 drafter' $R/serve-$t.log | grep -o 'head, .*'); $(grep -m1 'vram_kv' $R/serve-$t.log | tr -s ' ')"
  else say "serve $t: no health ($(grep -m1 '^E ' $R/serve-$t.log | cut -c1-200))"; fi
  down
}

gpu() {   # gpu FUNCTION...: under the card lock (exclusive) and ram.lock (shared)
  ( flock 5; flock -s 4; say "card $CARD + ram (shared) held"; "$@"; say "card $CARD released" ) 5>$L/card$CARD.lock 4>$L/ram.lock
}

st_gtest() {
  local B=$W/pb/exl3i
  for t in "test_rerank $B/test_rerank" "test_r16 $B/test_r16" "exl3i_gpu_test $B/exl3i_gpu_test"; do
    set -- $t
    [ -f $R/$1.txt ] && grep -q "PASS" $R/$1.txt && continue
    say "gtest $1"
    ROCR_VISIBLE_DEVICES=$CARD timeout 1800 $2 $(find $B -name exl3i.so | head -1) > $R/$1.txt 2>&1
    say "gtest $1 rc $? $(tail -n 1 $R/$1.txt)"
  done
}
st_kld() {
  kld P3-s0 $P/P3-mx.rad "" --num-speculative-tokens 0
  kld P3-main $P/P3-mx.rad ""
  kld P1-copy $P/P1-mx-rr.rad ""
  kld P3-none $P/P3-mx.rad "QWEN35_EXL3_RERANK=0"
}
st_kld7() {   # the same with the drafter declared (KLD mode's auto depth is 0: no drafter at all)
  kld P3-main-s7 $P/P3-mx.rad "" --num-speculative-tokens 7
  kld P1-copy-s7 $P/P1-mx-rr.rad "" --num-speculative-tokens 7
  kld P3-none-s7 $P/P3-mx.rad "QWEN35_EXL3_RERANK=0" --num-speculative-tokens 7
}
st_serve() {
  srv P3off $P/P3-mx.rad "" t --num-speculative-tokens 0
  for r in 1 2 3; do
    local w=b; [ $r = 1 ] && w=tb
    srv main-r$r $P/P3-mx.rad "" $w
    srv copy-r$r $P/P1-mx-rr.rad "" $w
    srv none-r$r $P/P3-mx.rad "QWEN35_EXL3_RERANK=0" $w
  done
}
st_note() {   # the plugin's startup notes (GET /graph: the drafter's says which head rescores) + the rescoring ops declared
  for x in main:P3-mx.rad: copy:P1-mx-rr.rad: none:P3-mx.rad:QWEN35_EXL3_RERANK=0; do
    IFS=: read t m e <<< "$x"
    [ -s $R/note-$t.txt ] && continue
    if up note-$t $P/$m "$e"; then
      curl -s -m 60 http://127.0.0.1:$PORT/graph > $R/graph-$t.json
      python3 -I -c "import json, sys, collections
j = json.load(open(sys.argv[1]))
for n in j.get('notes', []): print('note:', n)
c = collections.Counter(o.get('op') for o in j.get('ops', []) if isinstance(o, dict))
print('ops:', {k: c[k] for k in ('exl3i_rerank', 'logit_rerank', 'row_topk', 'dflash_select')})" $R/graph-$t.json > $R/note-$t.txt 2>&1
    fi
    down
    say "note $t: $(grep -o 'head, .*' $R/note-$t.txt | cut -c1-110) $(grep '^ops' $R/note-$t.txt)"
  done
}
st_serve2() {   # the final build (alignbit): three more interleaved bench rounds of the three configurations
  for r in 4 5 6; do
    srv main-r$r $P/P3-mx.rad "" b
    srv copy-r$r $P/P1-mx-rr.rad "" b
    srv none-r$r $P/P3-mx.rad "QWEN35_EXL3_RERANK=0" b
  done
}
st_acc() {   # acceptance on fixed prompts: gen-prompts (16) x 256 greedy tokens, C 1, a fresh server per configuration
  for x in acc-off:P3-mx.rad::--num-speculative-tokens:0 acc-main:P3-mx.rad: acc-copy:P1-mx-rr.rad: acc-none:P3-mx.rad:QWEN35_EXL3_RERANK=0; do
    IFS=: read t m e a1 a2 <<< "$x"
    [ -f $R/$t.done ] && continue
    say "serve $t"
    if up $t $P/$m "$e" $a1 $a2; then
      python3 -I $T/gen.py $PORT $PR/gen-prompts.jsonl $R/txt-$t.jsonl 1 256 > /dev/null 2>&1
      curl -s -m 10 http://127.0.0.1:$PORT/metrics > $R/metrics-$t.txt
      touch $R/$t.done
      say "serve $t: $(grep -c '"error"' $R/txt-$t.jsonl) errors; $(grep -E '^radiance:draft_(tokens|accepted)_total' $R/metrics-$t.txt | awk '{printf "%s ", $2}')"
    else say "serve $t: no health"; fi
    down
  done
  for t in acc-main acc-copy acc-none; do
    python3 -I $T/dcmp.py $R/txt-acc-off.jsonl $R/txt-$t.jsonl > $R/dcmp-txt-acc-off-vs-$t.txt 2>&1
    python3 -I $T/tiemargin.py $R/txt-acc-off.jsonl $R/txt-$t.jsonl > $R/tie-txt-acc-off-vs-$t.txt 2>&1
  done
}
st_cmp() {
  for x in P3off:main-r1 P3off:copy-r1 P3off:none-r1 main-r1:copy-r1 main-r1:none-r1; do
    a=${x%%:*}; b=${x#*:}
    [ -f $R/txt-$a.jsonl ] && [ -f $R/txt-$b.jsonl ] && python3 -I $T/dcmp.py $R/txt-$a.jsonl $R/txt-$b.jsonl > $R/dcmp-txt-$a-vs-$b.txt 2>&1 \
      && python3 -I $T/tiemargin.py $R/txt-$a.jsonl $R/txt-$b.jsonl > $R/tie-txt-$a-vs-$b.txt 2>&1
  done
  say "cmp done"
}
for s in "$@"; do
  case $s in
    gtest) gpu st_gtest ;;
    kld) gpu st_kld ;;
    kld7) gpu st_kld7 ;;
    note) gpu st_note ;;
    serve2) gpu st_serve2 ;;
    acc) gpu st_acc ;;
    serve) gpu st_serve ;;
    cmp) st_cmp ;;
  esac
done
say "END $*"
