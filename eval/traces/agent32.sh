#!/bin/bash
# agent32.sh [STAGE...] -- the chunked agentic slices on gpubox (eval/traces/README.md, w67). Stages (default: all, in
# order; each idempotent, a rerun resumes):
#   chunk     (CPU) trace tokens / gen from the published calib golden, chunk.py (32k, --gold check) -> chunks/; the
#             manifest must equal the frozen repo copy (code/eval/traces/manifest-32k.json), else FAIL
#   whole     (CPU) golden-agent-verify-whole: rich + express whole, cut from the published golden (no card)
#   probe     (card) kernel ladder (LADDER: stages, attn chunk, aotriton flash on/off, fla on/off). Each set must first
#             pass the GATE: mkgold27 on 2 docs whose context the published golden has (ripgrep's 2.3k prefix and its
#             26.7k chunk 0) agrees with it (KL <= 0.005, top-1 >= 0.95; a check of the ROCm stack, never run there before),
#             then fit the longest chunk (qad32's 32,761 tokens; VRAM_CAP_GB=30). Nothing fits -> chunks re-cut at 24k
#             (chunks24/, slices *24) and the ladder again. probe.env records the set and the projected golden time
#   golden    (card) mkgold27 bf16 goldens agent-search32 and agent-verify32 (both with the probe's N / ATTN_CHUNK);
#             refuses if the projection > A32_MAXH h (default 3)
#   finalize  (CPU) gen.u8 = chunk.py's mask (header and non-assistant rows off), chunk fields into kld.json, sanity
#   refgap    (CPU) chunk 0 of each trace vs the published golden (same context): the HF-vs-radiance reference gap
#   score     (card) 8 mixes on agent-search32, as swap27.sh's score (radiance 1.16.0 + plugins, bf16 KV) but
#             --max-model-len 32768; agent-verify32 is NOT scored (verify: finished mixes only)
#   report    report32.py -> report.txt / report.json (per arm mean / p99 / top-1, pairs with 3 bootstraps)
# GPU steps: /root/card0.lock, then /root/ram.lock shared, card 0 only; mkgold27 under memguard 2500 MB.
# Dry run (no card): A32_DIR=<scratch> A32_FAKE=1 RXM_WORK=<own lock dir> (fake mkgold27 / radiance; A32_FAKE_OOM="3:0 ..."
# makes those probe configs fail).
: ${RXM_WORK:=/root}; export RXM_WORK
D=${A32_DIR:-/data/agent32} C=0 MAXH=${A32_MAXH:-3}
CODE=$D/code PY=/data/venv-exl3/bin/python I=/data/install-116 H=/data/swap27/home
CAL=/data/challenge/golden/calib CORP=/data/challenge/corpus/calib.jsonl POOLS=/data/challenge/pools-v3.json
M=/data/workspace/models/qwen38-27b-bf16
export PATH=/opt/rocm/core-10.1/bin:$PATH ROCM_PATH=/opt/rocm/core-10.1
cd $D || exit 1
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $D/result.txt; }
declare -A MIX=([base]=/data/swap27/mix-base.rad [s1]=/data/swap27/mix-s1.rad [baseswj]=/data/swj27/mix-baseswj.rad
  [s1swj]=/data/swj27/mix-s1swj.rad [base4v4]=/data/verify27/mix-base4v4.rad [base4swj]=/data/swj27/mix-base4swj.rad
  [bn4v4]=/data/verify27/mix-bn4v4.rad [bn4swj]=/data/swj27/mix-bn4swj.rad)
ARMS="base baseswj s1 s1swj base4v4 base4swj bn4v4 bn4swj"
if [ -n "$A32_FAKE" ]; then MKG="$PY -I $CODE/eval/traces/fakemkgold.py"; RAD="bash $CODE/eval/traces/fakeradiance.sh"
else MKG="$PY $CODE/eval/gold27/mkgold27.py"; RAD="$I/bin/radiance"; fi
# the slice set in use (32k unless the probe fell back to 24k)
CH=chunks SFX=32; [ -f $D/probe.env ] && . $D/probe.env

chunk() {
  [ -f $D/chunks/manifest.json ] && cmp -s $D/chunks/manifest.json $CODE/eval/traces/manifest-32k.json && return 0
  nice $PY -I $CODE/eval/traces/agent32.py traces $CAL $D/traces 905 1241 1772 2300 2651 2675 4029 4723 > $D/traces.log 2>&1 \
    || { log "chunk: trace cut FAIL ($D/traces.log)"; return 1; }
  nice $PY $CODE/eval/traces/chunk.py $CORP $POOLS $D/chunks --tokenizer $M/tokenizer.json --gold $D/traces > $D/chunk.log 2>&1 \
    || { log "chunk: chunk.py FAIL ($D/chunk.log)"; return 1; }
  cmp -s $D/chunks/manifest.json $CODE/eval/traces/manifest-32k.json \
    || { log "chunk: manifest differs from the frozen manifest-32k.json FAIL"; return 1; }
  log "chunk: $(grep -c '' $D/chunks/agent-search32.jsonl) search + $(grep -c '' $D/chunks/agent-verify32.jsonl) verify chunks, manifest = frozen"
}

whole() {
  [ -f $D/golden-agent-verify-whole/kld.json ] && return 0
  nice $PY -I $CODE/eval/traces/agent32.py whole $CAL $D/chunks $D/golden-agent-verify-whole > $D/whole.log 2>&1 \
    && log "whole: $(tail -n 1 $D/whole.log)" || { log "whole: FAIL ($D/whole.log)"; return 1; }
}

# mkgold27 CORPUS OUT SCRATCH LOG, with the kernel choice in N, AC, AOT, BLK: card0.lock, ram.lock shared, memguard
mkg() { ( exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7
    ROCR_VISIBLE_DEVICES=$C VRAM_CAP_GB=30 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=$AOT P2B_BLOCK=$BLK OMP_NUM_THREADS=4 \
      TOKENIZERS_PARALLELISM=false A32_FAKE_CFG="$N:$AC:$AOT:$BLK" nice -n 5 bash /data/swap27/tools/memguard.sh 2500 $MKG $M \
      $1 $2 --scratch $3 --stages $N --scratch-dtype bf16 --attn-chunk $AC --model-name Qwen/Qwen3.8-27B ) > $4 2>&1; }

longest() { $PY -I -c "
import json, sys
best = max((json.loads(l)['tokens'], f, k) for f in sys.argv[1:] for k, l in enumerate(open(f)))
print(best[1], best[2] + 1, best[0])" "$@"; }

isoom() { grep -qiE "out of memory|OutOfMemory|memguard" $1; }

# kernel ladder: label N attn_chunk aotriton-experimental blocked-packages. Each must pass the gate (correctness vs the
# published golden on the same context, 2.3k and 26.7k tokens), then the memory probe (the longest chunk).
LADDER="k1:3:0:1: k2:3:2048:1: k3:4:2048:0: k4:3:0:1:fla k5:4:2048:0:fla"

gate1() { local t=$D/probe/gate-$1
  [ -f $t.pass ] && return 0; [ -f $t.fail ] && return 1; rm -rf $t.out $t.scr
  mkg $D/probe/gate.jsonl $t.out $t.scr $t.log; local rc=$?
  if [ $rc != 0 ] || [ ! -f $t.out/kld.json ]; then
    isoom $t.log && { echo oom > $t.fail; log "gate $1 (N=$N ac=$AC aotriton=$AOT block=${BLK:-none}): out of memory"; return 1; }
    echo "rc $rc" > $t.fail; log "gate $1: mkgold27 rc=$rc, not a memory error ($t.log)"; return 1
  fi
  nice $PY -I $CODE/eval/traces/agent32.py gate $t.out $CAL $D/probe/gate.jsonl $t.json > $t.txt 2>&1; rc=$?
  log "gate $1 (N=$N ac=$AC aotriton=$AOT block=${BLK:-none}): $(grep -h '^GATE' $t.txt) [all-row $(grep -h 'DONE' $t.log | grep -o 'ppl [0-9.]*')]"
  rm -rf $t.out; [ $rc = 0 ] && touch $t.pass && return 0; echo "gate $rc" > $t.fail; return 1; }

probe() {
  [ -f $D/probe.env ] && [ -n "$KSET" ] && return 0
  mkdir -p $D/probe
  [ -f $D/probe/gate.jsonl ] || A32_TOKENIZER=$M/tokenizer.json $PY -I $CODE/eval/traces/agent32.py gatecorpus $D/chunks $CORP \
    $D/probe/gate.jsonl > $D/probe/gate-corpus.log 2>&1 || { log "probe: gate corpus FAIL"; return 1; }
  local cut ch sf f ln nt cfg L t rc w tot proj
  for cut in 32768 24576; do ch=chunks; sf=${cut:0:2}
    if [ $cut != 32768 ]; then ch=chunks24
      [ -f $D/chunks24/manifest.json ] || nice $PY $CODE/eval/traces/chunk.py $CORP $POOLS $D/chunks24 --max 24576 \
        --tokenizer $M/tokenizer.json --gold $D/traces > $D/chunk24.log 2>&1 || { log "probe: chunk.py 24k FAIL"; return 1; }
    fi
    read f ln nt < <(longest $D/$ch/agent-search$sf.jsonl $D/$ch/agent-verify$sf.jsonl $D/$ch/qad$sf.jsonl)
    sed -n "${ln}p" $f > $D/probe/corpus-$cut.jsonl
    for cfg in $LADDER; do IFS=: read L N AC AOT BLK <<< "$cfg"; t=$D/probe/$cut-$L
      gate1 $L || continue
      [ -f $t.fail ] && continue
      if [ ! -f $t.ok ]; then
        rm -rf $t.out $t.scr; w=$(date +%s)
        mkg $D/probe/corpus-$cut.jsonl $t.out $t.scr $t.log; rc=$?; w=$(( $(date +%s) - w ))
        if [ $rc != 0 ] || [ ! -f $t.out/kld.json ]; then
          isoom $t.log && { echo "rc=$rc" > $t.fail; log "probe $L: does not fit $nt tokens (rc $rc): $(grep -iE 'out of memory|memguard' $t.log | tail -n 1 | cut -c1-160)"; continue; }
          log "probe $L: FAIL rc=$rc, not a memory error ($t.log)"; return 1
        fi
        $PY -I -c "import json; r=[json.loads(l) for l in open('$t.scr/timing.jsonl')]; print('PEAKS=\"' + ' '.join(f\"{x['peak_gib']:.1f}\" for x in r) + '\"'); print('RUN=%d' % max(1, round(sum(x['run_s'] for x in r))))" > $t.ok
        echo "TOK=$nt WALL=$w PPL=$(grep -h DONE $t.log | grep -o 'ppl [0-9.]*' | cut -d' ' -f2)" >> $t.ok; rm -rf $t.out
      fi
      . $t.ok
      tot=$(( $(cat $D/$ch/agent-search$sf.tokens.i32 $D/$ch/agent-verify$sf.tokens.i32 | wc -c) / 4 ))
      proj=$(( tot * RUN / TOK + 2 * N * 60 ))
      log "probe $L: $TOK-token doc ($(basename $f)) fits: peak GiB per stage $PEAKS, run ${RUN}s, wall ${WALL}s, all-row ppl $PPL (user / tool rows included: ~1e4 is normal, see README); goldens ~$(( proj / 60 )) min for $tot tokens"
      printf 'CH=%s\nSFX=%s\nN=%s\nAC=%s\nAOT=%s\nBLK=%s\nPROJ=%s\nKSET=%s\n' $ch $sf $N $AC $AOT "$BLK" $proj $L > $D/probe.env.part \
        && mv $D/probe.env.part $D/probe.env
      . $D/probe.env; return 0
    done
  done
  log "probe: no kernel set passes the gate and fits FAIL"; return 1
}

golden() { [ -f $D/probe.env ] && [ -n "$KSET" ] && [ -f $D/probe/gate-$KSET.pass ] || { log "golden: no gate-passed kernel set in probe.env FAIL"; return 1; }
  [ -f $D/golden-agent-verify$SFX/kld.json ] || [ $PROJ -le $(( MAXH * 3600 )) ] \
    || { log "golden: projected $(( PROJ / 60 )) min > ${MAXH} h FAIL (lead: A32_MAXH=N to accept)"; return 1; }
  for name in agent-search$SFX agent-verify$SFX; do G=$D/golden-$name
    [ -f $G/kld.json ] && continue
    local t0=$(date +%s); mkg $D/$CH/$name.jsonl $G $D/scratch/$name $D/golden-$name.log
    rc=$?; [ $rc = 0 ] && [ -f $G/kld.json ] && log "golden $name: done in $(( ($(date +%s) - t0) / 60 )) min, $(grep -h DONE $D/golden-$name.log | tail -n 1 | cut -c1-200)" \
      || { log "golden $name: mkgold27 rc=$rc FAIL ($D/golden-$name.log)"; return 1; }
  done; }

finalize() { for name in agent-search$SFX agent-verify$SFX; do
    nice $PY -I $CODE/eval/traces/agent32.py finalize $D/golden-$name $D/$CH $name >> $D/finalize.log 2>&1 \
      && log "$(tail -n 1 $D/finalize.log)" || { log "finalize $name FAIL ($D/finalize.log)"; return 1; }
  done; }

refgap() { for name in agent-search$SFX agent-verify$SFX; do
    [ -f $D/refgap-$name.json ] && continue
    nice $PY -I $CODE/eval/traces/agent32.py refgap $D/golden-$name $CAL --json $D/refgap-$name.json > $D/refgap-$name.txt 2>&1 \
      && cat $D/refgap-$name.txt >> $D/result.txt || { log "refgap $name FAIL ($D/refgap-$name.txt)"; return 1; }
  done; }

score1() { local arm=$1 G=$D/golden-agent-search$SFX O=$D/search; mkdir -p $O
  [ -f $O/$arm.json ] && [ -f $O/$arm.json.rows ] && return 0
  [ -f ${MIX[$arm]} ] || { log "score $arm: ${MIX[$arm]} missing FAIL"; return 1; }
  ( exec 5>$RXM_WORK/card$C.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; s=$(date +%s)
    ROCR_VISIBLE_DEVICES=$C RADIANCE_HOME=$H $RAD --model ${MIX[$arm]} --tp 1 --kv-cache-dtype bf16 \
      --max-model-len 32768 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $O/$arm.json.part > $O/score-$arm.log 2>&1
    rc=$?; r=$(( $(stat -c %s $G/rows.f32 2>/dev/null) / 12 * 16 ))
    [ $rc = 0 ] && [ "$(stat -c %s $O/$arm.json.part.rows 2>/dev/null)" = "$r" ] && mv $O/$arm.json.part.rows $O/$arm.json.rows && mv $O/$arm.json.part $O/$arm.json
    echo "$(date -u +%H:%M:%S) score $arm on agent-search$SFX rc=$rc $(( $(date +%s) - s ))s" ) | tee -a $D/result.txt
  [ -f $O/$arm.json ] || { log "score $arm FAIL ($O/score-$arm.log)"; return 1; }; }
score() { f=0; for a in $ARMS; do score1 $a || f=1; done; return $f; }

report() { $PY -I $CODE/eval/traces/report32.py $D/golden-agent-search$SFX $D/search --json $D/report.json \
    --paired $CODE/eval/traces/paired.py > $D/report.txt 2> $D/report.err && cat $D/report.txt >> $D/result.txt \
    || { log "report FAIL ($D/report.err)"; return 1; }; }

[ $# = 0 ] && set -- chunk whole probe golden finalize refgap score report
for stage in "$@"; do
  case $stage in
    chunk|whole|probe|golden|finalize|refgap|score|report) $stage || { log "AGENT32 stage $stage FAIL"; exit 1; } ;;
    *) echo "agent32.sh [chunk|whole|probe|golden|finalize|refgap|score|report]..."; exit 2 ;;
  esac
done
log "AGENT32 $* DONE"
