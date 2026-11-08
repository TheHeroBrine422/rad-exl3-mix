#!/bin/bash
# run90.sh -- w90 A4 items 2-3 (README.md): top-512 + tail healing target and data starvation (1/4, 1/8) vs d1104's
# heal0, K3 / K2, seeds 0 / 1, plus one heal0 rerun (noise floor). 4070, one GPU unit at a time under /data/yb/gpu.lock,
# VRAM_CAP_GB 7.2. Waits for #89-06c (W79 HEAL DONE) unless $O/NOWAIT. Idempotent; stop: touch /data/yb/w90/STOP.
P=/workspace/rad-exl3-mix/proxy2b W=$P/exp/w90-a4heal E6=$P/exp/w06-heal
M=/data/models/qwen35-08b S=/data/g08-search1 D4=/data/yb/d1104 O=/data/yb/w90
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
export POOLS=/workspace/rad-exl3-mix/eval/pools/pools-v3.json
mkdir -p $O; cd /tmp
log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
stop() { [ -f $O/STOP ]; }
lock() { exec 9>/data/yb/gpu.lock; flock 9; }; unlock() { flock -u 9; }
sfx() { [ $1 = 0 ] && echo "" || echo "_s$1"; }
until grep -q "W79 HEAL DONE" /data/yb/w79/result.txt 2>/dev/null || [ -f $O/NOWAIT ]; do stop && exit 0; sleep 120; done
st() { echo $D4/q-k$1edyswjasym_r0$(sfx $2)/wqall; }                     # stack wq
ctl() { echo $D4/k$1edyswjasym_r0$(sfx $2)_heal.f32; }                  # heal0 score (d1104 naming: seed before _heal)
stk() { echo $D4/k$1edyswjasym_r0$(sfx $2).f32; }                       # stack score
hm() { [ -f $1/model/.done ] && echo $1/model || echo $M; }
pair() { # pair A.f32 B.f32 NAME -> pairs.jsonl
  [ -f $1 ] && [ -f $2 ] || { log "pair $3: missing $1 or $2"; return 1; }
  local r=$(python3 $P/pcmp.py $S $1 $2 | tail -n 1); echo "{\"pair\": \"$3\", \"r\": $r}" >> $O/pairs.jsonl
  log "pair $3: $(echo "$r" | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"; }
# arm TAG K SEED ENV ARGS..: heal (heal90.py; ENV e.g. W90_TOPK=512 or -), fold, score, pair vs heal0 and vs the stack
arm() { local t=$1 k=$2 s=$3 e=$4; shift 4; local d=$O/h-$t
  [ -f $O/$t.f32 ] && return 0; lock; mkdir -p $d
  [ -f $d/params.pt ] || env ${e/#-/W90_TOPK=0} python3 $W/heal90.py train $d --wq $(st $k $s) "$@" >> $d.log 2>&1 \
    || { log "heal FAIL $t ($d.log)"; unlock; return 1; }
  log "$t: $(grep '^HEAL DONE' $d.log | tail -n 1)"
  [ -f $d/wq/.done ] || python3 $E6/fold.py $d/params.pt $(st $k $s) $d $M >> $d.log 2>&1 || { log "fold FAIL $t"; unlock; return 1; }
  local x=$(python3 $P/evalq.py $(hm $d) $S --wq $d/wq --out $O/$t.f32.part 2>/dev/null | tail -n 1)
  [ -f $O/$t.f32.part ] && mv $O/$t.f32.part $O/$t.f32 || { log "eval FAIL $t"; unlock; return 1; }
  unlock; log "$t: $(echo "$x" | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"
  pair $(ctl $k $s) $O/$t.f32 "heal0 -> $t"; pair $(stk $k $s) $O/$t.f32 "stack -> $t"; }
for k in 3 2; do for s in 0 1; do
  for f in $(st $k $s) $(ctl $k $s) $(stk $k $s); do [ -e $f ] || { log "FAIL missing input $f"; exit 1; }; done; done; done
grep -q "W79S DONE" $O/result.txt 2>/dev/null || bash $W/w79s.sh || log "FAIL w79s.sh (#89-06c student path)"
stop && exit 0
log "start (rules: README.md, written 18:40 UTC before data)"
for k in 3 2; do for s in 0 1; do stop && exit 0; arm k${k}t512$(sfx $s) $k $s W90_TOPK=512; done; done
stop && exit 0; arm k3rr 3 0 -
for q in "q4 256 8" "q8 128 16"; do set -- $q; for k in 3 2; do for s in 0 1; do stop && exit 0
  arm k${k}$1$(sfx $s) $k $s - --ntrain $2 --epochs $3; done; done; done
python3 $W/readout90.py $O | tee -a $O/result.txt
log "W90 A4 DONE"
