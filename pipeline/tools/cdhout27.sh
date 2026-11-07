#!/bin/bash
# cdhout27.sh -- the 27B K4 "does no harm" pair of tile CD WITH H_out on attn_qg (rule: tools/CDHOUT27.md, w76, written
# before any data). One arm per run: the 16 attn_qg (self_attn.q_proj) tensors of an H_out K4 level BASE (H27's bn4 /
# bn4s1) requantized by rxq with BASE's own settings, seeds, calib H and H_out PLUS tile CD (3 sweeps, the two-sided
# objective tr(E^T H E M), rxq/tilecd.py); everything else is BASE's. Pack (exl3i, mul1i8, rate 4), mix into V2 4.00,
# census, the swap-pair hygiene check (the arm's mix == the control mix outside attn_qg: rxq.pairs), score on
# golden-search27 with swap27's own functions, pair against the control (the arm's existing score).
# gpubox, card 0 under /root/card0.lock + /root/ram.lock (shared); packs under /root/conv.lock + /root/ram.lock
# EXCLUSIVE (lead 11-07, as c9w27.sh). Idempotent: rerun after any failure. The lead schedules it (insert-cdhout27.sh).
#   pre     BASE (64 shards; rxq levelargs: its settings expressible in rxq's config), the staged code tree, disk
#   noop    the 16 q_proj requantized by rxq with BASE's settings WITHOUT tile CD: bit-identical to BASE (rxq.pairs same:
#           PASS) or not (FAIL: the control becomes <ARM>n = BASE + these tensors, packed and scored here, so that
#           rxq-vs-ourq or build drift is never read as the CD effect)
#   arm     the same with tile CD (rxq's default settings: quant.tile_cd on, attn_qg)
#   asm     rxq.pairs splice BASE + new q_proj -> level-cd (and level-noop if the noop gate failed); rxq.pairs levels:
#           level-cd == BASE outside attn_qg
#   pack    swap27k's packmixk: rad-convert -> mix into V2 4.00 -> census (tag <ARM>cd; <ARM>n if needed)
#   hyg     rxq.pairs mixes mix-<ARM>cd.rad CONTROL_MIX --arm attn_qg: refuses the pair if anything outside attn_qg
#           differs (CONTROL_MIX = /data/swap27/mix-<ARM>.rad, or mix-<ARM>n.rad after a failed noop gate; if the
#           control mix is gone, BASE is packed again as <ARM>ctl and scored, and that is the control)
#   score   swap27's score on golden-search27 (card 0, bf16 KV)
#   pair    tools/paired.py control rows -> <ARM>cd rows -> OUT/pair.json, result.txt
# Env: ARM (bn4 / bn4s1), BASE (level dir), A (/data/cdhout27), CODE ($A/code), SWEEPS (3, checked against the config),
#      GUARD_MB (3000), MINMEM_GB (6), STEPS, DRY=1. Log $A/<ARM>/cdhout27.log, results $A/result.txt.
# Code: $A/code, staged from the VM (CDHOUT27.md "Stage"): pipeline/ (rxq), v3/e3/{exl3frac2,exl3frac,exl3cb},
#      store/census.py. Stop: touch $A/<ARM>/STOP (between steps).
: ${RXM_WORK:=/root}; export RXM_WORK
ARM=${ARM:?ARM} BASE=${BASE:?BASE}
V2IN=${V2:-/data/release/qwen3.8-27b-exl3-v2-4.00bpw.rad}
A=${A:-/data/cdhout27}; O=$A/$ARM; CODE=${CODE:-$A/code}; SWEEPS=${SWEEPS:-3}
GUARD_MB=${GUARD_MB:-3000} MINMEM_GB=${MINMEM_GB:-6}
MODEL=${MODEL:-/data/workspace/models/qwen38-27b-bf16} HESS=${HESS:-/root/challenge/hessians/calib}
STEPS=${STEPS:-pre noop arm asm pack hyg score pair}
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=/data/exl3frac/build PYTORCH_HIP_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4

# swap27's pack / mix / census / score (sourced as cd27.sh / c9w27.sh do), then everything they write redirected to $A
source <(sed -n '/^: \${RXM_WORK/,/^quant() /p' /data/swap27/tools/swap27.sh | sed '$d')
source <(sed -n '/^packmixk() /,/^STOPF=/p' /data/swap27/tools/swap27k.sh | sed '$d')
S=$A V2=$V2IN K=4; G=/data/swap27/golden-search27
# packmixk with the lead's 11-07 rule for every pack on iceman (c9w27.sh's text): conv.lock + ram.lock EXCLUSIVE, a
# stale .part removed before the convert and after a failed one; otherwise swap27k's packmixk unchanged
packmixk() { arm=$1 src=$2 P=$S/pack-$1.rad M=$S/mix-$1.rad
  [ -f $M ] && [ -f $S/census-$arm.ok ] && return 0
  [ $(ls $src/q-L*.safetensors 2>/dev/null | wc -l) = 64 ] || { log "$arm: $src has not 64 layers"; return 1; }
  TRUNK | sed 's/^blk\.[0-9]*\./blk.*./' | sort -u | sed "s|\$| exl3i src=$src codebook=mul1i8 rate=$K|" > $S/$arm.recipe
  if [ ! -f $P ]; then
    ( exec 9>$RXM_WORK/conv.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock 7; cd /workspace; s=$(date +%s); rm -f $P.part
      RADIANCE_HOME=$H RADIANCE_HOST_DEVICES=1 nice $T/memguard.sh 2500 $I/bin/rad-convert models/qwen38-27b-bf16 \
        --recipe $S/$arm.recipe --set quantization_config.quant_method=exl3 -o $P.part > $S/conv-$arm.log 2>&1 && mv $P.part $P
      echo "pack $arm rc=$? $(( $(date +%s) - s ))s" ) | tee -a $S/result.txt
    [ -f $P ] || { rm -f $P.part; log "pack $arm FAIL ($S/conv-$arm.log)"; return 1; }
  fi
  TRUNK | sed "s|\$| $P|" > $S/$arm.mix
  $T/radmix $M.part $V2 $S/$arm.mix > $S/mix-$arm.log 2>&1 && mv $M.part $M || { log "radmix $arm FAIL"; return 1; }
  $PY -I $T/census.py $M --mix $S/$arm.mix --base $V2 > $S/census-$arm.txt 2>&1 || { log "census $arm FAIL"; return 1; }
  grep -qE "exl3i  *$K(\.0+)?  *mul1i8 *f16  *336 " $S/census-$arm.txt || { log "$arm: not 336 exl3i K$K trunk weights"; return 1; }
  touch $S/census-$arm.ok; rm -f $P; log "$arm: mix ready"; }
mkdir -p $O; cd $A
L=$O/cdhout27.log; log() { echo "$(date -u +%H:%M:%S) $ARM: $*" | tee -a $L >> $A/result.txt; tail -n 1 $L; }
want() { [[ " $STEPS " == *" $1 "* ]] && [ ! -f $O/STOP ]; }
die() { log "$*: stop"; exit 1; }
memok() { local m=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  [ -n "$DRY" ] || [ $m -ge $MINMEM_GB ] || die "MemAvailable $m GB < $MINMEM_GB GB before $1 (rerun later)"; }
card() { if [ -n "$DRY" ]; then echo "DRY: $*"; return 0; fi
  ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; cd $RXM_WORK
    $T/memguard.sh $GUARD_MB nice -n 5 "$@" ); }
host() { if [ -n "$DRY" ]; then echo "DRY: $*"; return 0; fi; nice -n 10 "$@"; }
RX=(env PYTHONPATH=$CODE/pipeline RXQ_RXM=$CODE RXQ_EXL3_SRC=${SRC:-/data/exl3frac/src} $PY)     # rxq (no -I: PYTHONPATH)

X=$("${RX[@]}" -m rxq levelargs $BASE) || die "rxq levelargs: $BASE has settings rxq's config cannot express"
[[ " $X " == *" --hout "* ]] || die "$BASE has no H_out: this pair is tile CD on top of H_out"
[ "$("${RX[@]}" -c 'from rxq import config as C; c = C.load(); d = c["quant"]["tile_cd"]; print(d["enable"], d["sweeps"], d["families"])')" \
  = "True $SWEEPS ['attn_qg']" ] || die "the staged config's tile_cd is not on / $SWEEPS sweeps / attn_qg"
KEYS=$("${RX[@]}" -c "print(','.join(f'model.language_model.layers.{l}.self_attn.q_proj' for l in range(3, 64, 4)))")
log "start: BASE $BASE ($X), V2 $(basename $V2), sweeps $SWEEPS, steps: $STEPS"

# ---- pre
if want pre && [ ! -f $O/pre.done ]; then
  [ $(ls $BASE/q-L*.safetensors 2>/dev/null | wc -l) = 64 ] && [ -f $BASE/model.safetensors.index.json ] || die "$BASE: not 64 layer shards"
  for f in pipeline/rxq/tilecd.py pipeline/rxq/pairs.py pipeline/config/v3.toml v3/e3/exl3frac2/exl3frac2.py store/census.py; do
    [ -f $CODE/$f ] || die "$CODE/$f missing"; done
  [ -f /data/swap27/$ARM.json.rows ] || die "no control rows /data/swap27/$ARM.json.rows"
  free=$(df --output=avail -BG $A | tail -1 | tr -dc 0-9)
  [ $free -ge 40 ] || { [ -n "$DRY" ] && echo "DRY: disk only $free GiB (a real run stops here)"; } || die "disk: $free GiB free under $A, an arm needs ~40 GiB"
  log "rxq args: $X; disk $free GiB free"; [ -n "$DRY" ] || touch $O/pre.done
fi

quant() { # quant DIR rxq-args..: the 16 q_proj with BASE's settings
  local d=$1; shift
  [ $(ls $d/t-*.safetensors 2>/dev/null | wc -l) = 16 ] && return 0
  memok "quant $(basename $d)"; s=$(date +%s)
  card "${RX[@]}" -m rxq quant --keys $KEYS $X --ckpt $MODEL --hess $HESS -o $d --device 0 --no-prof "$@" >> $d.log 2>&1
  [ -n "$DRY" ] || [ $(ls $d/t-*.safetensors 2>/dev/null | wc -l) = 16 ] || die "quant $d FAIL ($d.log; a rerun resumes per tensor)"
  log "$(basename $d): $(( $(date +%s) - s )) s"; }
if want noop && [ ! -f $O/noop.pass ] && [ ! -f $O/noop.fail ]; then
  quant $O/q-noop --set quant.tile_cd.enable=false
  r=$(host "${RX[@]}" -m rxq.pairs same $BASE $O/q-noop | tail -n 1); rc=$?; log "noop gate: $r"
  [ -n "$DRY" ] || { [[ "$r" == "PAIR OK"* ]] && touch $O/noop.pass || touch $O/noop.fail; }
fi
NOOPFAIL=$([ -f $O/noop.fail ] && echo 1)
if want arm; then
  quant $O/q-cd
  [ -n "$DRY" ] || log "tile CD (two-sided rel in the quantization domain, cd$SWEEPS / ldlq_2hess): $($PY -c "
import json, math
r = list({v['key']: v for v in map(json.loads, open('$O/q-cd/tilecd.jsonl'))}.values())   # a rerun's lines win
x = [v['ratio'] for v in r]; h = sum(1 for v in r if v.get('hout'))
print(f'{len(r)} tensors ({h} two-sided), geo {math.exp(sum(map(math.log, x)) / len(x)):.4f}, range {min(x):.4f}..{max(x):.4f}, CD {sum(sum(v[\"sec\"].values()) for v in r):.0f} s total')")"
  [ -n "$DRY" ] || log "q_proj two-sided proxy_err cd / noop (after the scale refit): $($PY -c "
import json, math
p = lambda f: {j['key']: j['proxy_err'] for j in map(json.loads, open(f))}
a, b = p('$O/q-noop/log.jsonl'), p('$O/q-cd/log.jsonl'); x = [b[k] / a[k] for k in b if k in a]
print(f'{len(x)} tensors, geo {math.exp(sum(map(math.log, x)) / len(x)):.4f}, range {min(x):.4f}..{max(x):.4f}')")"
fi
if want asm; then
  [ -f $O/level-cd/spliced.json ] || host "${RX[@]}" -m rxq.pairs splice $BASE $O/q-cd $O/level-cd --arm attn_qg | tee -a $L || die "splice FAIL"
  r=$(host "${RX[@]}" -m rxq.pairs levels $BASE $O/level-cd --arm attn_qg | tail -n 1); log "level hygiene: $r"
  [ -n "$DRY" ] || [[ "$r" == "PAIR OK"* ]] || die "level-cd differs from BASE outside attn_qg"
  if [ -n "$NOOPFAIL" ] && [ ! -f $O/level-noop/spliced.json ]; then
    host "${RX[@]}" -m rxq.pairs splice $BASE $O/q-noop $O/level-noop --arm attn_qg | tee -a $L || die "splice (noop) FAIL"; fi
fi
CTLTAG=$ARM CTLMIX=/data/swap27/mix-$ARM.rad CR=/data/swap27/$ARM.json.rows
[ -n "$NOOPFAIL" ] && CTLTAG=${ARM}n CTLMIX=$A/mix-${ARM}n.rad CR=$A/${ARM}n.json.rows
[ -z "$NOOPFAIL" ] && [ ! -f $CTLMIX ] && CTLTAG=${ARM}ctl CTLMIX=$A/mix-${ARM}ctl.rad CR=$A/${ARM}ctl.json.rows
if want pack; then
  if [ -n "$DRY" ]; then echo "DRY: packmixk ${ARM}cd $O/level-cd (K$K into $V2); control $CTLTAG ($CTLMIX)"
  else packmixk ${ARM}cd $O/level-cd || die "pack ${ARM}cd FAIL"
       [ "$CTLTAG" = ${ARM}n ] && { packmixk ${ARM}n $O/level-noop || die "pack ${ARM}n FAIL"; }
       [ "$CTLTAG" = ${ARM}ctl ] && { packmixk ${ARM}ctl $BASE || die "pack ${ARM}ctl FAIL"; }; fi
fi
if want hyg && [ ! -f $O/hyg.pass ]; then
  r=$(host "${RX[@]}" -m rxq.pairs mixes $A/mix-${ARM}cd.rad $CTLMIX --arm attn_qg | tail -n 1); log "mix hygiene vs $CTLTAG: $r"
  [ -n "$DRY" ] || { [[ "$r" == "PAIR OK"* ]] && touch $O/hyg.pass || die "the arm's mix differs from $CTLTAG outside attn_qg: not scored"; }
fi
if want score; then
  if [ -n "$DRY" ]; then echo "DRY: score ${ARM}cd (and $CTLTAG if it is new)"
  else [ -f $O/hyg.pass ] || die "no hygiene pass: not scored"
       C=0 score ${ARM}cd || die "score ${ARM}cd FAIL"
       [ "$CTLTAG" = $ARM ] || { C=0 score $CTLTAG || die "score $CTLTAG FAIL"; }; fi
fi
if want pair && [ -z "$DRY" ] && [ -f $A/${ARM}cd.json.rows ]; then
  [ -f $CR ] || die "no control rows $CR"
  $PY -I $T/paired.py $G $CR $A/${ARM}cd.json.rows | tail -n 1 > $O/pair.json || die "paired.py FAIL"
  echo "{\"arm\": \"$ARM\", \"control\": \"$CR\", \"cd\": \"$A/${ARM}cd.json.rows\", \"noop\": \"$([ -n "$NOOPFAIL" ] && echo fail || echo pass)\", \"hygiene\": \"pass\"}" > $O/pair-meta.json
  log "pair $(basename ${CR%.json.rows}) -> ${ARM}cd: $(cat $O/pair.json)"
fi
log "CDHOUT27 $ARM DONE ($STEPS)"
