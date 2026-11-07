#!/bin/bash
# cd27.sh -- the 27B "does no harm" check of tile CD on attn_qg (rule: v3/e3/settle/CD27.md) on one swap-test arm:
# the 16 attn_qg (self_attn.q_proj) tensors of the level BASE requantized with --tile-cd 3 (BASE's own ourq args and
# seeds, calib H); everything else is BASE's. Pack (exl3i, mul1i8, rate K), mix into V2, census and score on
# golden-search27 with swap27's own functions (same build, golden and census as every swap arm), pair against CTL.
# gpubox, card 0 under /root/card0.lock + /root/ram.lock (shared), as asym27.sh. Idempotent: rerun after any failure.
#   pre    BASE (64 shards; its ourq args via levelargs.py), the private code tree (ourq.py + ourq-tilecd.patch), disk
#   noop   the 16 q_proj requantized with BASE's args WITHOUT --tile-cd: bit-identical to BASE (PASS) or not (FAIL:
#          then the control is BASE + these tensors, "<ARM>n", packed and scored here, so that code / build drift
#          since BASE is not read as the CD effect)
#   arm    the 16 q_proj requantized with BASE's args + --tile-cd 3 --tile-cd-families attn_qg
#   asm    BASE + new q_proj -> level-cd (and level-noop if the noop gate failed)
#   pack   swap27k's packmixk: rad-convert -> mix into V2 -> census  (arm tag <ARM>cd)
#   score  swap27's score on golden-search27 (card 0, bf16 KV)
#   pair   tools/paired.py CTL rows -> <ARM>cd rows -> OUT/pair.json, result.txt
# Env: ARM (tag: base / s1), BASE (level dir), CTL (control arm scored on BASE: /data/swap27/CTL.json.rows),
#      V2 (3.00 release for K2.5), A (/data/cd27), SWEEPS (3), GUARD_MB (3000), MINMEM_GB (6), STEPS, DRY=1.
# Code: $A/code (staged from the VM: CD27.md "Stage"): quantize/ourq.py (patched), v3/e3/{tilecd,exl3frac2,exl3frac,
#      exl3cb}, levelargs.py, assemble.py, cd27verdict.py. Log $A/<ARM>/cd27.log, results $A/result.txt.
# Stop: touch $A/<ARM>/STOP (between steps).
: ${RXM_WORK:=/root}; export RXM_WORK
ARM=${ARM:?ARM} BASE=${BASE:?BASE} CTL=${CTL:?CTL}
V2IN=${V2:-/data/release/qwen3.8-27b-exl3-v2-3.00bpw.rad}
A=${A:-/data/cd27}; O=$A/$ARM; CODE=$A/code; SWEEPS=${SWEEPS:-3}
GUARD_MB=${GUARD_MB:-3000} MINMEM_GB=${MINMEM_GB:-6}
OURQ=$CODE/quantize/ourq.py
STEPS=${STEPS:-pre noop arm asm pack score pair}
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=/data/exl3frac/build PYTORCH_HIP_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4

# swap27's pack / mix / census / score (sourced as swap27k.sh / asym27.sh do), then everything they write redirected to $A
source <(sed -n '/^: \${RXM_WORK/,/^quant() /p' /data/swap27/tools/swap27.sh | sed '$d')
source <(sed -n '/^packmixk() /,/^STOPF=/p' /data/swap27/tools/swap27k.sh | sed '$d')
S=$A V2=$V2IN; G=/data/swap27/golden-search27
mkdir -p $O; cd $A
L=$O/cd27.log; log() { echo "$(date -u +%H:%M:%S) $ARM: $*" | tee -a $L >> $A/result.txt; tail -n 1 $L; }
want() { [[ " $STEPS " == *" $1 "* ]] && [ ! -f $O/STOP ]; }
die() { log "$*: stop"; exit 1; }
memok() { local m=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  [ -n "$DRY" ] || [ $m -ge $MINMEM_GB ] || die "MemAvailable $m GB < $MINMEM_GB GB before $1 (rerun later)"; }
card() { if [ -n "$DRY" ]; then echo "DRY: $*"; return 0; fi
  ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; cd $RXM_WORK
    $T/memguard.sh $GUARD_MB nice -n 5 "$@" ); }
host() { if [ -n "$DRY" ]; then echo "DRY: $*"; return 0; fi; nice -n 10 "$@"; }

X=$($PY -I $CODE/levelargs.py $BASE) || die "levelargs: $BASE"
K=$(echo "$X" | awk '{for (i = 1; i < NF; i++) if ($i == "-K") print $(i + 1)}')
$PY -c "
for l in range(3, 64, 4): print(f'model.language_model.layers.{l}.self_attn.q_proj')" > $O/keys.txt
log "start: BASE $BASE (K$K), CTL $CTL, V2 $(basename $V2), sweeps $SWEEPS, steps: $STEPS"

# ---- pre
if want pre && [ ! -f $O/pre.done ]; then
  [ $(ls $BASE/q-L*.safetensors 2>/dev/null | wc -l) = 64 ] && [ -f $BASE/model.safetensors.index.json ] || die "$BASE: not 64 layer shards"
  [[ " $X " == *" --hout "* || " $X " == *" --hout-diag "* ]] && die "$BASE has an H_out: tile CD is one-sided only"
  grep -q -- '"--tile-cd"' $OURQ || die "$OURQ has no --tile-cd (stage the patched copy: CD27.md)"
  for f in levelargs.py assemble.py cd27verdict.py v3/e3/tilecd/tilecd.py v3/e3/exl3frac2/exl3frac2.py; do
    [ -f $CODE/$f ] || die "$CODE/$f missing"; done
  [ -f /data/swap27/$CTL.json.rows ] || die "no control rows /data/swap27/$CTL.json.rows"
  free=$(df --output=avail -BG $A | tail -1 | tr -dc 0-9)
  [ $free -ge 40 ] || { [ -n "$DRY" ] && echo "DRY: disk only $free GiB (a real run stops here)"; } || die "disk: $free GiB free under $A, an arm needs ~40 GiB (two levels + pack + mix)"
  log "base args: $X; disk $free GiB free"; [ -n "$DRY" ] || touch $O/pre.done
fi

quant() { # quant DIR ourq-args..: the 16 q_proj with BASE's args
  local d=$1; shift
  [ $(ls $d/q-L*.safetensors 2>/dev/null | wc -l) = 16 ] && return 0
  memok "quant $(basename $d)"; s=$(date +%s)
  card $PY -I $OURQ $X -o $d --device 0 --only $O/keys.txt "$@" >> $d.log 2>&1
  [ -n "$DRY" ] || [ $(ls $d/q-L*.safetensors 2>/dev/null | wc -l) = 16 ] || die "quant $d FAIL ($d.log; rerun resumes per layer)"
  log "$(basename $d): $(( $(date +%s) - s )) s"; }
if want noop && [ ! -f $O/noop.pass ] && [ ! -f $O/noop.fail ]; then
  quant $O/q-noop
  r=$(host $PY -I $CODE/assemble.py same $BASE $O/q-noop); rc=$?; log "$r"
  [ -n "$DRY" ] || { [ $rc = 0 ] && touch $O/noop.pass || touch $O/noop.fail; }
fi
NOOPFAIL=$([ -f $O/noop.fail ] && echo 1)
if want arm; then
  quant $O/q-cd --tile-cd $SWEEPS --tile-cd-families attn_qg
  [ -n "$DRY" ] || log "tile CD (quantization-domain rel, cd$SWEEPS / ldlq): $($PY -c "
import json, math
r = list({v['key']: v for v in map(json.loads, open('$O/q-cd/tilecd.jsonl'))}.values())   # a rerun's lines win
x = [v['ratio'] for v in r]
print(f'{len(r)} tensors, geo {math.exp(sum(map(math.log, x)) / len(x)):.4f}, range {min(x):.4f}..{max(x):.4f}, CD {sum(sum(v[\"sec\"].values()) for v in r):.0f} s total')")"
  [ -n "$DRY" ] || log "q_proj proxy_err cd / noop (after the scale refit): $($PY -c "
import json, math
p = lambda f: {j['key']: j['proxy_err'] for j in map(json.loads, open(f))}
a, b = p('$O/q-noop/proxy.jsonl'), p('$O/q-cd/proxy.jsonl'); x = [b[k] / a[k] for k in b if k in a]
print(f'{len(x)} tensors, geo {math.exp(sum(map(math.log, x)) / len(x)):.4f}, range {min(x):.4f}..{max(x):.4f}')")"
fi
if want asm; then
  [ -f $O/level-cd/assembled.json ] || host $PY -I $CODE/assemble.py build $BASE $O/q-cd $O/level-cd $O/keys.txt | tee -a $L \
    || die "assemble FAIL"
  [ -z "$NOOPFAIL" ] || [ -f $O/level-noop/assembled.json ] || host $PY -I $CODE/assemble.py build $BASE $O/q-noop $O/level-noop \
    $O/keys.txt | tee -a $L || die "assemble (noop) FAIL"
fi
if want pack; then
  if [ -n "$DRY" ]; then echo "DRY: packmixk ${ARM}cd $O/level-cd (K$K into $V2)"
  else packmixk ${ARM}cd $O/level-cd || die "pack ${ARM}cd FAIL"
       [ -z "$NOOPFAIL" ] || packmixk ${ARM}n $O/level-noop || die "pack ${ARM}n FAIL"; fi
fi
if want score; then
  if [ -n "$DRY" ]; then echo "DRY: score ${ARM}cd"
  else C=0 score ${ARM}cd || die "score ${ARM}cd FAIL"
       [ -z "$NOOPFAIL" ] || C=0 score ${ARM}n || die "score ${ARM}n FAIL"; fi
fi
if want pair && [ -z "$DRY" ] && [ -f $A/${ARM}cd.json.rows ]; then
  CR=$([ -n "$NOOPFAIL" ] && echo $A/${ARM}n.json.rows || echo /data/swap27/$CTL.json.rows)
  [ -f $CR ] || die "no control rows $CR"
  $PY -I $T/paired.py $G $CR $A/${ARM}cd.json.rows | tail -n 1 > $O/pair.json || die "paired.py FAIL"
  echo "{\"arm\": \"$ARM\", \"control\": \"$CR\", \"cd\": \"$A/${ARM}cd.json.rows\", \"noop\": \"$([ -n "$NOOPFAIL" ] && echo fail || echo pass)\"}" > $O/pair-meta.json
  log "pair $(basename ${CR%.json.rows}) -> ${ARM}cd: $(cat $O/pair.json)"
fi
log "CD27 $ARM DONE ($STEPS)"
