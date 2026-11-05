#!/bin/bash
# sib27.sh -- #87 at 27B (README.md): the 128 writer tensors (64 down_proj, 48 GDN out_proj, 16 o_proj) of one trunk
# level requantized against inputs from that level's own quantized siblings, asymmetric target W F (lam 1, rho 0.5),
# everything else the level's own. gpubox, card 0, idempotent (rerun after any failure; each step skips what is done).
#   pre    checks: base level (64 shards, stamp.json, codebook i8 / mul1), ourq --wpre, the lean mkdenoise, disk;
#          the trellis decode vs the C oracle on 4 base tensors (if the plugin sources are on the box)
#   gate   sibmom27 on layers 2-3 (one GDN, one attention) at GATE_TOK tokens: tok/s -> the estimate for TOK;
#          stops if it exceeds MAXH hours (raise MAXH or lower TOK)
#   mom    the sibling moments, TOK tokens of qad windows (WINS=moments: moments.py's list, the 0.8B-validated one;
#          long = collect.py's), layer-streamed, resumable per layer; half-window snapshots of layers 0, 3, 31, 63
#   mk     mkdenoise (lean path: 17408-wide groups) -> dn/hin (G) + dn/wpre (F^T)
#   dose   the same on the half snapshots: how far F moves between N/2 and N (reported, no gate)
#   noop   the writers requantized with the base's own settings and the calib H: must equal the base bit-exactly
#          (else the base is stale: code / settings changed since; NOOP_SOFT=1 continues, then run ctrl too)
#   arm    the writers requantized with --hin dn/hin --wpre dn/wpre (the base's settings and seeds otherwise)
#   asm    the assembled level: base + requantized writers -> $O/level-asym (an ourq-style export: exl3i src=...)
#   ctrl   (not in the default STEPS) writers-only noise floor: the writers at seed offset 7 on the calib H -> level-rs7
# Env: BASE (/data/v3ladder2/K3), TAG (K3), O (/data/sib27/$TAG), TOK (1000000), WINS (moments), LAM / RHO (1 / 0.5),
#      GATE_TOK (131072), MAXH (6), STEPS, NOOP_SOFT, DRY=1 (print the commands, run nothing).
# Copy this folder to $S (/root/sib27) first, with lean/mkdenoise.py as $S/mkdenoise.py until mkdenoise-lean.patch is
# merged (then MKD=<the merged copy>), and quantize/ourq.py to /root/ourq/ if it predates --wpre.
# Log $O/sib27.log; stop: touch $O/STOP (between steps and between layers of mom). Nothing here deletes data except
# the gate's own scratch ($O/gate) and the boundary files mom consumes.
: ${RXM_WORK:=/root}; export RXM_WORK
S=${SIB27_DIR:-/root/sib27}
BASE=${BASE:-/data/v3ladder2/K3} TAG=${TAG:-K3}; O=${O:-/data/sib27/$TAG}
TOK=${TOK:-1000000} WINS=${WINS:-moments} LAM=${LAM:-1} RHO=${RHO:-0.5} GATE_TOK=${GATE_TOK:-131072} MAXH=${MAXH:-6}
PY=${PY:-/data/venv-exl3/bin/python} CKPT=${RXM_CKPT:-/data/workspace/models/qwen38-27b-bf16}
META=$RXM_WORK/challenge/golden/calib HESS=$RXM_WORK/challenge/hessians/calib OURQ=$RXM_WORK/ourq/ourq.py
MKD=${MKD:-$S/mkdenoise.py} ORACLE_SRC=${ORACLE_SRC:-/root/rhv3/plugin/exl3import/common}
STEPS=${STEPS:-pre gate mom mk dose noop arm asm}
DN=$O/dn-l$LAM-r$RHO
export POOLS=${POOLS:-$RXM_WORK/challenge/pools-v3.json}
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=/data/exl3frac/build TRITON_CACHE_AUTOTUNING=1 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
mkdir -p $O; L=$O/sib27.log; log() { echo "$(date -u +%H:%M) $*" | tee -a $L; }
want() { [[ " $STEPS " == *" $1 "* ]] && [ ! -f $O/STOP ]; }
# card: one GPU job on card 0 (card lock, then host RAM shared, as ladder2 / night27)
card() { if [ -n "$DRY" ]; then echo "DRY: $*"; return 0; fi
  ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; "$@" ); }
host() { if [ -n "$DRY" ]; then echo "DRY: $*"; return 0; fi; "$@"; }
die() { log "$*: stop"; exit 1; }
mark() { [ -n "$DRY" ] || touch "$@"; }
cd $RXM_WORK
log "start: BASE $BASE, TOK $TOK ($WINS), lam $LAM rho $RHO, steps: $STEPS"

# ---- pre
X=$($PY -I $S/levelargs.py $BASE) || die "levelargs: $BASE"
K=$(echo "$X" | awk '{for (i = 1; i < NF; i++) if ($i == "-K") print $(i + 1)}')
CB=$(echo "$X" | grep -o -- '--codebook [a-z0-9]*' | awk '{print $2}'); CB=${CB:-mul1}
$PY -c "
for l in range(64):
    p = f'model.language_model.layers.{l}'
    print(f'{p}.self_attn.o_proj' if l % 4 == 3 else f'{p}.linear_attn.out_proj'); print(f'{p}.mlp.down_proj')" > $O/writers.txt
if want pre && [ ! -f $O/pre.done ]; then
  [ $(ls $BASE/q-L*.safetensors 2>/dev/null | wc -l) = 64 ] && [ -f $BASE/model.safetensors.index.json ] || die "$BASE: not 64 layer shards"
  case $CB in i8|mul1) ;; *) die "$BASE: codebook $CB (deq.py decodes i8 / mul1 only)";; esac
  grep -q -- '"--wpre"' $OURQ || die "$OURQ has no --wpre (copy quantize/ourq.py over)"
  grep -q lean_group $MKD || die "$MKD has no lean path (copy v3/sib27/lean/mkdenoise.py, or merge mkdenoise-lean.patch)"
  for f in sibmom27.py deq.py assemble.py levelargs.py cmpmom.py; do [ -f $S/$f ] || die "$S/$f missing"; done
  if [ -f $ORACLE_SRC/exl3_decode.c ]; then
    gcc -O2 -ffp-contract=off -shared -fPIC $ORACLE_SRC/exl3_decode.c $ORACLE_SRC/exl3_layout.c -I$ORACLE_SRC -o $O/exl3_decode.so -lm \
      || die "C oracle build"
    r=$(DEV=cpu nice $PY -I $S/deq.py oracle $BASE $O/exl3_decode.so 4 2>&1 | tail -1); log "decode vs C oracle: $r"
    [[ $r == *bit-exact* ]] || die "the trellis decode is not bit-exact on $BASE"
  else log "no C oracle sources at $ORACLE_SRC: decode check skipped (the 0.8B check covers the same code)"; fi
  need=$($PY -c "print(int(2 * $TOK * 5120 * 2 / 2**30 + 270 + 135 + 30))")   # boundaries, moments, dn, quants + level
  free=$(df --output=avail -BG $O | tail -1 | tr -dc 0-9)
  [ $free -ge $need ] || die "disk: $free GiB free under $O, the run needs ~$need GiB (README budget)"
  log "base args: $X; disk $free GiB free, ~$need GiB needed"; mark $O/pre.done
fi

# ---- gate: speed on one GDN + one attention layer -> estimate for TOK
if want gate && [ ! -f $O/gate.done ]; then
  rm -rf $O/gate
  card $PY -I $S/sibmom27.py $CKPT $META $O/gate --level $BASE --codebook $CB --layers 2-3 --tokens $GATE_TOK --wins $WINS \
    > $O/gate.log 2>&1 || die "gate FAIL ($O/gate.log)"
  if [ -z "$DRY" ]; then
    est=$($PY -c "
import re, sys
r = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r'^layer (\d+)(?: \(forward only\))?: \d+s, (\d+) tok/s', open('$O/gate.log').read(), re.M)}
h = $TOK * (48 / r[2] + 16 / r[3]) / 3600
print(f'{h:.2f} h for $TOK tokens (GDN {r[2]:.0f} tok/s, attention {r[3]:.0f} tok/s; forward-only {r.get(0, 0):.0f} tok/s)')")
    log "gate: $(grep -o 'peak GPU [0-9.]* GiB' $O/gate.log | sort -t' ' -k3 -n | tail -1); estimate $est"
    awk -v h="${est%% *}" -v m=$MAXH 'BEGIN { exit !(h <= m) }' || die "estimate over MAXH=$MAXH h"
    rm -rf $O/gate
  fi
  mark $O/gate.done
fi

# ---- mom
if want mom && [ ! -f $O/mom.done ]; then
  log "moments: start"
  card $PY -I $S/sibmom27.py $CKPT $META $O/mom --level $BASE --codebook $CB --tokens $TOK --wins $WINS --seed 0 \
    --snap-layers 0,3,31,63 >> $O/mom.log 2>&1
  rc=$?; [ $rc = 3 ] && { log "moments stopped (STOP); rerun to resume"; exit 0; }
  [ -n "$DRY" ] || grep -q "^MOMENTS DONE 128 groups" <(tail -n 2 $O/mom.log) || die "moments FAIL rc=$rc ($O/mom.log; rerun resumes per layer)"
  log "moments: $(tail -n 1 $O/mom.log)"; mark $O/mom.done
fi

# ---- mk (+ dose)
if want mk && [ ! -f $DN.done ]; then
  card env DEV=cuda $PY -I $MKD $HESS $O/mom $DN --lam $LAM --rho $RHO --ckpt $CKPT > $DN.log 2>&1 || die "mkdenoise FAIL ($DN.log)"
  [ -n "$DRY" ] || [ $(wc -l < $DN/stats.jsonl) = 128 ] || die "mkdenoise: not 128 groups"
  [ -n "$DRY" ] || log "mkdenoise: $($PY -c "
import json
r = [json.loads(l) for l in open('$DN/stats.jsonl')]
wf = [v for x in r for k, v in x.items() if k.startswith('wf_')]
print(f'{len(r)} groups; clip > 0 in {sum(x[\"clip\"] > 0 for x in r)} (max {max(x[\"clip\"] for x in r):.1e}); hsub median {sorted(x[\"hsub_vs_calib\"] for x in r)[len(r) // 2]:.3f};',
      f'F_dev {min(x.get(\"F_dev\", 0) for x in r):.3f}..{max(x.get(\"F_dev\", 0) for x in r):.3f}; |WF|/|W| {min(wf, default=1):.4f}..{max(wf, default=1):.4f}')")"
  mark $DN.done
fi
if want dose && [ ! -f $DN-half.done ] && [ -d $O/mom/half ]; then
  card env DEV=cuda $PY -I $MKD $HESS $O/mom/half $DN-half --lam $LAM --rho $RHO > $DN-half.log 2>&1 || die "mkdenoise (half) FAIL"
  r=$(card env DEV=cuda $PY -I $S/cmpmom.py fdose $DN $DN-half) || die "dose compare FAIL"; log "dose (N/2 vs N, layers 0 3 31 63): $r"
  mark $DN-half.done
fi

# ---- noop gate, arm, assembly
quant() { # quant DIR ourq-args..: the 128 writers, the base's settings
  local d=$1; shift
  [ $(ls $d/q-L*.safetensors 2>/dev/null | wc -l) = 64 ] && return 0
  s=$(date +%s); card $PY -I $OURQ $X -o $d --device 0 --only $O/writers.txt "$@" >> $d.log 2>&1
  [ -n "$DRY" ] || [ $(ls $d/q-L*.safetensors 2>/dev/null | wc -l) = 64 ] || die "quant $d FAIL ($d.log)"
  log "$(basename $d): $(( ($(date +%s) - s) / 60 )) min"; }
if want noop && [ ! -f $O/noop.done ]; then
  quant $O/q-noop
  r=$(host $PY -I $S/assemble.py same $BASE $O/q-noop); rc=$?; log "$r"
  [ $rc = 0 ] || [ -n "$NOOP_SOFT" ] || die "NOOP GATE FAIL (the base is not reproducible: NOOP_SOFT=1 and run ctrl)"
  mark $O/noop.done
fi
if want arm; then
  W=""; [ "$LAM" = "$RHO" ] || W="--wpre $DN/wpre"
  [ -n "$DRY" ] || [ -z "$W" ] || [ $(ls $DN/wpre/*.safetensors 2>/dev/null | wc -l) = 128 ] || die "$DN/wpre: not 128 transforms"
  quant $O/q-asym --hin $DN/hin $W
fi
if want asm && [ ! -f $O/level-asym/assembled.json ]; then
  host $PY -I $S/assemble.py build $BASE $O/q-asym $O/level-asym $O/writers.txt | tee -a $L || die "assemble FAIL"
fi
if want ctrl; then
  quant $O/q-rs7 --seed-offset 7
  [ -f $O/level-rs7/assembled.json ] || host $PY -I $S/assemble.py build $BASE $O/q-rs7 $O/level-rs7 $O/writers.txt | tee -a $L
fi
log "SIB27 DONE ($STEPS). Eval recipe line for the assembled level: exl3i src=$O/level-asym codebook=${CB/i8/mul1i8} rate=$K"
