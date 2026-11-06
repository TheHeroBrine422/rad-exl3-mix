#!/bin/bash
# asym27.sh -- #87 at 27B on one swap-test arm (rule: v3/yaqa/ASYM27.md): the 128 writer tensors (64 down_proj, 48 GDN
# out_proj, 16 o_proj) of the level BASE requantized against the sibling-sequential asymmetric target (lam 1, rho 0:
# the clean-output target W F, #87-05), on moments of 1M qad tokens taken through BASE's own quantized siblings;
# everything else is BASE's. Then pack (exl3i, mul1i8, rate K), mix into V2, census and score on golden-search27 with
# swap27's own functions (same build, golden and census as every swap arm), and pair against the control CTL.
# gpubox, card 0 under /root/card0.lock + /root/ram.lock (shared), as swap27k.sh. Idempotent: rerun after any failure;
# each step skips what is done (sibmom27 per layer, mkdenoise per 8-layer chunk, ourq per layer shard).
#   pre    BASE (64 shards, i8 / mul1, its ourq args via levelargs.py), the decode vs the plugin's exl3i C oracle on
#          4 sibling / writer tensors of BASE (bit-exact or stop), disk, CTL's score rows
#   gate   sibmom27 on layers 2-3 (one GDN, one attention) at GATE_TOK tokens -> estimate for TOK; stop over MAXH
#   mom    sibling moments: TOK tokens of qad windows (moments.py's list, the 0.8B arms'), layer-streamed; half
#          snapshots of layers 0 / 3 / 31 / 63 for the dose check
#   mk     mkdenoise (lean path, merged) -> dn/hin (G) + dn/wpre (F^T), in 8-layer chunks (resumable), merged by links
#   dose   F at N/2 vs N on the snapshot layers (reported)
#   noop   the writers requantized with BASE's own args and the calib H: bit-identical to BASE (PASS) or not (FAIL:
#          then the control is BASE + these writers, "<ARM>n", packed and scored here; a code / build change since
#          BASE would otherwise be read as the asym effect)
#   arm    the writers requantized with --hin dn/hin --wpre dn/wpre (BASE's args, seeds and H_out if any)
#   asm    BASE + new writers -> level-asym (and level-noop if the noop gate failed)
#   pack   swap27k's packmixk: rad-convert (exl3i src=level codebook=mul1i8 rate=K) -> mix into V2 -> census
#   score  swap27's score on golden-search27 (card 0, bf16 KV)
#   pair   tools/paired.py CTL rows -> <ARM>a rows (paired per document, bootstrap SE) -> OUT/pair.json, result.txt
# Env: ARM (tag, e.g. base), BASE (level dir), CTL (the swap-test arm scored on BASE: /data/swap27/CTL.json.rows),
#      V2 (release to mix into; 3.00 for K2.5, 4.00 for K4), A (/data/asym27), TOK (1000000), GATE_TOK (131072),
#      MAXH (5: moment hours allowed by the gate), GUARD_MB (3000: memguard for our own jobs, above the queue's 2500 so
#      ours die first), MINMEM_GB (6: a card step starts only with this much MemAvailable), STEPS, DRY=1.
# Code: $A/code (staged from the VM: ASYM27.md "Stage"). Log $A/<ARM>/asym27.log, results $A/result.txt.
# Stop: touch $A/<ARM>/STOP (between steps; sibmom27 between layers).
: ${RXM_WORK:=/root}; export RXM_WORK
ARM=${ARM:?ARM} BASE=${BASE:?BASE} CTL=${CTL:?CTL}
V2IN=${V2:-/data/release/qwen3.8-27b-exl3-v2-3.00bpw.rad}
A=${A:-/data/asym27}; O=$A/$ARM; CODE=$A/code
TOK=${TOK:-1000000} WINS=moments LAM=1 RHO=0 GATE_TOK=${GATE_TOK:-131072} MAXH=${MAXH:-5}
GUARD_MB=${GUARD_MB:-3000} MINMEM_GB=${MINMEM_GB:-6}
CKPT=${RXM_CKPT:-/data/workspace/models/qwen38-27b-bf16}
META=$RXM_WORK/challenge/golden/calib HESS=$RXM_WORK/challenge/hessians/calib OURQ=$RXM_WORK/ourq/ourq.py
ORACLE_SRC=${ORACLE_SRC:-/data/rh117/plugin/exl3import/common}
STEPS=${STEPS:-pre gate mom mk dose noop arm asm pack score pair}
DN=$O/dn-l$LAM-r$RHO
export POOLS=${POOLS:-$RXM_WORK/challenge/pools-v3.json}
export ROCR_VISIBLE_DEVICES=0 EXL3_QT_OPTIMIZED=1 ROCM_HOME=/opt/rocm/core-10.1 PYTORCH_ROCM_ARCH=gfx1201 \
       TORCH_EXTENSIONS_DIR=/data/exl3frac/build PYTORCH_HIP_ALLOC_CONF=expandable_segments:True \
       TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

# swap27's pack / mix / census / score (sourced as swap27k.sh does), then everything they write redirected to $A
source <(sed -n '/^: \${RXM_WORK/,/^quant() /p' /data/swap27/tools/swap27.sh | sed '$d')
source <(sed -n '/^packmixk() /,/^STOPF=/p' /data/swap27/tools/swap27k.sh | sed '$d')
S=$A V2=$V2IN; G=/data/swap27/golden-search27
mkdir -p $O; cd $A
L=$O/asym27.log; log() { echo "$(date -u +%H:%M:%S) $ARM: $*" | tee -a $L >> $A/result.txt; tail -n 1 $L; }
want() { [[ " $STEPS " == *" $1 "* ]] && [ ! -f $O/STOP ]; }
die() { log "$*: stop"; exit 1; }
mark() { [ -n "$DRY" ] || touch "$@"; }
memok() { local m=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  [ -n "$DRY" ] || [ $m -ge $MINMEM_GB ] || die "MemAvailable $m GB < $MINMEM_GB GB before $1 (rerun later)"; }
# card CMD..: one GPU job on card 0 (card lock, then host RAM shared), under our own memory guard
card() { if [ -n "$DRY" ]; then echo "DRY: $*"; return 0; fi
  ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; cd $RXM_WORK
    $T/memguard.sh $GUARD_MB nice -n 5 "$@" ); }
host() { if [ -n "$DRY" ]; then echo "DRY: $*"; return 0; fi; nice -n 10 "$@"; }

X=$($PY -I $CODE/levelargs.py $BASE) || die "levelargs: $BASE"
K=$(echo "$X" | awk '{for (i = 1; i < NF; i++) if ($i == "-K") print $(i + 1)}')
CB=$(echo "$X" | grep -o -- '--codebook [a-z0-9]*' | awk '{print $2}'); CB=${CB:-mul1}
$PY -c "
for l in range(64):
    p = f'model.language_model.layers.{l}'
    print(f'{p}.self_attn.o_proj' if l % 4 == 3 else f'{p}.linear_attn.out_proj'); print(f'{p}.mlp.down_proj')" > $O/writers.txt
log "start: BASE $BASE (K$K $CB), CTL $CTL, V2 $(basename $V2), TOK $TOK, lam $LAM rho $RHO, steps: $STEPS"

# ---- pre
if want pre && [ ! -f $O/pre.done ]; then
  [ $(ls $BASE/q-L*.safetensors 2>/dev/null | wc -l) = 64 ] && [ -f $BASE/model.safetensors.index.json ] || die "$BASE: not 64 layer shards"
  case $CB in i8|mul1) ;; *) die "$BASE: codebook $CB (deq.py decodes i8 / mul1 only)";; esac
  grep -q -- '"--wpre"' $OURQ || die "$OURQ has no --wpre"
  grep -q lean_group $CODE/mkdenoise.py || die "$CODE/mkdenoise.py has no lean path"
  [ -f /data/swap27/$CTL.json.rows ] || die "no control rows /data/swap27/$CTL.json.rows (score CTL first)"
  for f in sibmom27.py deq.py assemble.py levelargs.py cmpmom.py verdict.py; do [ -f $CODE/$f ] || die "$CODE/$f missing"; done
  [ -f $A/exl3i_oracle.so ] || gcc -O2 -ffp-contract=off -shared -fPIC $ORACLE_SRC/exl3i_decode.c $ORACLE_SRC/exl3_decode.c \
    $ORACLE_SRC/exl3_layout.c -I$ORACLE_SRC -o $A/exl3i_oracle.so -lm || die "exl3i C oracle build ($ORACLE_SRC)"
  memok oracle
  P=model.language_model.layers; KEYS=$P.0.linear_attn.in_proj_z,$P.0.linear_attn.out_proj,$P.3.self_attn.k_proj,$P.3.self_attn.o_proj
  r=$(host env DEV=cpu $T/memguard.sh $GUARD_MB $PY -I $CODE/deq.py oracle $BASE $A/exl3i_oracle.so $KEYS 2>&1 | tail -n 1)
  log "decode vs the exl3i C oracle: $r"
  [ -n "$DRY" ] || [[ $r == *bit-exact* ]] || die "the trellis decode is not bit-exact on $BASE"
  need=$($PY -c "print(int(2 * $TOK * 5120 * 2 / 2**30 + 262 + 135 + 10 + 2 * 8 + 12))")   # boundaries, moments, dn, quants, levels, mix
  free=$(df --output=avail -BG $A | tail -1 | tr -dc 0-9)
  [ $free -ge $need ] || die "disk: $free GiB free under $A, the arm needs ~$need GiB (ASYM27.md budget)"
  log "base args: $X; disk $free GiB free, ~$need GiB needed"; mark $O/pre.done
fi

# ---- gate: speed on one GDN + one attention layer -> estimate for TOK
if want gate && [ ! -f $O/gate.done ]; then
  memok gate; rm -rf $O/gate
  card $PY -I $CODE/sibmom27.py $CKPT $META $O/gate --level $BASE --codebook $CB --layers 2-3 --tokens $GATE_TOK --wins $WINS \
    > $O/gate.log 2>&1 || die "gate FAIL ($O/gate.log)"
  if [ -z "$DRY" ]; then
    est=$($PY -c "
import re
r = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r'^layer (\d+)(?: \(forward only\))?: \d+s, (\d+) tok/s', open('$O/gate.log').read(), re.M)}
h = $TOK * (48 / r[2] + 16 / r[3]) / 3600
print(f'{h:.2f} h for $TOK tokens (GDN {r[2]:.0f} tok/s, attention {r[3]:.0f} tok/s; forward-only {r.get(0, 0):.0f} tok/s)')")
    log "gate: $(grep -o 'peak GPU [0-9.]* GiB' $O/gate.log | sort -t' ' -k3 -n | tail -1), $(grep -o 'peak [0-9]* MiB' $O/gate.log | sort -k2 -n | tail -1) RSS; estimate $est"
    awk -v h="${est%% *}" -v m=$MAXH 'BEGIN { exit !(h <= m) }' || die "estimate over MAXH=$MAXH h"
    rm -rf $O/gate
  fi
  mark $O/gate.done
fi

# ---- mom
if want mom && [ ! -f $O/mom.done ]; then
  memok mom; log "moments: start"
  card $PY -I $CODE/sibmom27.py $CKPT $META $O/mom --level $BASE --codebook $CB --tokens $TOK --wins $WINS --seed 0 \
    --snap-layers 0,3,31,63 >> $O/mom.log 2>&1
  rc=$?; [ $rc = 3 ] && { log "moments stopped (STOP); rerun to resume"; exit 0; }
  [ $rc = 99 ] && die "moments killed by the memory guard (MemAvailable < $GUARD_MB MB; rerun resumes per layer)"
  [ -n "$DRY" ] || grep -q "^MOMENTS DONE 128 groups" <(tail -n 2 $O/mom.log) || die "moments FAIL rc=$rc ($O/mom.log; rerun resumes per layer)"
  log "moments: $(tail -n 1 $O/mom.log); $(grep -o 'peak GPU [0-9.]* GiB' $O/mom.log | sort -k3 -n | tail -1), $(grep -o 'peak [0-9]* MiB' $O/mom.log | sort -k2 -n | tail -1) RSS"
  mark $O/mom.done
fi

# ---- mk: mkdenoise per 8-layer chunk (a kill loses one chunk, not the 1-2 h run), merged into dn/{hin,wpre} by links
if want mk && [ ! -f $DN.done ]; then
  for c in $(seq 0 8 56); do
    [ -f $DN-c$c.done ] && continue
    memok "mk chunk $c"; rm -rf $O/momc/c$c $DN-c$c; mkdir -p $O/momc/c$c
    for l in $(seq $c $((c + 7))); do ln -s $O/mom/model.language_model.layers.$l.*.input.safetensors $O/momc/c$c/; done
    [ -n "$DRY" ] || [ $(ls $O/momc/c$c | wc -l) = 16 ] || die "mk chunk $c: not 16 moment files"
    card env DEV=cuda $PY -I $CODE/mkdenoise.py $HESS $O/momc/c$c $DN-c$c --lam $LAM --rho $RHO --ckpt $CKPT > $DN-c$c.log 2>&1 \
      || die "mkdenoise chunk $c FAIL ($DN-c$c.log)"
    [ -n "$DRY" ] || [ $(wc -l < $DN-c$c/stats.jsonl) = 16 ] || die "mkdenoise chunk $c: not 16 groups"
    mark $DN-c$c.done
  done
  if [ -z "$DRY" ]; then
    rm -rf $DN; mkdir -p $DN/hin $DN/wpre
    for c in $(seq 0 8 56); do ln -s $DN-c$c/hin/* $DN/hin/; ln -s $DN-c$c/wpre/* $DN/wpre/; cat $DN-c$c/stats.jsonl; done > $DN/stats.jsonl
    [ $(ls $DN/hin | wc -l) = 128 ] && [ $(ls $DN/wpre | wc -l) = 128 ] && [ $(wc -l < $DN/stats.jsonl) = 128 ] || die "dn merge: not 128 groups"
    log "mkdenoise: $($PY -c "
import json
r = [json.loads(l) for l in open('$DN/stats.jsonl')]
wf = [v for x in r for k, v in x.items() if k.startswith('wf_')]
print(f'{len(r)} groups; clip > 0 in {sum(x[\"clip\"] > 0 for x in r)} (max {max(x[\"clip\"] for x in r):.1e}); hsub median {sorted(x[\"hsub_vs_calib\"] for x in r)[len(r) // 2]:.3f};',
      f'F_dev {min(x.get(\"F_dev\", 0) for x in r):.3f}..{max(x.get(\"F_dev\", 0) for x in r):.3f}; |WF|/|W| {min(wf, default=1):.4f}..{max(wf, default=1):.4f}')")"
  fi
  mark $DN.done
fi
if want dose && [ ! -f $DN-half.done ] && [ -d $O/mom/half ]; then
  memok dose
  card env DEV=cuda $PY -I $CODE/mkdenoise.py $HESS $O/mom/half $DN-half --lam $LAM --rho $RHO > $DN-half.log 2>&1 || die "mkdenoise (half) FAIL"
  r=$(card env DEV=cuda $PY -I $CODE/cmpmom.py fdose $DN $DN-half | tail -n 1) || die "dose compare FAIL"
  log "dose (|F_N/2 - F_N| / |F_N - I|, layers 0 3 31 63): $r"; mark $DN-half.done
fi

# ---- noop gate, arm, assembly
quant() { # quant DIR ourq-args..: the 128 writers with BASE's args
  local d=$1; shift
  [ $(ls $d/q-L*.safetensors 2>/dev/null | wc -l) = 64 ] && return 0
  memok "quant $(basename $d)"; s=$(date +%s)
  card $PY -I $OURQ $X -o $d --device 0 --only $O/writers.txt "$@" >> $d.log 2>&1
  [ -n "$DRY" ] || [ $(ls $d/q-L*.safetensors 2>/dev/null | wc -l) = 64 ] || die "quant $d FAIL ($d.log; rerun resumes per layer)"
  log "$(basename $d): $(( ($(date +%s) - s) / 60 )) min"; }
if want noop && [ ! -f $O/noop.pass ] && [ ! -f $O/noop.fail ]; then
  quant $O/q-noop
  r=$(host $PY -I $CODE/assemble.py same $BASE $O/q-noop); rc=$?; log "$r"
  [ -n "$DRY" ] || { [ $rc = 0 ] && touch $O/noop.pass || touch $O/noop.fail; }
fi
NOOPFAIL=$([ -f $O/noop.fail ] && echo 1)
if want arm; then
  [ -n "$DRY" ] || [ $(ls $DN/wpre/*.safetensors 2>/dev/null | wc -l) = 128 ] || die "$DN/wpre: not 128 transforms"
  quant $O/q-asym --hin $DN/hin --wpre $DN/wpre
fi
if want asm; then
  [ -f $O/level-asym/assembled.json ] || host $PY -I $CODE/assemble.py build $BASE $O/q-asym $O/level-asym $O/writers.txt | tee -a $L \
    || die "assemble FAIL"
  [ -z "$NOOPFAIL" ] || [ -f $O/level-noop/assembled.json ] || host $PY -I $CODE/assemble.py build $BASE $O/q-noop $O/level-noop \
    $O/writers.txt | tee -a $L || die "assemble (noop) FAIL"
fi

# ---- pack, score, pair (swap27's functions; arms <ARM>a, and <ARM>n = the control if the noop gate failed)
if want pack; then
  if [ -n "$DRY" ]; then echo "DRY: packmixk ${ARM}a $O/level-asym (K$K into $V2)"
  else packmixk ${ARM}a $O/level-asym || die "pack ${ARM}a FAIL"
       [ -z "$NOOPFAIL" ] || packmixk ${ARM}n $O/level-noop || die "pack ${ARM}n FAIL"; fi
fi
if want score; then
  if [ -n "$DRY" ]; then echo "DRY: score ${ARM}a"
  else C=0 score ${ARM}a || die "score ${ARM}a FAIL"
       [ -z "$NOOPFAIL" ] || C=0 score ${ARM}n || die "score ${ARM}n FAIL"; fi
fi
if want pair && [ -z "$DRY" ] && [ -f $A/${ARM}a.json.rows ]; then
  CR=$([ -n "$NOOPFAIL" ] && echo $A/${ARM}n.json.rows || echo /data/swap27/$CTL.json.rows)
  [ -f $CR ] || die "no control rows $CR"
  $PY -I $T/paired.py $G $CR $A/${ARM}a.json.rows | tail -n 1 > $O/pair.json || die "paired.py FAIL"
  echo "{\"arm\": \"$ARM\", \"control\": \"$CR\", \"asym\": \"$A/${ARM}a.json.rows\", \"noop\": \"$([ -n "$NOOPFAIL" ] && echo fail || echo pass)\"}" > $O/pair-meta.json
  log "pair $(basename ${CR%.json.rows}) -> ${ARM}a: $(cat $O/pair.json)"
  [ -n "$NOOPFAIL" ] && log "pair $CTL -> ${ARM}n (code drift on the writers): $($PY -I $T/paired.py $G /data/swap27/$CTL.json.rows $A/${ARM}n.json.rows | tail -n 1)"
fi
log "ASYM27 $ARM DONE ($STEPS)"
