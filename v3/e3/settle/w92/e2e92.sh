#!/bin/bash
# e2e92.sh -- (gpubox, card 0, ~30 min) lead note 11-08 10:00 PM CST, report only: does the fs0 mix (#122's full-stack
# K2.5 level in V2 3.00) work end to end beyond KLD scoring?
#   a8     the A8 path: mix-fs0's trunk with act a8g128 on all 336 trunk weights (the fast variant's mode; radmix act lines,
#          base V2 3.00) -> mix-fs0a8.rad; scored on golden-search27c-c together with mix-fs0.rad on the SAME install
#          (install-117 + /data/a4cost/home, the home with the A8 / A4 kernels): KLD finite, ratio a8 / f16 (pair84.py)
#   gen    radiance's OpenAI server on 127.0.0.1 (loopback only) with mix-fs0.rad: 3 chat prompts (code, chat, a short
#          synthetic tool trace), greedy, 300 tokens -> $O/gen-*.json
#   speed  the same server: an ~8k-token prompt, max_tokens 1, 3 runs (the first warms) for mix-fs0.rad and V2 3.00;
#          prefill tok/s = prompt tokens / wall; VRAM = the server log's budget (static weights, activation arena, kv)
# Locks: card0.lock, then ram.lock shared for GPU steps; the radmix step holds ram.lock exclusive (+ conv.lock).
set -u
: ${RXM_WORK:=/root}
A=/data/w92; O=$A/e2e; PY=/data/venv-exl3/bin/python; I=/data/install-117 H=/data/a4cost/home
V2=/data/release/qwen3.8-27b-exl3-v2-3.00bpw.rad M=$A/mix/mix-fs0.rad MA=$A/mix/mix-fs0a8.rad PORT=18092
mkdir -p $O; cd $O
log() { echo "$(date -u +%H:%M:%S) $*" | tee -a $O/result.txt; }
cardrun() { ( exec 5>$RXM_WORK/card0.lock; flock 5; exec 7>$RXM_WORK/ram.lock; flock -s 7; "$@" ); }
TRUNK() { env PYTHONPATH=$A/code/pipeline $PY -c "from rxq import mix; print(''.join(n + chr(10) for n in mix.trunk_names()), end='')"; }

# ---- a8: the mix and two scores on the calib part
if [ ! -f $MA ]; then
  { TRUNK | sed "s|\$| $M|"; TRUNK | sed 's|^|act |; s|$| a8g128|'; } > $O/fs0a8.mix
  ( exec 9>$RXM_WORK/conv.lock; flock 9; exec 7>$RXM_WORK/ram.lock; flock 7
    /data/a4cost/tools/radmix $MA.part $V2 $O/fs0a8.mix > $O/radmix-a8.log 2>&1 && mv $MA.part $MA ) || { log "E2E FAIL radmix a8"; exit 1; }
  $PY -I /data/swap27/tools/census.py $MA --mix $O/fs0a8.mix --base $V2 > $O/census-a8.txt 2>&1
  log "a8 mix: $(grep -E 'exl3i' $O/census-a8.txt | head -2 | tr -s ' ' | tr '\n' ';') $(tail -1 $O/census-a8.txt)"
fi
score() { local tag=$1 m=$2 G=$A/golden-search27c-c o=$O/$1-c
  [ -f $o.json.rows ] && return 0
  cardrun bash -c "ROCR_VISIBLE_DEVICES=0 RADIANCE_HOME=$H $I/bin/radiance --model $m --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 \
    --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $o.json.part > $O/score-$tag.log 2>&1 && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json"
  [ -f $o.json.rows ] && log "score $tag (install-117): $(grep -o '"kld[^,]*,' $o.json | head -1)" || { log "E2E FAIL score $tag"; exit 1; }
}
score fs0f16 $M; score fs0a8 $MA
$PY -I - $A $O > $O/a8.txt <<'EOF'
import sys, numpy as np
sys.path.insert(0, f"{sys.argv[1]}/code/v3/e3/settle/w84")
from pair84 import pair, fmt, doc_sums
A, O = sys.argv[1:3]; G = f"{A}/golden-search27c-c"
R = np.fromfile(f"{O}/fs0a8-c.json.rows", np.float32).reshape(-1, 4)
print("a8 rows finite:", bool(np.isfinite(R[:, 0]).all()))
p = pair([(G, f"{O}/fs0f16-c.json.rows", f"{O}/fs0a8-c.json.rows")])
print("f16 -> a8 (calib part, install-117):", fmt(p))
r8 = np.fromfile(f"{A}/scores/fs0-c.json.rows", np.float32).reshape(-1, 4); r7 = np.fromfile(f"{O}/fs0f16-c.json.rows", np.float32).reshape(-1, 4)
print("fs0 f16 rows, install-116 vs install-117: identical" if np.array_equal(r8, r7) else
      f"fs0 f16 rows, install-116 vs install-117: max |dKL| {np.abs(r8[:, 0] - r7[:, 0]).max():.3g}")
print(f"A8 {p['r']:.4f} z {p['z']:+.2f}")
EOF
log "$(tail -1 $O/a8.txt) ($O/a8.txt)"

# ---- gen + speed: one server per container
serve() { local tag=$1 m=$2
  [ -f $O/speed-$tag.json ] && return 0
  cardrun bash $A/code/v3/e3/settle/w92/serve92.sh $tag $m $O $PORT $I $H || { log "E2E FAIL serve $tag ($O/serve-$tag.log)"; exit 1; }
  log "serve $tag: $(cat $O/speed-$tag.json)"
}
serve fs0 $M; serve v2 $V2
log "E2E92 DONE"
