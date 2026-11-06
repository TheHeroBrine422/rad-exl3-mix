#!/bin/bash
# run.sh CORPUS.jsonl OUT -- a 27B bf16 reference golden on gpubox card 0 with mkgold27.py (README.md beside it).
# Env: N stages (default 3), DT scratch dtype (bf16 default: lossless, the residual stream is bf16), BLOCK documents per
# pass (default 0 = all at once; set it to bound the scratch, see README), ATTN_CHUNK (0 default; 2048 if sdpa runs
# out of memory on long documents), M the checkpoint, SCR the scratch dir, LOCK the card lock, VRAM_CAP_GB.
# Restartable: rerun the same command after a kill; finished (stage, document) pairs are skipped. The lock is taken
# per block, so other card-0 jobs can interleave between blocks.
set -euo pipefail
[ $# -eq 2 ] || { sed -n 2,7p "$0"; exit 2; }
CORPUS=$1; OUT=$2
D=$(cd "$(dirname "$0")" && pwd)
M=${M:-/data/workspace/models/qwen38-27b-bf16}
SCR=${SCR:-/data/gold27-scratch/$(basename "$OUT")}
LOCK=${LOCK:-/root/card0.lock}
N=${N:-3}; DT=${DT:-bf16}; BLOCK=${BLOCK:-0}; ATTN_CHUNK=${ATTN_CHUNK:-0}
export ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-0} VRAM_CAP_GB=${VRAM_CAP_GB:-30}
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}   # flash sdpa on gfx1201
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} TOKENIZERS_PARALLELISM=false
mkdir -p "$SCR" "$(dirname "$OUT")"
ND=$(grep -c . "$CORPUS"); TOKS=$(python3 -c "import sys,json; print(sum(len(json.loads(l)['prompt']) for l in open(sys.argv[1]) if l.strip()) // 4)" "$CORPUS")
B=$BLOCK; [ "$B" -gt 0 ] || B=$ND
ES=$([ "$DT" = bf16 ] && echo 2 || echo 4)
echo "corpus $CORPUS: $ND documents, ~$TOKS tokens (chars/4); $N stages, scratch $DT in $SCR, blocks of $B documents"
echo "needs ~$((TOKS * 4096 / 1000000000 + 1)) GB in $(dirname "$OUT") (top.bin) and ~$((TOKS * B / ND * 5120 * ES / 1000000000 + 1)) GB peak scratch"
df -h "$(dirname "$OUT")" "$SCR" | sed 1d
python3 -c "import sys; sys.path.insert(0, sys.argv[1]); from common import kernel_paths; import torch; print('kernels', kernel_paths(), 'device', torch.cuda.get_device_name(0))" "$D/../../proxy2b"
ARGS=(--scratch "$SCR" --stages "$N" --scratch-dtype "$DT" --attn-chunk "$ATTN_CHUNK")
for ((lo = 0; lo < ND; lo += B)); do
  hi=$((lo + B)); [ $hi -le $ND ] || hi=$ND
  echo "== documents $lo:$hi ($(TZ=America/Chicago date +'%-I:%M %p %Z'))"
  flock "$LOCK" nice -n 5 python3 "$D/mkgold27.py" "$M" "$CORPUS" "$OUT" "${ARGS[@]}" --docs "$lo:$hi"
done
[ -f "$OUT/kld.json" ] && echo "golden done: $OUT" || { echo "not finished: rerun the same command"; exit 1; }
