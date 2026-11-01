#!/bin/bash
# speedcal.sh OUT.jsonl -- speed calibration for V3's time table (mining merge A4/B7), one card (ROCR_VISIBLE_DEVICES).
#  1. shapes as one TP1 card sees them and as one TP2 rank sees them (column-split weights: half N; row-split: half K):
#     attn q+gate, GDN qkv, gate_up, down, o_proj / ssm_out; f16 (tlgemm, T2 core) vs iu8 vs iu4 (igemm, per-token)
#  2. M sweep 16 ... 4096 on gate_up and down (is "M < 512: f16 as fast" true? where does int start to pay?)
# Each line: {"kern", "shape", "split", "M", "N", "K", "ms", "tops", "watts_cap"}. WATTS (optional): record the power
# cap in effect (set it on the host first, e.g. rocm-smi --setpoweroverdrive 210; the container cannot), default "stock".
# Run twice (stock, then 210 W) for the board-condition column. Builds the binaries once in $BIN (default /data/bench).
set -u
O=$1; B=${BIN:-/data/bench}; mkdir -p $B; S=$(dirname "$(readlink -f "$0")")
W=${WATTS:-stock}; IT=${ITERS:-30}
[ -x $B/tlg ] || hipcc -O3 --offload-arch=gfx1201 -DMT=4 -DNPW=4 -DWM=4 -DWN=2 -DKC=2 -DWLDS=1 -DGROUP=8 $S/tlgemm.hip -o $B/tlg || exit 1
[ -x $B/ig8 ] || hipcc -O3 --offload-arch=gfx1201 -DOP=8 $S/igemm.hip -o $B/ig8 || exit 1
[ -x $B/ig4 ] || hipcc -O3 --offload-arch=gfx1201 -DOP=4 $S/igemm.hip -o $B/ig4 || exit 1
one() {   # kern shape split M N K
  local bin=${1/f16/tlg}; bin=${bin/iu8/ig8}; bin=${bin/iu4/ig4}
  local r; r=$($B/$bin $4 $5 $6 $IT 2>&1 | tail -1)
  local ms=$(echo "$r" | grep -o '[0-9.]* ms' | cut -d' ' -f1) tp=$(echo "$r" | grep -oE '[0-9.]+ T(FL)?OPS' | cut -d' ' -f1)
  [ -z "$ms" ] && { echo "{\"kern\": \"$1\", \"shape\": \"$2\", \"split\": \"$3\", \"M\": $4, \"N\": $5, \"K\": $6, \"error\": \"$(echo $r | tr -d '\"')\"}" >> $O; return; }
  echo "{\"kern\": \"$1\", \"shape\": \"$2\", \"split\": \"$3\", \"M\": $4, \"N\": $5, \"K\": $6, \"ms\": $ms, \"tops\": $tp, \"watts_cap\": \"$W\"}" >> $O
}
# name N K split-type (col = output rows split across ranks, row = input dim split)
SHAPES="attn_qg 12288 5120 col
gdn_qkv 10240 5120 col
gdn_z 6144 5120 col
gate_up 34816 5120 col
down 5120 17408 row
attn_o 5120 6144 row
ssm_out 5120 6144 row"
for k in f16 iu8 iu4; do
  while read -r n N K t; do
    one $k $n tp1 4096 $N $K
    if [ $t = col ]; then one $k $n tp2 4096 $((N / 2)) $K; else one $k $n tp2 4096 $N $((K / 2)); fi
  done <<< "$SHAPES"
  for M in 16 32 64 128 256 384 512 768 1024 2048; do
    one $k gate_up tp1 $M 34816 5120; one $k down tp1 $M 5120 17408
  done
done
echo "SPEEDCAL DONE $(grep -c '"ms"' $O) points, $(grep -c error $O) errors" >&2
