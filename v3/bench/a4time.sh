#!/bin/bash
# a4time.sh OUT.jsonl -- mining merge A4: what the int8 prefill path costs WITH its trellis decode, next to the f16 T2
# path and speedcal's bare iu8 matmul. One GEMM per run (ntbench, random codes, the NFQ build's exl3_r4d.so):
#   f16  = the T2 two-pass prefill (wdec to f16 + exl3r4d_gemm_t2)          NFQ_MASK unset
#   nfq8 = the int8 path (nfq_act_i8 + nfq_wdec_i8 + nfq_gemm_i8)            NFQ_MASK=8 (ntbench's one weight = index 0)
# shapes as one TP1 card sees them, K 2..6, M 512 / 2048 / 4096. Each line: {"path", "shape", "M", "N", "K", "bits",
# "ms"}; at M 2048 a second, profiled run adds "kern_ms" {kernel: ms per call} (rocprofv3 kernel trace) -- the
# decode share. Card: ROCR_VISIBLE_DEVICES (the caller holds the card lock). ~10 min.
set -u
O=$1; SO=${SO:-/root/nfk/home/kernels/exl3_r4d.so}; NB=${NB:-/root/ntbench}; IT=${ITERS:-20}
nm -D $SO | grep -q nfq_ || strings $SO | grep -q nfq_wdec || { echo "no NFQ kernels in $SO" >&2; exit 1; }
TMP=$(mktemp -d)
# name N K stacked
SHAPES="gate_up 34816 5120 1
down 5120 17408 0
attn_qg 12288 5120 0
gdn_qkv 10240 5120 0
attn_o 5120 6144 0"
one() {  # path shape N K stacked bits M
  local env=""; [ $1 = nfq8 ] && env="NFQ_MASK=8"
  local r; r=$(env $env $NB $SO $7 $3 $4 $6 $IT $5 2>&1 | tail -1)
  local ms=$(echo "$r" | grep -o '[0-9.]* ms' | cut -d' ' -f1) h=$(echo "$r" | grep -o 'out-hash [0-9a-f]*' | cut -d' ' -f2)
  if [ -z "$ms" ]; then echo "{\"path\": \"$1\", \"shape\": \"$2\", \"M\": $7, \"bits\": $6, \"error\": \"$(echo $r | tr -d '\"')\"}" >> $O; return; fi
  local kj="null"
  if [ $7 = 2048 ]; then
    rm -rf $TMP/p; env $env /opt/rocm/bin/rocprofv3 --kernel-trace --stats --output-format csv -d $TMP/p -o rp -- \
      $NB $SO $7 $3 $4 $6 $IT $5 > /dev/null 2>&1
    local st=$(find $TMP/p -name '*kernel_stats.csv' | head -1)
    [ -n "$st" ] && kj=$(python3 -c "
import csv, json, sys
r = {}
for d in csv.DictReader(open(sys.argv[1])):
    n = d['Name'].split('(')[0].split('<')[0].strip(); r[n] = round(float(d['AverageNs']) / 1e6, 4)
print(json.dumps(r))" $st)
  fi
  echo "{\"path\": \"$1\", \"shape\": \"$2\", \"M\": $7, \"N\": $3, \"K\": $4, \"bits\": $6, \"ms\": $ms, \"hash\": \"$h\", \"kern_ms\": $kj}" >> $O
}
while read -r n N K s; do
  for b in 2 3 4 5 6; do for M in 512 2048 4096; do one f16 $n $N $K $s $b $M; one nfq8 $n $N $K $s $b $M; done; done
done <<< "$SHAPES"
rm -rf $TMP
echo "A4TIME DONE $(grep -c '"ms"' $O) points, $(grep -c error $O) errors" >&2
