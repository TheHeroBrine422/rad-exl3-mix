#!/bin/bash
# dkm.sh CARD ROUNDS "RATES" "ROWS" -- decode GEMV bench (bench_twopass, M 1, every trunk shape + the LM head, --cold), rows
# interleaved within each round and rate; ROW = BUILD:CODEBOOK:MODE[:ENV] (MODE a8 = the act a8g128 row: mad's GEMV, mul1i8's
# nt_pass; f16 = the f16 row), e.g. "b1:mad:a8 b0:mul1i8:a8 b1:mad:a8:EXL3I_MAD_GEMV=0". Card lock held for the whole run.
# CSV + txt into res/$DKO (dk)/rR-TAG-kK.*; skips finished units. (w150, #177-02; w137's dk.sh with codebooks and --cold)
C=$1; N=$2; RATES=$3; ROWS=$4; W=/data/q27/w150; O=$W/res/${DKO:-dk}; mkdir -p $O
export HIP_VISIBLE_DEVICES=$C LD_LIBRARY_PATH=/opt/rocm-10.1/lib
exec 9>/data/q27/locks/card$C.lock; flock 9
SH=${SH:-gate_up,down,gdn_qkv,gdn_z,attn_qg,attn_k,attn_v,attn_o,ssm_out,head}
for r in $(seq 1 $N); do
  for k in $RATES; do
    for row in $ROWS; do
      IFS=: read b cb md ev <<< "$row"; tag=$(echo "$row" | tr ':=' '__')
      B=$W/$b/pb/exl3i; f=$O/r$r-$tag-k$k
      [ -f $f.csv ] && continue
      env $ev $B/bench_twopass $B/radiance_home/kernels/exl3i.so --rate $k --codebook $cb --m ${MS:-1} --modes $md --reps ${REPS:-40} \
        --shapes $SH --cold ${BURST:+--burst $BURST} --csv $f.csv.part > $f.txt 2>&1 && mv $f.csv.part $f.csv || echo "FAIL r$r $row k$k"
    done
  done
done
echo "dkm done $(date -u +%T)"
