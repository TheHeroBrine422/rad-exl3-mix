#!/bin/bash
# r100.sh PART PAR [PARD] -- #100 (README): IQR's computed code (trellis4.py codebook iqrh) vs mul1_i8 on the 10
# #90-02 tensors (i4run.sh) + the 10 #90-03 ones (rep.sh's i4b list), K 0.25..7, scale grid 0.5..1.5, 128 rows,
# seed 0, LDLQ only (CD=0). Same rows / seed / factors for both codebooks, so the ratio is paired.
# PART a (cpu1, 10 job slots): 1 ffn_down (~11 GB a job) + 7 others; PART b (cpu2, 24 slots): 3 ffn_down + 9;
# PART c: all 20 (follow-ups, e.g. the tuned global A: CBS=iqrh:0xd0474b:0x2545f491 CHD="2 3 4 5 6" CHO="2,3 4,5 6").
# Env CBS (codebooks, default "mul1_i8 iqrh"), CHD / CHO (K chunks of the ffn_down / other jobs), ONLY=down|other
# (with PARD 0: one kind of job only).
# One thread a job; PAR jobs at once, of which at most PARD ffn_down jobs (memory). A job = (tensor, codebook,
# K chunk); a job is skipped when its chunk's last K is already in the output (rerun = resume; the report dedups).
# DRY=1 lists the jobs left. Lives in its own code dir with e3.py, trellis.py, trellis4.py, viterbi.c (libviterbi.so built there, -march=native).
cd "$(dirname "$(readlink -f "$0")")"
export E3_FACTORS=0.5,0.7,0.85,1.0,1.15,1.3,1.5 E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CD=0
P=$1; PAR=${2:-10}; PARD=${3:-0}; O=$PWD/r100-$P.jsonl; CBS=${CBS:-mul1_i8 iqrh}
case $P in
  a) T="10:ffn_down 10:ssm_qkv 10:ssm_out 10:ffn_gate_up 50:ssm_qkv 50:ffn_gate_up 19:attn_qg 19:attn_output" ;;
  b) T="30:ffn_down 26:ffn_down 39:ffn_down 43:attn_v 6:ssm_qkv 46:ssm_z 39:attn_output 6:ffn_gate_up 26:ssm_out 46:ffn_gate_up 58:ssm_qkv 39:attn_qg" ;;
  c) T="10:ffn_down 30:ffn_down 26:ffn_down 39:ffn_down 10:ssm_qkv 10:ssm_out 10:ffn_gate_up 50:ssm_qkv 50:ffn_gate_up 19:attn_qg 19:attn_output 43:attn_v 6:ssm_qkv 46:ssm_z 39:attn_output 6:ffn_gate_up 26:ssm_out 46:ffn_gate_up 58:ssm_qkv 39:attn_qg" ;;
  *) echo "PART a|b|c"; exit 1 ;;
esac
mkdir -p logs
jobs_for() {   # $1 = down|other
  for t in $T; do l=${t%%:*}; w=${t#*:}
    [ "$1" = down ] && [ $w != ffn_down ] && continue; [ "$1" = other ] && [ $w = ffn_down ] && continue
    if [ $w = ffn_down ]; then CH=${CHD:-0.25,0.5 1,1.5 2,2.5 3,3.5 4,5 6,7}; else CH=${CHO:-0.25,0.5,1,1.5 2,2.5,3,3.5 4,5,6,7}; fi
    for ks in $CH; do for cb in $CBS; do
      kl=${ks##*,}; [[ $kl == *.* ]] || kl=$kl.0   # the last K as json.dumps writes it
      grep -qF "\"layer\": $l, \"w\": \"$w\", \"method\": \"ldlq_$cb\", \"bits\": $kl," $O 2>/dev/null && continue
      echo "python3 -W ignore trellis4.py $l $w $cb $ks 128 $O > logs/r100-$l-$w-$cb-${ks//,/_}.log 2>&1"
    done; done
  done
}
[ -n "$DRY" ] && { jobs_for down; jobs_for other; exit; }   # DRY=1: list the jobs left
if [ "$PARD" -gt 0 ]; then
  jobs_for down | xargs -P $PARD -I{} bash -c "nice -n 10 {}" &
  jobs_for other | xargs -P $((PAR - PARD)) -I{} bash -c "nice -n 10 {}"
  wait
else
  jobs_for ${ONLY:-all} | xargs -P $PAR -I{} bash -c "nice -n 10 {}"   # ONLY=down|other: one kind
fi
echo R100 DONE $P
