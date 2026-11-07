#!/bin/bash
# run4070.sh UNIT -- w52's checks on the RTX 4070 box (shared: one hold of /data/yb/gpu.lock per unit, < 15 min in
# total, VRAM_CAP_GB 7.2). Code is read from /workspace/rad-exl3-mix/pipeline (shared folder); outputs /data/yb/w52/.
#   A  27B per-tensor jobs at K2.5 (ladder settings) on layer 10 (GDN + MLP: 6 tensors) and layer 39 attention (q/k/v/o)
#      -> bit-identity against hconv's ourq level /data/yb/hconv/q-none (same settings, 11-04), stage profile
#   B  the same 10 tensors at K2, and at K4 / K6 without ffn_down (lock budget), stage profile
#   C  0.8B writers' asym path: rxq mkdenoise (lam 1, rho 0) on sib27test/ref2, then the 48 writers through rxq quant
#      with hin / wpre at q-k3base's settings -> bit-identity against d1103's q-k3asym_r0 (as asym27 test08 G1)
#   BCD  B, C and D in one hold of the lock (~9 min)
#   D  H_out (two-sided ldlq_2hess): 27B layer 10 in_proj_z + gate_proj, layer 39 o_proj at K2.5 with hconv's bn H_out
#      (train-1664, sigma_out 0.01) -> bit-identity against hconv/q-bn1664 (ourq), stage profile
set -u
U=${1:?A|B|C|D|BCD}; R=/workspace/rad-exl3-mix/pipeline O=/data/yb/w52; mkdir -p $O
export PYTHONPATH=$R:/data/src/exllamav3 TORCH_EXTENSIONS_DIR=/data/torch_ext RXQ_EXL3_SRC=/data/src/exllamav3/exllamav3/exllamav3_ext \
       RXQ_RXM=/workspace/rad-exl3-mix RXQ_REFIT_CPU_FALLBACK=1 VRAM_CAP_GB=${VRAM_CAP_GB:-7.2} \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
M27=/workspace/models/qwen38-27b-bf16 H27=/data/challenge/dl/hessians/calib
P=model.language_model.layers
KEYS=$P.10.linear_attn.in_proj_qkv,$P.10.linear_attn.in_proj_z,$P.10.linear_attn.out_proj,$P.10.mlp.gate_proj,$P.10.mlp.up_proj,$P.10.mlp.down_proj
KEYS=$KEYS,$P.39.self_attn.q_proj,$P.39.self_attn.k_proj,$P.39.self_attn.v_proj,$P.39.self_attn.o_proj
# the references are 11-04 / 11-06 levels (no tile CD, no H_out unless D): w76's config turns tile CD on (attn_qg) and H_out
# on at K >= 4 and C9 on (writers), so every unit turns them off first (D turns H_out back on below)
OLD="--set quant.tile_cd.enable=false --set inputs.hout.enable=false --set post.c9.enable=false"
log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
exec 9>/data/yb/gpu.lock; t0=$(date +%s); flock 9; t1=$(date +%s)
log "unit $U: lock after $(( t1 - t0 ))s"
cd /tmp
b_run() {
  # A's one difference (ffn_down: 1-ulp suh / svh after the CPU-fallback refit, codes equal): rerun it with torch's
  # default thread count, as hconv's ourq run had (OMP_NUM_THREADS unset), and compare again
  ( unset OMP_NUM_THREADS; timeout 200 nice -n 10 python3 -m rxq quant $OLD --keys $P.10.mlp.down_proj -K 2.5 --codebook i8 --ckpt $M27 \
    --hess $H27 -o $O/a2-K2.5 > $O/a2.log 2>&1 ); log "B down K2.5 default threads rc=$? $(( $(date +%s) - t1 ))s"
  # ffn_down at K2 only: its refit falls back to the CPU on this card (~100 s at 4 threads), the lock budget
  timeout 300 python3 -m rxq quant $OLD --keys $KEYS -K 2 --codebook i8 --ckpt $M27 --hess $H27 -o "$O/b-K{K}" > $O/b.log 2>&1
  log "B quant K2 rc=$? $(( $(date +%s) - t1 ))s"
  timeout 300 python3 -m rxq quant $OLD --keys ${KEYS/$P.10.mlp.down_proj,/} -K 4,6 --codebook i8 --ckpt $M27 --hess $H27 -o "$O/b-K{K}" > $O/b4.log 2>&1
  log "B quant K4,6 rc=$? $(( $(date +%s) - t1 ))s"; }
b_check() {
  CUDA_VISIBLE_DEVICES= nice -n 19 python3 $R/tests/check_quant.py /data/yb/hconv/q-none $O/a2-K2.5 > $O/a2-check.txt 2>&1
  tail -n 2 $O/a2-check.txt | tee -a $O/result.txt; }
c_run() {
  M08=/data/models/qwen35-08b H08=/data/h08/calib T=/data/yb/sib27test
  W=$(python3 -c "
for l in range(24):
    p = f'$P.{l}'
    print(f'{p}.self_attn.o_proj' if l % 4 == 3 else f'{p}.linear_attn.out_proj'); print(f'{p}.mlp.down_proj')" | paste -sd,)
  [ -f $O/c-dn/.done ] || timeout 300 python3 -m rxq.asym denoise $H08 $T/ref2 $O/c-dn --lam 1 --rho 0 --ckpt $M08 > $O/c-dn.log 2>&1
  log "C mkdenoise rc=$? $(( $(date +%s) - t1 ))s"
  timeout 400 python3 -m rxq quant $OLD --keys $W -K 3 --codebook i8 --ckpt $M08 --hess $H08 --hin $O/c-dn/hin --wpre $O/c-dn/wpre \
    -o $O/c-asym > $O/c.log 2>&1
  log "C quant rc=$? $(( $(date +%s) - t1 ))s"; }
c_check() {
  CUDA_VISIBLE_DEVICES= nice -n 19 python3 $R/tests/check_quant.py /data/yb/d1103/q-k3asym_r0 $O/c-asym > $O/c-check.txt 2>&1
  tail -n 1 $O/c-check.txt | tee -a $O/result.txt; }
d_run() {
  HK=$P.10.linear_attn.in_proj_z,$P.10.mlp.gate_proj,$P.39.self_attn.o_proj
  timeout 300 python3 -m rxq quant $OLD --keys $HK -K 2.5 --codebook i8 --ckpt $M27 --hess $H27 --set inputs.hout.enable=true \
    --set inputs.hout.apply_k_min=0 --hout /data/yb/hconv/half-bn/train-1664 -o $O/d-K2.5 > $O/d.log 2>&1
  log "D quant rc=$? $(( $(date +%s) - t1 ))s"; }
d_check() {
  CUDA_VISIBLE_DEVICES= nice -n 19 python3 $R/tests/check_quant.py /data/yb/hconv/q-bn1664 $O/d-K2.5 > $O/d-check.txt 2>&1
  tail -n 1 $O/d-check.txt | tee -a $O/result.txt; }
unlock() { flock -u 9; log "$U lock released, held $(( $(date +%s) - t1 ))s"; }
case $U in
A)
  timeout 600 python3 -m rxq quant $OLD --keys $KEYS -K 2.5 --codebook i8 --ckpt $M27 --hess $H27 -o $O/a-K2.5 > $O/a.log 2>&1
  log "A quant rc=$? $(( $(date +%s) - t1 ))s"; unlock
  CUDA_VISIBLE_DEVICES= nice -n 19 python3 $R/tests/check_quant.py /data/yb/hconv/q-none $O/a-K2.5 > $O/a-check.txt 2>&1
  tail -n 1 $O/a-check.txt | tee -a $O/result.txt ;;
B) b_run; unlock; b_check ;;
C) c_run; unlock; c_check ;;
D) d_run; unlock; d_check ;;
BCD) b_run; c_run; d_run; unlock; b_check; c_check; d_check ;;
esac
log "UNIT $U DONE"
