#!/bin/bash
# evrocprof.sh CARD MODEL OUTDIR [flags] -- one calib-A eval under rocprofv3 (kernel + memory-copy trace, stats)
: ${RXM_WORK:=/root} ${RXM_RADIANCE:=$RXM_WORK/install-card} ${RXM_KLDBIN:=$RXM_WORK/radiance-kldgold-fast}; export RXM_WORK RXM_RADIANCE RXM_KLDBIN   # paths (README: Paths)
C=$1 M=$2 O=$3; shift 3
cd ${RXM_WORK}; exec 9>${RXM_WORK}/card$C.lock; flock 9
export PATH=${RXM_RADIANCE}/bin:$PATH RADIANCE_HOME=${RXM_RADIANCE}/share/radiance RADIANCE_KLD_GOLDEN=golden-calA
export ROCR_VISIBLE_DEVICES=$C RADIANCE_KLD_MAXTOK=8192
rm -rf $O ${RXM_WORK}/evprof/run-rp; mkdir -p $O
/opt/rocm/bin/rocprofv3 --kernel-trace --stats --output-format csv -d $O -o rp -- \
  ${RXM_KLDBIN} --model $M --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 --host-headroom-mib 16384 \
  --max-num-seqs 1 --kld-record ${RXM_WORK}/evprof/run-rp --kld-corpus corpus/calA.jsonl "$@" > $O/run.log 2>&1
echo rc=$?; rm -rf ${RXM_WORK}/evprof/run-rp
