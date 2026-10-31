#!/bin/bash
# gold2.sh MODEL GOLDEN CORPUS OUT KVDTYPE MAXLEN [flags] -- the patched recorder, scored inline; report to OUT.json.
# Card: $CARD (default 0, the fast link), home: $RHOME (default the frozen install). Kills nothing.
: ${RXM_WORK:=/root} ${RXM_RADIANCE:=$RXM_WORK/install-card} ${RXM_KLDBIN0:=$RXM_WORK/radiance-kldgold}; export RXM_WORK RXM_RADIANCE RXM_KLDBIN0   # paths (README: Paths)
M=$1 G=$2 C=$3 O=$4 KV=$5 ML=$6; shift 6
export PATH=${RXM_RADIANCE}/bin:$PATH RADIANCE_HOME=${RHOME:-${RXM_RADIANCE}/share/radiance} RADIANCE_KLD_GOLDEN=$G
export ROCR_VISIBLE_DEVICES=${CARD:-0}
# async scored-row copies are validated at TP1 only (2026-10-30): TP2 keeps the synchronous path
[ "${TP:-1}" = 2 ] && export RADIANCE_KLD_SYNC=1
rm -rf $O; s=$(date +%s)
${BIN:-${RXM_KLDBIN0}} --model $M --tp ${TP:-1} --kv-cache-dtype $KV --max-model-len $ML --host-headroom-mib ${HEADROOM:-16384} --max-num-seqs ${SEQS:-1} \
  --kld-record $O --kld-corpus $C "$@" > $O.log 2>&1
rc=$?
python3 -I ${RXM_WORK}/aggscore.py $G $O/score.f32 > $O.json 2>$O.err
echo "rc=$rc wall=$(( $(date +%s) - s ))s errs=$(grep -cE "^E |fatal" $O.log)"
