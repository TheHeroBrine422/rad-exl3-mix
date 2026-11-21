cd /data/work/cptcv; mkdir -p logs/cbk
export CPTC_VIT3=1 CPTC_LIBDIR=/data/work/cptcv/v3/e3 CPTC_E3DIR=/data/work/cptcv/v3/e3 OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive KEFF_NTRY=4
echo "$JOBS" | tr ',' '\n' | xargs -P 20 -L 1 sh -c 'f=logs/cbk/$0-K$1.log; grep -q "^{" $f 2>/dev/null && exit 0; python3 cptc/runs/cbkeff.py $0 $1 16384 > $f.part 2>&1 && mv $f.part $f; echo "$(date -u +%H:%M) $0 K$1 rc=$?" >> logs/cbk/done-$TAG'
echo "$TAG ALL DONE $(date -u +%H:%M)" >> logs/cbk/done-$TAG
