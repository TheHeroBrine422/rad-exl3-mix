cd /data/work/cptcv; mkdir -p logs/scr
export CPTC_VIT3=1 CPTC_LIBDIR=/data/work/cptcv/v3/e3 CPTC_E3DIR=/data/work/cptcv/v3/e3 OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive KEFF_NTRY=1 CB_ROUNDS=15
echo "$JOBS" | tr ',' '\n' | xargs -P $P -L 1 sh -c 'f=logs/scr/$0-K$1.log; grep -q "^{" $f 2>/dev/null && exit 0; python3 cptc/runs/cbkeff.py $0 $1 4096 > $f.part 2>&1 && mv $f.part $f'
echo "SCR ALL DONE $(date -u +%H:%M)" >> logs/scr/done
