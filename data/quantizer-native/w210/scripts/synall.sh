#!/bin/bash
# w210 box2: the synthetic batch measurements, card 2 (restartable: finished tags are skipped)
cd /data/q27/w210
./syn.sh 2 gu-B16-pair   --n 2304 --k 5120 --count 128 --B 16 --share 2 --K 2,4
./syn.sh 2 down-B16      --n 5120 --k 2304 --count 128 --B 16 --share 1 --K 2,4
./syn.sh 2 gu-B16-own    --n 2304 --k 5120 --count 64 --B 16 --share 1 --K 4
./syn.sh 2 gu-B16-pooled --n 2304 --k 5120 --count 64 --B 16 --share 0 --K 4
./syn.sh 2 gu-B1         --n 2304 --k 5120 --count 16 --B 1 --share 1 --K 4 --no-pipe
./syn.sh 2 gu-B4         --n 2304 --k 5120 --count 32 --B 4 --share 2 --K 4
./syn.sh 2 gu-B32-pair   --n 2304 --k 5120 --count 128 --B 32 --share 2 --K 4
./syn.sh 2 gu-B16-nopipe --n 2304 --k 5120 --count 64 --B 16 --share 2 --K 4 --no-pipe
./syn.sh 2 down-B32      --n 5120 --k 2304 --count 128 --B 32 --share 1 --K 4
echo DONE > out/syn/done
