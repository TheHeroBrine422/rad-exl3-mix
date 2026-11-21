#!/bin/bash
# w210 box2: the gate / up pair batch on cards 2 and 3 at once (do two cards hold the one-card rate with the host shared?)
cd /data/q27/w210
./syn.sh 2 two-c2 --n 2304 --k 5120 --count 128 --B 16 --share 2 --K 4 &
./syn.sh 3 two-c3 --n 2304 --k 5120 --count 128 --B 16 --share 2 --K 4 &
wait
echo DONE > out/syn/two.done
