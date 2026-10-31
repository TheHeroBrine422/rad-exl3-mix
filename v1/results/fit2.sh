#!/bin/bash
# largest model that fits 262k bounds all smaller; then step the cap down for the bigger ones
for b in 3.00 4.00 5.00; do /root/fit.sh $b 262144 | head -1; done
for b in 4.00 5.00 6.00; do /root/fit.sh $b 196608 | head -1; done
echo FITDONE
