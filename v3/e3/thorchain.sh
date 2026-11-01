#!/bin/bash
# after the seed test: the fine gain sweep
until grep -q "SEEDTEST DONE" /data/logs/seed.log; do sleep 60; done
cd /data/work/gain && ./gainsweep.sh
