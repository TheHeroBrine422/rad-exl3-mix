#!/bin/bash
# q1.sh -- #175 (w137) card 1 queue after the clk rows: interleaved e2e rounds, v3base (today's device code) vs v3off
# (the asm-literal fix), FUSE off both.
cd /data/q27/w137
while pgrep -f "pf.sh 1 clk" > /dev/null; do sleep 10; done
bash pf.sh 1 e2e3 v3base v3off; bash pf.sh 1 e2e4 v3off v3base; bash pf.sh 1 e2e5 v3base v3off
bash kb.sh 1 3
bash dclk.sh 1 v3base iqrs
bash rp.sh 1 v3base iqrs v3on
