#!/bin/bash
# finalconv.sh -- every final container, Lloyd embedding by size (Nano 3, Tiny/Small 4, 5.00 5, 6.00 6 bits;
# 72 s to change later with --reuse). Each waits for its export's download; convemb.sh takes the heavy lock.
cd /root
while pgrep -f "^/root/radiance-kldgold" >/dev/null || pgrep -x radiance >/dev/null; do sleep 10; done   # the unlocked run in flight
dl() { while ! grep -q DLDONE /root/dl-$1.out 2>/dev/null; do sleep 30; done; }
sc() {  # BPW BRANCH BITS
  dl $2
  [ -f /root/models/final/q38-27b-sc$1-e$3.rad ] || /root/convemb.sh sc$1-e$3 /workspace/references/td-$2 $3 /root/models/q38-27b-exl3-$1bpw.rad
}
plain() {  # BPW BITS
  dl $1bpw
  [ -f /root/models/final/q38-27b-plain$1-e$2.rad ] || /root/convemb.sh plain$1-e$2 /workspace/references/td-$1bpw $2
}
plain 2.00 3; plain 2.50 3; plain 3.00 4; plain 3.50 4; plain 4.00 4; plain 5.00 5; plain 6.00 6
sc 1.40 SC_1.40bpw_H3 3; sc 1.60 SC_1.60bpw_H3 3; sc 1.80 SC_1.80bpw_H3 3; sc 2.20 SC_2.20bpw_H3 3
sc 3.00 SC_3.00bpw_H4 4; sc 5.00 SC_5.00bpw_H6 5; sc 6.00 SC_6.00bpw_H6 6
echo FCDONE >> /root/models/final/conv.out
