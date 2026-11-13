#!/bin/bash
# s2a.sh PAR -- #177 stage 2a on colo (cd to the code dir with hscreen.py + libvitL.so): A searches (64 iid tiles, seed 11,
# K 2-6, 512 random odd A a family in 8 chunks) and the fixed variant set (512 tiles, seed 12, K 1-6, one job a spec).
# Resumable: a job whose output file has its last line is skipped. Out res/. (w139)
cd "$(dirname "$(readlink -f "$0")")"; mkdir -p res logs
export OMP_NUM_THREADS=1
B=0x2545F491
jobs() {
  for fam in "h16iq:h:{A}:$B:0x3F3F3F3F:16" "h16top:h:{A}:$B:0xFCFCFCFC:16" "h24top:h:{A}:$B:0xFCFCFCFC:24"; do
    n=${fam%%:*}; r=${fam#*:}; bits=${r##*:}; f=${r%:*}
    for c in 1 2 3 4 5 6 7 8; do
      o=res/as-$n-$c.jsonl; [ -f $o ] && [ $(wc -l < $o) -ge $((64 * 5)) ] && continue
      echo "python3 hscreen.py asearch '$f' 2,3,4,5,6 64 11 rand:$c$bits:64:$bits $o > logs/as-$n-$c.log 2>&1"
    done
  done
  for s in m1i8 m1 h:0xd0474b:$B:0x3F3F3F3F h:0xB5297B:$B:0x3F3F3F3F h:0xd0474b:$B:0xFCFCFCFC hg:0xd0474b:$B:0x3F3F3F3F \
           hc:0.02:0xd0474b:$B:0x3F3F3F3F hc:-0.02:0xd0474b:$B:0x3F3F3F3F hyb:0xd0474b:$B:8 hyb:0xd0474b:$B:10 \
           hyb:0xd04b:$B:9; do
    o=res/fx-$(echo $s | tr ':' '_').jsonl; [ -f $o ] && [ $(wc -l < $o) -ge 6 ] && continue
    echo "python3 hscreen.py eval $s 1,2,3,4,5,6 16 512 12 $o > logs/fx-$(echo $s | tr ':' '_').log 2>&1"
  done
}
jobs | xargs -P ${1:-20} -I{} bash -c "nice -n 19 {}"
echo S2A DONE
