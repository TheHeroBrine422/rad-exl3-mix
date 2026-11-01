#!/bin/bash
# tdverify-all.sh -- every turboderp download vs the pack converted from it (store/tdverify.py), sequential, low priority.
# Log: /data/store/tdverify.log (one summary line per pack + problems).
cd /tmp; PY=/data/venv-exl3/bin/python; R=/data/workspace/references; F=/data/root/models/final; L=/data/store/tdverify.log
for p in "td-SC_1.40bpw_H3 sc1.40-e3" "td-SC_1.60bpw_H3 sc1.60-e3" "td-SC_1.80bpw_H3 sc1.80-e3" "td-SC_2.00bpw_H3 sc2.00-e3" \
         "td-SC_2.20bpw_H3 sc2.20-e3" "td-SC_3.00bpw_H4 sc3.00-e4" "td-SC_4.00bpw_H5 sc4.00-e4" "td-SC_5.00bpw_H6 sc5.00-e5" \
         "td-SC_6.00bpw_H6 sc6.00-e6" "td-2.00bpw plain2.00-e3" "td-2.50bpw plain2.50-e3" "td-3.00bpw plain3.00-e4" \
         "td-3.50bpw plain3.50-e4" "td-4.00bpw plain4.00-e4" "td-5.00bpw plain5.00-e5" "td-6.00bpw plain6.00-e6"; do
  set -- $p
  nice -n 15 ionice -c3 $PY -I /data/store/tdverify.py $R/$1 $F/q38-27b-$2.rad >> $L 2>&1
  echo "$(date -u +%H:%M) rc=$? $1" >> $L
done
echo TDVERIFY DONE >> $L
