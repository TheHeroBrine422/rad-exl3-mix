#!/bin/bash
# p198sh.sh (w186, #198, RULES198 amendment 1: arm SH, eval only; box2) -- restartable like p198.sh.
#   0 unit: the shrink soup of N identical members (D0 x2, D0 x4) equals the plain mean soup bitwise (unitshrink)
#   1 SH4 = shrink(D0, D1, Q1D0, T1) (the S4 members) and SH2s = shrink(D0, A-s1) (the S2s pair): built on the GPU exp,
#     served through the same path and scored on x181g + fe38
#   2 readout over every arm (analysis-sh/: the same bootstrap draws as analysis/, so the old contrasts reproduce) and
#     supp198 on it
# Usage: CARD=2 setsid nohup bash code/v3/qad198/p198sh.sh > logs/p198sh.out 2>&1 < /dev/null &
set -u
cd /data/q27/w186 && source env.sh
CARD=${CARD:-2}
mkdir -p soups served logs analysis-sh unit
say() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" | tee -a $W/p198.log; }
ram_wait() { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt "$1" ]; do sleep 120; done; }
disk_wait() { while [ "$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)" -lt 100 ]; do say "/data below 100 GB: waiting"; sleep 600; done; }
M1=$W170/runs/L0B0T0Q0D0/ckpt-1e+06; M2=$W170/runs/L0B0T0Q0D1/ckpt-1e+06
M3=$W170/runs/L0B0T0Q1D0/ckpt-1e+06; M4=$W170/runs/L0B0T1Q0D0/ckpt-1e+06
A1=$W/runs/A-s1/ckpt-1e+06
SOUP=$CODE/v3/qad198/soup198.py

shrink() { # shrink NAME MEMBER... (exp on card CARD's GPU, under its lock)
  local n=$1; shift
  [ -e soups/$n/.done ] && return 0
  flock /data/q27/locks/card$CARD.lock env ROCR_VISIBLE_DEVICES=$CARD $PY $SOUP build soups/$n shrink "$@" --start $START --device cuda:0 \
     > logs/soup-$n.log 2>&1 || { say "soup $n FAILED (logs/soup-$n.log)"; exit 1; }
  say "soup $n built (shrink, exp on the GPU)"
}
served() { # served TAG CKPT_DIR
  local tag=$1 ck=$2
  [ -e served/$tag/result.json ] && return 0
  disk_wait; ram_wait 31
  $PY $CODE/v3/qad196/eval196.py $tag served --vlevel $VL --template $TPL --host $HOSTC --pack-home /data/q27/rad10-e157b/home \
     --card $CARD --ckpt $ck --slice x181g=$G1 --slice fe38=$G2 > logs/served-$tag.out 2>&1
  local rc=$?; say "served $tag rc $rc"; [ $rc -eq 0 ] || exit 1
}

say "p198sh start on card $CARD"
# 0. unit: N identical members -> shrink == plain mean, bitwise (CPU exp, no card needed)
if [ ! -e unit/shrink-identical.json ]; then
  { $PY $SOUP unitshrink unit/ushr2 $M1 2 --start $START; $PY $SOUP unitshrink unit/ushr4 $M1 4 --start $START; } \
    > unit/shrink-identical.json 2> logs/unitshrink.err || { say "UNIT SHRINK FAILED (logs/unitshrink.err)"; exit 1; }
fi
say "unit shrink (N identical members) exact: $(tr '\n' ' ' < unit/shrink-identical.json)"
# 1. the two SH soups, built and served
shrink SH4 $M1 $M2 $M3 $M4;     served SH4 $W/soups/SH4
shrink SH2s $M1 $A1;            served SH2s $W/soups/SH2s
# 2. readout over every arm
if [ ! -e analysis-sh/analysis.json ]; then
  $PY $CODE/v3/qad198/analyse198.py analysis-sh \
    --arm ref=$W170/served/L0B0T0Q0D0/result.json --arm REF2=$W/served/REF2/result.json \
    --arm D1=$W170/served/L0B0T0Q0D1/result.json \
    --arm Q1D0=$W170/served/L0B0T0Q1D0/result.json --arm T1=$W170/served/L0B0T1Q0D0/result.json \
    --arm A=/data/q27/w182/served/A/result.json --arm As1=$W/served/A-s1/result.json \
    --arm S2=$W/served/S2/result.json --arm S4=$W/served/S4/result.json --arm C2=$W/served/C2/result.json \
    --arm S2s=$W/served/S2s/result.json --arm SO1=$W/served/SO1/result.json --arm ctl=$W170/served/ctl/result.json \
    --arm SH4=$W/served/SH4/result.json --arm SH2s=$W/served/SH2s/result.json \
    --arm S2cpu=$W/served-cpuexp/S2/result.json --arm SO1cpu=$W/served-cpuexp/SO1/result.json \
    --pair REF2:ref --pair SO1:ref --pair SO1cpu:ref --pair S2cpu:S2 \
    --pair A:ref --pair As1:ref --pair As1:A --pair D1:ref --pair Q1D0:ref --pair T1:ref --pair ctl:ref \
    --pair S2:ref --pair S4:ref --pair C2:ref --pair S2s:ref \
    --best S2:ref,D1 --best S4:ref,D1,Q1D0,T1 --best C2:ref,D1 --best S2s:ref,As1 \
    --pair S2:A --pair S4:A \
    --pair SH4:S4 --best SH4:ref,D1,Q1D0,T1 --pair SH2s:S2s --best SH2s:ref,As1 > logs/analyse-sh.out 2>&1 \
    || { say "analysis FAILED"; exit 1; }
  $PY $CODE/v3/qad198/supp198.py analysis-sh > /dev/null 2>logs/supp-sh.err || { say "supp FAILED"; exit 1; }
  say "analysis-sh written"
fi
say "p198sh done"
