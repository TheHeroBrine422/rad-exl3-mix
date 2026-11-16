#!/bin/bash
# p198.sh (w186, #198, box2) -- the whole #198 chain; restartable (each step is skipped once its output exists).
#   1 soups (CPU): SO1 = soup of one (L0B0T0Q0D0 alone), S2 = mean(L0B0T0Q0D0, L0B0T0Q0D1), S4 = mean of the four #191 runs
#     without adapters / block stage, C2 = sign consensus of the S2 pair
#   2 served (eval196.py = the #196 served path, card CARD; the scoring step takes the card lock): SO1 first as the unit
#     gate (served KL within 0.5 % of L0B0T0Q0D0, else the chain stops), then S2, S4, C2
#   3 training A-s1 (run198.sh: arm A with --seed 1; card CARD) -- after the eval-only arms
#   4 served A-s1, soup S2s = mean(L0B0T0Q0D0, A-s1), served S2s
#   5 analysis (analyse198.py) -> analysis/
# Usage: CARD=2 nohup bash code/v3/qad198/p198.sh > logs/p198.out 2>&1 < /dev/null &
set -u
cd /data/q27/w186 && source env.sh
CARD=${CARD:-2}
mkdir -p soups served runs logs analysis unit
say() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" | tee -a $W/p198.log; }
ram_wait() { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt "$1" ]; do sleep 120; done; }
disk_wait() { while [ "$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)" -lt 100 ]; do say "/data below 100 GB: waiting"; sleep 600; done; }
M1=$W170/runs/L0B0T0Q0D0/ckpt-1e+06; M2=$W170/runs/L0B0T0Q0D1/ckpt-1e+06
M3=$W170/runs/L0B0T0Q1D0/ckpt-1e+06; M4=$W170/runs/L0B0T1Q0D0/ckpt-1e+06
SOUP=$CODE/v3/qad198/soup198.py

soup() { # soup NAME MODE MEMBER...
  local n=$1 mode=$2; shift 2
  [ -e soups/$n/.done ] && return 0
  $PY $SOUP build soups/$n $mode "$@" --start $START > logs/soup-$n.log 2>&1 || { say "soup $n FAILED (logs/soup-$n.log)"; exit 1; }
  say "soup $n built ($mode)"
}
served() { # served TAG CKPT_DIR  (CKPT_DIR empty = the untrained control)
  local tag=$1 ck=$2
  [ -e served/$tag/result.json ] && return 0
  disk_wait; ram_wait 31
  local C=""; [ -n "$ck" ] && C="--ckpt $ck"
  $PY $CODE/v3/qad196/eval196.py $tag served --vlevel $VL --template $TPL --host $HOSTC --pack-home /data/q27/rad10-e157b/home \
     --card $CARD $C --slice x181g=$G1 --slice fe38=$G2 > logs/served-$tag.out 2>&1
  local rc=$?; say "served $tag rc $rc"; [ $rc -eq 0 ] || exit 1
}
gate_so1() { # the unit gate: served SO1 vs the member it copies, pooled KL within 0.5 %
  $PY - <<'PY' || { say "UNIT GATE FAILED: SO1 served KL differs from L0B0T0Q0D0 by more than 0.5 %; chain stopped"; exit 1; }
import json, sys
a = json.load(open("/data/q27/w186/served/SO1/result.json")); b = json.load(open("/data/q27/w170/served/L0B0T0Q0D0/result.json"))
def pooled(r):
    s = sum(v["kld"] * v["positions"] for v in r["slices"].values()); n = sum(v["positions"] for v in r["slices"].values()); return s / n
x, y = pooled(a), pooled(b); rel = abs(x / y - 1)
print(f"SO1 {x:.6f} vs L0B0T0Q0D0 {y:.6f}: rel {rel:.2e}")
sys.exit(0 if rel <= 0.005 else 1)
PY
  say "unit gate passed (SO1 = L0B0T0Q0D0 served)"
}

say "p198 start on card $CARD"
# 0. the unit check on the files (state bitwise; scales vs the member's own files)
if [ ! -e unit/report.json ]; then
  $PY $SOUP unit unit/SO1-file $M1 --start $START > unit/report.json 2> logs/unit.err || { say "unit check FAILED (logs/unit.err)"; exit 1; }
  rm -rf unit/SO1-file; say "unit check (files) passed: $(cat unit/report.json)"
fi
# 1-2. eval-only arms
soup SO1 mean $M1
served SO1 $W/soups/SO1; gate_so1
soup S2 mean $M1 $M2;             served S2 $W/soups/S2
soup S4 mean $M1 $M2 $M3 $M4;     served S4 $W/soups/S4
soup C2 consensus $M1 $M2;        served C2 $W/soups/C2
# 3. training arm (after the eval-only arms)
if [ ! -e runs/A-s1/ckpt-1e+06/.done ]; then
  ram_wait 38
  while true; do ./code/v3/qad198/run198.sh A-s1 $CARD --seed 1; rc=$?; [ $rc -eq 75 ] && { sleep 60; continue; }; break; done
  [ $rc -eq 0 ] || { say "train A-s1 FAILED rc $rc"; exit 1; }
  say "train A-s1 done"
fi
# 4. the training arm's served read and the clean same-recipe soup
served A-s1 $W/runs/A-s1/ckpt-1e+06
soup S2s mean $M1 $W/runs/A-s1/ckpt-1e+06; served S2s $W/soups/S2s
# 5. readout
if [ ! -e analysis/analysis.json ]; then
  $PY $CODE/v3/qad198/analyse198.py analysis \
    --arm ref=$W170/served/L0B0T0Q0D0/result.json --arm D1=$W170/served/L0B0T0Q0D1/result.json \
    --arm Q1D0=$W170/served/L0B0T0Q1D0/result.json --arm T1=$W170/served/L0B0T1Q0D0/result.json \
    --arm A=/data/q27/w182/served/A/result.json --arm As1=$W/served/A-s1/result.json \
    --arm S2=$W/served/S2/result.json --arm S4=$W/served/S4/result.json --arm C2=$W/served/C2/result.json \
    --arm S2s=$W/served/S2s/result.json --arm SO1=$W/served/SO1/result.json --arm ctl=$W170/served/ctl/result.json \
    --pair A:ref --pair As1:ref --pair As1:A --pair D1:ref --pair Q1D0:ref --pair T1:ref --pair SO1:ref --pair ctl:ref \
    --pair S2:ref --pair S4:ref --pair C2:ref --pair S2s:ref \
    --best S2:ref,D1 --best S4:ref,D1,Q1D0,T1 --best C2:ref,D1 --best S2s:ref,As1 \
    --pair S2:A --pair S4:A > logs/analyse.out 2>&1 || { say "analysis FAILED"; exit 1; }
  say "analysis written: analysis/tables.md"
fi
say "p198 done"
