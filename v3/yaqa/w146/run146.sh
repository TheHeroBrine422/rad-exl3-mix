#!/bin/bash
# run146.sh [CARDS] -- #162-03 (w146, RULES.md) on ds8: identity gate of the patched qtf2 build (w140's eighths patch,
# applied in place to w119's tree; originals in $V/orig), then the 14 blocked R16 head donors, scores, price table, wq drop.
# Restartable: gate.ok / gate.fail decide, finished units are skipped by run122.sh. Stop: touch /data/q27/heads/STOP.
# CARDS = ds8 cards for the workers (default "0 1 6 7"; each waits on its card lock).
set -u; H=/data/q27/heads; V=$H/w146; R122=$H/code/v3/yaqa/w122/run122.sh; CARDS=${1:-"0 1 6 7"}
PY=/data/q27/venv/bin/python; X=/data/q27/w119/code/v3/e3/exl3frac2
mkdir -p $V/logs $V/rep; exec >> $V/run146.log 2>&1
say() { echo "$(date -u +%T) $*"; }
sha() { [ -f $1 ] && sha256sum < $1 | cut -c1-16 || echo none; }
work() {   # work QUEUE: one run122 worker per card, wait for all
  local q=$1 c; for c in $CARDS; do bash $R122 worker $c $q > $V/logs/w-$(basename $q .txt)-c$c.out 2>&1 & done; wait
}
# patched tree in place (checked every run)
for f in $(ls $V/patched); do cmp -s $V/patched/$f $X/$f || { say "tree: $f is not the patched file: stop"; exit 1; }; done
say "start (tree patched: $(ls $V/patched | wc -l) files)"
# ---- identity gate (RULES): requant three existing levels, compare bytes and stamps
if [ ! -f $V/gate.ok ] && [ ! -f $V/gate.fail ]; then
  printf "w146chk-i8 4 0.03 i8\nw146chk-m1 6 0.03 mul1\nw146chk-i8 6.25 0.03 i8\n" > $V/q-gate.txt
  work $V/q-gate.txt
  ok=1
  for p in "w146chk-i8/K4 R16s003/K4" "w146chk-m1/K6 R16s003m1/K6" "w146chk-i8/K6.25 R16s003/K6.25"; do
    set -- $p; a=$H/q/$1 b=$H/q/$2
    [ -f $a/wq/lm_head.rowerr.npz ] || { say "gate $1: not built (see $a.log)"; exit 1; }
    for f in q-head.safetensors wq/lm_head.rowerr.npz wq/lm_head.safetensors; do
      [ $f = wq/lm_head.safetensors ] && [ ! -f $b/$f ] && continue
      x=$(sha $a/$f) y=$(sha $b/$f); [ "$x" = "$y" ] && say "gate $2 $f SAME $x" || { say "gate $2 $f DIFFERS new $x ref $y"; ok=0; }
    done
    sa=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))[-1]['stamp'])" $a/stamp.json)
    sb=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))[-1]['stamp'])" $b/stamp.json)
    [ "$sa" = "$sb" ] && say "gate $2 stamp SAME $sa" || { say "gate $2 stamp DIFFERS new $sa ref $sb"; ok=0; }
  done
  [ $ok = 1 ] && { touch $V/gate.ok; say "GATE PASS"; } || { touch $V/gate.fail; say "GATE FAIL: revert the tree from $V/orig"; exit 1; }
fi
[ -f $V/gate.ok ] || { say "gate failed earlier: nothing built"; exit 1; }
# ---- build: the 14 donors w140 left BLOCKED
for u in $(awk '{print $1 "/K" $2}' $V/q-build.txt); do
  [ -f $H/q/$u.claim/BLOCKED ] && { rm -f $H/q/$u.claim/BLOCKED; rmdir $H/q/$u.claim; say "unblocked $u"; }
done
work $V/q-build.txt
n=$(awk '{print $1 "/K" $2}' $V/q-build.txt | while read u; do [ -f $H/q/$u/wq/lm_head.rowerr.npz ] && echo; done | wc -l)
say "built $n / 14"
# ---- scores, price table, wq drop (as post140.sh, on the first card)
c=${CARDS%% *}
bash $R122 score $c fit; bash $R122 score $c search
W1=$H/w119; SC=$H/scores DOCS=$W1/code/v3/yaqa/w119/docs119.json SHARE=$W1/code/v3/yaqa/w119/calibshare119.json PERMS=$W1/perms \
  OUT=$V/rep H122=$H/rep/hcurve122.json R119=$W1/code/v3/yaqa/w119 $PY -I $H/w140/code/rep140.py table
k=0
while read -r name kk sig cb; do
  f=$H/q/$name/K$kk/wq/lm_head.safetensors; s=$name-K$kk
  [ -f $f ] && [ -f $H/scores/fit/$s.npz ] && [ -f $H/scores/search/$s.npz ] && [ -f $H/q/$name/K$kk/q-head.safetensors ] && { rm -f $f; k=$((k + 1)); }
done < $V/q-build.txt
say "wq dropped for $k donor(s); $(df -h /data | awk 'NR == 2 {print $4}') free"
say "RUN146 DONE"
