#!/bin/bash
# post140.sh CARD -- #162-02 (w140, RULES amendment 1) on ds8, after donors land: score119 (run122.sh score) on fit and search
# for every unscored donor (CARD's lock), the price table (rep140.py table -> w140/rep/hcurve140.json), then delete
# wq/lm_head.safetensors of w140's donors (q-w140.txt + q-w140b.txt) whose two score files exist (keeps wq/lm_head.rowerr.npz,
# run122's done marker, and q-head.safetensors). Safe to rerun.
set -u; H=/data/q27/heads; W=$H/w140; W1=$H/w119; c=${1:-7}
bash $H/code/v3/yaqa/w122/run122.sh score $c fit; bash $H/code/v3/yaqa/w122/run122.sh score $c search
SC=$H/scores DOCS=$W1/code/v3/yaqa/w119/docs119.json SHARE=$W1/code/v3/yaqa/w119/calibshare119.json PERMS=$W1/perms OUT=$W/rep \
  H122=$H/rep/hcurve122.json R119=$W1/code/v3/yaqa/w119 /data/q27/venv/bin/python -I $W/code/rep140.py table
n=0
while read -r name k sig cb; do
  [ -z "$name" ] && continue; f=$H/q/$name/K$k/wq/lm_head.safetensors; s=$name-K$k
  [ -f $f ] && [ -f $H/scores/fit/$s.npz ] && [ -f $H/scores/search/$s.npz ] && [ -f $H/q/$name/K$k/q-head.safetensors ] && { rm -f $f; n=$((n + 1)); }
done < <(cat $W/q-w140.txt $W/q-w140b.txt)
echo "$(date -u +%T) post140: wq dropped for $n donor(s); $(df -h /data | awk 'NR == 2 {print $4}') free"
