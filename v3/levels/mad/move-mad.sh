#!/bin/bash
# move-mad.sh [--delete-local] -- (ds8) the mad ladder's per-tensor copy to box2 /data/q27/ladder-src-mad/ over the 10 GbE link
# (as the lead's ladmove.sh for the i8 ladder): every level that is DONE at the current scales and TRIMMED (final/ + the
# small files: DONE, final.recipe, level/ minus shards, specs, dup, logs, out/ diag + logs), then a second rsync pass with
# --checksum that must transfer nothing -> ROOT/<level>/MOVED. --delete-local: afterwards remove final/ on ds8 for MOVED
# levels whose pack exists in /data/q27/packs-mad (only then: packs are made from the local final/). Box2 keeps >= 100 GB
# free (checked before each level). Log mad/logs/move.log. (#177-03, w151)
set -u; Q=/data/q27; R=$Q/ladder-mad; D=/data/q27/ladder-src-mad; LG=$Q/mad/logs/move.log; PY=$Q/venv/bin/python
BOX="ssh -i /root/.ssh/dbox -o StrictHostKeyChecking=no -o Compression=no -c aes128-gcm@openssh.com"
say() { echo "$(date -u +%T) $*" >> $LG; }
$BOX root@10.29.6.6 "mkdir -p $D"
for L in $R/q27v3-*-mad; do
  n=$(basename $L); [ -f $L/MOVED ] && continue
  [ -f $L/TRIMMED ] && [ -f $L/DONE ] && [ -d $L/final ] || continue
  free=$($BOX root@10.29.6.6 "df -BG --output=avail /data | tail -n 1 | tr -dc 0-9")
  need=$(( $(du -sBG $L | cut -f1 | tr -dc 0-9) + 100 ))
  [ "$free" -ge "$need" ] || { say "box2 has $free GB free, $n needs $need (incl. the 100 GB floor): stop"; exit 1; }
  nice -n 10 rsync -a --whole-file -e "$BOX" $L/ root@10.29.6.6:$D/$n/ || { say "$n: copy FAIL"; continue; }
  c=$(nice -n 10 rsync -a --checksum --dry-run --itemize-changes -e "$BOX" $L/ root@10.29.6.6:$D/$n/ | grep -c '^[<>c]')
  [ "$c" = 0 ] && { touch $L/MOVED; say "$n: moved + verified ($(du -sh $L/final | cut -f1))"; } || say "$n: verify FAIL ($c files differ)"
done
$BOX root@10.29.6.6 "echo '$(date -u +%FT%TZ) w151 #177-03: the mad ladder q27v3-635bac02-*-mad per-tensor copy (final/ etc.), from ds8 ladder-mad, checksum-verified' > $D/README.txt"
if [ "${1:-}" = --delete-local ]; then
  for L in $R/q27v3-*-mad; do n=$(basename $L); s=$($PY -c "import json; print((json.load(open('$L/DONE')).get('scales') or '')[:8])")
    p=$Q/packs-mad/$n${s:+-s$s}.rad
    [ -f $L/MOVED ] && [ -f $p ] && [ -d $L/final ] && { rm -rf $L/final; say "$n: local final/ removed (pack $p, box2 copy verified)"; }
  done
fi
say "move pass done: $(ls $R/*/MOVED 2>/dev/null | wc -l) levels moved"
