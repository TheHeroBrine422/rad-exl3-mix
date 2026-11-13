#!/bin/bash
# hd129.sh -- the search's R16 head donor containers on ds8 (lead note 11-12 11:45 PM CST; TODO "Heads before w124's
# un-permute"): one container per plan of w122's price table (hcurve122.json: u1 ... u6.5 = R16s003/K*, u7 = R16s003m1/K7),
# each = a copy of the smallest level pack (K0.125, bf16 head) with output.weight overwritten by the plan's R16 head,
# dequantized to bf16 and un-permuted to vocab order (w122's headslab.py build, as e3d122.sh). radmix takes only
# output.weight from a donor, so the base's other weights do not matter (smallest base = least disk: ~5.6 GB a donor).
# Out: /data/q27/heads/donors/hd-<plan>.rad. conv.lock + ram.lock shared per donor; restartable (.part, skip finished).
set -eu
Q=/data/q27; D=$Q/heads/donors; PY=$Q/venv/bin/python; C=$Q/stage/code; HC=$C/v3/yaqa/data/w122/rep/hcurve122.json
PERM=$Q/heads/w119/perms/R16.npy; LV=q27v3-144dd6e1-k0.125-i8
mkdir -p $D
[ -f $D/plans-u.json ] || $PY -c "
import json; h = json.load(open('$HC'))['plans']
json.dump({'plans': {n: {'K_blocks': [p['avg_bits']] * 1940, 'avg_bits': p['avg_bits'], 'donor': p['donor']} for n, p in h.items()}}, open('$D/plans-u.json', 'w'))"
s=$($PY -c "import json; print(json.load(open('$Q/ladder/$LV/DONE')).get('scales') or '')"); B=$Q/packs/$LV${s:+-s${s:0:8}}.rad
[ -f $B ] || bash $Q/w129/pack129.sh 1 $LV
[ -f $B ] || { echo "no base pack $B"; exit 1; }
for pn in $($PY -c "import json; print(' '.join(json.load(open('$D/plans-u.json'))['plans']))"); do
  O=$D/hd-$pn.rad; [ -f $O ] && continue
  dn=$($PY -c "import json; print(json.load(open('$D/plans-u.json'))['plans']['$pn']['donor'])")      # R16s003-K2.5 / R16s003m1-K7
  qd=$Q/heads/q/${dn%-K*}
  ( exec 9>$Q/locks/conv.lock; flock 9; exec 7>$Q/locks/ram.lock; flock -s 7
    rm -f $O.part; cp $B $O.part && nice -n 10 $PY -I $C/v3/yaqa/headslab.py build $D/plans-u.json $pn $PERM $qd $O.part > $D/hd-$pn.log 2>&1 \
      && mv $O.part $O || { rm -f $O.part; echo "FAIL $pn ($D/hd-$pn.log)"; exit 1; } )
  echo "$(date -u +%T) $O"
done
echo HD129 DONE
