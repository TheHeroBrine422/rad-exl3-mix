#!/bin/bash
# swap182.sh -- (ds8, CPU + ram.lock shared) #182 swap test (w156): K2.5 with one group of trunk weights taken from K6
# (radmix; the 336 trunk linears of w151's a8.mix, act keys unchanged): layer quartiles L0-15 / L16-31 / L32-47 / L48-63
# and weight types attn (attn_qg/k/v/output) / gdn (ssm_qkv/z/out) / mlp (ffn_gate_up/down). Which group's upgrade
# lowers the image rows' KL more than the text control's points at where vision is more sensitive than text.
# Out W/packs/S-*.rad + W/cands-swap.txt. Restartable.
W=/data/q27/w156; Q=/data/q27; P=$W/packs; L=$Q/locks; A=$Q/mad/tools/a8.mix
mkdir -p $P/swap; : > $W/cands-swap.txt
names() { awk '{print $2}' $A; }
grp() {   # grp NAME FILTER(awk on the layer number l and the weight w)
  local n=$1 f=$2
  names | awk -v d=$P/K6.rad "{ split(\$1, a, \".\"); l = a[2] + 0; w = a[3]; if ($f) print \$1, d }" > $P/swap/$n.mix
  if [ ! -f $P/S-$n.rad ]; then
    ( flock -s 6; $W/tools/radmix $P/S-$n.rad.part $P/K2.5.rad $P/swap/$n.mix > $P/swap/$n.log 2>&1 && mv $P/S-$n.rad.part $P/S-$n.rad ) 6>$L/ram.lock
  fi
  echo "$(date -u +%T) S-$n: $(wc -l < $P/swap/$n.mix) weights from K6; $(tail -n 1 $P/swap/$n.log)"
  [ -f $P/S-$n.rad ] && echo "S-$n $P/S-$n.rad" >> $W/cands-swap.txt
}
grp L0-15 'l < 16'; grp L16-31 'l >= 16 && l < 32'; grp L32-47 'l >= 32 && l < 48'; grp L48-63 'l >= 48'
grp attn 'w ~ /^attn_/'; grp gdn 'w ~ /^ssm_/'; grp mlp 'w ~ /^ffn_/'
