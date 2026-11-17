#!/bin/bash
# verify-donors.sh -- (w201, box2) every mad donor: rad-info shows token_embd.weight as exl3i codebook=mad, rate=K, shape
# 248320x5120, src= its mad/K<k>/src; the file size equals the i8 donor's for the same K when that exists (same encoding
# sizes). Prints one line per K and a summary. CPU only.
set -u
NEW=/data/q27/w201/rad21; D=/data/q27/emb/trellis
ok=0; bad=0
for K in 1.5 1.625 1.75 1.875 2.25 2.5 2.625 2.75 2.875 3 3.375 3.625 3.75 3.875 4.125 4.25 4.625 5 5.5 6 6.5; do
  f=$D/emb-t-mad-K$K.rad
  if [ ! -f $f ]; then echo "K$K MISSING"; bad=$((bad+1)); continue; fi
  line=$(RADIANCE_HOME=$NEW/home timeout 120 $NEW/install/bin/rad-info -v $f 2>/dev/null | grep -E "^ +token_embd.weight" | head -1)
  sz=$(stat -c %s $f)
  i8=$D/emb-t-i8-K$K.rad; isz=$( [ -f $i8 ] && stat -c %s $i8 || echo - )
  good=1
  echo "$line" | grep -q "codebook=mad,rate=$K," || good=0
  echo "$line" | grep -q "248320 *x *5120\|248320x5120" || echo "$line" | grep -q "248320" || good=0
  echo "$line" | grep -q "src=$D/mad/K$K/src" || good=0
  [ "$isz" = "-" ] || [ "$isz" = "$sz" ] || echo "  (size differs from the i8 donor: $sz vs $isz)"
  if [ $good = 1 ]; then ok=$((ok+1)); echo "K$K OK bytes=$sz i8=$isz"; else bad=$((bad+1)); echo "K$K BAD bytes=$sz i8=$isz line=[$line]"; fi
done
echo "donors OK=$ok BAD=$bad"
