#!/bin/bash
# join-mik.sh DIR -- w212b: join mikoskinen's container parts (README procedure), record sizes + sha256 of the joined file, check SHA256SUMS.
cd "$1" || exit 1
ls -l deepseek-v4.1-flash-1017-exl3-2.25bpw.rad.part-0[0-5] > parts-sizes.txt
OUT=deepseek-v4.1-flash-1017-exl3-2.25bpw.rad
[ -e join.done ] || {
for p in $OUT.part-0[0-5]; do [ -e $p ] && { cat "$p" >> $OUT.part && rm "$p"; }; done   # restartable only on part boundaries; see README
mv $OUT.part $OUT; touch join.done; }
sha256sum -c SHA256SUMS > sha-check.txt 2>&1
