#!/bin/bash
# copy195.sh -- on ds8: push the 15 s195 slices (golden dirs + sidecars) to box2 (same path, 10 GbE), then md5 every file on both sides
# and compare. The ssh line to box2 is read from the lead's push-box2.sh. Output: strata/md5-golden-s195-{ds8,box2}.txt, "MD5 IDENTICAL".
set -eu
Q=/data/q27; W=$Q/w180; eval "$(sed -n 3p $Q/srch183/push-box2.sh | sed 's/^set -e; //')"
$B $H "mkdir -p $W/slices"
rsync -a --partial -e "$B" $W/slices/golden-s195* $H:$W/slices/
L='cd /data/q27/w180/slices && find . -maxdepth 2 -type f \( -name "golden-s195*" -o -path "./golden-s195*/*" \) | sort | xargs -P 8 -n 8 md5sum | sort -k2'
nice -n 19 bash -c "$L" > $W/strata/md5-golden-s195-ds8.txt
$B $H "nice -n 19 bash -c '$L'" > $W/strata/md5-golden-s195-box2.txt
wc -l $W/strata/md5-golden-s195-*.txt
cmp $W/strata/md5-golden-s195-ds8.txt $W/strata/md5-golden-s195-box2.txt && echo "MD5 IDENTICAL"
$B $H "cd $W/batch/gold && md5sum */kld.json */rows.f32 */tokens.i32 */top.bin" | sort -k2 > $W/strata/md5-batch-box2.txt
grep -v gm.u8 $W/strata/md5-batch-ds8.txt | sort -k2 | cmp - $W/strata/md5-batch-box2.txt && echo "BATCH MD5 IDENTICAL"
$B $H "df -h /data | tail -1"
