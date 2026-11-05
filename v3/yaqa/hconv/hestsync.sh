#!/bin/bash
# hestsync.sh -- (VM) copy the 27B Sketch B ("") and xw snapshots to the 4070 (/data/yb/hconv/snaps), as sync.sh does dy.
B=/workspace/rad-exl3-mix/local/bin
have=$($B/ssh-4070.sh 'ls /data/yb/hconv/snaps' | tr '\n' ' ')
for d in $($B/ssh-gpubox.sh 'cd /data/yaqa/hdiag && ls -d hdiag-snap* | grep -E "hdiag-snap[0-9]+(-xw)?$"' < /dev/null); do
  [[ " $have " == *" $d "* ]] && continue
  n=$($B/ssh-gpubox.sh "find /data/yaqa/hdiag/$d -name '*.safetensors' -mmin +2 | wc -l" < /dev/null)
  [ "$n" -ge 400 ] || { echo "$d: $n files, later"; continue; }
  $B/ssh-gpubox.sh "tar -C /data/yaqa/hdiag -cf - $d" < /dev/null | $B/ssh-4070.sh 'tar -C /data/yb/hconv/snaps -xf -' && echo "synced $d"
done
