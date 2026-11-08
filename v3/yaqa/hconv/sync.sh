#!/bin/bash
# sync.sh -- (VM) copy new Σdy² snapshots of the 27B collection from gpubox to the 4070 (/data/yb/hconv/snaps); small.
B=/workspace/rad-local/bin
have=$($B/ssh-4070.sh 'mkdir -p /data/yb/hconv/snaps; ls /data/yb/hconv/snaps' | tr '\n' ' ')
for d in $($B/ssh-gpubox.sh 'cd /data/yaqa/hdiag && ls -d hdiag-snap*-dy' < /dev/null); do
  [[ " $have " == *" $d "* ]] && continue
  n=$($B/ssh-gpubox.sh "find /data/yaqa/hdiag/$d -name '*.safetensors' -mmin +2 | wc -l" < /dev/null)
  [ "$n" -ge 400 ] || { echo "$d: $n files (incomplete or fresh), later"; continue; }
  $B/ssh-gpubox.sh "tar -C /data/yaqa/hdiag -cf - $d" < /dev/null | $B/ssh-4070.sh 'tar -C /data/yb/hconv/snaps -xf -' && echo "synced $d"
done
