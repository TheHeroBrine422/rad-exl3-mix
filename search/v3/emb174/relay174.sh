#!/bin/bash
# relay174.sh V2SIZE... -- (w135, #174) radfetch.py's sparse donor-source files on ds8 (/data/q27/emb/dl/), streamed
# through the VM: the VM range-reads HF (~40 MB/s) and pipes each 64 MiB chunk into ds8 (dd at its offset; ds8's own
# HF reads ran at 0.4 MiB/s). Restartable: finished chunks are listed in OUT.part.done on the VM side.
# Run on the VM; then emb174.sh on ds8 builds the donors from the files.
set -eu -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd); SSH=/workspace/rad-local/bin/ssh-ds8.sh; D=/data/q27/emb/dl
U=https://huggingface.co/Hero422/qwen3.8-27b-exl3-v2-rad/resolve/main; ST=${RELAY_STATE:-/workspace/rad-local/workers/w135/relay}
CH=$((64 << 20)); mkdir -p $ST
for s in "$@"; do
  V=qwen3.8-27b-exl3-v2-${s}bpw.rad; O=$D/$V
  $SSH "test -f $O" && { echo "$V: done"; continue; }
  plan=$(python3 -I $HERE/radfetch.py $U/$V /dev/null --plan); fb=$(echo "$plan" | python3 -c "import json,sys; print(json.load(sys.stdin)['file_bytes'])")
  $SSH "mkdir -p $D && touch $O.part && truncate -s $fb $O.part"
  touch $ST/$V.done
  echo "$plan" | python3 -c "
import json,sys; p=json.load(sys.stdin)
for o,n in p['regions']:
    for c in range(o, o+n, $CH): print(c, min($CH, o+n-c))" | while read off n; do
    grep -qx "$off" $ST/$V.done && continue
    for k in 1 2 3 4 5; do
      if curl -sfL -r $off-$((off + n - 1)) $U/$V | $SSH "cat > $O.chunk && [ \$(stat -c %s $O.chunk) = $n ] && dd if=$O.chunk of=$O.part bs=4M seek=$off oflag=seek_bytes conv=notrunc status=none && rm $O.chunk"; then
        echo $off >> $ST/$V.done; break
      fi
      echo "retry $k at $off" >&2; sleep $((5 * k))
    done
    grep -qx "$off" $ST/$V.done || { echo "$V: chunk $off failed" >&2; exit 1; }
  done
  $SSH "mv $O.part $O"; echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S CST') $V relayed"
done
