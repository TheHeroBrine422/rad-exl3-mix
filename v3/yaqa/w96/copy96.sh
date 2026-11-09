#!/bin/bash
# copy96.sh -- gpubox -> Thor (streamed through the VM, as backup-gpubox.sh): w96 inputs. Idempotent (per part .done).
B=/workspace/rad-local/bin L=/workspace/rad-local/workers/w96/logs/copy96.log
log() { echo "$(TZ=America/Chicago date +%T) $*" >> $L; }
part() {  # NAME SRCDIR FILES...
  local n=$1 s=$2; shift 2
  $B/ssh-thor.sh "test -f /data/work/w96/in/$n/.done" < /dev/null && return 0
  local t=$(date +%s)
  $B/ssh-iceman.sh "cd $s && nice -n 19 ionice -c3 tar cf - $*" < /dev/null | \
    $B/ssh-thor.sh "mkdir -p /data/work/w96/in/$n && tar xf - -C /data/work/w96/in/$n && touch /data/work/w96/in/$n/.done" \
    && log "$n OK $(( $(date +%s) - t )) s" || log "$n FAIL"
}
part hs /data/headslab3 headsens.npz perm.npy plan.json K3/wq/lm_head.rowerr.npz K4/wq/lm_head.rowerr.npz \
  K5/wq/lm_head.rowerr.npz K6/wq/lm_head.rowerr.npz K7/wq/lm_head.rowerr.npz
part misc /data/w91 golden-search27b-x.docs
part dump /data/w94/dump search qad
for k in 3 4 5 6 7; do part K$k /data/headslab3/K$k/wq lm_head.safetensors; done
part gold /root/challenge/golden/calib kld.json gen.u8 tokens.i32 top.bin
log ALL DONE
