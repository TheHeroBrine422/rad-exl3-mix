#!/bin/bash
# ext-pull.sh MACHINE (run on the VM; ssh-agent with the card key: SSH_AUTH_SOCK) -- ship the extended plan's new goldens (xc8k-b1, xc8k-b2-c2, tr32k-b1 from Thor's /data/backup/gold27, ~34 GB), the six gen masks and the plan file
# to QROOT/ext/ on box2 (streamed through the VM, ~7 MB/s) or iceman (Thor -> iceman on the LAN with the forwarded agent, ~100 MB/s). Restartable: a dir is extracted to NAME.part and renamed. Log /workspace/logs/ext-pull-MACHINE.log.
M=$1; K=/workspace/card_session_ed25519; O="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30 -i $K"
export SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-$(cat /tmp/agent.sock)}
TH="ssh $O -p 2224 root@192.168.1.2"; L=/workspace/logs/ext-pull-$M.log; log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> $L; }
case $M in box2) D="ssh $O -p 2241 root@127.0.0.1"; E=/data/q27/queue/ext;; iceman) D="ssh $O -p 2223 root@192.168.1.7"; E=/data/queue/ext;; esac
P=/workspace/data-ext/plan
$D "mkdir -p $E/gold $E/genmask" < /dev/null
tar cf - -C $P genmask ext-plan.json | $D "tar xf - -C $E" && log "masks + plan shipped"
for d in xc8k-b2-c2 tr32k-b1 xc8k-b1; do
  $D "test -e $E/gold/$d/.done" < /dev/null && { log "skip $d"; continue; }
  AV=$($D "df -BG --output=avail $E | tail -1 | tr -dc 0-9" < /dev/null); [ "$AV" -ge 150 ] || { log "ABORT: only $AV GB free"; exit 1; }
  log "start $d"; t=$(date +%s)
  if [ $M = iceman ]; then
    ssh -A $O -p 2223 root@192.168.1.7 "rm -rf $E/gold/$d.part && mkdir -p $E/gold/$d.part && ssh $O -p 2224 root@192.168.1.2 'tar cf - -C /data/backup/gold27 $d' | tar xf - -C $E/gold/$d.part --strip-components=1 && touch $E/gold/$d.part/.done && mv $E/gold/$d.part $E/gold/$d" < /dev/null
  else
    $TH "tar cf - -C /data/backup/gold27 $d" < /dev/null | $D "rm -rf $E/gold/$d.part && mkdir -p $E/gold/$d.part && tar xf - -C $E/gold/$d.part --strip-components=1 && touch $E/gold/$d.part/.done && mv $E/gold/$d.part $E/gold/$d"
  fi && log "done $d in $(( $(date +%s) - t )) s" || log "FAIL $d"
done
log "EXT-PULL $M DONE"
