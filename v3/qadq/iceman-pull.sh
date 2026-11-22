#!/bin/bash
# iceman-pull.sh (run on the VM; needs an ssh-agent holding the card key: SSH_AUTH_SOCK, iceman reaches Thor over the LAN with the forwarded agent, ~100 MB/s; the VM relay did 0.1 MB/s) -- stage w214's inputs on iceman /data/queue (w222): the three xcorp / trace goldens from Thor's backup over the LAN (streamed through the VM),
# the small files and j217 from box2. Restartable: each directory is extracted to NAME.part and renamed. Log /workspace/logs/iceman-pull.log.
K=/workspace/card_session_ed25519; O="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=30 -i $K"
export SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-$(cat /tmp/agent.sock)}
IC="ssh $O -p 2223 root@192.168.1.7"; TH="ssh $O -p 2224 root@192.168.1.2"; B2="ssh $O -p 2241 root@127.0.0.1"
L=/workspace/logs/iceman-pull.log; log() { echo "$(TZ=America/Chicago date '+%m-%d %-I:%M %p CST') $*" >> $L; }
Q=/data/queue
$IC "mkdir -p $Q/gold $Q/w214 $Q/slices $Q/ctl" < /dev/null
for d in xc8k-b2-c0 xc8k-b2-c1 tr32k-b2-c0; do
  $IC "test -e $Q/gold/$d/.done" < /dev/null && { log "skip $d"; continue; }
  log "start $d"; t=$(date +%s)
  $IC -A "rm -rf $Q/gold/$d.part && mkdir -p $Q/gold/$d.part && ssh $O -p 2224 root@192.168.1.2 'tar cf - -C /data/backup/gold27 $d' | tar xf - -C $Q/gold/$d.part --strip-components=1 && touch $Q/gold/$d.part/.done && rm -rf $Q/gold/$d && mv $Q/gold/$d.part $Q/gold/$d" < /dev/null \
    && log "done $d in $(( $(date +%s) - t )) s" || log "FAIL $d"
done
# small files from box2 (w214: chunks, pools, genmask, pylib-fla, judge args; w217: j217 slice + the three ctl rows)
$IC "test -e $Q/small.done" < /dev/null || {
  log "start small"
  $B2 "cd /data/q27/w214 && tar cf - chunks214.txt pools214.json genmask pylib-fla judge214.args" < /dev/null | $IC "cd $Q/w214 && tar xf -" && \
  $B2 "cd /data/q27/w217 && tar cf - judge217.args scores/ctl-2.50.json scores/ctl-2.50.json.rows scores/ctl-3.00.json scores/ctl-3.00.json.rows scores/ctl-4.00.json scores/ctl-4.00.json.rows" < /dev/null | $IC "cd $Q/ctl && tar xf - --strip-components=0" && \
  $IC "touch $Q/small.done" < /dev/null && log "done small" || log "FAIL small"; }
$IC "test -e $Q/slices/golden-j217/.done" < /dev/null || {
  log "start j217"; t=$(date +%s)
  $B2 "cd /data/q27/w217/slices && tar cf - golden-j217 j217.jsonl" < /dev/null | $IC "rm -rf $Q/slices/j217.part && mkdir -p $Q/slices/j217.part && tar xf - -C $Q/slices/j217.part && touch $Q/slices/j217.part/golden-j217/.done && mv $Q/slices/j217.part/j217.jsonl $Q/slices/ && rm -rf $Q/slices/golden-j217 && mv $Q/slices/j217.part/golden-j217 $Q/slices/ && rmdir $Q/slices/j217.part" \
    && log "done j217 in $(( $(date +%s) - t )) s" || log "FAIL j217"; }
log "ICEMAN-PULL DONE"
