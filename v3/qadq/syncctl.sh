#!/bin/bash
# syncctl.sh (run on the VM) -- copy control reads between the machines' scores dirs so the paired-vs-control columns resolve across machines (the .json.rows files are 2.8 MB each):
#   syncctl.sh box2 iceman C30-ckpt-2e+05 C30-ckpt-5e+05 ...   copies SRC/scores/<name>.json.rows to DST/scores/ (rows only, never .json: no job reads as done), then qrow.py fill on DST.
K=/workspace/card_session_ed25519; O="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i $K"
declare -A SSH=([box2]="ssh $O -p 2241 root@127.0.0.1" [iceman]="ssh $O -p 2223 root@192.168.1.7") ROOT=([box2]=/data/q27/queue [iceman]=/data/queue)
SRC=$1 DST=$2; shift 2
for n in "$@"; do
  ${SSH[$SRC]} "cat ${ROOT[$SRC]}/scores/$n.json.rows" < /dev/null | ${SSH[$DST]} "cat > ${ROOT[$DST]}/scores/$n.json.rows.part && mv ${ROOT[$DST]}/scores/$n.json.rows.part ${ROOT[$DST]}/scores/$n.json.rows"
done
case $DST in box2) G=/data/q27/w217/slices/golden-j217; C=/data/q27/w217/scores; PY=/data/q27/venv/bin/python;; iceman) G=/data/queue/slices/golden-j217; C=/data/queue/ctl/scores; PY=/data/venv-exl3/bin/python;; esac
${SSH[$DST]} "$PY ${ROOT[$DST]}/code/v3/qadq/qrow.py fill ${ROOT[$DST]} --golden $G --ctldir $C" < /dev/null
