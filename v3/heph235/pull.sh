#!/bin/bash
# pull.sh -- copy box2's raw jsonl results into data/heph235/ (run on the VM)
SSHO="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i /workspace/card_session_ed25519 -p 2241 root@127.0.0.1"
D=/workspace/rad-exl3-mix/data/heph235; mkdir -p $D
$SSHO 'cd /data/q27/w235 && tar cf - out logs/*.done' | tar xf - -C $D
