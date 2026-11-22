#!/bin/bash
# deploy.sh (run on the VM) -- ship v3/qadq/ (this directory) to QROOT/code/v3/qadq on both machines. The base code tree (QROOT/code: eval, mix, pipeline, store, v3 = w221's, trainer qad220.py unchanged)
# was copied once from box2 /data/q27/w221/code. Idempotent.
K=/workspace/card_session_ed25519; O="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i $K"
D=$(dirname "$(readlink -f "$0")")
tar cf - -C $D/.. qadq | ssh $O -p 2241 root@127.0.0.1 'mkdir -p /data/q27/queue/code/v3 && tar xf - -C /data/q27/queue/code/v3 && chmod +x /data/q27/queue/code/v3/qadq/*.sh' 
tar cf - -C $D/.. qadq | ssh $O -p 2223 root@192.168.1.7 'mkdir -p /data/queue/code/v3 && tar xf - -C /data/queue/code/v3 && chmod +x /data/queue/code/v3/qadq/*.sh'
