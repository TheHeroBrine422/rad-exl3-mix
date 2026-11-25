#!/bin/bash
# ship231.sh (VM) -- w231's code dir on box2: /data/q27/w231/code = a copy of the queue's code tree (/data/q27/queue/code) with this repo's
# v3/qad, qad181, qad220, qad225, qad226, qad231 and qadq over it. The queue's own copy is not touched (installing is a separate step: deploy.sh).
K=/workspace/card_session_ed25519; O="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i $K -p 2241 root@127.0.0.1"
D=$(dirname "$(readlink -f "$0")")/..
ssh $O 'set -e; [ -d /data/q27/w231/code/pipeline ] || { mkdir -p /data/q27/w231/code; cp -a /data/q27/queue/code/. /data/q27/w231/code/; }'
tar cf - --exclude=__pycache__ -C $D qad qad181 qad220 qad225 qad226 qad231 qadq | ssh $O 'tar xf - -C /data/q27/w231/code/v3 && chmod +x /data/q27/w231/code/v3/qad231/*.sh /data/q27/w231/code/v3/qadq/*.sh'
