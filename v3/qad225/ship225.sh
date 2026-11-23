#!/bin/bash
# ship225.sh (VM) -- w225's code dir on box2: /data/q27/w225/code = a copy of the queue's code tree (/data/q27/queue/code, = repo HEAD's v3)
# with this repo's v3/qad, v3/qad220, v3/qad225 over it. The queue's own copy is not touched (install = a separate step after G1).
K=/workspace/card_session_ed25519; O="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i $K -p 2241 root@127.0.0.1"
D=$(dirname "$(readlink -f "$0")")/..
ssh $O 'set -e; [ -d /data/q27/w225/code/pipeline ] || { mkdir -p /data/q27/w225/code; cp -a /data/q27/queue/code/. /data/q27/w225/code/; }'
tar cf - --exclude=__pycache__ -C $D qad qad220 qad225 | ssh $O 'tar xf - -C /data/q27/w225/code/v3 && chmod +x /data/q27/w225/code/v3/qad225/*.sh'
