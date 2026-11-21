#!/bin/bash
# pull.sh -- copy the 11-21 kernel pass's raw results from box2:/data/q27/w209/out2 into this directory
S="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i /workspace/card_session_ed25519 -p 2241"
cd "$(dirname "$0")" && rsync -a -e "$S" --include="*.jsonl" --include="old-K*/" --include="old-K*/proxy.jsonl" --include="old-K*/stamp.json" --include="old-K*.log" --exclude="*" root@127.0.0.1:/data/q27/w209/out2/ . < /dev/null
