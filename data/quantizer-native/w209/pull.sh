#!/bin/bash
# pull.sh -- copy w209's raw results from box2:/data/q27/w209/out into this directory
S="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i /workspace/card_session_ed25519 -p 2241"
cd "$(dirname "$0")" && rsync -a -e "$S" --include="*.jsonl" --include="old*-K*/" --include="old*-K*.log" --exclude="*.safetensors" --exclude="*.f32" --exclude="*.u16" --exclude="*.part" --exclude="debug/" --exclude="h-test*" --exclude="*.err" --exclude="h.tmp" --exclude="*.done" --exclude="cpuref.log" --exclude="prod.log" root@127.0.0.1:/data/q27/w209/out/ . < /dev/null
