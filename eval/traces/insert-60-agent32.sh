#!/bin/bash
# w67: chunked agentic slices (eval/traces/README.md). Runs as a q2.sh insert, before the next arm. Card 0 (card0.lock,
# then ram.lock shared, per step): a 32k memory probe of mkgold27 (~5 min; falls back to attn-chunk / 4 stages / 24k
# chunks), bf16 goldens agent-search32 + agent-verify32 (~1.0M tokens, projected by the probe; refused above 3 h), then 8
# K2.5 / K4 mixes scored on agent-search32 (~6-8 min each). CPU stages (chunk, whole) were done in the dry run;
# agent32.sh redoes any missing piece. Restartable: q2 marks this insert done whatever its exit, so after a FAIL rerun it
# by hand (`bash $0`); it resumes per stage / arm.
cd /data/agent32 || exit 1
bash /data/agent32/code/eval/traces/agent32.sh < /dev/null
rc=$?
if [ $rc = 0 ]; then m="done: $(grep -hE '^(base -> baseswj|base4v4 -> base4swj) ' /data/agent32/report.txt | awk '{printf "%s->%s %s; ", $1, $3, $4}')(report /data/agent32/report.txt)"
else m="FAIL exit $rc (/data/agent32/result.txt)"; fi
echo "$(date -u +%T) agent32 $m" >> /data/q1106/q.log
