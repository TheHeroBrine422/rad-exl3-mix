#!/bin/bash
# streamcheck.sh -- one-screen health check of the streaming pipeline (run on the GPU box)
: ${RXM_WORK:=/root}; export RXM_WORK   # paths (README: Paths)
cd ${RXM_WORK}; A=${RXM_WORK}/auto
echo "== $(date +%H:%M:%S)  driver: $(kill -0 $(cat $A/state/driver.pid) 2>/dev/null && echo up || echo DOWN)  w0: $(kill -0 $(cat $A/state/w0.pid) 2>/dev/null && echo up || echo DOWN)  w1: $(kill -0 $(cat $A/state/w1.pid) 2>/dev/null && echo up || echo DOWN)  stream fit next: s$(cat $A/state/streamfit)"
echo "running: $(ls evq/claim | while read t; do [ -f evq/scores/$t.f32 ] || grep -q "\"tag\": \"$t\", .*fail" evq/results.jsonl || echo $t; done | tr '\n' ' ')"
pend() { awk '{print $1}' "$1" | while read t; do [ -z "$t" ] && continue; [ -f evq/scores/$t.f32 ] || grep -q "\"tag\": \"$t\", .*fail" evq/results.jsonl || echo $t; done | wc -l; }
echo "pending: prio $(pend $A/q.prio), main $(pend $A/q)   results total $(wc -l < evq/results.jsonl), fails $(grep -c fail evq/results.jsonl)"
python3 -I - <<'EOF'
import json, os, time
now = time.time(); n = 0; kinds = {}
rs = [json.loads(l) for l in open(os.environ["RXM_WORK"] + "/evq/results.jsonl")]
for r in rs[-40:]:
    t = r["tag"]
    k = "swap" if t.startswith("cal-") else "thompson" if "-ts" in t else "mean" if "-knap-" in t else "other"
    kinds[k] = kinds.get(k, 0) + 1
f = os.environ["RXM_WORK"] + "/evq/results.jsonl"
print(f"last 40 results by kind: {kinds}   results file age {now - os.path.getmtime(f):.0f}s")
EOF
grep -E "stream fit s[0-9]+ \(" $A/auto.log | tail -4 | cut -c1-150
grep -E "CV log-RMSE" $A/auto.log | tail -1 | cut -c1-120
