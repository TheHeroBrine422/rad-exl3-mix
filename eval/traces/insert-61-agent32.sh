#!/bin/bash
# w67: chunked agentic slices (eval/traces/README.md), second try. w67 stopped insert-60 at 07:56 UTC on its all-row ppl
# (30,626 / 57,305). That was a false alarm: the published golden gives 57,556 on the same rows, because the 27B puts
# p ~0.98 on <|im_end|> inside user / tool turns; gen rows have ppl ~1.4. This run first GATES each kernel set
# (a check of the ROCm stack): mkgold27 on 2 docs whose context the published golden has (2.3k and 26.7k tokens) must
# agree with it (KL <= 0.005, top-1 >= 0.95). The first set that passes and fits the 32,761-token chunk makes the goldens:
# k1 default, k2 attn-chunk 2048, k3 + aotriton off, k4 fla off, k5 both off.
# Card 0 (card0.lock, then ram.lock shared, per step): gates ~2-10 min a set; probe ~2 min; goldens ~1.0M tokens
# (projected by the probe; refused above 3 h); then 8 mixes on agent-search32 (~6-8 min each). The gate lines go to q.log
# too, as gate-ok / gate-no (no FAIL word: the watchdog's q.log rule; the k1 result says whether the default ROCm torch
# stack is right at 2.3k / 26.7k tokens).
# Restartable: after a FAIL rerun it by hand (`bash $0`); it resumes per stage / kernel set / arm.
cd /data/agent32 || exit 1
bash /data/agent32/code/eval/traces/agent32.sh < /dev/null
rc=$?
grep -h "^[0-9:]* gate k" /data/agent32/result.txt | sed 's/^[0-9:]* /agent32 /; s/GATE FAIL/gate-no/; s/GATE PASS/gate-ok/' | cut -c1-230 | while read -r l; do echo "$(date -u +%T) $l" >> /data/q1106/q.log; done
if [ $rc = 0 ]; then m="done: $(grep -hE '^(base -> baseswj|base4v4 -> base4swj) ' /data/agent32/report.txt | awk '{printf "%s->%s %s; ", $1, $3, $4}')(report /data/agent32/report.txt)"
else m="FAIL exit $rc (/data/agent32/result.txt)"; fi
echo "$(date -u +%T) agent32 $m" >> /data/q1106/q.log
