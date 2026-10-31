#!/bin/bash
# summary.sh -- $RXM_WORK/auto/SUMMARY.md: pipeline status, latest fit, per-round predicted vs measured, held-out tables
: ${RXM_WORK:=/root}; export RXM_WORK   # paths (README: Paths)
cd ${RXM_WORK}; A=${RXM_WORK}/auto; O=$A/SUMMARY.md
{
echo "# Overnight pipeline summary ($(date '+%Y-%m-%d %H:%M') card time)"; echo
echo "## Status"; echo '```'
echo "deadline: $(date -d @$(cat $A/state/deadline) '+%m-%d %H:%M'); round: $(cat $A/state/round 2>/dev/null)"
grep -E "\[c1\]|OURCHAINDONE" ourchain.log | tail -n 12
grep -E "SMOKE|smoke ok|excluded" $A/auto.log
echo "excluded: $(cat $A/state/exclude 2>/dev/null | tr '\n' ' ')"
echo "failed evals: $(grep -c fail evq/results.jsonl)"; echo '```'
L=$(ls -t $A/fit-*.txt 2>/dev/null | head -1)
if [ -n "$L" ]; then echo; echo "## Latest fit ($(basename $L .txt))"; echo '```'; head -2 $L | cut -c1-400
  sed -n '/log10 s at knots/,/head /p' $L; echo '```'; fi
echo; echo "## Rounds: predicted vs measured (mean-model mixes, on the round's fit slice)"; echo '```'
python3 -I - <<'PY'
import json, glob, os, re
W = os.environ["RXM_WORK"]
res = {}
for l in open(f"{W}/evq/results.jsonl"):
    r = json.loads(l)
    if "fail" not in r: res[r["tag"]] = r
for f in sorted(glob.glob(f"{W}/auto/frontier-r*.txt"), key=lambda p: int(re.search(r"r(\d+)", p).group(1))):
    rnd = re.search(r"frontier-(r\d+)", f).group(1)
    rows = []
    for l in open(f):
        m = re.match(r"(knap-\S+)\s+bpw ([\d.]+)\s+pred KLD ([\d.]+)", l)
        if not m or "-ts" in m.group(1): continue
        t = next((t for t in res if t.endswith(m.group(1)) and not t.startswith(("w-", "v2-"))), None)
        k = res[t]["kld"] if t else None
        rows.append(f"  {m.group(1)[5:]:28} bpw {float(m.group(2)):.3f}  pred {float(m.group(3)):.5f}  meas {k if k is None else round(k, 5)}"
                    + (f"  meas/pred {k/float(m.group(3)):.2f}" if k else ""))
    print(rnd); print("\n".join(rows))
PY
echo '```'
for P in "w- golden-ver1" "v2- golden-ver2"; do set -- $P
  if ls evq/scores/$1knap-* > /dev/null 2>&1; then echo; echo "## Held out: $2 (mixes vs reference containers, 95% bootstrap CI)"; echo '```'
    python3 -I knap/verreport.py $1 $2 2>&1 | cut -c1-130; echo '```'; fi
done
} > $O.part 2>&1 && mv $O.part $O
