#!/bin/bash
# run.sh -- w03 A9 (ranked.md): forward-only analytic SwiGLU diagonal for the MLP gate / up H_out, 0.8B, 3 seeds (README).
# Needs #33-18's collection (/data/yb/c18: hdiag-snap1024 and -dy). Arms reuse yaqaFact7's eB controls (built here
# with yaqaFact7's exact command when missing). A9VAR=a9x (default) | a9d picks the analytic variant (README: CPU gate).
P=/workspace/rad-exl3-mix/proxy2b E=$P/exp/w03-swiglu-diag M=/data/models/qwen35-08b G=/data/g08-search1
Y=/data/yb C=/data/yb/c18 O=/data/yb/w03-swiglu-diag V=${A9VAR:-a9x}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
mkdir -p $O; cd /tmp
log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
q() { d=$1 tag=$2 k=$3; shift 3
  [ -f $d/$tag.f32 ] && return
  [ $(ls $d/q-$tag/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || $P/q2b.sh 08 $k i8 $d/q-$tag "$@" > $d/q-$tag.log 2>&1
  log "$tag: $(python3 $P/evalq.py $M $G --wq $d/q-$tag/wq --out $d/$tag.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
[ -f $C/hdiag-snap1024/halves.json ] && [ -d $C/hdiag-snap1024-dy ] || { log "c18 snap1024 (+ -dy) missing: run yaqaFact7.sh's collection (#33-18) first"; exit 1; }
# 1. forward-only statistics over the same 1,024 calib qad windows as c18 (~5 min)
if ! grep -q "^A9 DONE" $O/fwd.log 2>/dev/null; then
  python3 $E/swiglu_diag.py $M /data/challenge/calib-meta /workspace/rad-exl3-mix/eval/pools/pools-v3.json $O/fwd 1024 \
    --snap 256,1024 > $O/fwd.log 2>&1
  grep -q "^A9 DONE" $O/fwd.log || { log "FWD FAIL"; exit 1; }
  log "forward stats: $(grep '^A9 DONE' $O/fwd.log)"
fi
# 2. the cosine table against the 1K Sketch B diagonal (CPU, seconds)
[ -f $O/a9cmp.txt ] || CUDA_VISIBLE_DEVICES= python3 $E/a9cmp.py $O/fwd/raw-snap1024 $M $C/hdiag-snap1024 \
  --xw $C/hdiag-snap1024-xw --dy $C/hdiag-snap1024-dy --out $O/a9cmp.jsonl > $O/a9cmp.txt && cat $O/a9cmp.txt >> $O/result.txt
# 3. hout dirs: B everywhere but gate/up (one-sided there = noGU; analytic there = $V)
python3 $E/a9mk.py $C/hdiag-snap1024 none $O/hout-noGU >> $O/result.txt
python3 $E/a9mk.py $C/hdiag-snap1024 $O/fwd/$V-snap1024 $O/hout-$V >> $O/result.txt
# C1 (README): a9 median cosine >= 0.95 and min >= 0.90 for gate and up; the a9 arms run only if it holds (FORCE_A9=1:
# anyway); noGU (the gate/up share of the gain, ranked.md B3) runs regardless
C1=$(python3 -c "
import json, numpy as np; r = [json.loads(l) for l in open('$O/a9cmp.jsonl')]
ok = all(np.median([x['cos_$V'] for x in r if x['proj'] == p]) >= 0.95 and min(x['cos_$V'] for x in r if x['proj'] == p) >= 0.90 for p in ('gate_proj', 'up_proj'))
print('pass' if ok else 'fail')")
log "C1 ($V cosine vs Sketch B): $C1"
A9ON=$([ "$C1" = pass ] || [ -n "$FORCE_A9" ] && echo 1)
# 4. quants, 3 rotation seeds x K3 / K2
for s in 0 1 2; do for k in 3 2; do
  sfx=$([ $s = 0 ] && echo "" || echo "_s$s")
  q $Y k${k}eB$sfx $k --hout $C/hdiag-snap1024 --sigma-reg-out 0.01 --seed-offset $s      # = yaqaFact7's arm
  q $O k${k}noGU$sfx $k --hout $O/hout-noGU --sigma-reg-out 0.01 --seed-offset $s
  [ -n "$A9ON" ] && q $O k${k}$V$sfx $k --hout $O/hout-$V --sigma-reg-out 0.01 --seed-offset $s
done; done
# 5. summary and the pass rule's numbers
for k in 3 2; do
  f() { echo $1/k$k$2.f32,$1/k$k$2_s1.f32,$1/k$k$2_s2.f32; }
  python3 $E/seedsumm.py base=$(f $Y base) eB=$(f $Y eB) noGU=$(f $O noGU) $V=$(f $O $V) --ref base | sed "s/^/K$k /" | tee -a $O/result.txt
done
python3 - $O/result.txt $V <<'PY' | tee -a $O/result.txt
import json, sys
R = {}
for l in open(sys.argv[1]):
    if l.startswith("K") and "{" in l: k, j = l.split(" ", 1); R[(k, json.loads(j)["arm"])] = json.loads(j)
v = sys.argv[2]
for k in ("K3", "K2"):
    if not all((k, a) in R for a in ("base", "eB", "noGU")): continue
    m = lambda a: R[(k, a)]["mean"]; sd = R[(k, "base")]["sd_rel"] * m("base")
    gu = m("noGU") - m("eB")
    print(f"VERDICT {k} G0: gate/up gain (noGU - eB) {gu:.5f} = {gu / sd:.2f} base seed-SDs ({gu / m('noGU'):+.1%} of noGU)")
    if (k, v) in R:
        print(f"VERDICT {k} A9: {v} keeps {(m('noGU') - m(v)) / gu:.0%} of it; every {v} seed beats every noGU seed: "
              f"{max(R[(k, v)]['seeds']) < min(R[(k, 'noGU')]['seeds'])}")
PY
log W03-A9 DONE
