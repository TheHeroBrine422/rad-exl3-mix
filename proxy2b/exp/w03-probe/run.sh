#!/bin/bash
# run.sh -- w03 A10 (ranked.md): sqrt-Fisher logit probe instead of the sampled label in the diagonal H_out collector,
# 0.8B: no-op gate, one 1,024-window collection on c18's windows, precision vs c18, quants at 256 / 512, 3 seeds (README).
# PROBE=rad (default) | gauss. Needs #33-18's c18 collection; the eB controls are yaqaFact7's (built here when missing).
P=/workspace/rad-exl3-mix/proxy2b E=$P/exp/w03-probe M=/data/models/qwen35-08b G=/data/g08-search1 S=/workspace/rad-exl3-mix/v3/yaqa
Y=/data/yb C=/data/yb/c18 O=/data/yb/w03-probe PR=${PROBE:-rad}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
mkdir -p $O/src; cd /tmp
log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
q() { d=$1 tag=$2 k=$3; shift 3
  [ -f $d/$tag.f32 ] && return
  [ $(ls $d/q-$tag/wq/*.safetensors 2>/dev/null | wc -l) -ge 150 ] || $P/q2b.sh 08 $k i8 $d/q-$tag "$@" > $d/q-$tag.log 2>&1
  log "$tag: $(python3 $P/evalq.py $M $G --wq $d/q-$tag/wq --out $d/$tag.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
[ -f $C/hdiag-snap1024/halves.json ] || { log "c18 snap1024 missing: run yaqaFact7.sh's collection (#33-18) first"; exit 1; }
# the patched and unpatched collector side by side (works before and after the lead merges collect-probe.patch)
if [ ! -f $O/src/collect.py ]; then
  if grep -q YAQA_PROBE $S/collect.py; then cp $S/collect.py $O/src/collect.py; patch -R -s -o $O/src/collect-orig.py $S/collect.py $E/collect-probe.patch
  else cp $S/collect.py $O/src/collect-orig.py; patch -s -o $O/src/collect.py $S/collect.py $E/collect-probe.patch; fi || { log "PATCH FAIL"; exit 1; }
  cp $S/bf16x.py $O/src/
fi
W=/data/c08work; mkdir -p $W/challenge/golden; ln -sfn /data/challenge/calib-meta $W/challenge/golden/calib
col() { out=$1 n=$2 py=$3; shift 3      # collect.py YAQA_DIAG on c18's windows and settings (yaqaFact7.sh)
  env RXM_WORK=$W RXM_CKPT=$M YAQA_POOLS=/workspace/rad-exl3-mix/eval/pools/pools-v3.json YAQA_FP32=1 YAQA_DIAG=1 YAQA_SEED=0 "$@" \
    python3 $O/src/$py $out $n 2048 >> $out.log 2>&1; }
# 0. no-op gate (~4 min): patched with YAQA_PROBE unset vs unpatched, 8 windows: bit-identical (or within the unpatched
#    collector's own run-to-run difference, if it has one); the probe run finite
if [ ! -f $O/noop.ok ]; then
  rm -rf $O/noop-a $O/noop-a2 $O/noop-b $O/noop-p $O/noop-*.log
  col $O/noop-a 8 collect-orig.py; col $O/noop-a2 8 collect-orig.py; col $O/noop-b 8 collect.py; col $O/noop-p 8 collect.py YAQA_PROBE=$PR
  python3 - $O <<'PY' | tee -a $O/result.txt
import glob, os, sys, torch
from safetensors import safe_open
o = sys.argv[1]; ld = lambda f: safe_open(f, "pt").get_tensor("hout_diag")
fa = sorted(glob.glob(f"{o}/noop-a/hdiag/*.safetensors")); assert len(fa) == 150, len(fa)
dmax = lambda x: max(float((ld(f) - ld(f"{o}/{x}/hdiag/{os.path.basename(f)}")).abs().max() / ld(f).abs().max()) for f in fa)
d_b, d_a2 = dmax("noop-b"), dmax("noop-a2")       # patched vs unpatched; unpatched vs itself (run-to-run floor)
same = d_b == 0.0 or d_b <= d_a2
fin = all(bool(torch.isfinite(ld(f"{o}/noop-p/hdiag/{os.path.basename(f)}")).all()) for f in fa)
cs = sorted(float(torch.nn.functional.cosine_similarity(ld(f).double(), ld(f"{o}/noop-p/hdiag/{os.path.basename(f)}").double(), 0)) for f in fa)
print(f"NOOP {'PASS' if same and fin else 'FAIL'}: patched (probe off) vs unpatched max rel diff {d_b:.2e} (unpatched rerun: "
      f"{d_a2:.2e}); probe finite: {fin}; "
      f"8-window probe vs sampled cos median {cs[75]:.3f} min {cs[0]:.3f}")
if same and fin: open(f"{o}/noop.ok", "w").close()
PY
  [ -f $O/noop.ok ] || exit 1
fi
# 1. the probe collection: c18's 1,024 windows, snapshots 128-1024 (~25 min; resumes from hdiag-state.pt)
if [ ! -f $O/cprobe/hdiag-snap1024/halves.json ]; then
  col $O/cprobe 1024 collect.py YAQA_PROBE=$PR YAQA_SNAP=128,256,512,1024 YAQA_SAVE_EVERY=64
  [ -f $O/cprobe/hdiag-snap1024/halves.json ] || { log "COLLECT FAIL"; exit 1; }
  log "collected ($PR): $(grep '^DONE' $O/cprobe.log | tail -1)"
fi
# 2. precision (noise only): halves and cross-cosines vs c18
[ -f $O/probecmp.txt ] || { CUDA_VISIBLE_DEVICES= python3 $E/probecmp.py $C $O/cprobe > $O/probecmp.txt && cat $O/probecmp.txt >> $O/result.txt; }
# P1 (README) decides whether the 12 quants are worth running: stop here unless it holds (FORCE_Q=1 runs them anyway)
P1=$(python3 -c "
import json, sys; r = {x['n']: x for x in json.loads(open('$O/probecmp.txt').read().split('PROBECMP ', 1)[1])}
s, p = r[1024]['sampled'], r[512]['probe']; print('pass' if p[0] >= s[0] and p[1] >= s[1] else 'fail')")
log "P1 (probe halves at 512 >= sampled at 1024, median and p5): $P1"
[ "$P1" = pass ] || [ -n "$FORCE_Q" ] || { log "W03-A10 STOP after P1 (no quants)"; exit 0; }
# 3. quants on base settings + the diagonal, 3 rotation seeds x K3 / K2; eB* = yaqaFact7's arms (same commands)
NS="256 512 ${PROBE_Q1024:+1024}"
for s in 0 1 2; do for k in 3 2; do
  sfx=$([ $s = 0 ] && echo "" || echo "_s$s")
  for n in 1024 256 512; do b=$([ $n = 1024 ] && echo "" || echo $n)
    q $Y k${k}eB$b$sfx $k --hout $C/hdiag-snap$n --sigma-reg-out 0.01 --seed-offset $s; done
  for n in $NS; do q $O k${k}eP$n$sfx $k --hout $O/cprobe/hdiag-snap$n --sigma-reg-out 0.01 --seed-offset $s; done
done; done
# 4. summary and the pass rule's numbers
for k in 3 2; do
  f() { echo $1/k$k$2.f32,$1/k$k$2_s1.f32,$1/k$k$2_s2.f32; }
  python3 $E/seedsumm.py base=$(f $Y base) eB=$(f $Y eB) eB256=$(f $Y eB256) eB512=$(f $Y eB512) \
    $(for n in $NS; do echo eP$n=$(f $O eP$n); done) --ref base | sed "s/^/K$k /" | tee -a $O/result.txt
done
python3 - $O/result.txt <<'PY' | tee -a $O/result.txt
import json, sys
R = {}
for l in open(sys.argv[1]):
    if l.startswith("K") and "{" in l: k, j = l.split(" ", 1); R[(k, json.loads(j)["arm"])] = json.loads(j)
for k in ("K3", "K2"):
    if (k, "eB") not in R: continue
    gB = R[(k, "eB")]["gain"]; sd = R[(k, "base")]["sd_rel"] * R[(k, "base")]["mean"]
    for n in (256, 512, 1024):
        a = R.get((k, f"eP{n}")); b = R.get((k, f"eB{n}" if n != 1024 else "eB"))
        if not a: continue
        print(f"VERDICT {k} probe {n}: keeps {a['gain'] / gB:.0%} of the 1K sampled gain; every seed beats base: {a['every_beats_ref']}; "
              f"vs sampled at {n}: {(a['mean'] - b['mean']) / sd:+.2f} base seed-SDs" if b else "")
PY
log W03-A10 DONE
