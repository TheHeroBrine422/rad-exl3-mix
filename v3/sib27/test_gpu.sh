#!/bin/bash
# test_gpu.sh -- the 0.8B GPU equivalence test of sibmom27.py (#87 at 27B), for the lead to run on the 4070 (one GPU job,
# ~30-40 min, VRAM_CAP_GB 7.2 as the proxy runs). Reference: w07-seqsib's moments.py --mode sib collection
# /data/yb/w07-seqsib/mom-k3base (K3 base siblings, 1M qad tokens, seed 0) and its asym arm k3asym (search 0.08490).
#   1. floor:  moments.py once more with the same arguments (-> ref2): the GPU's own run-to-run difference
#   2. ours:   sibmom27 --wq (the same siblings) and --level (the siblings decoded from q-k3base's shards: the 27B path)
#   3. rule:   cmpmom.py verdict: floor 0 -> ours (--wq) bit-exact-level (max rel <= 1e-6); else every stat (H, C, N)
#              within 2x the floor (worst group, relative Frobenius); n equal. --level vs --wq: reported (CPU test:
#              the decode moves C / N exactly as much as a same-size perturbation of the siblings)
#   4. mkdenoise (the patched lean copy, MKD_LEAN=1: the 27B code path on the GPU) on ours --wq: files vs w07's
#      dn-asym-k3base (bit-identical if the moments are; reported either way)
#   5. end to end, the 27B path: mkdenoise on ours --level -> the 48 writers requantized (q2b.sh, seed 0) -> search KL
#      PASS: |log(KL / KL(k3asym))| <= 0.01 (w07's writers-only control SD at K3 was 0.0083) and KL / KL(k3base) <= 0.87
#      (w07: 0.8585 at seed 0)
# Out /data/yb/sib27test/result.txt. Idempotent (steps skip what exists).
P=/workspace/rad-exl3-mix/proxy2b S=/workspace/rad-exl3-mix/v3/sib27 D7=$P/exp/w07-denoise M=/data/models/qwen35-08b
META=/data/challenge/calib-meta Y=/data/yb W7=/data/yb/w07-seqsib O=/data/yb/sib27test H=/data/h08/calib G=/data/g08-search1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2} POOLS=/workspace/rad-exl3-mix/eval/pools/pools-v3.json
mkdir -p $O; cd /tmp; log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
die() { log "$*: stop"; exit 1; }
ARGS="--seed 0 --tokens 1000000"
if [ ! -f $O/ref2.done ]; then
  for r in 0-11 12-23; do python3 $D7/moments.py $M $META $O/ref2 --mode sib --wq $Y/q-k3base/wq --layers $r $ARGS >> $O/ref2.log 2>&1 || die "ref2 $r"; done
  touch $O/ref2.done; log "ref2: $(grep '^MOMENTS DONE' $O/ref2.log | tail -1)"
fi
for v in wq lv; do [ -f $O/$v.done ] && continue
  src=$([ $v = wq ] && echo "--wq $Y/q-k3base/wq" || echo "--level $Y/q-k3base --codebook i8")
  python3 $S/sibmom27.py $M $META $O/$v $src $ARGS > $O/$v.log 2>&1 || die "sibmom27 $v ($O/$v.log)"
  touch $O/$v.done; log "$v: $(tail -1 $O/$v.log); $(grep -o 'peak GPU [0-9.]* GiB' $O/$v.log | sort -k3 -n | tail -1)"
done
log "$(python3 $S/cmpmom.py verdict $W7/mom-k3base $O/ref2 $O/wq)"
log "--level vs --wq: $(python3 $S/cmpmom.py diff $O/lv $O/wq | tail -1)"
for v in wq lv; do [ -f $O/dn-$v.done ] && continue
  MKD_LEAN=1 python3 $S/lean/mkdenoise.py $H $O/$v $O/dn-$v --lam 1 --rho 0.5 --ckpt $M > $O/dn-$v.log 2>&1 || die "mkdenoise $v"
  touch $O/dn-$v.done
done
log "dn (lean, GPU) from ours --wq vs w07's dn-asym-k3base: $(python3 - $O/dn-wq $W7/dn-asym-k3base <<'PY'
import glob, os, sys, torch
from safetensors import safe_open
a, b = sys.argv[1:3]; worst = {"hin": 0.0, "wpre": 0.0}; same = 0; n = 0
for f in sorted(glob.glob(f"{b}/*/*.safetensors")):
    r = os.path.relpath(f, b); n += 1
    with safe_open(f"{a}/{r}", "pt") as x, safe_open(f, "pt") as y:
        k = list(y.keys())[0]; u, v = x.get_tensor(k).double(), y.get_tensor(k).double()
    same += torch.equal(u, v); d = r.split("/")[0]
    if d == "wpre": v = v - torch.eye(v.shape[0], dtype=v.dtype); u = u - torch.eye(u.shape[0], dtype=u.dtype)
    worst[d] = max(worst[d], float((u - v).norm() / v.norm()))
print(f"{same} of {n} files bit-identical; worst ||d hin|| / ||hin|| {worst['hin']:.1e}, ||d (F - I)|| / ||F - I|| {worst['wpre']:.1e}")
PY
)"
t=lvasym
if [ ! -f $O/$t.f32 ]; then
  [ $(ls $O/q-$t/wq/*.safetensors 2>/dev/null | wc -l) -ge 48 ] || $P/q2b.sh 08 3 i8 $O/q-$t --only $W7/only.txt \
    --hin $O/dn-lv/hin --wpre $O/dn-lv/wpre > $O/q-$t.log 2>&1
  [ $(ls $O/q-$t/wq/*.safetensors 2>/dev/null | wc -l) = 48 ] || die "quant $t ($O/q-$t.log)"
  d=$O/q-$t/wqall; mkdir -p $d
  for f in $Y/q-k3base/wq/*.safetensors; do ln -sfn $(readlink -f $f) $d/; done
  for f in $O/q-$t/wq/*.safetensors; do ln -sfn $f $d/; done
  [ $(ls $d/*.safetensors | wc -l) = 150 ] || die "assemble $t"
  log "$t: $(python3 $P/evalq.py $M $G --wq $d --out $O/$t.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*')"
fi
python3 - $G $W7/k3asym.f32 $Y/k3base.f32 $O/$t.f32 $P/pcmp.py <<'PY' | tee -a $O/result.txt
import json, math, subprocess, sys
g, asym, base, ours, pc = sys.argv[1:6]
def cmp(a, b): return json.loads(subprocess.run([sys.executable, pc, g, a, b], capture_output=True, text=True).stdout.strip().splitlines()[-1])
r1, r2 = cmp(asym, ours), cmp(base, ours)
ok = abs(math.log(r1["ratio"])) <= 0.01 and r2["ratio"] <= 0.87
print(f"E2E {'PASS' if ok else 'FAIL'}: ours (level-decoded siblings, sibmom27) / w07 k3asym {r1['ratio']:.4f} (z {r1['z']:.1f}); "
      f"/ k3base {r2['ratio']:.4f} (w07: 0.8585)")
PY
log "TEST_GPU DONE"
