#!/bin/bash
# test08.sh -- the 0.8B gate of asym27 (ASYM27.md "Gate"), 4070 box, ONE hold of /data/yb/gpu.lock (VRAM_CAP_GB 7.2,
# ~8-10 min). References (kept from #87-05 / #87-06): /data/yb/sib27test/ref2 = moments.py --mode sib on q-k3base's
# --save-wq siblings (1M qad tokens, seed 0; = w07's mom-k3base bit for bit, #87-06), sib27test/lv = sibmom27 --level
# q-k3base (the 27B path's siblings), d1103's arm q-k3asym_r0 (48 writers, lam 1, rho 0, seed 0) and its search score.
#   G1   mkdenoise rho 0 on ref2 -> the 48 writers requantized as asym27.sh's quant() calls ourq (levelargs.py of
#        q-k3base + --only + --hin + --wpre) -> every tensor bit-identical to d1103's q-k3asym_r0 (assemble.py same)
#   G1m  asym27's sibmom27 (low-RAM save) --wq q-k3base/wq, layers 0-3 -> = ref2's layers 0-3 bit for bit (cmpmom diff 0)
#   G2   the 27B path end to end: mkdenoise rho 0 on lv -> the 48 writers as G1 -> assemble.py build over q-k3base (the
#        level asym27.sh packs) -> search KL of q-k3base/wq + the new writers' wq (as d1103 scored k3asym_r0);
#        deq.py check of the assembled level against those wq files (what is scored = what would be packed)
#   G3   asym27's sibmom27 --level on a FRACTIONAL level (d1106/q-k1.5edy, K1.5, i8), layers 0-3, on the GPU: H = G1m's
#        bit for bit (clean inputs do not depend on the siblings), trN / trH above K3's in every group
# Out /data/yb/w37/gate (result.txt). Idempotent per step.
P=/workspace/rad-exl3-mix/proxy2b S=/workspace/rad-exl3-mix/v3/sib27 X=/workspace/rad-exl3-mix/v3/yaqa/asym27
D7=$P/exp/w07-denoise M=/data/models/qwen35-08b META=/data/challenge/calib-meta Y=/data/yb T=/data/yb/sib27test
H=/data/h08/calib G=/data/g08-search1 O=/data/yb/w37/gate REF=/data/yb/d1103/q-k3asym_r0 ONLY=/data/yb/w07-seqsib/only.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2} POOLS=/workspace/rad-exl3-mix/eval/pools/pools-v3.json
mkdir -p $O; cd /tmp; log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
die() { log "$*: stop"; exit 1; }
g2verdict() {   # CPU: the paired comparisons (pcmp.py) and the assembled level against the wq files that were scored
  python3 - $G $P/pcmp.py /data/yb/d1103/k3asym_r0.f32 $Y/k3base.f32 $O/g2.f32 <<'PY' | tee -a $O/result.txt
import json, math, subprocess, sys
g, pc, asym, base, ours = sys.argv[1:6]
def cmp(a, b): return json.loads(subprocess.run([sys.executable, pc, g, a, b], capture_output=True, text=True).stdout.strip().splitlines()[-1])
r1, r2, r3 = cmp(asym, ours), cmp(base, ours), cmp(base, asym)
ok = abs(math.log(r1["ratio"])) <= 0.025 and r2["ratio"] <= 0.85
print(f"G2 {'PASS' if ok else 'FAIL'}: ours (27B path: level-decoded siblings, rho 0) / d1103 k3asym_r0 {r1['ratio']:.4f} "
      f"(z {r1['z']:.1f}, |log| {abs(math.log(r1['ratio'])):.4f} <= 0.025); / k3base {r2['ratio']:.4f} (z {r2['z']:.1f}; "
      f"d1103's k3asym_r0 / k3base {r3['ratio']:.4f})")
PY
  mkdir -p $O/wqcmp; for f in $O/q-g2/wqall/*.safetensors; do ln -sfn $(readlink -f $f) $O/wqcmp/; done
  log "G2 assembled level vs the scored wq: $(DEV=cpu python3 $X/deq.py check $O/level-g2 $O/wqcmp | tail -n 1)"; }
[ -n "$G2_ONLY" ] && { g2verdict; exit 0; }
t0=$(date +%s); log "gate start (lock held)"
# ourq exactly as asym27.sh's quant(): the base's own args from its stamp, + the 4070's environment (as q2b.sh)
XA=$(python3 -I $S/levelargs.py $Y/q-k3base) || die "levelargs"; log "base args: $XA"
W=/data/p08; mkdir -p $W/challenge/hessians; ln -sfn $H $W/challenge/hessians/calib
python3 -c "import json; c=json.load(open('$M/config.json')); c['quantization_config']={'quant_method': 'exl3'}; json.dump(c, open('$W/template.json', 'w'))"
ourq() { ( mkdir -p /tmp/w37q && cd /tmp/w37q && PYTHONPATH=/data/src/exllamav3 RXM_WORK=$W RXM_CKPT=$M RXM_TEMPLATE=$W/template.json \
  TORCH_EXTENSIONS_DIR=/data/torch_ext python3 /workspace/rad-exl3-mix/quantize/ourq.py $XA --device 0 --layers 0-23 --save-wq "$@" ); }
mkd() { [ -f $2.done ] || { python3 $D7/mkdenoise.py $H $1 $2 --lam 1 --rho 0 --ckpt $M > $2.log 2>&1 && touch $2.done; } || die "mkdenoise $2"; }

# G1
mkd $T/ref2 $O/dn-ref2
[ $(ls $O/q-g1/wq/*.safetensors 2>/dev/null | wc -l) = 48 ] || ourq -o $O/q-g1 --only $ONLY --hin $O/dn-ref2/hin --wpre $O/dn-ref2/wpre > $O/q-g1.log 2>&1 \
  || die "quant g1 ($O/q-g1.log)"
r=$(python3 -I $S/assemble.py same $REF $O/q-g1); log "G1 $([ $? = 0 ] && echo PASS || echo FAIL): $r"
log "G1 stamps: d1103 $(python3 -c "import json; print(json.load(open('$REF/stamp.json'))[-1]['stamp'])"), now $(python3 -c "import json; print(json.load(open('$O/q-g1/stamp.json'))[-1]['stamp'])")"
log "G1 wq files: $(python3 - $REF/wq $O/q-g1/wq <<'PY'
import glob, os, sys, torch
from safetensors import safe_open
a, b = sys.argv[1:3]; n = same = 0
for f in sorted(glob.glob(f"{b}/*.safetensors")):
    n += 1
    with safe_open(f, "pt") as x, safe_open(f"{a}/{os.path.basename(f)}", "pt") as y: same += torch.equal(x.get_tensor("weight"), y.get_tensor("weight"))
print(f"{same} of {n} bit-identical to d1103's")
PY
)"

# G1m
if [ ! -f $O/g1m.done ]; then
  python3 $X/sibmom27.py $M $META $O/mom-g1m --wq $Y/q-k3base/wq --layers 0-3 --seed 0 --tokens 1000000 > $O/g1m.log 2>&1 || die "sibmom27 g1m"
  touch $O/g1m.done; fi
mkdir -p $O/ref2-03; for l in 0 1 2 3; do ln -sfn $T/ref2/model.language_model.layers.$l.*.input.safetensors $O/ref2-03/; done
log "G1m sibmom27 (low-RAM save) vs ref2, layers 0-3: $(python3 $S/cmpmom.py diff $O/mom-g1m $O/ref2-03 | tail -n 1); $(grep -o 'peak GPU [0-9.]* GiB' $O/g1m.log | sort -k3 -n | tail -1)"

# G2
mkd $T/lv $O/dn-lv
[ $(ls $O/q-g2/wq/*.safetensors 2>/dev/null | wc -l) = 48 ] || ourq -o $O/q-g2 --only $ONLY --hin $O/dn-lv/hin --wpre $O/dn-lv/wpre > $O/q-g2.log 2>&1 \
  || die "quant g2 ($O/q-g2.log)"
d=$O/q-g2/wqall; mkdir -p $d
for f in $Y/q-k3base/wq/*.safetensors; do ln -sfn $(readlink -f $f) $d/; done
for f in $O/q-g2/wq/*.safetensors; do ln -sfn $f $d/; done
[ $(ls $d/*.safetensors | wc -l) = 150 ] || die "wqall g2"
[ -f $O/g2.f32 ] || log "G2 search: $(python3 $P/evalq.py $M $G --wq $d --out $O/g2.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*')"
python3 $S/assemble.py build $Y/q-k3base $O/q-g2 $O/level-g2 $ONLY | tee -a $O/result.txt
log "gate GPU part done: $(( $(date +%s) - t0 )) s"

# G3 (GPU, short)
if [ ! -f $O/g3.done ]; then
  python3 $X/sibmom27.py $M $META $O/mom-g3 --level $Y/d1106/q-k1.5edy --codebook i8 --layers 0-3 --seed 0 --tokens 1000000 > $O/g3.log 2>&1 \
    || die "sibmom27 g3"
  touch $O/g3.done; fi
log "gate GPU part done: $(( $(date +%s) - t0 )) s (incl. G3)"
g2verdict
log "G3 $(python3 - $O/mom-g1m $O/mom-g3 <<'PY'
import glob, os, sys, torch
from safetensors import safe_open
a, b = sys.argv[1:3]; hs = up = n = 0
for f in sorted(glob.glob(f"{b}/*.input.safetensors")):
    n += 1
    with safe_open(f, "pt") as x, safe_open(f"{a}/{os.path.basename(f)}", "pt") as y:
        hs += torch.equal(x.get_tensor("H"), y.get_tensor("H"))
        r = lambda z: float(z.get_tensor("N").diagonal().sum() / z.get_tensor("H").diagonal().sum())
        up += r(x) > r(y)
print(f"{'PASS' if n and hs == n and up == n else 'FAIL'}: K1.5 level (fractional decode on the GPU), {n} groups: H bit-identical to G1m's in {hs}, trN/trH above K3's in {up}")
PY
)"
