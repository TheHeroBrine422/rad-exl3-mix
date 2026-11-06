#!/bin/bash
# gate-prefix.sh -- realq.py --prefix gate (w30), one GPU unit (run under /data/yb/gpu.lock, < 15 min). REAL-Q's GD is
# not bit-reproducible run to run (CUDA kernels), and the trellis turns any difference into ~20% different codes, so
# the byte gate runs every quant in deterministic mode (DET: torch.use_deterministic_algorithms via a sitecustomize in
# a gate-only PYTHONPATH, CUBLAS_WORKSPACE_CONFIG) and compares the pre-edit realq.py (git HEAD, run from a temporary
# sibling dir so ourq's path and stamp are the same) with the edited one:
#   det    new code, --prefix bf16, layer 2 alone, twice: identical? (is DET deterministic at all)
#   quant  layer 1 with k3rq's own layer-0 shard and wq as the quantized prefix, old code vs new code (default --prefix
#          quant): q-L01, layer-1 wq, realq.jsonl losses and realq-stamp.json identical; the stamp also == k3rq's
#   bf16   --prefix bf16 on layers 1-2 in one run vs layer 2 alone: layer 2 identical (layers independent); layer 1
#          differs from the quant run's (the option changes the inputs)
# GATE=bf16: only det-a and bf16-12 (the independence check) with DET_STRICT=1: warn_only=False, which also makes
# flash-attention's backward deterministic (layer l+1's loss term runs through layer 3's full attention)
X=/workspace/rad-exl3-mix/proxy2b/exp/d1109 R=/data/yb/d1109/q-k3rq G=${1:-/data/yb/d1109/gate-prefix}
XO=/workspace/rad-exl3-mix/proxy2b/exp/.d1109-w30old M=/data/models/qwen35-08b W=/data/p08
HO="--hout /data/yb/c18/hdiag-snap1024-dy --sigma-reg-out 0.01"
rm -rf $G $XO; mkdir -p $G/q-quant-new/wq $G/det $XO; cd /tmp
git -C /workspace/rad-exl3-mix show HEAD:proxy2b/exp/d1109/realq.py > $XO/realq.py; cp $X/rqdata.py $XO/
cat > $G/det/sitecustomize.py <<EOF
import torch
WARN_ONLY = $([ "$DET_STRICT" = 1 ] && echo False || echo True)
torch.use_deterministic_algorithms(True, warn_only=WARN_ONLY)
exec(open("/usr/lib/python3.12/sitecustomize.py").read())
EOF
cp $R/q-L00.safetensors $G/q-quant-new/; cp $R/wq/model.language_model.layers.0.*.safetensors $G/q-quant-new/wq/
cp -r $G/q-quant-new $G/q-quant-old
# rqd REALQ_DIR OUT [realq opts] -- [ourq args]: rq.sh's command for K3 i8, in DET mode
rqd() { local x=$1 o=$2; shift 2; local rq=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do rq+=("$1"); shift; done; shift
  PYTHONPATH=$G/det:/data/src/exllamav3 CUBLAS_WORKSPACE_CONFIG=:4096:8 RXM_WORK=$W RXM_CKPT=$M RXM_TEMPLATE=$W/template.json \
  TORCH_EXTENSIONS_DIR=/data/torch_ext PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=7.2 \
  python3 $x/realq.py --fisher /data/yb/d1109/fisher/fisher.pt --max-sec 780 "${rq[@]}" -- -K 3 -o $o --device 0 \
    --layers 0-23 --codebook i8 --qtf2 --frac-ext /data/src/exllamav3/exllamav3/exllamav3_ext --scaled-hessian \
    --sigma-reg 0.025 --sigma-reg-in 0.005 --save-wq $HO --seed-offset 0 "$@" > $o.log 2>&1; echo "$(basename $o) rc $? $(($(date +%s) - t0))s"; }
t0=$(date +%s)
rqd $X $G/q-det-a --prefix bf16 -- --layers 2-2
if [ "$GATE" != bf16 ]; then rqd $X $G/q-det-b --prefix bf16 -- --layers 2-2
  rqd $X $G/q-quant-new -- --layers 1-1
  rqd $XO $G/q-quant-old -- --layers 1-1; fi
rqd $X $G/q-bf16-12 --prefix bf16 -- --layers 1-2
grep -h "streams at layer\|Warn.*determin" $G/*.log | sort | uniq -c | cut -c1-200
rm -rf $XO
python3 - $R $G $GATE <<'EOF'
import sys, glob, os, json, filecmp
R, G = sys.argv[1:3]; ONLY_BF16 = sys.argv[3:] == ['bf16']
def same(a, b): return os.path.exists(a) and os.path.exists(b) and filecmp.cmp(a, b, shallow=False)
def files(d, l): return [f"{d}/q-L{l:02d}.safetensors"] + sorted(glob.glob(f"{d}/wq/model.language_model.layers.{l}.*.safetensors"))
def losses(d, l): return [(r["key"], r["losses"]) for r in map(json.loads, open(f"{d}/realq.jsonl")) if r["layer"] == l]
def check(name, pairs, want):
    res = [same(a, b) for a, b in pairs]; ok = len(pairs) > 0 and all(r == want for r in res)
    print(f"{'PASS' if ok else 'FAIL'} {name}: {len(pairs)} files, {sum(res)} byte-identical (want {'all' if want else 'none'})"); return ok
D = lambda t: f"{G}/q-{t}"
if ONLY_BF16:
    ok = check("bf16 (strict DET): layer 2 of run 1-2 == run 2 alone", [(f, f.replace(D("bf16-12"), D("det-a"))) for f in files(D("bf16-12"), 2)], True)
    print("GATE-PREFIX-BF16", "PASS" if ok else "FAIL"); sys.exit()
det = check("DET control: bf16 layer 2, run a == run b", [(f, f.replace(D("det-a"), D("det-b"))) for f in files(D("det-a"), 2)], True)
ok = check("quant: old code == new code (q-L01, layer-1 wq)", [(f, f.replace(D("quant-new"), D("quant-old"))) for f in files(D("quant-new"), 1)], True)
lq = losses(D("quant-new"), 1) == losses(D("quant-old"), 1); ok &= lq
print(f"{'PASS' if lq else 'FAIL'} quant: old == new realq.jsonl losses ({sum(len(x) for _, x in losses(D('quant-new'), 1))} GD losses)")
ok &= check("quant: realq-stamp.json old == new == k3rq's", [(f"{D('quant-new')}/realq-stamp.json", f"{D('quant-old')}/realq-stamp.json"),
            (f"{D('quant-new')}/realq-stamp.json", f"{R}/realq-stamp.json")], True)
ok &= check("bf16: layer 2 of run 1-2 == run 2 alone", [(f, f.replace(D("bf16-12"), D("det-a"))) for f in files(D("bf16-12"), 2)], True)
ok &= check("bf16: layer-1 wq != quant's", [(f, f.replace(D("bf16-12"), D("quant-new"))) for f in files(D("bf16-12"), 1)[1:]], False)
print("GATE-PREFIX", "PASS" if ok else "FAIL", "" if det else "(DET mode is not deterministic: byte checks cannot pass)")
EOF
