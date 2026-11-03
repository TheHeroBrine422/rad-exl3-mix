#!/bin/bash
# run.sh -- w04-blockgain GPU arms on the 4070 (0.8B, ~35 min; README): B6 MLP neuron permutation (6 MLP-only quants +
# 6 evals + CPU proxy check), D6 slab split of GDN in_proj_qkv / attention q_proj (3 input-group quants at K2.5 + 9
# evals). Idempotent, resumable; controls are the existing bdiag arms in /data/yb. CPU analysis: run-cpu.sh.
P=/workspace/rad-exl3-mix/proxy2b F=$P/exp/w04-blockgain M=/data/models/qwen35-08b Y=/data/yb G=/data/g08-search1
O=/data/yb/w04-blockgain HIN=/data/h08/calib HD=/data/yb/dv8192/hout
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VRAM_CAP_GB=${VRAM_CAP_GB:-7.2}
mkdir -p $O; log() { echo "$(date -u +%T) $*" | tee -a $O/result.txt; }
ev() { tag=$1; [ -f $O/$tag.f32 ] && return
  log "$tag: $(python3 $P/evalq.py $M $G --wq $2 --out $O/$tag.f32 2>/dev/null | tail -1 | grep -o '"kld": [0-9.e-]*, "top1": [0-9.]*')"; }
f32() { [ -f $O/$1.f32 ] && echo $O/$1.f32 || echo $Y/$1.f32; }
pair() { log "pair $1 -> $2: $(python3 $P/pcmp.py $G $(f32 $1) $(f32 $2) | grep -o '"ratio": [0-9.]*, "diff": [0-9.e-]*, "se": [0-9.e-]*, "z": [0-9.e-]*')"; }
sfx() { [ $1 = 0 ] && echo "" || echo "_s$1"; }
cd /tmp

# ---- B6: MLP intermediate neurons permuted (balanced deal, data/perms08.npz), gate/up/down requantized, rest = bdiag donor
[ -f $O/perm/.done ] || { python3 $F/mkperm.py build $F/data/perms08.npz balanced $M $HIN $HD $O/perm && touch $O/perm/.done; } || exit 1
W=$O/perm/w
python3 -c "import json; c=json.load(open('$M/config.json')); c['quantization_config']={'quant_method': 'exl3'}; json.dump(c, open('$W/template.json', 'w'))"
qperm() { k=$1 s=$2; tag=k${k}bperm$(sfx $s)
  [ -f $O/q-$tag/wq/model.language_model.layers.23.mlp.down_proj.safetensors ] || (   # q2b.sh's settings, permuted model
    mkdir -p /tmp/q2b-run && cd /tmp/q2b-run && PYTHONPATH=/data/src/exllamav3 RXM_WORK=$W RXM_CKPT=$O/perm/model \
    RXM_TEMPLATE=$W/template.json TORCH_EXTENSIONS_DIR=/data/torch_ext \
    python3 /workspace/rad-exl3-mix/quantize/ourq.py -K $k -o $O/q-$tag --device 0 --layers 0-23 --codebook i8 --qtf2 \
      --frac-ext /data/src/exllamav3/exllamav3/exllamav3_ext --scaled-hessian --sigma-reg 0.025 --sigma-reg-in 0.005 --save-wq \
      --hout $O/perm/hout --sigma-reg-out 0.01 --seed-offset $s --only $O/perm/mlp.txt ) > $O/q-$tag.log 2>&1
  [ -f $O/q-$tag/wqn/.done ] || { python3 $F/mkperm.py unperm $F/data/perms08.npz balanced $O/q-$tag/wq $Y/q-k${k}bdiag$(sfx $s)/wq \
    $O/q-$tag/wqn && touch $O/q-$tag/wqn/.done; } || { log "UNPERM FAIL $tag"; exit 1; }
  ev $tag $O/q-$tag/wqn; pair k${k}bdiag$(sfx $s) $tag; }
for k in 3 2; do for s in 0 1 2; do qperm $k $s; done; done
for k in 3 2; do                                  # proxy check: measured h-weighted error perm / natural vs predicted
  [ -f $O/b6check-k$k.txt ] && continue
  W1=""; W2=""; for s in 0 1 2; do W1="$W1 --wq k${k}bperm$(sfx $s)=$O/q-k${k}bperm$(sfx $s)/wq"; W2="$W2 --wq k${k}bdiag$(sfx $s)=$Y/q-k${k}bdiag$(sfx $s)/wq"; done
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python3 $F/stats.py $O/perm/model $O/perm/hout $O/b6perm-k$k.npz --hin $W/challenge/hessians/calib --keys mlp. $W1 2>/dev/null
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python3 $F/stats.py $M $HD $O/b6nat-k$k.npz --hin $HIN --keys mlp. $W2 2>/dev/null
  python3 $F/b6check.py $O/b6nat-k$k.npz $O/b6perm-k$k.npz $F/data/perms08.npz $F/data/downperm08.json $k > $O/b6check-k$k.txt
  log "b6check K$k: $(tail -1 $O/b6check-k$k.txt)"
done

# ---- D6: half of the 128-row output blocks at K3 (highest predicted two-sided cost) and half at K2, vs uniform K2.5
python3 - <<PY
L = "model.language_model.layers"
with open("$O/d6keys.txt", "w") as k, open("$O/d6only.txt", "w") as o:
    for l in range(24):
        if l % 4 == 3: k.write(f"{L}.{l}.self_attn.q_proj\n"); o.write("".join(f"{L}.{l}.self_attn.{x}_proj\n" for x in "qkv"))
        else: k.write(f"{L}.{l}.linear_attn.in_proj_qkv\n"); o.write(f"{L}.{l}.linear_attn.in_proj_qkv\n{L}.{l}.linear_attn.in_proj_z\n")
PY
for s in 0 1 2; do x=$(sfx $s)
  [ -f $O/q-k25in$x/wq/model.language_model.layers.23.linear_attn.in_proj_qkv.safetensors ] || \
    $P/q2b.sh 08 2.5 i8 $O/q-k25in$x --hout $HD --sigma-reg-out 0.01 --seed-offset $s --only $O/d6only.txt > $O/q-k25in$x.log 2>&1
  B=$Y/q-k3bdiag$x/wq
  [ -f $O/d6u$x/.done ] || python3 $F/splice.py uniform $M $HD $O/d6keys.txt $B $O/d6u$x $O/q-k25in$x/wq && touch $O/d6u$x/.done
  [ -f $O/d6s$x/.done ] || python3 $F/splice.py sorted $M $HD $O/d6keys.txt $B $O/d6s$x $B $Y/q-k2bdiag$x/wq && touch $O/d6s$x/.done
  [ -f $O/d6r$x/.done ] || python3 $F/splice.py random $M $HD $O/d6keys.txt $B $O/d6r$x $B $Y/q-k2bdiag$x/wq && touch $O/d6r$x/.done
  ev d6u$x $O/d6u$x; ev d6s$x $O/d6s$x; ev d6r$x $O/d6r$x
  pair d6u$x d6s$x; pair d6r$x d6s$x; pair k3bdiag$x d6u$x
done
log W04 DONE
