cd /data/asym27/dbg; export ROCR_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 DEV=cuda
PY=/data/venv-exl3/bin/python; H=/root/challenge/hessians/calib; CK=/data/workspace/models/qwen38-27b-bf16
( exec 5>/root/card0.lock; flock 5; exec 7>/root/ram.lock; flock -s 7; cd /root
  rm -rf /data/asym27/dbg/mk-*
  for v in old new; do f=/data/asym27/code/mkdenoise.py; [ $v = new ] && f=/data/asym27/dbg/mkdenoise-new.py
    s=$(date +%s); MKD_LEAN=1 /data/swap27/tools/memguard.sh 3000 nice -n 5 $PY -I $f $H /data/asym27/dbg/mt/g6 /data/asym27/dbg/mk-$v-g6 --lam 1 --rho 0 --ckpt $CK 2>&1 | grep -v Warn | tail -2; echo "g6 $v $(( $(date +%s) - s )) s"; done
  s=$(date +%s); /data/swap27/tools/memguard.sh 3000 nice -n 5 $PY -I /data/asym27/dbg/mkdenoise-new.py $H /data/asym27/dbg/mt/g17 /data/asym27/dbg/mk-new-g17 --lam 1 --rho 0 --ckpt $CK 2>&1 | grep -v Warn | tail -2; echo "g17 new $(( $(date +%s) - s )) s"
  $PY -I - <<'PY'
import torch, json
from safetensors import safe_open
g = "model.language_model.layers.0.linear_attn.out_proj.input.safetensors"
d = "/data/asym27/dbg"
for sub in ("hin", "wpre"):
    key = "hessian_upper" if sub == "hin" else "Ft"
    with safe_open(f"{d}/mk-old-g6/{sub}/{g}", "pt") as a, safe_open(f"{d}/mk-new-g6/{sub}/{g}", "pt") as b:
        x, y = a.get_tensor(key), b.get_tensor(key)
        print(sub, "bit-identical" if torch.equal(x, y) else f"differ: {int((x != y).sum())} of {x.numel()} f32 values, max rel {float((x - y).abs().max() / x.abs().max()):.1e}")
for v in ("old", "new"): print(v, open(f"{d}/mk-{v}-g6/stats.jsonl").read().strip()[:400])
print("g17", open(f"{d}/mk-new-g17/stats.jsonl").read().strip()[:600])
PY
) 2>&1 | tee /data/asym27/dbg/mkcheck.log
