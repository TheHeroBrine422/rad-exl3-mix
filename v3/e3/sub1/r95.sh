#!/bin/bash
# r95.sh [PAR] [DOWNPAR] -- #95 (README "#95"): IQR-V4 ideas on the CPU proxy, trellis3.py, codebook cubic0.02_i8 (i8c).
#   95-01 act order: the 10 tensors of #90-02 at K2, 2.5, 3, 4, 5, E3_ORDER natural vs hdiag (paired rows / seed).
#   95-02 bf16 (row, 128) scales: attn_output L3/19/43/63, ssm_out L2/10/30/50/62 (+ ffn_down L10/30, reported only)
#         at K4, 4.25, 5, 5.25, 6, 6.25 (the quarter steps price the 0.125 extra bits), natural order, E3_RS128=1.
#   Natural-order runs always carry E3_RS128=1 (the refit is a cheap post-fit: free extra lines).
# Idempotent: one job per (tensor, order, K group) still missing from r95.jsonl (a K is done once its last line is
# there: rs128j for natural, ldlq for hdiag); a job already running (per-job flock) is skipped. At most PAR (16)
# jobs of ours at once, DOWNPAR (6) of them ffn_down (~13 GB each). Longest first.
cd /data/work/e3t; export E3_FACTORS=${E3_FACTORS:-0.85,1.0,1.15,1.3,1.5} E3_CKPT=/data/models/qwen38-27b-bf16 E3_HESS=/data/challenge/hessians/calib OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CD=0
O=/data/work/e3t/r95.jsonl; PAR=${1:-16}; DPAR=${2:-6}
mkdir -p logs r95lock
mapfile -t J < <(python3 - "$O" <<'EOF'
import json, os, sys
T1 = "10:ssm_qkv 10:ssm_out 10:ffn_gate_up 10:ffn_down 30:ffn_down 50:ssm_qkv 50:ffn_gate_up 19:attn_qg 19:attn_output 43:attn_v".split()
T2 = "3:attn_output 19:attn_output 43:attn_output 63:attn_output 2:ssm_out 10:ssm_out 30:ssm_out 50:ssm_out 62:ssm_out 10:ffn_down 30:ffn_down".split()
need = {}   # (layer, w, order) -> set of K
for t in T1:
    for o in ("natural", "hdiag"): need.setdefault((*t.split(":"), o), set()).update([2, 2.5, 3, 4, 5])
for t in T2: need.setdefault((*t.split(":"), "natural"), set()).update([4, 4.25, 5, 5.25, 6, 6.25])
done = set()
if os.path.exists(sys.argv[1]):
    for l in open(sys.argv[1]):
        try: r = json.loads(l)
        except ValueError: continue
        done.add((str(r["layer"]), r["w"], r.get("order"), r["method"], float(r["bits"])))
last = {"natural": "rs128j_cubic0.02_i8", "hdiag": "ldlq_cubic0.02_i8"}
jobs = []
for (l, w, o), ks in need.items():
    miss = sorted(k for k in ks if (l, w, o, last[o], float(k)) not in done)
    n = 1 if w == "ffn_down" else 3                     # ffn_down ~1,000 s a K, the rest ~250 s
    for i in range(0, len(miss), n):
        g = miss[i:i + n]
        jobs.append((len(g) * (4 if w == "ffn_down" else 1), w == "ffn_down", l, w, o, ",".join(f"{k:g}" for k in g)))
for c, d, l, w, o, ks in sorted(jobs, key=lambda j: -j[0]):
    print(int(d), l, w, o, ks)
EOF
)
echo "r95: ${#J[@]} jobs to run"
own() { pgrep -fc "^python3 -W ignore trellis3.py $1.* $O\$"; }
start() {   # start "d l w o ks"
  read -r d l w o ks <<< "$1"; rs=0; [ $o = natural ] && rs=1
  nohup flock -n r95lock/$l-$w-$o-$ks.lock nice -n 10 env E3_ORDER=$o E3_RS128=$rs \
    python3 -W ignore trellis3.py $l $w cubic0.02_i8 $ks 128 $O > logs/r95-$l-$w-$o-$ks.log 2>&1 < /dev/null &
}
D=(); N=(); for j in "${J[@]}"; do [ ${j%% *} = 1 ] && D+=("$j") || N+=("$j"); done
while [ ${#D[@]} -gt 0 ] || [ ${#N[@]} -gt 0 ]; do
  if [ $(own) -ge $PAR ]; then sleep 30
  elif [ ${#D[@]} -gt 0 ] && [ $(own '[0-9]+ ffn_down') -lt $DPAR ]; then start "${D[0]}"; D=("${D[@]:1}"); sleep 60   # memory shows
  elif [ ${#N[@]} -gt 0 ]; then start "${N[0]}"; N=("${N[@]:1}"); sleep 3
  else sleep 30; fi
done
wait
echo R95 DONE
