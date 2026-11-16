# env.sh (w187, #200, box2): copied to /data/q27/w187/env.sh. w182's #196 env (w170's 2.5 incumbent pieces, the judge
# sets, the search's served host) with W = /data/q27/w187. The student-build lock is w182's (host-wide, one build at a time).
Q=/data/q27; W=$Q/w187; CODE=$W/code; W170=$Q/w170; PY=$Q/venv/bin/python
export PYTHONPATH=$Q/w125/pylib-fla QAD_PIPELINE=$CODE/pipeline TRITON_DEFAULT_FP_FUSION=1 OMP_NUM_THREADS=6
MODEL=$Q/models/qwen38-27b-bf16; CALIB=$Q/challenge/golden/calib; POOLS=$CODE/eval/pools/pools-v3.json
PERM=$Q/w119/perms/R16.npy; G154=$Q/w154
VL=$W170/vl/inc2.5m; HD=$W170/heads/K2.875; EMB=$W170/emb/t1.75.bf16; AB=$W170/ab/ab-T4.safetensors
TPL=$Q/srch183/mixes/v3s183-2.5-inc-01334.mix; HOSTC=$Q/srch183/host.json
G1=$G154/slices/golden-x181g; G2=$W170/slices/sub-fe38
BUILD_LOCK=$Q/w182/build.lock
START=$Q/w182/runs/ONE/ckpt-1e+06
