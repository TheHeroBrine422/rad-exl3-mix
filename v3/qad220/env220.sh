# env220.sh (box2): source from /data/q27/w220 -- w220's paths (w217's env with W = w220; student pieces, data cuts, releases = w214's, the judge slice j217 = w217's, all read-only)
Q=/data/q27; W=$Q/w220; W14=$Q/w214; W17=$Q/w217; CODE=$W/code; PY=$Q/venv/bin/python
export PYTHONPATH=$W14/pylib-fla QAD_PIPELINE=$CODE/pipeline TRITON_DEFAULT_FP_FUSION=1 OMP_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1
MODEL=$Q/models/qwen38-27b-bf16; CALIB=$Q/challenge/golden/calib; POOLS=$W14/pools214.json
BUILD_LOCK=$W/build220.lock
J217=$W17/slices/golden-j217
