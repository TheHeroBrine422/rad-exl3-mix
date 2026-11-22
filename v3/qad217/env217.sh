# env217.sh (box2): source from /data/q27/w217 -- #217's paths (w214's env; the student pieces, data cuts and releases are w214's, read-only)
Q=/data/q27; W=$Q/w217; W14=$Q/w214; CODE=$W/code; PY=$Q/venv/bin/python
export PYTHONPATH=$W14/pylib-fla QAD_PIPELINE=$CODE/pipeline TRITON_DEFAULT_FP_FUSION=1 OMP_NUM_THREADS=6
MODEL=$Q/models/qwen38-27b-bf16; CALIB=$Q/challenge/golden/calib; POOLS=$W14/pools214.json
BUILD_LOCK=$W/build.lock
