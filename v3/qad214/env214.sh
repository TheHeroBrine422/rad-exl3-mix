# env214.sh (box2): source from /data/q27/w214 -- #214's paths (w200's env with the released-pick student pieces)
Q=/data/q27; W=$Q/w214; CODE=$W/code; PY=$Q/venv/bin/python
export PYTHONPATH=$W/pylib-fla QAD_PIPELINE=$CODE/pipeline TRITON_DEFAULT_FP_FUSION=1 OMP_NUM_THREADS=6
MODEL=$Q/models/qwen38-27b-bf16; CALIB=$Q/challenge/golden/calib; POOLS=$W/pools214.json
BUILD_LOCK=$W/build.lock
# lv() SIZE: the student pieces of a released pick (rad2level214.py's OUT)
lvdir() { echo $W/lv/$1; }
