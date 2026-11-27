# env.sh -- box2 environment of the w235 jobs (source it)
export W211_E3DIR=/data/q27/w235/code/e3 W211_CKPT=/data/ds41ck-full W211_CALIB=/data/q27/w235/calib W211_HFSCRIPTS=/data/q27/w211/hf/scripts
export W235_W211DIR=/data/q27/w235/code/w211 W235_ROWS_GU=288 W235_ROWS_DN=320 W211_SLICE=0
export E3_V2=3 E3_TB=1 OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/data/q27/venv/bin/python
