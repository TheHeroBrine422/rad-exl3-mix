# env-w139.sh -- source on box2: rxq job environment for #177 (w139): box2's env-box2.sh with the patched tree
# /data/q27/w139/code (= stage/code-old-8f73ee4, the ladder's code stamp 43e08d796b6bcfbf, + rxq-hash.patch) and own builds.
Q=/data/q27
export ROCM_HOME=$Q/venv/lib/python3.14/site-packages/_rocm_sdk_devel ROCM_PATH=$Q/venv/lib/python3.14/site-packages/_rocm_sdk_devel
export PATH=$Q/venv/bin:$PATH PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=$Q/w139/build EXL3_QT_OPTIMIZED=1 OMP_NUM_THREADS=4 MAX_JOBS=8
export PYTHONPATH=$Q/w139/code/pipeline:$Q/stage/exl3:$Q/extbase RXQ_RXM=$Q/w139/code RXQ_EXL3_SRC=$Q/stage/exl3/exllamav3/exllamav3_ext RXQ_REFIT_CPU_FALLBACK=1
export EXL3HASH_A=0xd0474b EXL3HASH_B=0x2545F491
PY=$Q/venv/bin/python
