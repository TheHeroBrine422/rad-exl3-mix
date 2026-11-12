# w115 job env on box2 (= /data/q27/tools/env-box2.sh with the w115 staged tree a84ed23 + rxq-w115.patch, own build dir)
Q=/data/q27
export ROCM_HOME=$Q/venv/lib/python3.14/site-packages/_rocm_sdk_devel ROCM_PATH=$Q/venv/lib/python3.14/site-packages/_rocm_sdk_devel
export PATH=$Q/venv/bin:$PATH PYTORCH_ROCM_ARCH=gfx1201 TORCH_EXTENSIONS_DIR=/data/w115/build EXL3_QT_OPTIMIZED=1 OMP_NUM_THREADS=4 MAX_JOBS=8
export PYTHONPATH=/data/w115/code/pipeline:$Q/stage/exl3:$Q/extbase RXQ_RXM=/data/w115/code RXQ_EXL3_SRC=$Q/stage/exl3/exllamav3/exllamav3_ext RXQ_REFIT_CPU_FALLBACK=1
PY=$Q/venv/bin/python
