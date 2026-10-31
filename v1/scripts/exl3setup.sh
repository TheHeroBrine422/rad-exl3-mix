#!/bin/bash
# exl3setup.sh -- PyTorch (ROCm) venv + exllamav3 built for gfx1201, for our own EXL3 quantization. Log: /root/exl3setup.log
set -e
cd /data; rm -rf exllamav3; tar xf exl3src.tar
python3 -m venv /data/venv-exl3
. /data/venv-exl3/bin/activate
pip install -q --upgrade pip
pip install -q "torch==2.15.0+rocm7.14" --index-url https://download.pytorch.org/whl/rocm7.14
python -c "import torch;print('torch',torch.__version__,torch.cuda.is_available(),torch.cuda.device_count())"
pip install -q ninja safetensors numpy
cd /data/exllamav3
ROCM_HOME=/opt/rocm PYTORCH_ROCM_ARCH=gfx1201 MAX_JOBS=12 pip install -v --no-build-isolation -e . > /root/exl3build.log 2>&1
python -c "import exllamav3, exllamav3_ext; print('exllamav3 ok')"
echo SETUPDONE
