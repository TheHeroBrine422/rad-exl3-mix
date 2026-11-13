#!/bin/bash
# llvmcheck10.sh [W] -- radiance's docker/llvm-check.sh on the host build W/build, with this host's paths: ROCm's clang
# (/opt/rocm-10.1) vs the shadow tree's (W/rocm-repeatable). Pairing off: both must emit the same device code (the
# rebuilt libLLVM is ROCm's apart from the patch); pairing on: two more compiles must equal the build's object. (w130)
W=${1:-/data/q27/rad10}
sed -e "s|^STOCK=.*|STOCK=/opt/rocm-10.1/lib/llvm/bin/clang++|" -e "s|^IMAGE=.*|IMAGE=$W/rocm-repeatable/lib/llvm/bin/clang++|" \
    -e "s|^OBJCOPY=.*|OBJCOPY=/opt/rocm-10.1/lib/llvm/bin/llvm-objcopy|" $W/src/docker/llvm-check.sh > $W/scripts/llvm-check-host.sh
PATH=/data/q27/venv/bin:$PATH bash $W/scripts/llvm-check-host.sh $W/build
