#!/bin/bash
# vgpr.sh -- registers / spills of the gemm_nt kernels in the dev build's exl3r4d.hip object
ssh-gpu.sh 'mkdir -p /tmp/co && cd /tmp/co && objcopy -O binary --only-section=.hip_fatbin /root/radiance-hero/plugin/b-card31-exl3r4d/CMakeFiles/exl3r4d.dir/exl3r4d.hip.o fb.bin && B=/opt/rocm/core-10.1/lib/llvm/bin/clang-offload-bundler; [ -x $B ] || B=$(ls /opt/rocm*/lib/llvm/bin/clang-offload-bundler /opt/rocm*/llvm/bin/clang-offload-bundler 2>/dev/null | head -1); R=$(dirname $B)/llvm-readelf; t=$($B --list --type=o --input=fb.bin | grep gfx1201 | head -1); $B --unbundle --type=o --targets=$t --input=fb.bin --output=k.co && $R --notes k.co > notes.txt && python3 - <<"PY"
import re
t=open("/tmp/co/notes.txt").read()
for blk in re.split(r"\n  - \.", t):
    m=re.search(r"\.name:\s+(\S+)", blk)
    if not m or "nt_kernel" not in m.group(1): continue
    if not re.search(r"ILi4ELi2ELi[0-9]+ELb1E", m.group(1)): continue
    g=lambda k:(re.search(r"\."+k+r":\s+(\S+)",blk) or [None,"?"])[1]
    print(m.group(1)[:58],"vgpr",g("vgpr_count"),"spill",g("vgpr_spill_count"),"priv",g("private_segment_fixed_size"),"lds",g("group_segment_fixed_size"))
PY'
