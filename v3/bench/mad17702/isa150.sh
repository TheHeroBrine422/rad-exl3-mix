#!/bin/bash
# isa150.sh BUILD OUT.s [UNIT] -- device asm of exl3i's UNIT (default exl3i_nt_cb3 = mad + its GEMV) as BUILD compiled it
# (BUILD/pb/exl3i/compile_commands.json), --offload-device-only -S; env XF: extra flags. (w150; w137's isa.sh)
B=/data/q27/w150/$1/pb/exl3i; O=$2; U=${3:-exl3i_nt_cb3}
cmd=$(python3 -I -c "
import json,sys
for e in json.load(open(sys.argv[1])):
    if e['file'].endswith('/'+sys.argv[2]+'.hip'): print(e['command'])
" $B/compile_commands.json $U)
cmd=$(echo "$cmd $XF" | sed -E 's/ -o [^ ]+\.hip\.o / -o '"${O//\//\\/}"' /; s/ -c / -S --offload-device-only /; s/-cuid=[^ ]+//')
export PATH=/opt/rocm-10.1/bin:$PATH
cd $B && nice -n 10 bash -c "$cmd"
