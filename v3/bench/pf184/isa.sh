#!/bin/bash
# isa.sh BUILD UNIT OUT.s -- device asm of exl3i's UNIT (e.g. exl3i_a8_prep) of box2 build /data/q27/w160/b/BUILD with that
# build's own flags (compile_commands.json), --offload-device-only -S; env XF: extra flags. (w137's isa.sh, #184 w160)
B=/data/q27/w160/b/$1/pb/exl3i; U=$2; O=$3
cmd=$(python3 -I -c "
import json,sys
for e in json.load(open(sys.argv[1])):
    if e['file'].endswith('/'+sys.argv[2]+'.hip'): print(e['command'])
" $B/compile_commands.json $U)
cmd=$(echo "$cmd $XF" | sed -E 's/ -o [^ ]+\.hip\.o / -o '"${O//\//\\/}"' /; s/ -c / -S --offload-device-only /; s/-cuid=[^ ]+//')
cd $B && nice -n 10 bash -c "$cmd"
