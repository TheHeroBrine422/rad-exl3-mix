#!/bin/bash
# lint.sh TC PB OUT [UNIT_FILTER] -- radisa check (Code000's rdna4-isa-suite, rad10-perf/src/tools/rdna4-isa-suite) over our
# plugins' HIP units as built in PB/<plugin> (CMake build dirs) with toolchain TC:
#   10  : ROCm 10.1 shadow clang (rad10/rocm-repeatable, AMD clang 24)     714 : pip ROCm SDK 7.14.1 (AMD clang 23)
# Source mode (each unit compiled with its build's flags, radisa's per-kernel cache) and object mode (the built .hip.o:
# the bytes that run). Flags: PB/<p>/compile_commands.json, else `ninja -t compdb`. Out OUT/{src,obj}/<unit>.{json,txt}.
# Restartable: a unit with a .txt is skipped. (w137, #175)
set -u
TC=$1; PB=$2; O=$3; FILT=${4:-}
Q=/data/q27; W=$Q/w137; S=$Q/rad10-perf/src/tools/rdna4-isa-suite
case $TC in
  10)  L=$Q/rad10/rocm-repeatable/lib/llvm/bin; export ROCM_PATH=/opt/rocm-10.1 ;;
  714) export ROCM_PATH=$Q/venv/lib/python3.14/site-packages/_rocm_sdk_devel; L=$ROCM_PATH/lib/llvm/bin ;;
  *) echo "TC 10|714"; exit 2 ;;
esac
export PATH=$L:$PATH RADISA_CACHE=$O/cache PYTHONPATH=$S
mkdir -p $O/src $O/obj $O/cc
for p in exl3i exl3dense exl3r4d; do
  cc=$PB/$p/compile_commands.json
  [ -f $cc ] || { cc=$O/cc/$p.json; (cd $PB/$p && $Q/venv/bin/ninja -t compdb) > $cc; }
  for f in $(python3 -I -c "import json,sys; print('\n'.join(sorted({e['file'] for e in json.load(open(sys.argv[1])) if e['file'].endswith('.hip')})))" $cc); do
    u=${p}__$(basename $f .hip); [ -n "$FILT" ] && [[ $u != *$FILT* ]] && continue
    if [ ! -f $O/src/$u.txt ]; then
      ( cd /tmp && timeout 3600 nice -n 19 python3 -m radisa check $f --compile-commands $cc --cxx $L/clang++ --llvm-mc $L/llvm-mc \
          --jobs 8 --json $O/src/$u.json > $O/src/$u.part 2>&1; echo "exit $?" >> $O/src/$u.part ) ; mv $O/src/$u.part $O/src/$u.txt
      echo "$(date -u +%T) src $u: $(tail -2 $O/src/$u.txt | head -1)"
    fi
  done
  for ob in $(find $PB/$p/CMakeFiles -name '*.hip.o' | sort); do
    u=${p}__$(basename $(dirname $ob) .dir)__$(basename $ob .hip.o); [ -n "$FILT" ] && [[ $u != *$FILT* ]] && continue
    [ -f $O/obj/$u.txt ] && continue
    ( cd /tmp && timeout 1800 nice -n 19 python3 -m radisa check $ob --llvm-mc $L/llvm-mc --objdump $L/llvm-objdump \
        --json $O/obj/$u.json > $O/obj/$u.part 2>&1; echo "exit $?" >> $O/obj/$u.part ); mv $O/obj/$u.part $O/obj/$u.txt
    echo "$(date -u +%T) obj $u: $(tail -2 $O/obj/$u.txt | head -1)"
  done
done
echo "$(date -u +%T) LINT DONE $TC $PB"
