#!/bin/bash
# w213: link the host SIMT-emulator tests by hand with libstdc++ / libm (box2's amdclang-24 C++ link lacks them) and run them
B=/data/q27/w213/src/build; L=/data/q27/w213/logs; D=$B/tests/r4d_emu
OBJS=$(ls $D/*.o | grep -v '/test_')
mkdir -p $B/tests/emu-w213
for t in ds41mr ds41mx ds41walk mhc moefuse dsa pick norm f32 attn; do
  /opt/rocm/llvm/bin/amdclang-24 $D/test_$t.o $OBJS -ldl -lstdc++ -lm -o $B/tests/emu-w213/r4d_emu_$t > $L/t4-link-$t.log 2>&1 || { echo "$t LINK_FAIL"; continue; }
  timeout 1200 $B/tests/emu-w213/r4d_emu_$t > $L/t4-run-$t.log 2>&1; echo "$t RC=$? $(tail -n 1 $L/t4-run-$t.log | cut -c1-150)"
done
echo T4_DONE
