#!/bin/bash
# audit.sh -- #175 step 0 (w137): rdna4-isa-suite run.sh (radisa collect + audit: every gfx1201 code object's descriptors,
# every byte decodes) over the final artifacts of the staged perf build (rad10-perf: install/bin, install/lib, home's
# kernels + architectures; ROCm 10.1) and of our plugins with the fix (b10v home: ROCm 10.1; b714v home: pip ROCm 7.14.1).
# run.sh does not walk symlinked dirs, so each set is copied (regular files, dereferenced) into audit/in-<name>/.
# Out /data/q27/w137/audit/<name>/ (+ .log, .exit). Niced; restartable.
Q=/data/q27; W=$Q/w137; S=$Q/rad10-perf/src/tools/rdna4-isa-suite; mkdir -p $W/audit
for spec in perf:10:$Q/rad10-perf/install/bin,$Q/rad10-perf/install/lib,$Q/rad10-perf/home b10v:10:$W/b10v/home/kernels,$W/b10v/home/architectures b714v:714:$W/b714v/home/kernels,$W/b714v/home/architectures; do
  IFS=: read n tc dirs <<< "$spec"
  [ -f $W/audit/$n.exit ] && continue
  case $tc in 10) L=$Q/rad10/rocm-repeatable/lib/llvm/bin ;; 714) L=$Q/venv/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin ;; esac
  I=$W/audit/in-$n; rm -rf $I $W/audit/$n; mkdir -p $I
  for d in ${dirs//,/ }; do
    find -L $d -type f \( -name '*.so*' -o -name '*.hsaco' -o -name '*.co' -o -perm -u+x \) 2>/dev/null | while read f; do
      file -L "$f" | grep -q ELF && cp -L "$f" $I/$(echo ${f#$Q/} | tr / _); done
  done
  ( cd /tmp && PATH=$L:$PATH RAD_ISA_OBJDUMP=$L/llvm-objdump nice -n 19 timeout 7200 bash $S/run.sh $I $Q/rad10-perf/src $W/audit/$n > $W/audit/$n.log 2>&1; echo $? > $W/audit/$n.exit )
  echo "$(date -u +%T) audit $n ($(ls $I | wc -l) files) exit $(cat $W/audit/$n.exit)"
done
