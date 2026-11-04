#!/bin/bash
# patchtest.sh -- w11: no-op / behaviour test of d1103-heal-norms.patch, on copies in a scratch dir (no GPU, no repo edit).
# 1. applies cleanly to the repo's d1103 run.sh / run2.sh; both parse (bash -n)
# 2. ev without a 5th argument runs evalq.py on $M exactly as before (every non-heal call site unchanged)
# 3. the heal call scores with DIR/model when fold.py wrote one (norms trained), else with $M
# 4. run2.sh's helper import (sed '/^P=/,/^step() /p' run.sh) picks up hm()
set -u; R=/workspace/rad-exl3-mix; T=$(mktemp -d); trap 'rm -rf $T' EXIT; ok=1
mkdir -p $T/repo/proxy2b/exp/d1103; cp $R/proxy2b/exp/d1103/run{,2}.sh $T/repo/proxy2b/exp/d1103/
( cd $T/repo && git apply $R/proxy2b/exp/w11-review/d1103-heal-norms.patch ) || { echo "FAIL apply"; exit 1; }
for f in run.sh run2.sh; do bash -n $T/repo/proxy2b/exp/d1103/$f || { echo "FAIL parse $f"; ok=0; }; done
# stub environment: python3 records its arguments; helpers sourced as run2.sh does
mkdir -p $T/bin $T/O/h-a/model $T/O/h-b; touch $T/O/h-a/model/.done
printf '#!/bin/bash\necho "$*" >> %s/calls\necho "{\\"kld\\": 0.1, \\"top1\\": 0.9}"\n' $T > $T/bin/python3; chmod +x $T/bin/python3
src=$(sed -n '/^P=/,/^step() /p' $T/repo/proxy2b/exp/d1103/run.sh)
grep -q '^hm()' <<< "$src" || { echo "FAIL hm() not in run2's helper import"; ok=0; }
PATH=$T/bin:$PATH bash -c "$src
O=$T/O; M=/MODEL; S=/GOLD; P=/P; mkdir -p \$O
ev plain /wq                           # old call form
ev fit /wq /FIT .fit                    # old call form with golden + suffix
d=$T/O/h-a; ev a_heal \$d/wq \$S '' \$(hm \$d)    # healed with norms
d=$T/O/h-b; ev b_heal \$d/wq \$S '' \$(hm \$d)    # healed, no norm file
" > /dev/null 2>&1
exp="/P/evalq.py /MODEL /GOLD --wq /wq --out $T/O/plain.f32
/P/evalq.py /MODEL /FIT --wq /wq --out $T/O/fit.fit.f32
/P/evalq.py $T/O/h-a/model /GOLD --wq $T/O/h-a/wq --out $T/O/a_heal.f32
/P/evalq.py /MODEL /GOLD --wq $T/O/h-b/wq --out $T/O/b_heal.f32"
if [ "$(cat $T/calls 2>/dev/null)" = "$exp" ]; then echo "calls as expected"; else echo "FAIL calls:"; cat $T/calls; ok=0; fi
[ $ok = 1 ] && echo "PATCHTEST PASS" || { echo "PATCHTEST FAIL"; exit 1; }
