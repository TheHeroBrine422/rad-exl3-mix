#!/bin/bash
# test-qrun233.sh (VM or any box, no GPU) -- w233: qrun.sh with --pipe-procs launches the stage processes (qenv's `test` machine, a fake trainer that logs
# its rank / world / card / QAD_STAGES): A,B = 2 processes (cards A, B), A,B+C,D = 4 (rank 2 x group + stage), without the flag A,B+C,D = 2 processes
# and A,B = 1 (unchanged). Exit 0 = all pass.
set -u
HERE=$(dirname "$(readlink -f "$0")"); QQD=$HERE/../qadq
R=$(mktemp -d); export QROOT_OVERRIDE=$R QTEST_CARDS="0 1 2 3" QCARD_LOCKED=1
mkdir -p $R/in $R/runs $R/code/v3; ln -s $(readlink -f $QQD) $R/code/v3/qadq
cat > $R/fake.py <<'EOF'
import os, sys, json
out = sys.argv[1]
open(os.path.join(out, "ranks.txt"), "a").write(json.dumps({"rank": int(os.environ.get("RANK", 0)), "world": int(os.environ.get("WORLD_SIZE", 1)),
                                                            "cards": os.environ.get("X"), "stages": os.environ.get("QAD_STAGES")}) + "\n")
EOF
export CODE_PY=$R/fake.py
fails=0
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1: $(cat $R/runs/$3/ranks.txt 2>/dev/null | sort | tr '\n' ' ')"; fails=$((fails + 1)); fi; }
got() { sort $R/runs/$1/ranks.txt | tr -d ' ' | tr '\n' ' '; }
$QQD/qrun.sh test p2 0,1 2.50 5e5 --pipe-procs --split-layers 32 > /dev/null 2>&1
check "--pipe-procs on 0,1: two processes (rank 0 card 0, rank 1 card 1, world 2, QAD_STAGES=2)" "[ \"\$(got p2)\" = '{\"rank\":0,\"world\":2,\"cards\":\"0\",\"stages\":\"2\"} {\"rank\":1,\"world\":2,\"cards\":\"1\",\"stages\":\"2\"} ' ]" p2
$QQD/qrun.sh test p4 0,1+2,3 2.50 5e5 --split-layers 32 --pipe-procs > /dev/null 2>&1
check "--pipe-procs on 0,1+2,3: four processes, rank 2 x group + stage on cards 0..3" "[ \"\$(got p4)\" = '{\"rank\":0,\"world\":4,\"cards\":\"0\",\"stages\":\"2\"} {\"rank\":1,\"world\":4,\"cards\":\"1\",\"stages\":\"2\"} {\"rank\":2,\"world\":4,\"cards\":\"2\",\"stages\":\"2\"} {\"rank\":3,\"world\":4,\"cards\":\"3\",\"stages\":\"2\"} ' ]" p4
$QQD/qrun.sh test d2 0,1+2,3 2.50 5e5 --split-layers 32 --pipeline > /dev/null 2>&1
check "without --pipe-procs: 0,1+2,3 = two dp processes on card pairs (QAD_STAGES=1)" "[ \"\$(got d2)\" = '{\"rank\":0,\"world\":2,\"cards\":\"0,1\",\"stages\":\"1\"} {\"rank\":1,\"world\":2,\"cards\":\"2,3\",\"stages\":\"1\"} ' ]" d2
$QQD/qrun.sh test s1 0,1 2.50 5e5 --split-layers 32 --pipeline > /dev/null 2>&1
check "without --pipe-procs: 0,1 = one process on both cards (no RANK / QAD_STAGES set)" "[ \"\$(got s1)\" = '{\"rank\":0,\"world\":1,\"cards\":\"0,1\",\"stages\":null} ' ]" s1
$QQD/qrun.sh test bad 0,1,2 2.50 5e5 --pipe-procs > $R/bad.out 2>&1; rc=$?
check "--pipe-procs with a 3-card group: refused (rc 2)" "[ $rc = 2 ] && [ ! -e $R/runs/bad/ranks.txt ]" bad
[ $fails = 0 ] && { echo ALL PASS; rm -rf $R; exit 0; } || { echo "FAILED ($fails), dir $R"; exit 1; }
