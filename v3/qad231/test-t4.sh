#!/bin/bash
# test-t4.sh (VM or any box, no GPU) -- w231: the queue's T4 kind end to end with qenv.sh's `test` machine (4 fake cards, a fake trainer that logs its
# rank / cards). Checks: (1) a T4 line is claimed by card 0, holds cards 1-3 while training (their dispatchers take nothing), runs two ranks with cards
# 0,1 and 2,3, and is done; (2) when rank 1 fails, rank 0 (which would wait for minutes) is stopped and the job is retried from its state, then fails;
# (3) T2 still runs one process on two cards. Exit 0 = all pass.
set -u
HERE=$(dirname "$(readlink -f "$0")"); QQD=$HERE/../qadq
R=$(mktemp -d); export QROOT_OVERRIDE=$R QTEST_CARDS="0 1 2 3"
mkdir -p $R/in $R/runs $R/code/v3; ln -s $(readlink -f $QQD) $R/code/v3/qadq
cat > $R/fake.py <<'EOF'
import os, sys, time, json
out = sys.argv[1]; a = sys.argv[2:]; bud = a[a.index("--budgets") + 1].split(",")[-1]
r = int(os.environ.get("RANK", 0)); w = int(os.environ.get("WORLD_SIZE", 1))
open(os.path.join(out, f"ranks.txt"), "a").write(json.dumps({"rank": r, "world": w, "cards": os.environ.get("X"), "port": os.environ.get("MASTER_PORT")}) + "\n")
if os.environ.get("FAIL_RANK") == str(r):
    open(os.path.join(out, "state.pt"), "w").write("x"); time.sleep(1); sys.exit(3)
if os.environ.get("FAIL_RANK") is not None: time.sleep(600); sys.exit(0)        # would block in the all-reduce
time.sleep(2)
if r == 0:
    ck = os.path.join(out, "ckpt-%.0e" % float(bud)); os.makedirs(ck, exist_ok=True); open(ck + "/.done", "w").close()
EOF
export CODE_PY=$R/fake.py
fails=0
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi; }
start() { for c in 0 1 2 3; do setsid nohup $QQD/qcard.sh test $c < /dev/null > $R/qcard-$c.out 2>&1 & echo $! > $R/pid$c; done; }
stop() { touch $R/stop; for c in 0 1 2 3; do p=$(cat $R/pid$c); for i in $(seq 120); do kill -0 $p 2>/dev/null || break; sleep 1; done; done; rm -f $R/stop; }
waitfor() { for i in $(seq $2); do eval "$1" && return 0; sleep 1; done; return 1; }

printf 'T4 dp1 2.50 5e5 - --lr 1e-4 --snap-at none @cards=0 @partner=1,2,3\nT2 sp1 2.50 5e5 - --lr 1e-4 --snap-at none @cards=1 @partner=3 @needs=dp1/ckpt-5e+05/.done\n' > $R/jobs.txt
start
check "T4 done" "waitfor '[ -e $R/done/T4-dp1 ]' 150"
check "T4 ran two ranks on cards 0,1 and 2,3 (world 2, port 29400)" "[ \"\$(sort $R/runs/dp1/ranks.txt | tr -d ' ' | tr '\n' ' ')\" = '{\"rank\":0,\"world\":2,\"cards\":\"0,1\",\"port\":\"29400\"} {\"rank\":1,\"world\":2,\"cards\":\"2,3\",\"port\":\"29400\"} ' ]"
check "T2 after it: one process on cards 1,3" "waitfor '[ -e $R/done/T2-sp1 ]' 200 && grep -q '\"rank\": 0, \"world\": 1, \"cards\": \"1,3\"' $R/runs/sp1/ranks.txt"
check "no other job ran while T4 held the cards (q.log order)" "! grep -q 'takes T2-sp1' <(sed -n '/takes T4-dp1/,/T4-dp1: ok/p' $R/q.log)"
stop
printf 'T4 dpf 2.50 5e5 - --lr 1e-4 --snap-at none @cards=0 @partner=1,2,3\n' > $R/jobs.txt
export FAIL_RANK=1
start
check "T4 with a failing rank 1: rank 0 stopped, retried from state.pt, then failed" "waitfor '[ -e $R/failed/T4-dpf ]' 900"
check "the surviving rank was stopped each time (no 600 s waits): 11 attempts logged" "[ \$(grep -c 'rank 1 rc 3: stopping the other ranks' $R/run.log) -eq 11 ]"
stop
unset FAIL_RANK
echo "queue log:"; cat $R/q.log | sed 's/^/  /' | head -40
[ $fails = 0 ] && { echo ALL PASS; rm -rf $R; exit 0; } || { echo "FAILED ($fails), dir $R"; exit 1; }
