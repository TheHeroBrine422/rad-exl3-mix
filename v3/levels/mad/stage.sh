#!/bin/bash
# stage.sh HOST COMMIT -- (VM side) restage HOST (ds8 | box2) at rad-exl3-mix COMMIT for #177-03 (w151), the ds8.md
# procedure: git archive -> /data/q27/code-<c>.tgz, extract to stage/code.new, the old stage/code kept as
# stage/code-old-<its COMMIT>, swap, then build the mad kernels (prebuild-mad.py: exl3frac + exl3hash's exl3cb i8 /
# exl3frac2 mul1 with EXL3_CB_HASH and the settings' A / B; no card) and print the code stamp and hostd's tree stamp.
# Idempotent: a host already at COMMIT only re-runs the prebuild. Log: HOST /data/q27/mad/stage.log
set -eu
H=$1 C=$2
B=/workspace/rad-local/bin; case $H in ds8) S=$B/ssh-ds8.sh;; box2) S=$B/ssh-box2.sh;; *) echo "host ds8 | box2"; exit 1;; esac
R=/workspace/rad-exl3-mix; c=$(git -C $R rev-parse --short=7 $C)
cur=$($S "cat /data/q27/stage/code/COMMIT 2>/dev/null || echo none")
if [ "$cur" != "$c" ]; then
  # ssh-box2.sh does not pass stdin: box2 gets the archive from ds8 over the 10 GbE link
  git -C $R archive $c | $B/ssh-ds8.sh "cat > /data/q27/code-$c.tar"
  [ $H = box2 ] && $B/ssh-ds8.sh "scp -q -i /root/.ssh/dbox -o StrictHostKeyChecking=no /data/q27/code-$c.tar root@10.29.6.6:/data/q27/code-$c.tar"
  $S "set -e; cd /data/q27/stage; rm -rf code.new; mkdir code.new; tar -xf /data/q27/code-$c.tar -C code.new; echo $c > code.new/COMMIT
      old=\$(cat code/COMMIT); [ -e code-old-\$old ] && rm -rf code || mv code code-old-\$old; mv code.new code
      mkdir -p /data/q27/mad; echo \"\$(date -u +%FT%TZ) staged $c (old \$old)\" >> /data/q27/mad/stage.log"
fi
$S "set -e; mkdir -p /data/q27/mad; cd /data/q27; source tools/env-$H.sh
    nice -n 10 \$PY /data/q27/stage/code/v3/levels/mad/prebuild-mad.py >> /data/q27/mad/stage.log 2>&1; tail -n 1 /data/q27/mad/stage.log
    env PYTHONPATH=/data/q27/stage/code/pipeline \$PY -m rxrun.hostd /data/q27/multi/host.json --stamp 2>&1 | tail -n 2"
