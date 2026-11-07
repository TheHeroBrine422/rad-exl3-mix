#!/bin/bash
# mkpatch.sh -- regenerate autoresearch-v3-kv-t2.patch: the kv-task branch (git archive, nothing in the repo touched)
# + the edits held in /tmp/w66-ar (kvjudge.t2_rlib / rlib_run, BIN_FILES, the profile's ruler block, README) + a
# copy of this harness's rlib.py. Apply in autoresearch-v3 on kv-task: git apply <patch>.
# /tmp is wiped on reboot: to edit again, rebuild W from the patch (mkdir W; git -C autoresearch-v3 archive kv-task
# README.md tasks/kv.py tasks/kv/kvjudge.py tasks/kv/models | tar -x -C W; git init + commit; git apply <patch>).
set -e
P="$(dirname "$(readlink -f "$0")")"; W=${W:-/tmp/w66-ar}
(echo '# Copy of rad-exl3-mix eval/ruler/rlib.py (the RULER harness library; the harness is the source: copy, do not edit).'
 cat "$P/../rlib.py") > "$W/tasks/kv/rlib.py"
cd "$W" && git add -A && git diff --cached --binary > "$P/autoresearch-v3-kv-t2.patch"
git diff --cached --stat
