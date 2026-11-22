#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""qhold.py ROOT release|hold [REGEX] -- un-hold / hold queue lines (w222): `release` strips the `# HOLD ` prefix from every held line whose text matches REGEX (default all); `hold` prefixes the matching
unheld T lines. Done under QROOT/queue.lock. The int8 batch was appended held on 11-22 (user review): `qhold.py /data/q27/queue release` (box2) / `qhold.py /data/queue release` (iceman) puts it in."""
import fcntl, os, re, sys
root, cmd = sys.argv[1], sys.argv[2]; rx = re.compile(sys.argv[3] if len(sys.argv) > 3 else ".")
lk = open(f"{root}/queue.lock", "a"); fcntl.flock(lk, fcntl.LOCK_EX)
L = open(f"{root}/jobs.txt").read().split("\n"); n = 0
for i, l in enumerate(L):
    if cmd == "release" and l.startswith("# HOLD ") and rx.search(l): L[i] = l[len("# HOLD "):]; n += 1
    elif cmd == "hold" and l.startswith("T ") and rx.search(l): L[i] = "# HOLD " + l; n += 1
open(f"{root}/jobs.txt.new", "w").write("\n".join(L)); os.replace(f"{root}/jobs.txt.new", f"{root}/jobs.txt"); print(cmd, n, "lines")
