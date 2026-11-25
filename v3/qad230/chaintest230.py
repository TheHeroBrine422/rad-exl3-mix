#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""chaintest230.py CUTS_DIR TOKENIZER WORKDIR [--n 60] -- no-GPU test of chain230.py: WORKDIR gets the first N cuts of each set (whole sessions are not
needed: gen230 groups what is there), fake radiance servers / recorders (fake_radiance230.py; a TP2 recorder fails for max-model-len > 20000 so the
TP4 fallback runs), and the chain runs as on box2. Checks: all four merged goldens + masks + corpora exist, the TP2-failed chunks were recorded on TP4."""
import json, os, shutil, subprocess, sys

cuts_dir, tkp, work = sys.argv[1:4]; N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 60
HERE = os.path.dirname(os.path.abspath(__file__))
shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
for n in ("t1", "t1-judge", "t2", "t2-judge"):
    r = f"{work}/render-{n}"; os.makedirs(r)
    cs = [json.loads(l) for l in open(f"{cuts_dir}/{n}/cuts.jsonl")][:N]
    open(f"{r}/cuts.jsonl", "w").writelines(json.dumps(c) + "\n" for c in cs); os.symlink(os.path.abspath(f"{cuts_dir}/{n}/sess"), f"{r}/sess")
for d in ("slices", "ext/gold", "ext/genmask"): os.makedirs(f"{work}/q/{d}")
for f, body in (("rg", "gen"), ("rr", "rec")):
    open(f"{work}/{f}", "w").write(f"#!/bin/bash\nexec {sys.executable} -I {HERE}/fake_radiance230.py \"$@\"\n"); os.chmod(f"{work}/{f}", 0o755)
env = dict(os.environ, W230_ROOT=work, W230_QROOT=f"{work}/q", W230_CODE=HERE, W230_PY=sys.executable, W230_GEN_RB=f"{work}/rg", W230_REC_RB=f"{work}/rr",
           W230_MODEL=tkp, W230_PORT_A="18020", W230_PORT_B="18021", W230_MIN_GB="1", FAKE_TP2_MAX="20000", PYTHONPATH="")
r = subprocess.run([sys.executable, f"{HERE}/chain230.py"], env=env, capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-3000:]); print(open(f"{work}/chain.log").read())
q = f"{work}/q"
need = [f"{q}/slices/golden-jt1/.done", f"{q}/slices/golden-jt2/.done", f"{q}/ext/gold/tset1/.done", f"{q}/ext/gold/tset2/.done", f"{q}/ext/genmask/tset1.u8", f"{q}/ext/genmask/tset2.u8",
        f"{q}/slices/golden-jt1.jsonl", f"{work}/chain.done"]
bad = [p for p in need if not os.path.exists(p)]
log = open(f"{work}/chain.log").read()
assert not bad, f"missing {bad}"
print("chaintest230 OK; TP4 fallback used:", "all four cards TP4" in log)
