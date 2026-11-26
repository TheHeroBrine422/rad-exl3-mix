#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""chaintest232.py TOKENIZER DOCSDIR WORKDIR [N=4] -- no-GPU test of the whole chain232.py: a mini docs dir (the first N documents of each of sel232.py's chunks),
a fake radiance (fake_radiance232.py; TP2 fails above max-model-len FAKE_TP2_MAX_MML = 20,000 -> the deepest chunks must go through the TP4 fallback), temp lock files. Both lanes run
(lane B is opportunistic: the temp locks are free). Expect: every chunk recorded, TP4 used for the failed ones, merged + checked golden, chain.done."""
import json, os, shutil, subprocess, sys
tkp, docs, work = sys.argv[1:4]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 4
here = os.path.dirname(os.path.abspath(__file__))
shutil.rmtree(work, ignore_errors=True)
for d in ("docs", "locks", "queue", "code", "logs"): os.makedirs(f"{work}/{d}")
chunks = json.load(open(f"{docs}/xg-fin.chunks.json")); out = []; allL = []; allM = []
for ch in chunks:
    L = open(f"{docs}/{ch['file']}").read().rstrip("\n").split("\n")[:N]; M = [json.loads(l) for l in open(f"{docs}/{ch['file']}.meta.jsonl")][:N]
    open(f"{work}/docs/{ch['file']}", "w").write("\n".join(L) + "\n"); open(f"{work}/docs/{ch['file']}.meta.jsonl", "w").write("".join(json.dumps(m) + "\n" for m in M))
    out.append(dict(ch, docs=len(L), tokens=sum(m["tokens"] for m in M), max_tokens=max(m["tokens"] for m in M), rows=sum(m["tokens"] - 1 for m in M), gen_rows=sum(m["gen_rows"] for m in M)))
    allL += L; allM += M
json.dump(out, open(f"{work}/docs/xg-fin.chunks.json", "w"))
open(f"{work}/docs/xg-fin.jsonl", "w").write("\n".join(allL) + "\n"); open(f"{work}/docs/xg-fin.jsonl.meta.jsonl", "w").write("".join(json.dumps(m) + "\n" for m in allM))
env = dict(os.environ, W232_ROOT=work, W232_QROOT=f"{work}/queue", W232_CODE=here, W232_PY=sys.executable, W232_MODEL=tkp, W232_LOCKS=f"{work}/locks", W232_MIN_GB="1",
           W232_RB=f"{work}/fakerad.sh", FAKE_TP2_MAX_MML="20000", FAKE_TOK=tkp)
open(f"{work}/fakerad.sh", "w").write(f"#!/bin/bash\nexec {sys.executable} -I {here}/fake_radiance232.py \"$@\"\n"); os.chmod(f"{work}/fakerad.sh", 0o755)
os.makedirs(f"{work}/code/v3"); shutil.copytree(here, f"{work}/code/v3/qad232"); os.makedirs(f"{work}/code/v3/qad181"); shutil.copy(f"{here}/../qad181/genmask181.py", f"{work}/code/v3/qad181/")
env["W232_CODE"] = f"{work}/code/v3/qad232"
r = subprocess.run([sys.executable, "-I", f"{here}/chain232.py"], env=env, capture_output=True, text=True); print(r.stdout[-800:], r.stderr[-1500:])
log = open(f"{work}/chain.log").read(); print(log)
assert r.returncode == 0 and os.path.exists(f"{work}/chain.done") and os.path.exists(f"{work}/queue/ext/gold/xg-fin/.done"), "chain did not finish"
assert "TP4 all four cards: recording chunk 0" in log and "failed on TP2" in log and "check232 rc 0" in log
assert not os.path.exists(f"{work}/rec/c0") and os.path.exists(f"{work}/queue/ext/genmask/xg-fin.u8")
print("chaintest232 OK")
