#!/usr/bin/env python3
"""test_hostd_census.py PIPELINE_DIR -- offline check of hostd-census.patch (w142, #178): rxrun.hostd.run_eval with stub pack /
mix / score commands (search/v3/test_hostd_es.py's) and a stub census_cmd that fails on tags containing "badc" and counts
its calls. Checks: a passing mix is scored and leaves <mix>.census; a failing one returns rc 1 with a census error and
no score; a second eval of the passed mix does not run the census again; without census_cmd nothing changes.
PIPELINE_DIR: a pipeline/ tree with the patch applied."""
import json, os, shutil, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(os.path.join(HERE, "..", "..", "test_hostd_es.py")).read()
head = src[:src.index("ok = True")]                 # the stubs, the golden, FakeHost, E_ (runs at import: makes W)
g = {"__name__": "stubs", "__file__": os.path.join(HERE, "..", "..", "test_hostd_es.py")}
exec(compile(head, "test_hostd_es.py (stubs)", "exec"), g)
hostd, W, E_, FakeHost, LV = g["hostd"], g["W"], g["E_"], g["FakeHost"], g["LV"]
CS = f"{W}/census_stub.py"; CN = f"{W}/census_calls"
open(CS, "w").write("import os, sys\nopen(%r, 'a').write(sys.argv[1] + '\\n')\nsys.exit(1 if 'badc' in os.path.basename(sys.argv[1]) else 0)\n" % CN)
ok = True


def check(name, cond, msg=""):
    global ok
    ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'} {name} {msg}")


def run(tag, root, census=True):
    os.makedirs(f"{root}/unit-{tag}", exist_ok=True)
    E = E_(root, prog=False)
    if census: E["census_cmd"] = [sys.executable, CS, "{mix}", "{mixfile}", "{base}"]
    req = {"tag": tag, "mix": {"base": f"{W}/base.rad", "trunk": LV}, "slices": [{"name": "s", "parts": {"m": "GM"}}]}
    return hostd.run_eval(FakeHost(root, E), f"{root}/unit-{tag}", req)


calls = lambda: open(CN).read().split() if os.path.exists(CN) else []
r = run("good", f"{W}/a")
check("pass: scored", r["rc"] == 0 and "s" in r["slices"])
check("pass: <mix>.census written", os.path.exists(f"{W}/a/mixes/good.rad.census"))
check("pass: census called once with the mix", calls() == [f"{W}/a/mixes/good.rad"], str(calls()))
b = run("badc", f"{W}/a")
check("fail: rc 1, census error", b["rc"] == 1 and "census" in b.get("err", ""), b.get("err", ""))
check("fail: not scored", not os.path.exists(f"{W}/a/scores/badc-s-m.json.rows"))
check("fail: no .census", not os.path.exists(f"{W}/a/mixes/badc.rad.census"))
n = len(calls()); r2 = run("good", f"{W}/a")
check("rerun of a passed mix: census not run again", r2["rc"] == 0 and len(calls()) == n)
p = run("good", f"{W}/p", census=False)
check("no census_cmd: same result, no census", p["rc"] == 0 and p["slices"]["s"]["kld"] == r["slices"]["s"]["kld"] and len(calls()) == n)
print("ALL PASS" if ok else "FAILED", W)
if ok: shutil.rmtree(W)
sys.exit(0 if ok else 1)
