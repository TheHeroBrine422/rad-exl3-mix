#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""test_hostd_act.py PIPELINE_DIR -- offline check of the hostd half of hostd-driver-a8.patch (w132, #172): rxrun.hostd
.run_eval with stub pack / mix / score commands (test_hostd_es.py's stubs); the mix file hostd hands radmix is read back.
  none     no "act" in the request: the mix file is the same as before the patch (trunk + donor lines, no act line)
  string   "act": "a8g128": one "act <w> a8g128" line per trunk weight (336), after the weight lines, nothing else changed
  dict     "act": {w: mode}: exactly those lines
  bad      "act": "a4": refused before radmix (rc 1), no mix written
PIPELINE_DIR: a pipeline/ tree with the patch applied."""
import json, os, shutil, sys, tempfile

PIPE = os.path.abspath(sys.argv[1]); sys.path.insert(0, PIPE)
from rxrun import hostd   # noqa: E402
from rxq import mix as M   # noqa: E402

W = tempfile.mkdtemp(prefix="w132-hostd-")
STUB = f"{W}/stub.py"
open(STUB, "w").write("""import sys
op = sys.argv[1]
if op == "pack": open(sys.argv[3], "w").write("pack\\n")
elif op == "mix": open(sys.argv[2], "w").write(open(sys.argv[4]).read())
""")
LV = "lvA"
os.makedirs(f"{W}/ladder/{LV}"); json.dump({"level": LV}, open(f"{W}/ladder/{LV}/DONE", "w"))
open(f"{W}/ladder/{LV}/final.recipe", "w").write("recipe A\n"); open(f"{W}/base.rad", "w").write("base\n")


class FakeHost:
    def __init__(self, root, E):
        self.root = root; self.name = "t"; self.rc = {"root": f"{W}/ladder"}; self.hc = {"eval": E}

    def current(self, n): return n == LV


def E_(root):
    return {"packs": f"{W}/packs", "mixes": f"{root}/mixes", "scores": f"{root}/scores", "card": 7, "card_lock": f"{W}/card7.lock",
            "ram_lock": f"{W}/ram.lock", "conv_lock": f"{W}/conv.lock",
            "pack_cmd": [sys.executable, STUB, "pack", "{recipe}", "{out}"], "mix_cmd": [sys.executable, STUB, "mix", "{out}", "{base}", "{mixfile}"],
            "score_cmd": ["false"]}


def mixfile(tag, act=None):
    root = f"{W}/r-{tag}"; os.makedirs(f"{root}/unit", exist_ok=True)
    mx = {"base": f"{W}/base.rad", "trunk": LV, "donors": {"output.weight": "/d/head.rad"}}
    if act is not None: mx["act"] = act
    r = hostd.run_eval(FakeHost(root, E_(root)), f"{root}/unit", {"tag": tag, "mix": mx, "slices": []})
    p = f"{root}/mixes/{tag}.mix"
    return r, (open(p).read().splitlines() if os.path.exists(p) else None)


ok = True


def check(name, cond, msg=""):
    global ok
    ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'} {name} {msg}")


T = M.trunk_names(); pk = f"{W}/packs/{LV}.rad"
base_lines = [f"{w} {pk}" for w in T] + ["output.weight /d/head.rad"]
r, L = mixfile("none")
check("none", r["rc"] == 0 and L == base_lines, f"{len(L or [])} lines, no act line")
r, L = mixfile("str", "a8g128")
acts = [l for l in (L or []) if l.startswith("act ")]
check("string", r["rc"] == 0 and L[:len(base_lines)] == base_lines and acts == [f"act {w} a8g128" for w in T] and len(L) == len(base_lines) + len(T),
      f"{len(acts)} act lines for {len(T)} trunk weights")
r, L = mixfile("dict", {T[0]: "a8g128", T[5]: "a8t"})
check("dict", r["rc"] == 0 and L[len(base_lines):] == [f"act {T[0]} a8g128", f"act {T[5]} a8t"])
r, L = mixfile("bad", "a4")
check("bad", r["rc"] == 1 and "a4" in r.get("err", "") and L is None, r.get("err", ""))
print("ALL PASS" if ok else "FAILED", W)
if ok: shutil.rmtree(W)
sys.exit(0 if ok else 1)
