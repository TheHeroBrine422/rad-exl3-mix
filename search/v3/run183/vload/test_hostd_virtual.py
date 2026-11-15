#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""test_hostd_virtual.py PIPELINE_DIR -- offline check of hostd-virtual.patch (w162, #185): rxrun.hostd.run_eval with stub
pack / mix / census / bytes / score commands (no GPU, no radmix).
  old     eval without eval.virtual: the mix step runs (radmix stub writes the .rad), the census runs on the .rad, the
          served size is the file's
  virtual eval.virtual: no mix step (the mix stub fails if it is called), the manifest <tag>.mix is `base BASE` + the
          mixfile lines (<tag>.mixfile, no base line), the census gets the manifest and the mixfile, the size comes
          from virtual_bytes_cmd, the scorer gets the manifest; the result's bytes / bpw / slice are the same shape
  rerun   a second run of the same tag (the manifest is rewritten, the scores are kept): same result
PIPELINE_DIR: a pipeline/ tree with the patch applied."""
import json, os, sys, tempfile

import numpy as np

PIPE = os.path.abspath(sys.argv[1]); sys.path.insert(0, PIPE)
from rxrun import hostd   # noqa: E402

W = tempfile.mkdtemp(prefix="w162-hostd-virt-")
STUB = f"{W}/stub.py"
open(STUB, "w").write(r"""import json, os, sys
op = sys.argv[1]; log = os.environ["STUB_LOG"]
def rec(*a): open(log, "a").write(json.dumps(list(a)) + "\n")
if op == "pack":
    open(sys.argv[3], "w").write("pack of " + sys.argv[2])
elif op == "mix":                       # radmix: must not run in virtual mode (STUB_NO_MIX makes it fail)
    rec("mix", sys.argv[2])
    if os.environ.get("STUB_NO_MIX"): sys.exit(3)
    open(sys.argv[2], "w").write("mix of " + sys.argv[4] + " + " + sys.argv[3] + "\n")
elif op == "census":                    # census MIX MIXFILE BASE
    rec("census", sys.argv[2], sys.argv[3], sys.argv[4]); open(sys.argv[2] + ".checked", "w").write("ok\n")
elif op == "bytes":
    rec("bytes", sys.argv[2]); print(4242)
elif op == "score":                     # score MIX GOLDEN OUT CARD
    mix, g, o = sys.argv[2:5]
    rec("score", mix, open(mix).readline().strip())
    import struct
    n = 5
    rows = b"".join(struct.pack("<ffff", 0.25, 0.0, 0.0, 1.0) for _ in range(n))
    open(o + ".json.rows", "wb").write(rows)
    json.dump({"all": {"positions": n, "top1_agreement": 1.0}}, open(o + ".json", "w"))
""")

G = f"{W}/golden"; os.makedirs(G)
open(f"{G}/tokens.i32", "wb").write(np.arange(6, dtype=np.int32).tobytes())
open(f"{G}/gen.u8", "wb").write(bytes([1, 1, 1, 1, 1]))
json.dump({"format": "radiance-kld-top-1", "model": "t", "top": 1, "n_vocab": 10, "rows": 5,
           "docs": [{"source": "t", "tokens": 6, "score_from": 0, "row0": 0}]}, open(f"{G}/kld.json", "w"))
LV = "lvA"
os.makedirs(f"{W}/ladder/{LV}")
json.dump({"level": LV}, open(f"{W}/ladder/{LV}/DONE", "w"))
open(f"{W}/ladder/{LV}/final.recipe", "w").write("recipe A\n")
BASE = f"{W}/base.rad"; open(BASE, "w").write("base\n")
ALIAS = {"GOLD": G}


class FakeHost:
    def __init__(self, root, E):
        self.root = root; self.name = "t"; self.rc = {"root": f"{W}/ladder"}; self.hc = {"eval": E}

    def current(self, n): return n == LV


def E_(root, virtual):
    st = [sys.executable, STUB]
    E = {"packs": f"{W}/packs", "mixes": f"{root}/mixes", "scores": f"{root}/scores", "aliases": ALIAS,
         "card": 7, "card_lock": f"{W}/card7.lock", "ram_lock": f"{W}/ram.lock", "conv_lock": None,
         "pack_cmd": st + ["pack", "{recipe}", "{out}"],
         "mix_cmd": st + ["mix", "{out}", "{base}", "{mixfile}"],
         "census_cmd": st + ["census", "{mix}", "{mixfile}", "{base}"],
         "census_virtual_cmd": st + ["census", "{mix}", "{mixfile}", "{base}"],
         "virtual_bytes_cmd": st + ["bytes", "{mix}"],
         "score_cmd": st + ["score", "{mix}", "{golden}", "{out}", "{card}"]}
    if virtual: E["virtual"] = True
    else:
        E.pop("virtual_bytes_cmd"); E.pop("census_virtual_cmd")
    return E


ok = True


def check(name, cond, msg=""):
    global ok
    ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'} {name} {msg}")


def run(tag, root, virtual):
    os.makedirs(f"{root}/unit-{tag}", exist_ok=True)
    os.environ["STUB_LOG"] = f"{W}/{tag}.log"
    os.environ["STUB_NO_MIX"] = "1" if virtual else ""
    if os.path.exists(os.environ["STUB_LOG"]): os.remove(os.environ["STUB_LOG"])
    req = {"tag": tag, "mix": {"base": BASE, "trunk": LV}, "slices": [{"name": "s", "parts": {"g": "GOLD"}}]}
    return hostd.run_eval(FakeHost(root, E_(root, virtual)), f"{root}/unit-{tag}", req)


def calls(tag):
    p = f"{W}/{tag}.log"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


root = f"{W}/run"
os.makedirs(f"{root}/mixes"); os.makedirs(f"{root}/scores")

# old path: the mix step runs and writes the .rad; the census is on it; the size is the file's
r = run("old", root, virtual=False)
c = calls("old")
mix_rad = f"{root}/mixes/old.rad"
check("old: ok", r.get("rc") == 0, str(r.get("err", ""))[:120])
check("old: mix step ran", any(x[0] == "mix" for x in c))
check("old: the .rad exists and the census gets it", os.path.exists(mix_rad) and any(x[0] == "census" and x[1] == mix_rad for x in c))
check("old: size is the file's", r.get("bytes") == os.path.getsize(mix_rad) and r["bytes"] != 4242, str(r.get("bytes")))

# virtual: no mix step; the manifest and the mixfile; the census and the size and the scorer get the manifest
r = run("virt", root, virtual=True)
c = calls("virt")
man = f"{root}/mixes/virt.mix"; mf = f"{root}/mixes/virt.mixfile"
check("virt: ok", r.get("rc") == 0, str(r.get("err", ""))[:160])
check("virt: no mix step (the mix stub would have exited 3)", not any(x[0] == "mix" for x in c))
check("virt: no .rad written", not os.path.exists(f"{root}/mixes/virt.rad"))
lines = open(man).read().splitlines() if os.path.exists(man) else []
check("virt: manifest starts with `base BASE`", lines[:1] == [f"base {BASE}"], str(lines[:1]))
check("virt: manifest = base line + the mixfile", lines[1:] == open(mf).read().splitlines() and os.path.exists(mf))
check("virt: mixfile has no base line", not any(l.startswith("base ") for l in open(mf).read().splitlines()))
check("virt: census gets the manifest and the mixfile", any(x[0] == "census" and x[1] == man and x[2] == mf for x in c))
check("virt: size from virtual_bytes_cmd", r.get("bytes") == 4242 and abs(r.get("bpw", 0) - 4242 * 8 / hostd.PARAMS) < 1e-12, str(r.get("bytes")))
check("virt: scorer gets the manifest", any(x[0] == "score" and x[1] == man and x[2] == f"base {BASE}" for x in c))
sl = r.get("slices", {}).get("s", {})
check("virt: slice result", sl.get("positions") == 5 and sl.get("bytes") == 4242, str(sl)[:120])

# rerun: the manifest is rewritten and the census runs again; the scores are kept
r2 = run("virt", root, virtual=True)
check("virt rerun: same result", r2.get("rc") == 0 and r2.get("bytes") == 4242 and r2["slices"]["s"]["kld"] == sl.get("kld"))

print("ALL PASS" if ok else "FAILURES")
sys.exit(0 if ok else 1)
