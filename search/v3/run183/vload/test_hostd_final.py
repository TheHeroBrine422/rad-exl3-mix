#!/usr/bin/env python3
"""test_hostd_final.py PIPELINE_DIR -- offline check of hostd-final.patch (w162, #185 B): rxrun.hostd.run_eval with stub pack /
mix / census / bytes / score commands (no GPU, no radmix, no pack).
  final    eval.final_root: no pack step (the pack stub would write one; none may appear), no mix step, the manifest
           <tag>.mix is `base BASE` + the mixfile, whose trunk lines are `<weight> final ROOT/<level>/final codebook=mad`
           (336 of them) and whose donor / act lines are as the virtual path writes them; the census gets the manifest;
           the size comes from virtual_bytes_cmd; the scorer gets the manifest.
  missing  the same with no final/ index for the level: the unit fails before any census or score (rc 1).
  virtual  eval.virtual alone (no final_root): still the packs' manifest (the virtual test's own check, kept here).
PIPE_DIR: a pipeline/ tree with the patch applied."""
import json, os, sys, tempfile

import numpy as np

PIPE = os.path.abspath(sys.argv[1]); sys.path.insert(0, PIPE)
from rxrun import hostd   # noqa: E402
from rxq import mix as M  # noqa: E402

W = tempfile.mkdtemp(prefix="w162-hostd-final-")
STUB = f"{W}/stub.py"
open(STUB, "w").write(r"""import json, os, sys
op = sys.argv[1]; log = os.environ["STUB_LOG"]
def rec(*a): open(log, "a").write(json.dumps(list(a)) + "\n")
if op == "pack":
    rec("pack", sys.argv[2]); open(sys.argv[3], "w").write("pack of " + sys.argv[2])
elif op == "mix":
    rec("mix", sys.argv[2]); sys.exit(3)
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
LV = "q27v3-635bac02-k2.5-mad"
FR = f"{W}/finals"
os.makedirs(f"{FR}/{LV}/final")
open(f"{FR}/{LV}/final/model.safetensors.index.json", "w").write("{}\n")
os.makedirs(f"{W}/ladder/{LV}")
json.dump({"level": LV}, open(f"{W}/ladder/{LV}/DONE", "w"))
open(f"{W}/ladder/{LV}/final.recipe", "w").write("recipe A\n")
BASE = f"{W}/base.rad"; open(BASE, "w").write("base\n")
ALIAS = {"GOLD": G}


class FakeHost:
    def __init__(self, root, E):
        self.root = root; self.name = "t"; self.rc = {"root": f"{W}/ladder"}; self.hc = {"eval": E}

    def current(self, n): return n == LV


def E_(root, final_root):
    st = [sys.executable, STUB]
    E = {"packs": f"{W}/packs", "mixes": f"{root}/mixes", "scores": f"{root}/scores", "aliases": ALIAS,
         "card": 7, "card_lock": f"{W}/card7.lock", "ram_lock": f"{W}/ram.lock", "conv_lock": None,
         "pack_cmd": st + ["pack", "{recipe}", "{out}"],
         "mix_cmd": st + ["mix", "{out}", "{base}", "{mixfile}"],
         "census_cmd": st + ["census", "{mix}", "{mixfile}", "{base}"],
         "census_virtual_cmd": st + ["census", "{mix}", "{mixfile}", "{base}"],
         "virtual_bytes_cmd": st + ["bytes", "{mix}"],
         "score_cmd": st + ["score", "{mix}", "{golden}", "{out}", "{card}"],
         "virtual": True}
    if final_root: E["final_root"] = final_root      # the final path; without it, eval.virtual alone (the packs' manifest)
    return E


ok = True


def check(name, cond, msg=""):
    global ok
    ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'} {name} {msg}")


def run(tag, root, final_root):
    os.makedirs(f"{root}/unit-{tag}", exist_ok=True)
    os.environ["STUB_LOG"] = f"{W}/{tag}.log"
    if os.path.exists(os.environ["STUB_LOG"]): os.remove(os.environ["STUB_LOG"])
    req = {"tag": tag, "mix": {"base": BASE, "trunk": LV, "donors": {"output.weight": f"{W}/head.rad"}, "act": "a8g128"},
           "slices": [{"name": "s", "parts": {"g": "GOLD"}}]}
    return hostd.run_eval(FakeHost(root, E_(root, final_root)), f"{root}/unit-{tag}", req)


def calls(tag):
    p = f"{W}/{tag}.log"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


root = f"{W}/run"
os.makedirs(f"{root}/mixes"); os.makedirs(f"{root}/scores"); os.makedirs(f"{W}/packs")
trunk = M.trunk_names()
check("trunk list is the 336 projections", len(trunk) == 336, str(len(trunk)))

r = run("fin", root, FR)
c = calls("fin")
man = f"{root}/mixes/fin.mix"; mf = f"{root}/mixes/fin.mixfile"
check("final: ok", r.get("rc") == 0, str(r.get("err", ""))[:160])
check("final: no pack step", not any(x[0] == "pack" for x in c) and not os.listdir(f"{W}/packs"))
check("final: no mix step (the mix stub would have exited 3)", not any(x[0] == "mix" for x in c))
lines = open(man).read().splitlines() if os.path.exists(man) else []
check("final: manifest starts with `base BASE`", lines[:1] == [f"base {BASE}"], str(lines[:1]))
want = [f"{w} final {FR}/{LV}/final codebook=mad" for w in trunk] + [f"output.weight {W}/head.rad", "act " + trunk[0] + " a8g128"] + \
       [f"act {w} a8g128" for w in trunk[1:]]
check("final: trunk lines are final/ lines (no packs), then the donor and act lines, in the order the virtual path writes them",
      lines[1:] == open(mf).read().splitlines() and lines[1:] == want and len(want) == 336 + 1 + 336, str(lines[1:3]))
check("final: no pack path in the manifest", not any("packs" in l for l in lines))
check("final: census gets the manifest", any(x[0] == "census" and x[1] == man for x in c))
check("final: size from virtual_bytes_cmd", r.get("bytes") == 4242, str(r.get("bytes")))
check("final: scorer gets the manifest", any(x[0] == "score" and x[1] == man for x in c))

bad = run("miss", root, f"{W}/nowhere")
check("missing final/: rc 1, nothing scored", bad.get("rc") == 1 and "final/ not on this host" in bad.get("err", "") and
      not any(x[0] in ("census", "score") for x in calls("miss")), str(bad.get("err", ""))[:120])

v = run("virt", root, None)
check("virtual alone: still the pack manifest (packs in the lines, pack step ran)",
      v.get("rc") == 0 and any(x[0] == "pack" for x in calls("virt")) and any("packs" in l for l in open(f"{root}/mixes/virt.mix").read().splitlines()),
      str(v.get("err", ""))[:120])

print("ALL PASS" if ok else "FAILURES")
sys.exit(0 if ok else 1)
