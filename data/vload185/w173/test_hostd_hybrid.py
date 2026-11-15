#!/usr/bin/env python3
"""test_hostd_hybrid.py PIPELINE_DIR -- offline check of the hybrid hostd (w173, #185): rxrun.hostd.run_eval with stub pack /
mix / census / bytes / score commands (no GPU, no radmix, no real pack). PIPELINE_DIR: a pipeline/ tree with hostd-hybrid.patch
applied (hostd-final.patch is superseded by it).

  hybrid   two levels: LV_READY (DIR/LV_READY/READY present, final/ index present) and LV_PACK (no READY, built on the host).
           The trunk lines of LV_READY's weights are `<weight> final DIR/LV_READY/final codebook=mad`; LV_PACK's are pack
           lines, and the pack step runs for LV_PACK only; no mix step; the census gets the manifest; the scorer gets it.
  allpack  final_root set, no level READY: every trunk line is a pack line (the pure pack path); a pack that exists is reused.
  no-census  final_root set without eval.census_virtual_cmd: rc 1 before any pack or mix step (fail closed).
  missing  READY but no final/ index: rc 1 before any pack, mix, census or score.
  unbuilt  no READY and not built on the host: rc 1 before any pack or mix step.
  census-fail  the census exits non-zero: rc 1, nothing scored.
"""
import json, os, sys, tempfile

import numpy as np

PIPE = os.path.abspath(sys.argv[1]); sys.path.insert(0, PIPE)
from rxrun import hostd   # noqa: E402
from rxq import mix as M  # noqa: E402

W = tempfile.mkdtemp(prefix="w173-hostd-hybrid-")
STUB = f"{W}/stub.py"
open(STUB, "w").write(r"""import json, os, sys
op = sys.argv[1]; log = os.environ["STUB_LOG"]
def rec(*a): open(log, "a").write(json.dumps(list(a)) + "\n")
if op == "pack":
    rec("pack", sys.argv[2]); open(sys.argv[3], "w").write("pack of " + sys.argv[2])
elif op == "mix":
    rec("mix", sys.argv[2]); sys.exit(3)
elif op == "census":                    # census MIX MIXFILE BASE
    rec("census", sys.argv[2], sys.argv[3], sys.argv[4])
    sys.exit(int(os.environ.get("CENSUS_RC", "0")))
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
LR, LP = "q27v3-635bac02-k7.5-mad", "q27v3-635bac02-k2.5-mad"
FR = f"{W}/finals"
os.makedirs(f"{FR}/{LR}/final"); open(f"{FR}/{LR}/final/model.safetensors.index.json", "w").write("{}\n")
open(f"{FR}/{LR}/READY", "w").write("READY\n")
os.makedirs(f"{FR}/{LP}/final"); open(f"{FR}/{LP}/final/model.safetensors.index.json", "w").write("{}\n")   # not READY
for lv in (LP, LR):                       # the ladder's DONE and recipe (read when a level is packed)
    os.makedirs(f"{W}/ladder/{lv}")
    json.dump({"level": lv}, open(f"{W}/ladder/{lv}/DONE", "w"))
    open(f"{W}/ladder/{lv}/final.recipe", "w").write(f"recipe {lv}\n")
BASE = f"{W}/base.rad"; open(BASE, "w").write("base\n")
ALIAS = {"GOLD": G}


class FakeHost:
    def __init__(self, root, E, built):
        self.root = root; self.name = "t"; self.rc = {"root": f"{W}/ladder"}; self.hc = {"eval": E}; self.built = built

    def current(self, n): return n in self.built


def E_(root, final_root, census=True):
    st = [sys.executable, STUB]
    E = {"packs": f"{W}/packs", "mixes": f"{root}/mixes", "scores": f"{root}/scores", "aliases": ALIAS,
         "card": 7, "card_lock": f"{W}/card7.lock", "ram_lock": f"{W}/ram.lock", "conv_lock": None,
         "pack_cmd": st + ["pack", "{recipe}", "{out}"],
         "mix_cmd": st + ["mix", "{out}", "{base}", "{mixfile}"],
         "census_cmd": st + ["census", "{mix}", "{mixfile}", "{base}"],
         "virtual_bytes_cmd": st + ["bytes", "{mix}"],
         "score_cmd": st + ["score", "{mix}", "{golden}", "{out}", "{card}"],
         "virtual": True}
    if census: E["census_virtual_cmd"] = st + ["census", "{mix}", "{mixfile}", "{base}"]
    if final_root: E["final_root"] = final_root
    return E


ok = True


def check(name, cond, msg=""):
    global ok
    ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'} {name} {msg}")


def run(tag, root, final_root, levels, built, census=True, census_rc="0"):
    os.makedirs(f"{root}/unit-{tag}", exist_ok=True)
    os.environ["STUB_LOG"] = f"{W}/{tag}.log"; os.environ["CENSUS_RC"] = census_rc
    if os.path.exists(os.environ["STUB_LOG"]): os.remove(os.environ["STUB_LOG"])
    req = {"tag": tag, "mix": {"base": BASE, "levels": levels, "donors": {"output.weight": f"{W}/head.rad"}, "act": "a8g128"},
           "slices": [{"name": "s", "parts": {"g": "GOLD"}}]}
    return hostd.run_eval(FakeHost(root, E_(root, final_root, census), built), f"{root}/unit-{tag}", req)


def calls(tag):
    p = f"{W}/{tag}.log"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


root = f"{W}/run"
for d in ("mixes", "scores"): os.makedirs(f"{root}/{d}")
os.makedirs(f"{W}/packs")
trunk = M.trunk_names()
check("trunk list is the 336 projections", len(trunk) == 336, str(len(trunk)))
levels = {w: (LR if i % 2 == 0 else LP) for i, w in enumerate(trunk)}
n_ready = sum(1 for v in levels.values() if v == LR)

r = run("hyb", root, FR, levels, {LR, LP})
c = calls("hyb")
man = f"{root}/mixes/hyb.mix"; mf = f"{root}/mixes/hyb.mixfile"
lines = open(man).read().splitlines() if os.path.exists(man) else []
check("hybrid: ok", r.get("rc") == 0, str(r.get("err", ""))[:160])
check("hybrid: manifest starts with `base BASE`", lines[:1] == [f"base {BASE}"], str(lines[:1]))
want = []
for w in trunk:
    if levels[w] == LR: want.append(f"{w} final {FR}/{LR}/final codebook=mad")
    else: want.append(f"{w} {W}/packs/{LP}.rad")
check("hybrid: READY levels as final lines, the others as pack lines, in trunk order",
      lines[1:1 + 336] == want, str(lines[1:3]))
check("hybrid: the pack step ran for the pack level only", [x[1] for x in c if x[0] == "pack"] == [f"{W}/ladder/{LP}/final.recipe"],
      str([x[:2] for x in c if x[0] == "pack"]))
check("hybrid: no mix step", not any(x[0] == "mix" for x in c))
check("hybrid: the census gets the manifest", any(x[0] == "census" and x[1] == man for x in c))
check("hybrid: the scorer gets the manifest", any(x[0] == "score" and x[1] == man for x in c))
check("hybrid: donor and act lines after the trunk", lines[337:] == [f"output.weight {W}/head.rad"] + [f"act {w} a8g128" for w in trunk],
      str(lines[337:339]))
check(f"hybrid: {n_ready} final lines, {336 - n_ready} pack lines", sum(l.startswith(tuple(trunk)) and " final " in l for l in lines) == n_ready)

os.remove(f"{FR}/{LR}/READY")              # no level READY: the same mix must come out as the pure pack path
ap = run("allpack", root, FR, levels, {LR, LP})
la = open(f"{root}/mixes/allpack.mix").read().splitlines()
check("allpack: no READY level: no final line at all (the pure pack path)", ap.get("rc") == 0 and not any(" final " in l for l in la),
      str(ap.get("err", ""))[:120])
check("allpack: only the level without a pack is packed (the k2.5 pack from the hybrid unit is reused)",
      [x[1] for x in calls("allpack") if x[0] == "pack"] == [f"{W}/ladder/{LR}/final.recipe"], str([x[:2] for x in calls("allpack") if x[0] == "pack"]))
open(f"{FR}/{LR}/READY", "w").write("READY\n")

nc = run("nocensus", root, FR, levels, {LR, LP}, census=False)
check("no-census: rc 1, no pack / mix / census / score",
      nc.get("rc") == 1 and "census_virtual_cmd" in nc.get("err", "") and not any(x[0] in ("pack", "mix", "census", "score") for x in calls("nocensus")),
      str(nc.get("err", ""))[:120])

os.remove(f"{FR}/{LR}/final/model.safetensors.index.json")
mi = run("missing", root, FR, levels, {LR, LP})
check("missing final/ on a READY level: rc 1, nothing scored or censused",
      mi.get("rc") == 1 and "READY, so no pack" in mi.get("err", "") and not any(x[0] in ("census", "score") for x in calls("missing")),
      str(mi.get("err", ""))[:120])
open(f"{FR}/{LR}/final/model.safetensors.index.json", "w").write("{}\n")

ub = run("unbuilt", root, FR, levels, {LR})
check("unbuilt pack level: rc 1 before any pack step",
      ub.get("rc") == 1 and "levels not built on this host" in ub.get("err", "") and not any(x[0] in ("pack", "mix") for x in calls("unbuilt")),
      str(ub.get("err", ""))[:120])

cf = run("cfail", root, FR, levels, {LR, LP}, census_rc="5")
check("census fails: rc 1, not scored", cf.get("rc") == 1 and "census" in cf.get("err", "") and not any(x[0] == "score" for x in calls("cfail")),
      str(cf.get("err", ""))[:120])

print("ALL PASS" if ok else "FAILURES")
sys.exit(0 if ok else 1)
