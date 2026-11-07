#!/usr/bin/env python3
"""check_knap.py OUTDIR [RXM] -- S7 driver check (CPU, ~2 min single-threaded): the 11-07 ladder as rxq's library (knap5's
default V3R lists: i4 K1 + K1.125-3, i8 K1.125-5 in 1/8 + K5.25-7 in 1/4, m1 K6-8 in 1/4), the settings file's [knap]
switches -> rxq.knap env + solve at 2.5 bpw, accurate (f16 only, lambda 0); the output line must equal
data/levels/reprice/swj2/out/main-acc-2.5.jsonl (w63's run of the same solve) byte for byte. Then the hand-off: that line
-> a radmix mix file (packs and donors faked): every trunk weight once, in TRUNK order; options outside the library
(the head, V2 donors such as sc3.00-e4) from the donor map.
Last line "KNAP CHECKS n/m PASS"."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from rxq import config as C, knap, mix  # noqa: E402

O = sys.argv[1]; RXM = sys.argv[2] if len(sys.argv) > 2 else os.path.normpath(os.path.join(HERE, "..", ".."))
os.makedirs(O, exist_ok=True); res = []


def report(name, ok, why):
    res.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'} {name}: {why}", flush=True)


st = lambda a, b, s: [a + i * s for i in range(int(round((b - a) / s)) + 1)]
lib = {}
for cb, ks in (("i4", [1.0] + st(1.125, 3, 0.125)), ("i8", st(1.125, 5, 0.125) + st(5.25, 7, 0.25)), ("mul1", st(6, 8, 0.25))):
    for K in ks: lib[knap.level_name(cb, K)] = {"dir": f"/lv/{cb}/K{K:g}", "K": K, "codebook": cb, "pack": f"/packs/{cb}-K{K:g}.rad"}
cfg = C.load()
e = knap.env(cfg, lib, accurate=True)
want = {"KN_V3R": "1", "KN_EMBT": "2", "KN_HEADSLAB": "1", "KN_LEVELS": "../../data/v3/levels", "KN_DAMP": "1", "KN_PICKS": "1",
        "KN_NOA4": "1", "KN_SWJ": "all", "KN_SWJ_PTS": "2:0.91,2.5:0.935,4:0.989,5:1.0", "KN_MODES": "f16"}
same_env = all(e.get(k) == v for k, v in want.items()) and "KN_CD" not in e
report("env", same_env, f"{len(e)} switches; w63 main arm's set {'contained' if same_env else 'NOT matched'} "
       f"(KN_I8 = {e['KN_I8'][:40]}.., KN_TP {e['KN_TP']})")
out = f"{O}/main-acc-2.5.jsonl"
if os.path.exists(out): os.remove(out)
rc = knap.solve(cfg, lib, out, RXM, [2.5], [0.0], accurate=True)
ref = open(f"{RXM}/data/levels/reprice/swj2/out/main-acc-2.5.jsonl").read()
got = open(out).read() if os.path.exists(out) else ""
report("solve", rc == 0 and got == ref, f"rc {rc}; output {'byte-identical to' if got == ref else 'DIFFERS from'} w63's main-acc-2.5 "
       f"({len(got)} vs {len(ref)} bytes)")
if got:
    line = json.loads(got.splitlines()[0])
    don = {o: f"/donors/{o}.rad" for w, (o, _) in line["picks"].items() if o not in lib}   # head, embedding, V2 donors (sc3.00-e4)
    p, fl = knap.handoff(line, lib, f"{O}/mixes", "chk", donors=don, frontier=f"{O}/frontier.txt")
    rows = [l.split() for l in open(p) if l.strip()]
    names = [r[0] for r in rows if r[0] != "act"]
    tr = mix.trunk_names()
    ok = names[:len(tr)] == tr and len(names) == len(set(names)) == len(line["picks"]) and all(len(r) == 2 for r in rows)
    report("handoff", ok, f"{os.path.basename(p)}: {len(names)} weights ({len(tr)} trunk in TRUNK order), "
           f"{sum(r[0] == 'act' for r in rows)} act lines; frontier '{fl}'")
print(f"KNAP CHECKS {sum(res)}/{len(res)} PASS")
sys.exit(0 if all(res) else 1)
