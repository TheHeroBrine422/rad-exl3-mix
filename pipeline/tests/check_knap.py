#!/usr/bin/env python3
"""check_knap.py OUTDIR [RXM] -- S7 driver check (CPU, ~2 min single-threaded): the 11-07 ladder as rxq's library (knap5's
default V3R lists: i4 K1 + K1.125-3, i8 K1.125-5 in 1/8 + K5.25-7 in 1/4, m1 K6-8 in 1/4), the settings file's [knap]
switches -> rxq.knap env + solve at 2.5 bpw, accurate (f16 only, lambda 0); the output line must equal
data/levels/reprice/swj2/out/main-acc-2.5.jsonl (w63's run of the same solve) byte for byte. Then the hand-off: that line
-> a radmix mix file (packs and donors faked): every trunk weight once, in TRUNK order; options outside the library
(the head, V2 donors such as sc3.00-e4) from the donor map.
V3 part (w120, 11-12; --part v3 | all): the 49-level i8 ladder (rxq.ladder.levels) as the library, the settings
file's V3 [knap] (swj_mode gu, the measured anchors swj-anchors.json, KN_SWJ_KMAX = post.swj.k_max) + the 27B sub-1
prices (data/levels/reprice/sub1/sub1ratios-27b.json, KN_SUB1_FIT = sub1-anchors.json), solved at 2.5 and 1.25 bpw
(lambda 0): every solved anchor reproduced (|model - measured| <= 1e-3), no swj factor above K4.5 and none on other
families, sub-1 options priced, picks cover the trunk; and without sub-1 prices the sub-1 anchors are skipped.
Last line "KNAP CHECKS n/m PASS"."""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from rxq import config as C, knap, mix  # noqa: E402

args = [x for x in sys.argv[1:] if not x.startswith("--part")]
PART = next((x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--part=")), "all")
O = args[0]; RXM = args[1] if len(args) > 1 else os.path.normpath(os.path.join(HERE, "..", ".."))
os.makedirs(O, exist_ok=True); res = []


def report(name, ok, why):
    res.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'} {name}: {why}", flush=True)


if PART in ("legacy", "all"):
    st = lambda a, b, s: [a + i * s for i in range(int(round((b - a) / s)) + 1)]
    lib = {}
    for cb, ks in (("i4", [1.0] + st(1.125, 3, 0.125)), ("i8", st(1.125, 5, 0.125) + st(5.25, 7, 0.25)), ("mul1", st(6, 8, 0.25))):
        for K in ks: lib[knap.level_name(cb, K)] = {"dir": f"/lv/{cb}/K{K:g}", "K": K, "codebook": cb, "pack": f"/packs/{cb}-K{K:g}.rad"}
    cfg = C.load()
    e = knap.env(cfg, lib, accurate=True)
    want = {"KN_V3R": "1", "KN_EMBT": "2", "KN_HEADSLAB": "1", "KN_LEVELS": "../../data/v3/levels", "KN_DAMP": "1", "KN_PICKS": "1",
            "KN_NOA4": "1", "KN_SWJ": "all", "KN_SWJ_PTS": "2:0.91,2.5:0.935,4:0.989,5:1.0", "KN_MODES": "f16"}
    same_env_x = "KN_SWJ_ANCHORS" not in e and "KN_SWJ_KMAX" not in e
    same_env = all(e.get(k) == v for k, v in want.items()) and "KN_CD" not in e and same_env_x
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


def anchors(log):
    """(id, measured, model, check) of every anchor line knap5 printed"""
    out = []
    for l in open(log):
        m = re.match(r"(swj|sub-1) anchor (\S+): measured (?:r|KL ratio) ([\d.]+), model ([\d.]+)( \(check\))?", l)
        if m: out.append((m.group(2), float(m.group(3)), float(m.group(4)), bool(m.group(5))))
    return out


if PART in ("v3", "all"):
    from rxq import ladder
    cfg = C.load()
    lib3 = {}
    for v in ladder.levels(cfg):
        n = knap.level_name(v["codebook"], v["K"]); lib3[n] = {"dir": f"/lv/{n}", "K": v["K"], "codebook": v["codebook"], "pack": f"/packs/{n}.rad"}
    e3 = knap.env(cfg, lib3)
    ok = (e3.get("KN_SWJ") == "gu" and e3.get("KN_SWJ_KMAX") == f"{cfg['post']['swj']['k_max']:g}" and "KN_SWJ_ANCHORS" in e3
          and "KN_SWJ_PTS" not in e3 and all(float(k) >= 1 for k in e3["KN_I8"].split(",")) and e3.get("KN_MODES") == "a8")
    report("env-v3", ok, f"{len(lib3)} levels ({sum(v['K'] < 1 for v in lib3.values())} sub-1); KN_SWJ {e3.get('KN_SWJ')}, KMAX "
           f"{e3.get('KN_SWJ_KMAX')}, anchors {e3.get('KN_SWJ_ANCHORS')}, KN_SUB1 {e3.get('KN_SUB1', '(unset: warned)')}")
    S1 = {"KN_SUB1": "../../data/levels/reprice/sub1/sub1ratios-27b.json", "KN_SUB1_TENSORS": "1",
          "KN_I8_SUB1": ",".join(f"{v['K']:g}" for v in lib3.values() if v["K"] < 1),
          "KN_SUB1_FIT": "../levels/reprice/sub1-anchors.json", "KN_SUB1_FIT_OUT": os.path.abspath(f"{O}/sub1fit.json")}
    # a synthetic saturation anchor (ratio 20: below the model's raw U1 / U2.5 ratio) only to exercise KN_SAT's path
    sa = json.load(open(os.path.join(RXM, "v3", "levels", "reprice", "sub1-anchors.json")))
    for a_ in sa["anchors"]:
        if a_["id"] == "149-sat": a_.update(ratio=20.0, skip=False, src="SYNTHETIC (check_knap only)")
    json.dump(sa, open(f"{O}/sub1-anchors-sat.json", "w"), indent=1)
    SAT = dict(S1, KN_SUB1_FIT=os.path.abspath(f"{O}/sub1-anchors-sat.json"), KN_SUB1_FIT_OUT=os.path.abspath(f"{O}/sub1fit-sat.json"))
    for tag, extra, tg in (("s1", S1, [2.5, 1.25]), ("nos1", {}, [2.5]), ("sat", SAT, [1.25])):
        out = f"{O}/v3-{tag}.jsonl"; log = f"{O}/v3-{tag}.log"
        if os.path.exists(out): os.remove(out)
        rc = knap.solve(cfg, lib3, out, RXM, tg, [0.0], extra_env=extra, log=log)
        lines = [json.loads(x) for x in open(out)] if os.path.exists(out) else []
        an = anchors(log); txt = open(log).read()
        solved = [a for a in an if not a[3]]
        rep = all(abs(m - mm) <= 1e-3 for _, m, mm, _ in solved)
        kf = re.search(r"swj factor: (\d+) of (\d+) options != 1, K range with a factor (\S+)\.\.(\S+), min ([\d.]+)", txt)
        cap = kf is not None and float(kf.group(4)) <= cfg["post"]["swj"]["k_max"] + 1e-9
        tr = set(mix.trunk_names())
        cover = bool(lines) and all(tr <= set(l["picks"]) for l in lines)
        sub = re.search(r"V3R sub-1: (\d+) tensor-levels", txt)
        skip = re.findall(r"(?:swj|sub-1) anchor (\S+): skipped", txt)
        if tag == "s1":
            ok = rc == 0 and rep and cap and cover and sub is not None and int(sub.group(1)) > 0 and len(solved) >= 5
            why = f"sub-1 options {sub.group(1) if sub else 0}"
        elif tag == "nos1":
            ok = rc == 0 and rep and cap and cover and sub is None and {"145-ns", "145-sn"} <= set(skip)
            why = f"skipped {skip}"
        else:
            c = re.search(r"saturation: C ([\d.]+)", txt); fit = json.load(open(f"{O}/sub1fit-sat.json"))
            m = re.search(r"sub-1 anchor 149-sat: measured KL ratio ([\d.]+), model ([\d.]+)", txt)
            ok = (rc == 0 and rep and c is not None and fit.get("sat_c") and all("kld_sat" in l for l in lines) and m is not None
                  and abs(float(m.group(1)) - float(m.group(2))) <= 1e-3)
            why = f"synthetic sat anchor 20.0 -> C {c.group(1) if c else None}, kld_sat {[round(l.get('kld_sat', -1), 5) for l in lines]}"
        f = re.search(r"swj anchors gu: .*f\(K\) (\[.*?\])", txt)
        report(f"solve-v3-{tag}", ok, f"rc {rc}; {why}; anchors {[(i, m, round(mm, 4)) for i, m, mm, _ in an]}; f(K) {f.group(1) if f else '?'}; "
               f"swj factor on {kf.group(1) if kf else '?'} options, K {kf.group(3) if kf else '?'}..{kf.group(4) if kf else '?'}; "
               f"{[(l['target'], round(l['bpw'], 4), round(l['kld'], 5)) for l in lines]}")
print(f"KNAP CHECKS {sum(res)}/{len(res)} PASS")
sys.exit(0 if all(res) else 1)
