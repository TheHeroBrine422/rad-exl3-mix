# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""build190ef.py PLAN.json WORK CALIB_STRATA.json XVIEW [--x-only] -- builds the s190e / s190f parts from plan190ef.json (#190-03,
w179; build190cd.py's build with pickx190ef.py, the copy of pickx190cd.py). The goldmix seeds are 1903 + k. Steps per part: x (pickx190ef, then the board
gen mask by genmask181.py on the part, then the labels from the part's own tokens: strata186.label_golden); c (golden_pick on the calib
golden, idx:4096, as build190) and the goldmix (x / c interleaved, seed 1900 + k) only without --x-only and only when the plan's
c part is not blocked. Restartable: a finished golden (kld.json present) is skipped; parts are written as .part and renamed.
XVIEW: the xview dir (per batch: kld.json, tokens.i32, top.bin, rows.f32, corpus.meta.jsonl; w164/xview)."""
import json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import strata186 as ST  # noqa: E402

PICKX = os.path.join(HERE, "pickx190ef.py")
args = [a for a in sys.argv[1:] if a != "--x-only"]; X_ONLY = "--x-only" in sys.argv[1:]
PLAN, W, CAL, XV = args[:4]
CODE = os.environ.get("CODE", "/data/q27/w131/code")          # golden_pick.py (eval/)
PY = sys.executable; TOK = "/data/q27/models/qwen38-27b-bf16"; GENMASK = "/data/q27/w131/genmask181.py"
CALG = "/data/q27/challenge/golden/calib"; CORP = "/data/q27/challenge/corpus/calib.jsonl"
plan = json.load(open(PLAN)); os.makedirs(W, exist_ok=True)
calib = json.load(open(CAL))["docs"]
calib_kld = f"{CALG}/kld.json"
log = open(f"{W}/build.log", "a")


def say(msg):
    print(msg, flush=True); log.write(msg + "\n"); log.flush()


def run(cmd):
    say("$ " + " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode: raise SystemExit(f"failed ({r.returncode}): {r.stdout[-2000:]}")
    return r.stdout


def write_kld(d, m):
    json.dump(m, open(f"{d}/kld.json.part", "w"), indent=0); os.replace(f"{d}/kld.json.part", f"{d}/kld.json")


def build_x(name):
    out = f"{W}/golden-{name}-x"
    if os.path.exists(f"{out}/kld.json"): return out
    shutil.rmtree(f"{out}.part", ignore_errors=True)
    run([PY, "-I", PICKX, PLAN, name, XV, f"{out}.part", f"{out}.part.jsonl", calib_kld])
    run([PY, "-I", GENMASK, "make", f"{out}.part", f"{out}.part/gen.u8.new", TOK])
    os.replace(f"{out}.part/gen.u8.new", f"{out}.part/gen.u8")
    from transformers import AutoTokenizer
    lab = ST.label_golden(f"{out}.part", AutoTokenizer.from_pretrained(TOK), f"{out}.part/kld.json.lab")
    os.replace(f"{out}.part/kld.json.lab", f"{out}.part/kld.json")
    os.replace(f"{out}.part.jsonl", f"{out}.jsonl")
    os.replace(f"{out}.part", out)
    say(f"{out}: {len(lab['docs'])} pieces, N {lab['strata']['n_N']}")
    return out


def build_c(name, idx):
    out = f"{W}/golden-{name}-c"
    if os.path.exists(f"{out}/kld.json"): return out
    shutil.rmtree(f"{out}.part", ignore_errors=True)
    run([PY, "-I", f"{CODE}/eval/golden_pick.py", CALG, CORP, f"{out}.part", f"{out}.jsonl", *[f"{i}:4096" for i in idx]])
    m = json.load(open(f"{out}.part/kld.json"))
    for d in m["docs"]:
        cd = calib[d["orig"]]
        assert d["tokens"] <= min(cd["tokens"], 4096), (name, d["orig"])
        d["stratum"], d["script"] = cd["stratum"], cd["script"]
    write_kld(f"{out}.part", m)
    os.replace(f"{out}.part", out)
    say(f"{out}: {len(m['docs'])} calib docs, N {sum(d['stratum'] == 'N' for d in m['docs'])}")
    return out


def goldmix(name, gx, gc, seed):
    out = f"{W}/golden-{name}"
    if os.path.exists(f"{out}/kld.json"): return out
    run([PY, "-I", f"{HERE}/../slices.py", "goldmix", out, str(seed), f"x={gx}", f"c={gc}"])
    return out


for k, p in enumerate(plan["x"]):
    build_x(p["name"])
if not X_ONLY:
    for k, c in enumerate(plan["c"]):
        if c.get("blocked"): raise SystemExit(f"{c['name']}: c part blocked ({c['blocked']}); build with --x-only, or after the N decision")
        gc = build_c(c["name"], c["idx"]); gx = f"{W}/golden-{c['name']}-x"
        goldmix(c["name"], gx, gc, 1903 + k)
say("BUILD DONE" + (" (x only)" if X_ONLY else ""))
