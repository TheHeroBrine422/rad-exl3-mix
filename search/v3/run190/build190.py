# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""build190.py PLAN.json WORK CALIB_STRATA.json XVIEW -- builds the s190a / s190b pair from draw190.py's plan (#190, RULES-cooldown
amendment 11-14 11:40 PM CST). build186.py without the t186n monitor: per part c (golden_pick on the calib golden, idx:4096), x
(pickx186 from the plan's explicit pieces, cut as pickx; the board gen mask; labels from the part's own tokens), then goldmix
(x / c interleaved, seed 1900 + k). Runs on ds8 (CPU, nice; the bf16 tokenizer for the x parts' labels). Restartable: a finished
golden (kld.json present) is skipped; parts are written as .part and renamed.
XVIEW: a dir with xc8k-b1/{kld.json, tokens.i32, top.bin, rows.f32, corpus.meta.jsonl} (the b1 golden + colo's meta)."""
import json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import strata186 as ST  # noqa: E402

PICKX = os.path.join(HERE, "..", "run186w", "pickx186.py")      # w164's pickx (run186w), unchanged
PLAN, W, CAL, XV = sys.argv[1:5]
CODE = os.environ.get("CODE", "/data/q27/w131/code")          # golden_pick.py (eval/)
PY = sys.executable; TOK = "/data/q27/models/qwen38-27b-bf16"
CALG = "/data/q27/challenge/golden/calib"; CORP = "/data/q27/challenge/corpus/calib.jsonl"
plan = json.load(open(PLAN)); os.makedirs(W, exist_ok=True)
calib = json.load(open(CAL))["docs"]
calib_model = json.load(open(f"{CALG}/kld.json"))["model"]
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


def build_x(name, plan_path):
    out = f"{W}/golden-{name}-x"
    if os.path.exists(f"{out}/kld.json"): return out
    shutil.rmtree(f"{out}.part", ignore_errors=True)
    stub = f"{W}/xstub-b1.jsonl"
    if not os.path.exists(stub):
        n = sum(1 for _ in open(f"{XV}/xc8k-b1/corpus.meta.jsonl")); open(stub, "w").write("{}\n" * n)
    run([PY, "-I", PICKX, f"{XV}/xc8k-b1", stub, f"{XV}/xc8k-b1/corpus.meta.jsonl", plan_path, name,
         f"{out}.part", f"{out}.part.jsonl"])
    m = json.load(open(f"{out}.part/kld.json")); m["model"] = calib_model; write_kld(f"{out}.part", m)
    run([PY, "-I", "/data/q27/w131/genmask181.py", "make", f"{out}.part", f"{out}.part/gen.u8.new", TOK])
    os.replace(f"{out}.part/gen.u8.new", f"{out}.part/gen.u8")
    from transformers import AutoTokenizer
    lab = ST.label_golden(f"{out}.part", AutoTokenizer.from_pretrained(TOK), f"{out}.part/kld.json.lab")
    os.replace(f"{out}.part/kld.json.lab", f"{out}.part/kld.json")
    if os.path.exists(f"{out}.part.jsonl"): os.replace(f"{out}.part.jsonl", f"{out}.jsonl")
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


for k, c in enumerate(plan["c"]):
    name = c["name"]
    gc = build_c(name, c["idx"])
    gx = build_x(name, PLAN)
    goldmix(name, gx, gc, 1900 + k)
say("BUILD DONE")
