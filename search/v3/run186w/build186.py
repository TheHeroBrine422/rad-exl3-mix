# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""build186.py PLAN.json WORK CALIB_STRATA.json XVIEW -- builds s186a-f and t186n from draw186.py's plan (#186, RULES-strata
§3 / §7). Runs on ds8 (CPU, nice; the bf16 tokenizer for the x parts' labels). Restartable: a finished golden (kld.json
present) is skipped; parts are written as .part and renamed. Per slice:
  x  pickx186.py (the explicit pieces, cut as pickx) -> the calib model name -> genmask181 make (board gen mask) -> labels
     (strata186.label_golden on the part's own tokens)
  c  golden_pick.py on the calib golden (idx:4096, as draw171) -> labels copied from the calib labels by orig
  goldmix (x / c interleaved; seed 1860 + k; t186n: 1869) -> WORK/golden-NAME
XVIEW: a dir with xc8k-b1/{kld.json, tokens.i32, top.bin, rows.f32, corpus.meta.jsonl} (the b1 golden + colo's meta)."""
import json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import strata186 as ST  # noqa: E402

PLAN, W, CAL, XV = sys.argv[1:5]
CODE = os.environ.get("CODE", "/data/q27/w131/code")          # golden_pick.py (eval/)
PY = sys.executable; TOK = "/data/q27/models/qwen38-27b-bf16"
CALG = "/data/q27/challenge/golden/calib"; CORP = "/data/q27/challenge/corpus/calib.jsonl"
HEAVY = {"agentic", "tools", "long", "reasoning"}
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
    run([PY, "-I", f"{HERE}/pickx186.py", f"{XV}/xc8k-b1", stub, f"{XV}/xc8k-b1/corpus.meta.jsonl", plan_path, name,
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


def caps_for(idx):
    """pickx's cut for the heavy categories: scored positions beyond 2,048 are not kept"""
    X = [json.loads(l) for l in open(f"{XV}/xc8k-b1/corpus.meta.jsonl")]; D = json.load(open(f"{XV}/xc8k-b1/kld.json"))["docs"]
    return {str(i): 2048 for i in idx if X[i]["category"] in HEAVY and D[i]["tokens"] - 1 - D[i]["score_from"] > 2048}


for k, c in enumerate(plan["c"]):
    name = c["name"]
    gc = build_c(name, c["idx"])
    gx = build_x(name, PLAN)
    goldmix(name, gx, gc, 1860 + k)
t = plan["t186n"]
plan_t = dict(plan); plan_t["x"] = [{"name": "t186n", "idx": t["xcorp"], "caps": caps_for(t["xcorp"]), "N": t["xcorp"]}]
tp = f"{W}/plan-t186n.json"; json.dump(plan_t, open(tp, "w"))
gcn = build_c("t186n", t["calib"]); gxn = build_x("t186n", tp)
goldmix("t186n", gxn, gcn, 1869)
say("BUILD DONE")
