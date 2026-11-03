#!/usr/bin/env python3
"""test_hgen.py MODEL META WORK -- CPU test of hgen.py / mkgenh.py on the real 0.8B, layers 0-1, the first qad documents
of at most 3k tokens that have both scored and unscored positions:
  1. H_ng + H_gen = H_all (fp32 sums of disjoint position sets) and their rows add up;
  2. rows_gen = the scored positions counted directly from gen.u8 (row by row, score_from respected);
  3. --exclude drops a golden's documents (the count changes by the documents it shares);
  4. mkgenh: w0 = 1 gives H_calib back, w0 = 0 gives H_calib - H_ng, rows follow; a starved diagonal is refused."""
import glob, json, os, subprocess, sys
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file
MODEL, META, WORK = sys.argv[1:4]; HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(WORK, exist_ok=True)
ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name} {info}", flush=True)
m = json.load(open(f"{META}/kld.json")); gen = np.fromfile(f"{META}/gen.u8", np.uint8).astype(bool)
pools = json.load(open(os.environ["POOLS"]))["pool"]
pick = []
for i, d in enumerate(m["docs"]):
    if pools[i] != "qad" or d["tokens"] > 3000: continue
    k = d["tokens"] - 1 - d["score_from"]; g = gen[d["row0"]:d["row0"] + k]
    if 0 < g.sum() < len(g): pick.append(i)
    if len(pick) == 6: break
# a pool file where only the picked docs are "qad": hgen.py selects exactly them
pv = ["verify"] * len(pools)
for i in pick: pv[i] = "qad"
json.dump({"pool": pv}, open(f"{WORK}/pools.json", "w"))
ENV = dict(os.environ, DEV="cpu", POOLS=f"{WORK}/pools.json")
def run(out, *extra):
    r = subprocess.run([sys.executable, f"{HERE}/../hgen.py", MODEL, META, out, "--pools", "qad", "--layers", "0-1", *extra],
                       env=ENV, capture_output=True, text=True)
    if r.returncode: print(r.stderr[-1500:])
    return r.returncode == 0
def load(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/*.input.safetensors")):
        with safe_open(f, "pt") as h: out[os.path.basename(f)[:-12]] = (h.get_tensor("hessian_upper").double(), int(h.metadata()["rows"]), int(h.metadata()["docs"]))
    return out
for w in ("ng", "gen", "all"): check(f"hgen --weight {w} runs", run(f"{WORK}/{w}", "--weight", w))
NGd, GEd, ALd = load(f"{WORK}/ng"), load(f"{WORK}/gen"), load(f"{WORK}/all")
check("8 groups each", len(NGd) == len(GEd) == len(ALd) == 8)
wr = max(float((NGd[g][0] + GEd[g][0] - ALd[g][0]).norm() / ALd[g][0].norm()) for g in ALd)
check("1. H_ng + H_gen = H_all", wr < 1e-5, f"(max rel {wr:.1e})")
check("1. rows add up", all(NGd[g][1] + GEd[g][1] == ALd[g][1] for g in ALd), f"({next(iter(NGd.values()))[1]} + {next(iter(GEd.values()))[1]})")
want = 0; tot = 0
for i in pick:
    d = m["docs"][i]; tot += d["tokens"]
    for j in range(d["tokens"] - 1 - d["score_from"]): want += int(gen[d["row0"] + j])
check("2. rows_gen = scored rows counted from gen.u8", all(v[1] == want for v in GEd.values()) and all(v[1] == tot for v in ALd.values()),
      f"({want} scored of {tot} positions in {len(pick)} docs)")
os.makedirs(f"{WORK}/gold", exist_ok=True)
json.dump({"docs": [{"orig": pick[0]}, {"orig": pick[2]}]}, open(f"{WORK}/gold/kld.json", "w"))
check("3. --exclude runs", run(f"{WORK}/ex", "--weight", "all", "--exclude", f"{WORK}/gold"))
EX = load(f"{WORK}/ex")
check("3. --exclude drops the golden's documents", all(v[2] == len(pick) - 2 for v in EX.values())
      and all(v[1] == tot - m["docs"][pick[0]]["tokens"] - m["docs"][pick[2]]["tokens"] for v in EX.values()))
# 4. mkgenh on the 'all' sums as a stand-in calib
cal = f"{WORK}/all"
r = subprocess.run([sys.executable, f"{HERE}/../mkgenh.py", cal, f"{WORK}/ng", f"{WORK}/mk", "1,0"], capture_output=True, text=True)
check("4. mkgenh runs", r.returncode == 0, r.stderr[-400:])
M1, M0 = load(f"{WORK}/mk/w1"), load(f"{WORK}/mk/w0")
e1 = max(float((M1[g][0] - ALd[g][0]).norm() / ALd[g][0].norm()) for g in ALd)
e0 = max(float((M0[g][0] - GEd[g][0]).norm() / GEd[g][0].norm()) for g in ALd)
check("4. w0 = 1: H_calib; w0 = 0: H_calib - H_ng (= H_gen here)", e1 < 1e-7 and e0 < 1e-4, f"(rel {e1:.1e}, {e0:.1e})")
check("4. rows follow", all(M1[g][1] == ALd[g][1] and M0[g][1] == GEd[g][1] for g in ALd))
os.makedirs(f"{WORK}/bad", exist_ok=True)
with safe_open(glob.glob(f"{WORK}/all/*.input.safetensors")[0], "pt") as h: up, md = h.get_tensor("hessian_upper"), h.metadata()
save_file({"hessian_upper": up}, f"{WORK}/bad/{os.path.basename(glob.glob(f'{WORK}/all/*.input.safetensors')[0])}", metadata=md)
r = subprocess.run([sys.executable, f"{HERE}/../mkgenh.py", f"{WORK}/all", f"{WORK}/bad", f"{WORK}/mkbad", "-1"], capture_output=True, text=True)
check("4. a non-positive diagonal is refused (w0 = -1: calib - 2 x calib)", r.returncode != 0 and "AssertionError" in r.stderr)
print("ALL PASS" if ok else "SOME FAIL")
