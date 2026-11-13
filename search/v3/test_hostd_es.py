#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""test_hostd_es.py PIPELINE_DIR -- offline check of hostd-earlystop.patch (w126, #166): rxrun.hostd.run_eval with
stub pack / mix / score commands; the stub scorer behaves like radiance --kld-progress (documents in golden order, a
progress line + the document's per-position KL in FILE.kl as each finishes, ~50 ms a document; the rows file at the
end). Per-position KL = a base row (from the document's tokens) x the candidate's factor (from its tag: inc 1.0,
bad 1.15, good 0.9) x a small per-document jitter. Checks:
  incumbent     scored without early stop (no early_stop block): same result as score_cmd alone
  bad           stopped early by the rule (the run killed: no rows file, docs after the stop are null, "stopped" set,
                kld / positions over the scored documents == the progress sums)
  good          not stopped: the result (kld, p99, top1, docs) equals the plain score_cmd run's; progress sums == the
                rows file's sums
  copy          pipeline/rxrun/earlystop.py == search/v3/seqrule.py (byte for byte)
PIPELINE_DIR: a pipeline/ tree with the patch applied."""
import hashlib, json, os, shutil, sys, tempfile, time

import numpy as np

PIPE = os.path.abspath(sys.argv[1]); sys.path.insert(0, PIPE)
from rxrun import hostd   # noqa: E402

W = tempfile.mkdtemp(prefix="w126-hostd-")
STUB = f"{W}/stub.py"
open(STUB, "w").write(r"""import hashlib, json, os, sys, time
import numpy as np
op = sys.argv[1]
if op == "pack": open(sys.argv[3], "w").write("pack of " + open(sys.argv[2]).read()[:200])
elif op == "mix": open(sys.argv[2], "w").write(open(sys.argv[3]).read() + open(sys.argv[4]).read())
else:
    mix, g, o, card = sys.argv[2:6]; prog = sys.argv[6] if len(sys.argv) > 6 else None
    tag = os.path.basename(mix); f = 1.15 if "bad" in tag else 0.9 if "good" in tag else 1.0
    m = json.load(open(g + "/kld.json")); tok = np.fromfile(g + "/tokens.i32", np.int32)
    R = np.zeros((m["rows"], 4), np.float32); t0 = 0
    for i, d in enumerate(m["docs"]):
        n = d["tokens"] - 1 - d["score_from"]; t = tok[t0:t0 + d["tokens"]]; t0 += d["tokens"]
        rng = np.random.default_rng(int(hashlib.sha256(t.tobytes()).hexdigest()[:8], 16))
        jit = np.random.default_rng(int(hashlib.sha256(t.tobytes() + tag.encode()).hexdigest()[:8], 16)).lognormal(0, 0.03)
        R[d["row0"]:d["row0"] + n, 0] = rng.gamma(0.5, 0.2, n) * f * jit; R[d["row0"]:d["row0"] + n, 3] = rng.random(n) < 0.9
        if prog:
            kl = R[d["row0"]:d["row0"] + n, 0].copy(); off = os.path.getsize(prog + ".kl") if os.path.exists(prog + ".kl") else 0
            with open(prog + ".kl", "ab") as fk: fk.write(kl.tobytes())
            with open(prog, "a") as fp: fp.write(json.dumps({"doc": i, "row0": d["row0"], "rows": n, "kl_sum": float(kl.astype(np.float64).sum()), "top1": 0, "kl_off": off}) + "\n")
            time.sleep(0.05)
    R.tofile(o + ".json.part.rows"); os.replace(o + ".json.part.rows", o + ".json.rows")
    sel = np.concatenate([np.arange(d["row0"], d["row0"] + d["tokens"] - 1 - d["score_from"]) for d in m["docs"]])
    json.dump({"all": {"positions": int(len(sel)), "top1_agreement": float(R[sel, 3].astype(np.float64).mean()),
                       "kld": {"mean": float(R[sel, 0].mean())}}}, open(o + ".json", "w"))
""")


def golden(path, nd, seed, top=4):
    rng = np.random.default_rng(seed); os.makedirs(path); docs, r, toks = [], 0, []
    for i in range(nd):
        T = int(rng.integers(50, 400)); sf = int(rng.integers(0, 20)) if i % 3 == 0 else 0
        toks.append(rng.integers(0, 1000, T).astype(np.int32)); docs.append({"source": "t", "tokens": T, "score_from": sf, "row0": r, "orig": i, "part": "xc"[i % 2]})
        r += T - 1 - sf
    np.concatenate(toks).tofile(f"{path}/tokens.i32")
    (rng.random(r) < 0.8).astype(np.uint8).tofile(f"{path}/gen.u8")
    rng.integers(0, 255, r * top * 8).astype(np.uint8).tofile(f"{path}/top.bin")
    rng.random(r * 3).astype(np.float32).tofile(f"{path}/rows.f32")
    json.dump({"format": "radiance-kld-top-1", "model": "t", "top": top, "n_vocab": 1000, "rows": r, "docs": docs}, open(f"{path}/kld.json", "w"))
    return docs


DOCS = golden(f"{W}/gm", 60, 1)
LV = "lvA"
os.makedirs(f"{W}/ladder/{LV}"); json.dump({"level": LV}, open(f"{W}/ladder/{LV}/DONE", "w"))
open(f"{W}/ladder/{LV}/final.recipe", "w").write("recipe A\n"); open(f"{W}/base.rad", "w").write("base\n")


class FakeHost:
    def __init__(self, root, E):
        self.root = root; self.name = "t"; self.rc = {"root": f"{W}/ladder"}; self.hc = {"eval": E}

    def current(self, n): return n == LV


def E_(root, prog=True):
    E = {"packs": f"{W}/packs", "mixes": f"{root}/mixes", "scores": f"{root}/scores", "aliases": {"GM": f"{W}/gm"},
         "card": 7, "card_lock": f"{W}/card7.lock", "ram_lock": f"{W}/ram.lock", "conv_lock": f"{W}/conv.lock",
         "pack_cmd": [sys.executable, STUB, "pack", "{recipe}", "{out}"], "mix_cmd": [sys.executable, STUB, "mix", "{out}", "{base}", "{mixfile}"],
         "score_cmd": [sys.executable, STUB, "score", "{mix}", "{golden}", "{out}", "{card}"]}
    if prog: E["score_cmd_progress"] = [sys.executable, STUB, "score", "{mix}", "{golden}", "{out}", "{card}", "{progress}"]
    return E


ok = True


def check(name, cond, msg=""):
    global ok
    ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'} {name} {msg}")


def run(tag, root, es=None, prog=True):
    os.makedirs(f"{root}/unit-{tag}", exist_ok=True)
    req = {"tag": tag, "mix": {"base": f"{W}/base.rad", "trunk": LV}, "slices": [{"name": "s", "parts": {"m": "GM"}}]}
    if es: req["early_stop"] = es
    return hostd.run_eval(FakeHost(root, E_(root, prog)), f"{root}/unit-{tag}", req)


inc = run("inc", f"{W}/a"); inc_plain = run("inc", f"{W}/p", prog=False)
check("incumbent rc", inc["rc"] == 0 and inc_plain["rc"] == 0)
check("incumbent == plain", inc["slices"]["s"]["kld"] == inc_plain["slices"]["s"]["kld"] and inc["slices"]["s"]["parts"]["m"]["docs"] == inc_plain["slices"]["s"]["parts"]["m"]["docs"])
ref = inc["slices"]["s"]["parts"]["m"]["docs"]
es = {"rule": {"kind": "curve", "p": 2.0, "warmup": 0.05}, "ref": ref, "parts": [d["part"] for d in DOCS]}
t0 = time.time(); bad = run("bad", f"{W}/a", es); tb = time.time() - t0
sb = bad["slices"]["s"]; st = sb.get("stopped")
check("bad: stopped early", bad["rc"] == 0 and st is not None and st["k"] < len(DOCS), f"stopped {st and st['k']} of {len(DOCS)} docs in {tb:.1f} s, z {st and round(st['z'], 2)}")
d = sb["parts"]["m"]["docs"]
check("bad: docs after the stop are null, before are set", all(x is not None for x in d[:st["k"]]) and all(x is None for x in d[st["k"]:]))
check("bad: run killed (no rows file)", not os.path.exists(f"{W}/a/scores/bad-s-m.json.rows"))
S = sum(x[0] for x in d[:st["k"]]); n = sum(x[1] for x in d[:st["k"]])
check("bad: kld / positions over the scored documents", abs(sb["kld"] - S / n) < 1e-12 and sb["positions"] == n)
# what the docs would be without stopping (plain run) -> the scored prefix equals it
badp = run("bad", f"{W}/p", prog=False)["slices"]["s"]["parts"]["m"]["docs"]
check("bad: scored prefix == the plain run's per-document sums", d[:st["k"]] == badp[:st["k"]])
good = run("good", f"{W}/a", es); goodp = run("good", f"{W}/p", prog=False)
g, gp = good["slices"]["s"], goodp["slices"]["s"]
check("good: not stopped", "stopped" not in g)
check("good: result == plain run", all(g[k] == gp[k] for k in ("kld", "p99", "top1", "positions")) and g["parts"]["m"]["docs"] == gp["parts"]["m"]["docs"])
check("good: no progress-mismatch warning", "WARN progress" not in open(f"{W}/a/unit-good/eval.log").read())
here = os.path.dirname(os.path.abspath(__file__))
check("copy: rxrun/earlystop.py == search/v3/seqrule.py", open(f"{PIPE}/rxrun/earlystop.py", "rb").read() == open(f"{here}/seqrule.py", "rb").read())
print("ALL PASS" if ok else "FAILED", W)
if ok: shutil.rmtree(W)
sys.exit(0 if ok else 1)
