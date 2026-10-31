#!/usr/bin/env python3
"""abreport.py -- the knapsack check, with per-document bootstrap intervals (2000 resamples, paired where two runs
share a slice). Run on the GPU box from /root after the calib-A pass.

  1. calib A (the half the sensitivities were fitted on): measured KLD of every corrected mix and reference vs the
     model's prediction (frontier-fix.txt for mixes, KNAP_PRED for containers) -> the model's own error.
  2. calib B (held out): each mix vs the reference container of nearest size, paired dKLD with a 95% interval,
     and both put on a common size axis (log-KLD interpolated between references) -> does mixing beat the exports."""
import json, os, re, subprocess, sys
import numpy as np
# paths (README: Paths)
RXM_WORK = os.environ.get("RXM_WORK", "/root")

R = RXM_WORK
res = {}
for l in open(f"{R}/evq/results.jsonl"):
    r = json.loads(l)
    if "fail" not in r: res[r["tag"]] = r
PARAMS = 26895998464

def docsum(golden, tag):
    m = json.load(open(f"{R}/{golden}/kld.json"))
    s = np.fromfile(f"{R}/evq/scores/{tag}.f32", np.float32).reshape(-1, 4)[:, 0].astype(np.float64)
    gen = np.fromfile(f"{R}/{golden}/gen.u8", np.uint8).astype(bool)
    k, n = [], []
    for d in m["docs"]:
        a = d["row0"]; b = a + d["tokens"] - 1 - d["score_from"]; mk = gen[a:b]
        k.append(s[a:b][mk].sum()); n.append(mk.sum())
    return np.array(k), np.array(n)

rng = np.random.default_rng(0)
B = 2000
def ci_mean(k, n):
    idx = rng.integers(0, len(n), (B, len(n)))
    v = k[idx].sum(1) / n[idx].sum(1)
    return k.sum() / n.sum(), np.percentile(v, 2.5), np.percentile(v, 97.5)
def ci_rel(ka, kb, n):   # (B - A) / A, paired
    idx = rng.integers(0, len(n), (B, len(n)))
    v = kb[idx].sum(1) / ka[idx].sum(1) - 1
    return kb.sum() / ka.sum() - 1, np.percentile(v, 2.5), np.percentile(v, 97.5)

# predictions
pred = {}
for l in open(f"{R}/knap/frontier-fix.txt"):
    m = re.match(r"(knap-fix-\S+)\s+bpw ([\d.]+)\s+pred KLD ([\d.]+)", l)
    if m: pred["a-" + m.group(1)] = (float(m.group(2)), float(m.group(3)))
refs = ["sc2.00-e3", "plain2.00-e3", "plain2.50-e3", "sc3.00-e4", "plain3.50-e4", "sc4.00-e4", "plain4.00-e4", "sc5.00-e5", "sc6.00-e6"]
env = dict(os.environ, KNAP_TAG="chk-", KNAP_PRED=",".join(refs))
out = subprocess.run([sys.executable, "-I", f"{R}/knap/knap.py", f"{R}/knap", "3.0"], env=env, capture_output=True, text=True).stdout
for l in out.splitlines():
    if l.startswith("{"):
        r = json.loads(l); pred["a-ref-" + r["c"]] = (None, r["pred_kld"])
for f in os.listdir(f"{R}/knap/mixes"):
    if f.startswith("knap-chk-"): os.remove(f"{R}/knap/mixes/{f}")
res.setdefault("a-ref-sc4.00-e4", res.get("base-sc4.00-e4"))
if "a-ref-sc4.00-e4" in res and not os.path.exists(f"{R}/evq/scores/a-ref-sc4.00-e4.f32"):
    os.symlink(f"{R}/evq/scores/base-sc4.00-e4.f32", f"{R}/evq/scores/a-ref-sc4.00-e4.f32")

print("== 1. calib A: model prediction vs measurement (95% CI over documents)")
print(f"{'tag':34} {'bpw':>6} {'pred':>9} {'measured':>9} {'95% CI':>21} {'pred/meas':>9}")
for tag in sorted(pred, key=lambda t: res[t]["bytes"] if t in res else 0):
    if tag not in res: continue
    k, n = docsum("golden-calA", tag); m, lo, hi = ci_mean(k, n); p = pred[tag][1]
    print(f"{tag[2:]:34} {res[tag]['bytes']*8/PARAMS:6.3f} {p:9.5f} {m:9.5f} [{lo:9.5f},{hi:9.5f}] {p/m:9.2f}")

print("\n== 2. calib B: mixes vs reference containers (paired, 95% CI over documents)")
rb = {t: res["v-ref-" + t] for t in refs if "v-ref-" + t in res}
rbpw = sorted((v["bytes"] * 8 / PARAMS, v["kld"], t) for t, v in rb.items())
def ref_at(b):   # log-KLD interpolation over the reference containers' frontier
    xs = [x for x, _, _ in rbpw]; ys = [np.log(y) for _, y, _ in rbpw]
    return float(np.exp(np.interp(b, xs, ys)))
print(f"{'mix':26} {'bpw':>6} {'KLD':>9} {'refs@bpw':>9} {'vs refs':>8} | {'nearest ref (<= bpw)':20} {'paired dKLD':>10} {'95% CI':>17}")
for tag in sorted([t for t in res if t.startswith("v-knap-fix-")], key=lambda t: res[t]["bytes"]):
    b = res[tag]["bytes"] * 8 / PARAMS; kk = res[tag]["kld"]
    cand = [r for r in rbpw if r[0] <= b + 0.02] or rbpw[:1]
    rt = cand[-1][2]
    ka, n = docsum("golden-calB", "v-ref-" + rt); kb, _ = docsum("golden-calB", tag)
    d, lo, hi = ci_rel(ka, kb, n)
    print(f"{tag[11:]:26} {b:6.3f} {kk:9.5f} {ref_at(b):9.5f} {kk/ref_at(b)-1:+8.1%} | {rt:13} {rb[rt]['bytes']*8/PARAMS:6.3f} {d:+10.1%} [{lo:+7.1%},{hi:+7.1%}]")
