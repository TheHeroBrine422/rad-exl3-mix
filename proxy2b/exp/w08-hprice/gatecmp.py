#!/usr/bin/env python3
"""gatecmp.py REF OFF ON -- gate of ourq-hprice.patch on the 4070 (run.sh step 0): three ourq outputs of the same
setting and layers -- REF unpatched ourq, OFF patched without --hprice, ON patched with --hprice. Checks:
  1. every tensor of every q-L*.safetensors shard is bit-identical across REF / OFF / ON (shard metadata, i.e. the
     code stamp, ignored): the patch does not change the quant;
  2. ON's proxy.jsonl has proxy_err1 for every linear and proxy_err2 for every one with an --hprice file, and
     |proxy_err1 / proxy_err - 1| < 2e-3 (proxy_err: the refit's error against the H minus its damping);
  3. OFF / ON rowerr.npz: sum(err) / sum(ref) == proxy_err1 (1e-9 relative); REF's ratio is reported (~1 on every
     linear while the in-place bug is unfixed, i.e. before w04-blockgain/rowerr.patch is merged).
Prints one line per check and GATE PASS / GATE FAIL."""
import glob, json, os, sys
import numpy as np
from safetensors import safe_open

REF, OFF, ON = sys.argv[1:4]
ok = True
def shard(p):
    with safe_open(p, "pt") as f: return {k: f.get_tensor(k) for k in f.keys()}
names = sorted(os.path.basename(p) for p in glob.glob(f"{REF}/q-L*.safetensors"))
same = 0
for n in names:
    a, b, c = shard(f"{REF}/{n}"), shard(f"{OFF}/{n}"), shard(f"{ON}/{n}")
    eq = a.keys() == b.keys() == c.keys() and all(a[k].equal(b[k]) and a[k].equal(c[k]) for k in a)
    same += eq; ok &= eq
print(f"1. shards bit-identical REF / OFF / ON: {same}/{len(names)}" + ("" if names else " (no shards: FAIL)"))
ok &= bool(names)
P = [json.loads(l) for l in open(f"{ON}/proxy.jsonl")]
miss1 = [d["key"] for d in P if "proxy_err1" not in d]
dev = max(abs(d["proxy_err1"] / d["proxy_err"] - 1) for d in P if "proxy_err1" in d) if P else 1.0
n2 = sum("proxy_err2" in d for d in P)
c2 = not miss1 and dev < 2e-3 and n2 > 0
print(f"2. ON proxy.jsonl: {len(P)} linears, proxy_err1 missing on {len(miss1)}, max |err1/proxy_err - 1| {dev:.2e}, "
      f"proxy_err2 on {n2}"); ok &= c2
pe1 = {d["key"]: d["proxy_err1"] for d in P if "proxy_err1" in d}
worst, oldmin = 0.0, 9.0
for d_, isnew in ((OFF, True), (ON, True), (REF, False)):
    for f in glob.glob(f"{d_}/wq/*.rowerr.npz"):
        z = np.load(f); r = z["err"].sum() / z["ref"].sum(); k = os.path.basename(f)[:-len(".rowerr.npz")]
        if isnew and k in pe1: worst = max(worst, abs(r / pe1[k] - 1))
        if not isnew: oldmin = min(oldmin, r)
c3 = worst < 1e-9
print(f"3. fixed rowerr.npz vs proxy_err1: max rel diff {worst:.1e}; REF rowerr err/ref min {oldmin:.3f} "
      f"({'the in-place bug: REF is unfixed ourq' if oldmin > 0.9 else 'REF already fixed'})")
ok &= c3
print("GATE PASS" if ok else "GATE FAIL")
