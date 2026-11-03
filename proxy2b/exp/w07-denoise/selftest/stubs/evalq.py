#!/usr/bin/env python3
"""evalq.py stub: rows of [kl, top1, gen, 0] whose level depends on the wq dir's name (tag): base 1.0, every arm tag a
deterministic factor in 0.95..1.03, seeds +-1%"""
import hashlib, json, os, sys, numpy as np
def factor(wq):
    tag = os.path.basename(os.path.dirname(os.path.abspath(wq))).replace("q-", "")
    s = 1.0 + 0.01 * (("_s1" in tag) - ("_s2" in tag))
    if "base" in tag or tag == "noop": return s
    return s * (0.95 + 0.08 * (int(hashlib.md5(tag.split("_s")[0].encode()).hexdigest(), 16) % 1000) / 1000)
def score(gd, wq, out):
    m = json.load(open(f"{gd}/kld.json")); r = m["rows"]; rng = np.random.default_rng(7)
    base = rng.gamma(2.0, 0.05, r); gen = (rng.random(r) < 0.8).astype(np.float32)
    kl = (base * factor(wq) * (1 + 0.005 * np.random.default_rng(abs(hash(wq)) % 2**32).standard_normal(r))).astype(np.float32)
    np.stack([kl, np.zeros(r, np.float32), gen, np.zeros(r, np.float32)], 1).astype(np.float32).tofile(out)
    return float(kl[gen > 0].mean())
if __name__ == "__main__":
    a = sys.argv[1:]; wq = a[a.index("--wq") + 1]; out = a[a.index("--out") + 1]; gd = a[1]
    assert len([f for f in os.listdir(wq) if f.endswith(".safetensors")]) == 150, f"{wq}: not 150 tensors"
    print(json.dumps({"kld": score(gd, wq, out), "top1": 0.9}))
