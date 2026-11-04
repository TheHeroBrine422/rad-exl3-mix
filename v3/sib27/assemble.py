#!/usr/bin/env python3
"""assemble.py -- a level = a base level with some tensors replaced by a partial requant (ourq.py --only).

  assemble.py build BASE PART OUT KEYFILE   OUT/q-L*.safetensors = BASE's shards with every tensor of PART (an ourq
      --only export) put in place of BASE's: same keys, shapes and dtypes required, and the replaced linears must be
      exactly KEYFILE's (a writer missing from PART, or an extra one, is an error). BASE's config.json, an index for
      the new shards, assembled.json (provenance: both dirs, their stamps, the keys). Per shard, resumable.
  assemble.py same BASE PART                every tensor of PART bit-identical to BASE's (the no-op gate: PART = the
      writers requantized with BASE's own settings, seeds and calib H); exit 1 otherwise."""
import json, os, struct, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def index(d): return json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
def meta(p):
    with open(p, "rb") as fh: n = struct.unpack("<Q", fh.read(8))[0]; return json.loads(fh.read(n)).get("__metadata__") or {}
def stamp(d):
    s = f"{d}/stamp.json"; return json.load(open(s))[-1]["stamp"] if os.path.exists(s) else None
def codebook(d):
    s = f"{d}/stamp.json"
    if os.path.exists(s): return json.load(open(s))[-1]["args"].get("codebook", "mul1")
    s = f"{d}/assembled.json"; return json.load(open(s))["codebook"] if os.path.exists(s) else None
def tensors(d, ix, keys):
    out, byf = {}, {}
    for k in keys: byf.setdefault(ix[k], []).append(k)
    for f, ks in byf.items():
        with safe_open(f"{d}/{f}", "pt", device="cpu") as h:
            for k in ks: out[k] = h.get_tensor(k)
    return out


cmd = sys.argv[1]
if cmd == "same":
    B, P = sys.argv[2:4]; bi, pi = index(B), index(P)
    miss = sorted(set(pi) - set(bi))
    if miss: sys.exit(f"{len(miss)} tensors of {P} not in {B}: {miss[:3]}")
    bad, n = [], 0
    for f in sorted(set(pi.values())):
        ks = [k for k, v in pi.items() if v == f]
        pt, bt = tensors(P, pi, ks), tensors(B, bi, ks)
        for k in ks:
            n += 1
            if pt[k].dtype != bt[k].dtype or pt[k].shape != bt[k].shape or not torch.equal(pt[k], bt[k]): bad.append(k)
    lin = sorted({k.rsplit(".", 1)[0] for k in pi})
    print(f"noop gate: {len(lin)} linears, {n} tensors, {len(bad)} differ from {B} {bad[:3]} -> {'PASS' if n and not bad else 'FAIL'}")
    sys.exit(1 if bad or not n else 0)

assert cmd == "build", __doc__
B, P, O, KF = sys.argv[2:6]
keys = {l.strip() for l in open(KF) if l.strip() and not l.startswith("#")}
bi, pi = index(B), index(P)
plin = {k.rsplit(".", 1)[0] for k in pi}
if plin != keys:
    sys.exit(f"{P} holds {len(plin)} linears, KEYFILE {len(keys)}: missing {sorted(keys - plin)[:3]}, extra {sorted(plin - keys)[:3]}")
miss = sorted(set(pi) - set(bi))
if miss: sys.exit(f"{len(miss)} tensors of {P} not in {B}: {miss[:3]}")
for lin in plin:                      # the base's tensors of a replaced linear are all replaced (no stale suh / svh)
    bk = {k for k in bi if k.rsplit(".", 1)[0] == lin}; pk = {k for k in pi if k.rsplit(".", 1)[0] == lin}
    if bk != pk: sys.exit(f"{lin}: base tensors {sorted(bk)} != part's {sorted(pk)}")
if codebook(B) != codebook(P): sys.exit(f"codebooks differ: {B} {codebook(B)}, {P} {codebook(P)}")
os.makedirs(O, exist_ok=True)
prov = {"codebook": codebook(B), "base": os.path.abspath(B), "part": os.path.abspath(P), "base_stamp": stamp(B), "part_stamp": stamp(P),
        "keys": sorted(keys), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
tag = json.dumps({k: prov[k] for k in ("base", "part", "base_stamp", "part_stamp")}, sort_keys=True)
nrep = 0
for f in sorted(set(bi.values())):
    out = f"{O}/{f}"
    if os.path.exists(out) and meta(out).get("sib27_assembled") == tag: continue
    ks = [k for k, v in bi.items() if v == f]
    t = tensors(B, bi, ks); rep = [k for k in ks if k in pi]
    if rep:
        new = tensors(P, pi, rep)
        for k in rep:
            assert new[k].shape == t[k].shape and new[k].dtype == t[k].dtype, (k, new[k].shape, t[k].shape, new[k].dtype, t[k].dtype)
            t[k] = new[k]
    nrep += len(rep)
    save_file({k: v.contiguous() for k, v in t.items()}, out + ".part", metadata=dict(meta(f"{B}/{f}"), sib27_assembled=tag))
    os.replace(out + ".part", out)
json.dump({"metadata": {}, "weight_map": bi}, open(f"{O}/model.safetensors.index.json", "w"), indent=1)
for x in ("config.json",):
    if os.path.exists(f"{B}/{x}"): open(f"{O}/{x}", "w").write(open(f"{B}/{x}").read())
json.dump(prov, open(f"{O}/assembled.json", "w"), indent=1)
print(f"ASSEMBLED {O}: {len(set(bi.values()))} shards, {len(plin)} linears ({nrep} tensors this run) from {P} over {B}")
