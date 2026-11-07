#!/usr/bin/env python3
"""radiff.py ARM.rad CTL.rad SCALESDIR -- w75 #112-02's identity gate (../w75.md): the arm's container equals the
control's except the svh plane of exactly the writers SCALESDIR/<hf key>.safetensors names, and those svh values are
f16(svh_ctl * c) (swjapply.py's arithmetic on the shard; c from the scales file).
Per weight (census.py's reader, /data/swap27/tools): the same name set, shape, quantiser options (but the recipe's
src= dir), encoding struct and every plane's
relative offset / size / bytes. A changed writer may differ in its svh plane only; its decoded f16 svh must match
f16(f32(svh_ctl) * c) exactly on >= 99.9% of rows and within 1 ulp on all (else FAIL). Weights with a scales file whose
c is all 1 must be unchanged. Last line: "RADIFF PASS|FAIL {json}"; exit 1 on FAIL."""
import hashlib, json, os, struct, sys
import numpy as np
sys.path.insert(0, "/data/swap27/tools")
import census as C
from safetensors.numpy import load_file

arm, ctl, sd = sys.argv[1:4]
A, B = C.Rad(arm), C.Rad(ctl)


def hf2rad(key):                                    # model.language_model.layers.L.<mod>.<lin> -> blk.L.<name>.weight
    p = key.split("."); l, rest = p[3], ".".join(p[4:6])
    return f"blk.{l}." + {"self_attn.o_proj": "attn_output", "linear_attn.out_proj": "ssm_out", "mlp.down_proj": "ffn_down"}[rest] + ".weight"


scales = {}
for f in os.listdir(sd):
    if f.endswith(".safetensors"):
        c = load_file(f"{sd}/{f}")["c"]; scales[hf2rad(f[:-len(".safetensors")])] = c
changed_expected = {n for n, c in scales.items() if not np.all(c == 1)}


def planes(R, n):
    w = R.w[n]; eb = w["enc"]; k = struct.unpack_from("<I", eb, 48)[0]
    roles = [C.PLANE.unpack_from(eb, 56 + 64 * i)[0].split(b"\0")[0].decode() for i in range(k)]
    out = []
    for role, (po, pn) in zip(roles, w["planes"]):                    # sha256 in 16 MiB chunks; raw bytes for svh only
        R.f.seek(po); h = hashlib.sha256(); left = pn; raw = b""
        while left:
            b = R.f.read(min(C.CH, left)); h.update(b); left -= len(b)
            if role == "svh": raw += b
        out.append((role, po - w["offset"], pn, h.digest(), raw))
    return out


nosrc = lambda o: ",".join(t for t in o.split(",") if not t.startswith("src="))   # the recipe's source dir differs by design
bad, nsame, nchg, worst_ulp, exact = [], 0, 0, 0, []
if set(A.w) != set(B.w): bad.append(f"name sets differ: {sorted(set(A.w) ^ set(B.w))[:4]}")
for n in sorted(set(A.w) & set(B.w)):
    wa, wb = A.w[n], B.w[n]
    if wa["shape"] != wb["shape"] or wa["quant"] != wb["quant"] or nosrc(wa["opts"]) != nosrc(wb["opts"]):
        bad.append(f"{n}: shape / quantiser / options differ"); continue
    if wa["enc"] != wb["enc"]: bad.append(f"{n}: encoding struct differs"); continue
    pa, pb = planes(A, n), planes(B, n)
    diff = [x[0] for x, y in zip(pa, pb) if x[1:4] != y[1:4]]
    if not diff:
        if n in changed_expected: bad.append(f"{n}: expected a changed svh, identical")
        else: nsame += 1
        continue
    if n not in changed_expected or diff != ["svh"]: bad.append(f"{n}: planes {diff} differ (expected: {'svh' if n in changed_expected else 'none'})"); continue
    sa = np.frombuffer(dict((x[0], x[4]) for x in pa)["svh"], np.float16)
    sb = np.frombuffer(dict((x[0], x[4]) for x in pb)["svh"], np.float16)
    c = scales[n]
    if sa.shape != c.shape: bad.append(f"{n}: svh {sa.shape} vs c {c.shape}"); continue
    want = (sb.astype(np.float32) * c.astype(np.float32)).astype(np.float16)
    ulp = np.abs(want.view(np.int16).astype(np.int32) - sa.view(np.int16).astype(np.int32))
    exact.append(float((ulp == 0).mean())); worst_ulp = max(worst_ulp, int(ulp.max())); nchg += 1
    if (ulp == 0).mean() < 0.999 or ulp.max() > 1: bad.append(f"{n}: svh != f16(svh_ctl * c): exact {(ulp == 0).mean():.4f}, max ulp {ulp.max()}")
miss = changed_expected - {n for n in A.w}
if miss: bad.append(f"scaled writers missing from the container: {sorted(miss)[:3]}")
st = {"weights": len(A.w), "identical": nsame, "svh_changed": nchg, "expected_changed": len(changed_expected),
      "min_exact_share": min(exact) if exact else None, "worst_ulp": worst_ulp, "problems": len(bad)}
if nchg != len(changed_expected): bad.append(f"changed {nchg} != expected {len(changed_expected)}")
for b in bad[:20]: print("MISMATCH", b)
print(f"RADIFF {'PASS' if not bad else 'FAIL'} {json.dumps(st)}")
sys.exit(1 if bad else 0)
