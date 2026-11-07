#!/usr/bin/env python3
"""check_pairs.py OUTDIR -- CPU checks of rxq/pairs.py (swap-pair hygiene, splice, no-op gate) on synthetic 4-layer
levels and synthetic RAD2 containers (written here from store/census.py's struct layout, read back by census.Rad).
Each case expects OK or REFUSED; the last line is "PAIRS CHECKS n/m PASS"."""
import json, os, shutil, struct, sys

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from rxq import pairs, qwen35  # noqa: E402

O = sys.argv[1]; shutil.rmtree(O, ignore_errors=True); os.makedirs(O)
NL = 4
res = []


def expect(name, want_ok, got):
    ok, st, probs = got
    res.append(ok == want_ok)
    print(f"{'PASS' if ok == want_ok else 'FAIL'} {name}: {'OK' if ok else 'REFUSED'} (expected {'OK' if want_ok else 'REFUSED'})"
          + (f" -- {probs[0]}" if probs else ""), flush=True)


# ---------------------------------------------------------------- synthetic levels
def level(d, tweak=lambda n, t: t):
    os.makedirs(d, exist_ok=True); g = torch.Generator().manual_seed(0); wm = {}
    for l in range(NL):
        t = {}
        for _, ks in qwen35.layer_plan(l):
            for k in ks:
                for p, shp, dt in (("trellis", (4, 8, 32), torch.int16), ("suh", (64,), torch.float16), ("svh", (32,), torch.float16)):
                    x = (torch.randint(-999, 999, shp, generator=g, dtype=torch.int16) if dt == torch.int16 else torch.randn(shp, generator=g).half())
                    t[f"{k}.{p}"] = tweak(f"{k}.{p}", x)
        name = f"q-L{l:02d}.safetensors"; save_file(t, f"{d}/{name}"); wm.update({n: name for n in t})
    json.dump({"metadata": {}, "weight_map": wm}, open(f"{d}/model.safetensors.index.json", "w"))


P = qwen35.L
Q3 = f"{P}.3.self_attn.q_proj"
level(f"{O}/A")
level(f"{O}/B", lambda n, t: t + 1 if n.startswith(Q3 + ".") and n.endswith(".svh") else t)
level(f"{O}/C", lambda n, t: t + 1 if (n.startswith(Q3) or n.startswith(f"{P}.2.mlp.down_proj")) and n.endswith(".svh") else t)
level(f"{O}/D", lambda n, t: t + 1 if n.startswith(Q3 + ".") and n.endswith(".trellis") else t)
arm = lambda s: pairs.parse_arm(s, NL)
expect("levels arm only", True, pairs.cmp_levels(f"{O}/A", f"{O}/B", arm("attn_qg")))
expect("levels arm by key", True, pairs.cmp_levels(f"{O}/A", f"{O}/B", arm(Q3)))
expect("levels arm by layer", True, pairs.cmp_levels(f"{O}/A", f"{O}/B", arm("attn_qg@3")))
expect("levels wrong arm", False, pairs.cmp_levels(f"{O}/A", f"{O}/B", arm("attn_k")))
expect("levels differ outside", False, pairs.cmp_levels(f"{O}/A", f"{O}/C", arm("attn_qg")))
expect("levels no-op arm", False, pairs.cmp_levels(f"{O}/A", f"{O}/A", arm("attn_qg")))
expect("levels no-op allowed", True, pairs.cmp_levels(f"{O}/A", f"{O}/A", arm("attn_qg"), allow_same=True))
expect("levels planes svh ok", True, pairs.cmp_levels(f"{O}/A", f"{O}/B", arm("attn_qg"), {"svh"}))
expect("levels planes svh, trellis changed", False, pairs.cmp_levels(f"{O}/A", f"{O}/D", arm("attn_qg"), {"svh"}))

# ---------------------------------------------------------------- same / splice
J = f"{O}/job"; os.makedirs(J)
from safetensors import safe_open  # noqa: E402
ix = json.load(open(f"{O}/D/model.safetensors.index.json"))["weight_map"]
with safe_open(f"{O}/D/q-L03.safetensors", "pt") as f:
    save_file({n: f.get_tensor(n) for n in f.keys() if n.startswith(Q3 + ".")}, f"{J}/t-{Q3}.safetensors")
expect("same: job vs its own level", True, pairs.same(f"{O}/D", J))
expect("same: job vs another level", False, pairs.same(f"{O}/A", J))
pairs.splice(f"{O}/A", J, f"{O}/S", arm("attn_qg"))
expect("splice result vs base (arm attn_qg)", True, pairs.cmp_levels(f"{O}/A", f"{O}/S", arm("attn_qg")))
expect("splice result == D", True, pairs.cmp_levels(f"{O}/D", f"{O}/S", arm("attn_qg"), allow_same=True))
try:
    pairs.splice(f"{O}/A", J, f"{O}/S2", arm("attn_qg,attn_k")); res.append(False); print("FAIL splice with a wrong arm: accepted")
except SystemExit as x:
    res.append(True); print(f"PASS splice with a wrong arm: refused -- {str(x)[:70]}")

# ---------------------------------------------------------------- synthetic RAD2 containers
C = pairs._census()


def rad(path, weights, meta):
    """weights: {name: (quant, opts, shape, scheme, transform, [(role, bytes)])}, meta: {key: str}"""
    S = bytearray(b"\0"); sidx = {}

    def s(x):
        if x not in sidx: sidx[x] = len(S); S.extend(x.encode() + b"\0")
        return sidx[x]
    ents, planes, encs, blobs = [], [], [], []
    for name, (q, o, shp, sch, tr, pl) in weights.items():
        e = bytearray(C.ENC_SIZE); e[0:len(sch)] = sch.encode(); e[24:24 + len(tr)] = tr.encode()
        struct.pack_into("<I", e, 48, len(pl))
        for i, (role, b) in enumerate(pl): C.PLANE.pack_into(e, 56 + 64 * i, role.encode(), 2, 0, len(b), 0, 0, 0)
        ents.append([s(name), s(q), s(o), len(encs), len(shp), shp, len(planes), [len(b) for _, b in pl]])
        encs.append(bytes(e)); blobs.append([b for _, b in pl]); planes += [None] * len(pl)
    kvs = [(s(k), 2, 0, s(v)) for k, v in meta.items()]
    rec = s("recipe")
    hdr = C.HDR.size; so = hdr; mo = so + len(S); do = mo + len(kvs) * C.KV.size; po = do + len(ents) * C.ENT.size
    eo = po + len(planes) * 16; data = eo + len(encs) * C.ENC_SIZE
    P_, D_, cur, pi = bytearray(), bytearray(), data, 0
    for e, bl in zip(ents, blobs):
        off = cur
        for b in bl: P_ += struct.pack("<QQ", cur, len(b)); cur += len(b)
        shp = list(e[5]) + [0] * (6 - len(e[5]))
        D_ += C.ENT.pack(e[0], e[1], e[2], e[3], e[4], *shp, 0, 0, 0, len(bl), e[6], off, 0, 0, 0)
    h = [0] * 26; h[0], h[1], h[6] = C.MAGIC, C.VER, rec
    h[8:18] = [so, len(S), mo, len(kvs), do, len(ents), po, len(planes), eo, len(encs)]
    with open(path, "wb") as f:
        f.write(C.HDR.pack(*h)); f.write(S); f.write(b"".join(C.KV.pack(*kv) for kv in kvs)); f.write(D_); f.write(P_)
        f.write(b"".join(encs))
        for bl in blobs: f.write(b"".join(bl))


def container(path, change=None, act=None, src="/x/A"):
    W = {}
    for l in range(NL):
        for k in [k for _, ks in qwen35.layer_plan(l) for k in ks]:
            n = pairs.rad_name(k)
            if n in W: continue
            pl = [("trellis", bytes([l % 7]) * 64), ("suh", b"\1" * 16), ("svh", b"\2" * 16)]
            if change and n in change: pl = [(r, change[n].get(r, b)) for r, b in pl]
            W[n] = ("exl3i", f"src={src},codebook=mul1i8,rate=4", [32, 64], "exl3i", "mul1i8/h128/k4", pl)
    rad(path, W, {f"exl3.act.{n}": a for n, a in (act or {}).items()})


qg3 = "blk.3.attn_qg.weight"
container(f"{O}/a.rad")
container(f"{O}/b.rad", {qg3: {"svh": b"\3" * 16}}, src="/x/B")
container(f"{O}/c.rad", {qg3: {"svh": b"\3" * 16}, "blk.2.ffn_down.weight": {"suh": b"\4" * 16}})
container(f"{O}/d.rad", {qg3: {"trellis": b"\5" * 64}})
container(f"{O}/e.rad", {qg3: {"svh": b"\3" * 16}}, act={"blk.0.ssm_z.weight": "a8g128"})
assert set(C.Rad(f"{O}/a.rad").w) == {pairs.rad_name(k) for k in qwen35.trunk_keys(NL)}
expect("mixes arm only (src= differs)", True, pairs.cmp_mixes(f"{O}/a.rad", f"{O}/b.rad", arm("attn_qg")))
expect("mixes wrong arm", False, pairs.cmp_mixes(f"{O}/a.rad", f"{O}/b.rad", arm("ssm_out")))
expect("mixes differ outside", False, pairs.cmp_mixes(f"{O}/a.rad", f"{O}/c.rad", arm("attn_qg")))
expect("mixes act differs outside", False, pairs.cmp_mixes(f"{O}/a.rad", f"{O}/e.rad", arm("attn_qg")))
expect("mixes no-op arm", False, pairs.cmp_mixes(f"{O}/a.rad", f"{O}/a.rad", arm("attn_qg")))
expect("mixes planes svh ok", True, pairs.cmp_mixes(f"{O}/a.rad", f"{O}/b.rad", arm("attn_qg"), {"svh"}))
expect("mixes planes svh, trellis changed", False, pairs.cmp_mixes(f"{O}/a.rad", f"{O}/d.rad", arm("attn_qg"), {"svh"}))
expect("mixes gate+up family name", True, pairs.cmp_mixes(f"{O}/a.rad", f"{O}/a.rad", arm("ffn_gate_up"), allow_same=True))

print(f"PAIRS CHECKS {sum(res)}/{len(res)} PASS")
sys.exit(0 if all(res) else 1)
