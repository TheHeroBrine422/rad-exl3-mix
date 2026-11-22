#!/usr/bin/env python3
"""test220.py -- CPU unit checks of qad220's switches (no GPU, no student: pure torch). Functions are taken from qad220.py itself
(AST extraction, so the tests run the shipped code). Run: python3 test220.py   (exit 0 = all pass)."""
import ast, argparse, json, math, os, re, sys, tempfile, types
from fractions import Fraction
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TREE = ast.parse(open(os.path.join(HERE, "qad220.py")).read())
ns = {"os": os, "re": re, "argparse": argparse, "json": json, "math": math, "Fraction": Fraction, "np": np, "torch": torch, "HERE": HERE,
      "F": torch.nn.functional, "checkpoint": torch.utils.checkpoint.checkpoint, "types": types, "BF": torch.bfloat16}


def grab(*names, classes=()):
    for node in TREE.body:
        if (isinstance(node, ast.FunctionDef) and node.name in names) or (isinstance(node, ast.ClassDef) and node.name in classes):
            exec(compile(ast.Module([node], []), "qad220.py", "exec"), ns)
    return [ns[n] for n in list(names) + list(classes)]


fails = []
def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info else ""))
    if not ok: fails.append(name)


# ---------------------------------------------------------------- items 1, 2: trainable set, mask, frozen prefix
import mkmask220
layer_of, parse_range, apply_trainable, set_first_train, run_segment = grab("layer_of", "parse_range", "apply_trainable", "set_first_train", "run_segment", classes=("SegCache",))[:5]
SegCache = ns["SegCache"]


class FakeStu:
    def __init__(self, nl=6):
        self.layers = [None] * nl; self.P = {}
        for i in range(nl):
            for nm, k in (("self_attn.q_proj:r", 8), ("mlp.up_proj:c", 12), ("mlp.up_proj:r", 8)):
                self.P[f"model.language_model.layers.{i}.{nm}"] = torch.nn.Parameter(torch.zeros(k))
        self.P["model.language_model.norm.weight:g"] = torch.nn.Parameter(torch.zeros(4))


def targs(**kw):
    d = dict(train_layers=None, train_tensors=None, mask=None); d.update(kw); return argparse.Namespace(**d)


logs = []
st = FakeStu(); mk = apply_trainable(targs(), st, logs.append)
check("defaults: everything trainable, no mask, no log", mk is None and all(p.requires_grad for p in st.P.values()) and not logs)
check("layer_of", layer_of(st, "model.language_model.layers.17.mlp.up_proj:c") == 17 and layer_of(st, "model.language_model.norm.weight:g") == 6
      and layer_of(st, "lm_head:r") == 6)
check("parse_range", parse_range("32-63") == (32, 63) and parse_range("5") == (5, 5))
st = FakeStu(); apply_trainable(targs(train_layers="2-4"), st, logs.append); set_first_train(st, logs.append)
tr = sorted({layer_of(st, n) for n, p in st.P.items() if p.requires_grad})
check("--train-layers 2-4: layers 2..4 trainable, final norm frozen, first_train 2", tr == [2, 3, 4] and st.first_train == 2, f"{tr} first {st.first_train}")
st = FakeStu(); apply_trainable(targs(train_tensors=r"up_proj:c$"), st, logs.append)
check("--train-tensors regex", {n.split(".", 4)[-1] for n, p in st.P.items() if p.requires_grad} == {"mlp.up_proj:c"}
      and sum(p.requires_grad for p in st.P.values()) == 6)
st = FakeStu(); apply_trainable(targs(train_layers="3-5", train_tensors=r"q_proj"), st, logs.append); set_first_train(st, logs.append)
check("--train-layers and --train-tensors intersect", sorted(n for n, p in st.P.items() if p.requires_grad) ==
      [f"model.language_model.layers.{i}.self_attn.q_proj:r" for i in (3, 4, 5)] and st.first_train == 3)

# mask file: build from stats, apply
rng = np.random.default_rng(0)
nb = 40
z = {}
for n, p in FakeStu().P.items():
    sd = rng.uniform(0.5, 2.0, p.numel()); mu = rng.normal(size=p.numel()) * sd * 0.1
    z[n + "|mean"] = mu.astype(np.float32); z[n + "|std"] = sd.astype(np.float32)
k0 = "model.language_model.layers.0.mlp.up_proj:c"; z[k0 + "|mean"][:] = 0.0                                    # layer 0 up c: no signal at all
k1 = "model.language_model.layers.1.mlp.up_proj:c"; z[k1 + "|mean"][:] = np.float32(1e3); z[k1 + "|std"][:] = np.float32(1.0)   # all signal
z[k1 + "|mean"][3] = np.float32(-1e3)
k2 = "model.language_model.layers.2.mlp.up_proj:c"; z[k2 + "|mean"][:] = 0.0; z[k2 + "|mean"][:5] = np.float32(1e3); z[k2 + "|std"][:] = np.float32(1.0)   # 5 of 12
mask = mkmask220.build(z, nb, 3.0)
exp = {n: (np.abs(z[n + "|mean"].astype(np.float64)) > 3.0 * z[n + "|std"].astype(np.float64) / math.sqrt(nb)).astype(np.uint8) for n in mask}
check("mkmask: |mean| > Z std / sqrt(n), every vector", all(np.array_equal(mask[n], exp[n]) for n in exp) and len(mask) == len(FakeStu().P))
check("mkmask: all-zero and all-one vectors as built", mask[k0].sum() == 0 and mask[k1].all())
tmp = tempfile.mkdtemp(); mp = os.path.join(tmp, "m.npz"); np.savez(mp, **mask)
st = FakeStu(); mkd = apply_trainable(targs(mask=mp), st, logs.append); set_first_train(st, logs.append)
check("--mask: an all-zero vector is frozen, an all-one vector has no partial mask, the rest are partly masked",
      not st.P[k0].requires_grad and st.P[k1].requires_grad and st.P[k2].requires_grad and k1 not in mkd and k0 not in mkd and set(mkd) == {k2}
      and all(torch.equal(mkd[n].cpu(), torch.from_numpy(mask[n].astype(bool))) for n in mkd), f"{len(mkd)} partly masked")
sm = mkmask220.summary(mask)
check("mkmask summary kept counts", sm["0"]["kept"] == sum(int(mask[n].sum()) for n in mask if ".layers.0." in n))
np.savez(os.path.join(tmp, "bad.npz"), **{"nonexistent:r": np.ones(3, np.uint8)})
try: apply_trainable(targs(mask=os.path.join(tmp, "bad.npz")), FakeStu(), logs.append); ok = False
except AssertionError: ok = True
check("--mask with a name not in the student is an error", ok)

# masked Adam: grad * mask, masked entries stay put (also under AdamW decay, with the trainer's pin), kept ones move
torch.manual_seed(1)
p0 = torch.randn(50) * 0.1; m = torch.rand(50) > 0.6
for wd in (0.0, 20.0):
    q = torch.nn.Parameter(p0.clone())
    opt = torch.optim.AdamW([{"params": [q], "weight_decay": wd}], lr=1e-3, betas=(0.9, 0.99), weight_decay=0.0)
    for _ in range(3):
        q.grad = torch.randn(50) * m
        opt.step()
        with torch.no_grad(): q.copy_(torch.where(m, q, p0))
    check(f"masked scales do not move (wd {wd}), kept ones do", torch.equal(q[~m], p0[~m]) and bool((q[m] != p0[m]).all()))


class ToyLayer(torch.nn.Module):
    def __init__(self, d, tr=True):
        super().__init__(); self.w = torch.nn.Parameter(torch.randn(d, d) / d ** 0.5, requires_grad=tr); self.b = torch.nn.Parameter(torch.zeros(d), requires_grad=tr)
    def forward(self, x, **kw): return torch.tanh(x @ self.w + self.b) + x


class ToyStu:
    def __init__(self, nl, d=16, seed=0):
        torch.manual_seed(seed)
        self.layers = [ToyLayer(d) for i in range(nl)]; self.devs = [torch.device("cpu")]; self.q35 = None; self.d = d
        self.rotary = {self.devs[0]: lambda x, pid: None}; self.fnorm = lambda x: x
    def embed(self, ids): return torch.from_numpy(np.asarray(ids, np.float32))[None, :, None].expand(1, len(ids), self.d).contiguous() * 0.1


ids = np.arange(24)
def toy_grads(ft, prefix_path):
    st = ToyStu(6)
    for l in st.layers[:ft]: l.requires_grad_(False)
    st.first_train = ft if prefix_path else 0
    h, _ = run_segment(st, ids, 0, None)
    h.pow(2).sum().backward()
    return h.detach(), [None if p.grad is None else p.grad.clone() for l in st.layers for p in (l.w, l.b)]
ha, ga = toy_grads(3, True); hb, gb = toy_grads(3, False)
check("frozen prefix: loss and the trainable layers' gradients bitwise identical to the full-backward run (same trainable set)",
      torch.equal(ha, hb) and all((x is None and y is None) or torch.equal(x, y) for x, y in zip(ga, gb)))
check("frozen prefix: layers 0..2 get no gradient, layers 3..5 do", all(x is None for x in ga[:6]) and all(x is not None for x in ga[6:]))
h0, g0 = toy_grads(0, True); h0b, g0b = toy_grads(0, False)
check("first_train = 0 is the old loop (bitwise)", torch.equal(h0, h0b) and all(torch.equal(x, y) for x, y in zip(g0, g0b)))
print("\n" + ("ALL PASS" if not fails else f"FAILED: {fails}"))
sys.exit(1 if fails else 0)
