#!/usr/bin/env python3
"""test220.py -- CPU unit checks of qad220's switches (no GPU, no student: pure torch). Functions are taken from qad220.py itself
(AST extraction, so the tests run the shipped code). Run: python3 test220.py   (exit 0 = all pass)."""
import ast, argparse, json, math, os, re, sys, tempfile, types
from fractions import Fraction
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "qad"))
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

# ---------------------------------------------------------------- item 3: tied scales
import qmodel
from rxq import deq
ns["qmodel"] = qmodel
grab("tie_linear", "apply_ties", classes=("TiedQLinear",))
TiedQLinear, tie_linear, apply_ties = ns["TiedQLinear"], ns["tie_linear"], ns["apply_ties"]


def mkq(k=256, n=384, seed=0, gauge_r=False):
    q = qmodel.QLinear.__new__(qmodel.QLinear); torch.nn.Module.__init__(q)
    g = torch.Generator().manual_seed(seed)
    q.key = f"model.language_model.layers.{seed}.mlp.up_proj"; q.k, q.n = k, n; q.in_features, q.out_features = k, n
    q.register_buffer("wq", torch.randint(-20, 20, (k, n), generator=g).to(torch.int8), persistent=False); q.s = 0.05; q.exact = True
    q.register_buffer("suh", (torch.randint(0, 2, (k,), generator=g) * 2 - 1).float() * 0.8, persistent=False)
    q.register_buffer("svh", (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float() * 0.8, persistent=False)
    q.register_buffer("H", deq.hadamard128("cpu"), persistent=False); q.register_buffer("Hb", q.H.to(torch.bfloat16), persistent=False)
    q.r = None if gauge_r else torch.nn.Parameter(torch.zeros(k)); q.c = torch.nn.Parameter(torch.zeros(n)); q.la = q.lb = None
    return q


G = 32
x = torch.randn(10, 256).to(torch.bfloat16)
qt = mkq(); tie_linear(qt, G)
rt, ct = torch.randn(256 // G) * 0.05, torch.randn(384 // G) * 0.05
with torch.no_grad(): qt._parameters["r_t"].copy_(rt); qt._parameters["c_t"].copy_(ct)
qf = mkq()
with torch.no_grad(): qf.r.copy_(rt.repeat_interleave(G)); qf.c.copy_(ct.repeat_interleave(G))
check("tied: r / c read as the per-channel vectors (group value repeated)", torch.equal(qt.r.detach(), rt.repeat_interleave(G)) and torch.equal(qt.c.detach(), ct.repeat_interleave(G)))
check("tied: forward bitwise = the untied linear with the expanded scales", torch.equal(qt(x), qf(x)))
(qt(x).float() ** 2).sum().backward(); (qf(x).float() ** 2).sum().backward()
gr = qf.r.grad.view(-1, G).sum(1); gc = qf.c.grad.view(-1, G).sum(1)
check("tied: a tied parameter's gradient = the sum of its members' per-channel gradients",
      torch.allclose(qt._parameters["r_t"].grad, gr, rtol=1e-5, atol=1e-6 * float(gr.abs().max())) and
      torch.allclose(qt._parameters["c_t"].grad, gc, rtol=1e-5, atol=1e-6 * float(gc.abs().max())),
      f"max rel {float((qt._parameters['r_t'].grad - gr).abs().max() / gr.abs().max()):.1e}")
check("tied: only r_t / c_t are parameters (r / c are gone)", sorted(n for n, _ in qt.named_parameters()) == ["c_t", "r_t"])
sa, sb = qt.export(); sc_, sd_ = qf.export()
check("tied: export() (f16 suh / svh) equals the untied one", torch.equal(sa, sc_) and torch.equal(sb, sd_))
# the export round trip through write_ckpt: per-channel scales, format unchanged
write_ckpt, = grab("write_ckpt")
ns["save_atomic"] = lambda o, p: torch.save(o, p); 
import safetensors.torch as stt
ns["qmodel"] = qmodel
class WStu:
    def __init__(self, q): self.qlin = {q.key: q}; self.P = {q.key + ":r": q._parameters.get("r_t", q._parameters.get("r")), q.key + ":c": q._parameters.get("c_t", q._parameters.get("c"))}
    norm = None
    def state(self): return {k: v.detach().cpu().clone() for k, v in self.P.items()}
o1, o2 = os.path.join(tmp, "ck-tied"), os.path.join(tmp, "ck-full")
write_ckpt(WStu(qt), o1, {}); write_ckpt(WStu(qf), o2, {})
f1 = stt.load_file(f"{o1}/scales/{qt.key}.safetensors"); f2 = stt.load_file(f"{o2}/scales/{qf.key}.safetensors")
check("tied export round trip: scales/<key>.safetensors {r [k], c [n]} f32, identical to the untied student's with the expanded values",
      f1["r"].shape == (256,) and f1["c"].shape == (384,) and f1["r"].dtype == torch.float32 and torch.equal(f1["r"], f2["r"]) and torch.equal(f1["c"], f2["c"])
      and torch.equal(f1["r"], rt.repeat_interleave(G).exp()))
# gauge (r None) and apply_ties on a fake student: layers range, P rebuilt, frozen sides untouched
class TStu:
    def __init__(self):
        self.layers = [None] * 4; self.qlin = {}; self.P = {}
        for i in range(4):
            q = mkq(seed=i, gauge_r=(i == 3)); self.qlin[q.key] = q
            for nm in ("r", "c"):
                if getattr(q, nm) is not None: self.P[f"{q.key}:{nm}"] = getattr(q, nm)
ts = TStu(); lg = []
apply_ties(argparse.Namespace(tie_groups=64, tie_layers="1-3"), ts, lg.append)
check("apply_ties: layers 1-3 tied, layer 0 per channel; the gauge-fixed side (None) stays None; P points at the tied parameters",
      ts.P["model.language_model.layers.0.mlp.up_proj:r"].numel() == 256 and ts.P["model.language_model.layers.1.mlp.up_proj:r"].numel() == 4
      and ts.P["model.language_model.layers.2.mlp.up_proj:c"].numel() == 6 and "model.language_model.layers.3.mlp.up_proj:r" not in ts.P
      and ts.qlin["model.language_model.layers.3.mlp.up_proj"].r is None and ts.qlin["model.language_model.layers.3.mlp.up_proj"].c.numel() == 384
      and ts.P["model.language_model.layers.1.mlp.up_proj:r"] is ts.qlin["model.language_model.layers.1.mlp.up_proj"]._parameters["r_t"], lg[0])
ts2 = TStu(); apply_ties(argparse.Namespace(tie_groups=1, tie_layers=None), ts2, lg.append)
check("--tie-groups 1 (default) changes nothing", all(type(q) is qmodel.QLinear for q in ts2.qlin.values()))

print("\n" + ("ALL PASS" if not fails else f"FAILED: {fails}"))
sys.exit(1 if fails else 0)
