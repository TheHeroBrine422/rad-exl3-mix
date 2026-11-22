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
st9 = FakeStu(); apply_trainable(targs(train_layers="2-4"), st9, logs.append); set_first_train(st9, logs.append, True)
check("--no-frozen-prefix: first_train 0 (full path), first_trainable still 2", st9.first_train == 0 and st9.first_trainable == 2)
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


# ---------------------------------------------------------------- item 4: clip, beta2
ns["CLIP_MIN"], ns["CLIP_WIN"] = 5, 256
clip_factor, = grab("clip_factor")
check("clip: off (X = 0) and no clipping before 5 steps of history", clip_factor([1.0] * 100, 1e9, 0.0) == (1.0, None) and clip_factor([1.0] * 4, 1e9, 3.0) == (1.0, None))
f, thr = clip_factor([1.0, 2.0, 3.0, 4.0, 5.0], 12.0, 3.0)
check("clip: threshold = X x the median of the earlier norms, factor = thr / g when above", thr == 9.0 and abs(f - 0.75) < 1e-15, f"thr {thr} f {f}")
check("clip: g at or below the threshold is untouched", clip_factor([1.0] * 6, 3.0, 3.0)[0] == 1.0 and clip_factor([1.0] * 6, 2.0, 3.0)[0] == 1.0)
hist = list(np.r_[np.full(300, 10.0), np.full(256, 1.0)])
check("clip: the median runs over the last 256 norms only", clip_factor(hist, 5.0, 3.0)[1] == 3.0)
g = torch.randn(1000, dtype=torch.float64) * 7; gn = float(g.norm()); f, thr = clip_factor([gn / 4] * 10, gn, 2.0)
g2 = g * f
check("clip: the clipped gradient has norm X x median", abs(float(g2.norm()) - thr) < 1e-9 * thr)
# beta2: betas (0.9, b); the default 0.99 is the old optimizer (bitwise), another value changes the step
p1 = torch.nn.Parameter(torch.ones(8)); p2 = torch.nn.Parameter(torch.ones(8)); p3 = torch.nn.Parameter(torch.ones(8))
o1 = torch.optim.Adam([p1], lr=1e-3, betas=(0.9, 0.99), eps=1e-8); o2 = torch.optim.Adam([p2], lr=1e-3, betas=(0.9, 0.99), eps=1e-8)
o3 = torch.optim.Adam([p3], lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
for _ in range(3):
    gg = torch.randn(8)
    for p, o in ((p1, o1), (p2, o2), (p3, o3)): p.grad = gg.clone(); o.step()
check("beta2 0.99 = the old optimizer bitwise; 0.95 differs", torch.equal(p1, p2) and not torch.equal(p1, p3))

# ---------------------------------------------------------------- item 5: schedules, fork, decay data
sys.path.insert(0, os.path.join(HERE, "..", "qad181"))
import data181 as D
ns["D"] = D; ns["hashlib"] = __import__("hashlib")
lr_main, group_lr, parse_lr_steps, decay_mult, sched_lr, make_lr, decay_start_step, rows_cum, calib_cooldown, plan_sha = grab(
    "lr_main", "group_lr", "parse_lr_steps", "decay_mult", "sched_lr", "make_lr", "decay_start_step", "rows_cum", "calib_cooldown", "plan_sha")
sys.path.insert(0, os.path.join(HERE, "..", "qad181"))
def sargs(**kw):
    d = dict(lr_steps=None, decay_shape="linear", lr=1e-3, warmup=8, decay_frac=0.2); d.update(kw); return argparse.Namespace(**d)
cum = np.cumsum(np.full(200, 850))
ok = True
for last, fork, N, k0 in [(194, 155, 1, 0), (194, 155, 4, 0), (49, 39, 1, 5), (100, 80, 2, 0)]:
    f = make_lr(sargs(), cum, last, fork, k0, N)
    ok &= all(f(s) == group_lr(s, k0, last, fork, 1e-3, 8, N) for s in range(k0, last))
check("defaults: make_lr is group_lr exactly (N 1 / 4 / 2, block offset)", ok)
# sched_lr with linear shape and no phases == lr_main arithmetic, bitwise
U, FU = 49, 39
check("sched_lr(linear, no phases) == lr_main bitwise", all(sched_lr(g, U, FU, 8, 1e-3) == lr_main(g, U, FU, 1e-3, 8) for g in range(U)))
check("parse_lr_steps", parse_lr_steps("1e-3:0.25e6,3e-4:1e6") == [(1e-3, 0.25e6), (3e-4, 1e6)])
try: parse_lr_steps("1e-3:5e5,3e-4:1e5"); ok = False
except AssertionError: ok = True
check("parse_lr_steps: rows must increase", ok)
# piecewise: 850 rows a plan step, budget 1M rows
cum = np.cumsum(np.full(1300, 850.0))                    # 1.105M rows in 1300 steps
last = int(np.searchsorted(cum, 1e6)) + 1
a_ = sargs(lr_steps="1e-3:0.25e6,3e-4:0.9e6", lr=1e-3, warmup=4)
fl = decay_start_step(a_, cum, 1e6)
f = make_lr(a_, cum, last, fl, 0, 1)
l1 = [f(s) for s in range(last)]
ph1_end = int(np.searchsorted(cum, 0.25e6)) + 1
check("lr-steps: warmup to 1e-3, 1e-3 until 0.25M rows, 3e-4 to 0.9M, then linear decay from 3e-4 to ~0",
      abs(l1[3] - 1e-3) < 1e-15 and l1[10] == 1e-3 and all(x == 1e-3 for x in l1[4:ph1_end]) and all(x == 3e-4 for x in l1[ph1_end:fl])
      and l1[fl] == 3e-4 and l1[-1] < 3e-4 / (last - fl) * 1.0001 and all(b <= a for a, b in zip(l1[fl:], l1[fl + 1:])),
      f"phase1 ends step {ph1_end}, decay from {fl}, last lr {l1[-1]:.3e}")
check("lr-steps: the first row count at which a phase has ended switches the lr (rows seen before the step)", l1[ph1_end - 1] == 1e-3 and l1[ph1_end] == 3e-4 or cum[ph1_end - 1] >= 0.25e6)
# shapes at U = 100, decay start 80
g_ = list(range(100))
for shp in ("linear", "sqrt"):
    v = [sched_lr(g, 100, 80, 8, 1e-3, shp) for g in g_]
    check(f"shape {shp}: constant to 79, decays monotonically from 1e-3 at 80 to ~0", all(x == 1e-3 for x in v[8:80]) and v[80] == 1e-3 and all(b <= a for a, b in zip(v[80:], v[81:])) and 0 < v[-1] < 1e-3 * 0.1,
          f"last {v[-1]:.3e}")
v = [sched_lr(g, 100, 80, 8, 1e-3, "sqrt") for g in g_]; lv = [sched_lr(g, 100, 80, 8, 1e-3, "linear") for g in g_]
check("shape sqrt: 1 - sqrt(progress): at mid-decay (g = 90) 1 - sqrt(.5) of the peak, below linear's 0.5 throughout (falls fast, ends gently)", abs(v[90] - 1e-3 * (1 - math.sqrt(0.5))) < 1e-12 and all(v[g] < lv[g] for g in range(81, 100)), f"{v[90]:.3e}")
for shp in ("cosine-all", "linear-all"):
    FU = 8
    v = [sched_lr(g, 100, FU, 8, 1e-3, shp) for g in g_]
    check(f"shape {shp}: warmup then straight into the decay (peak at g = 8, no constant phase), monotone to ~0", v[8] == 1e-3 and all(b < a for a, b in zip(v[8:], v[9:])) and v[-1] < 1e-4, f"g=54 {v[54]:.3e} last {v[-1]:.3e}")
cs = [sched_lr(g, 100, 8, 8, 1e-3, "cosine-all") for g in g_]
check("cosine-all: half the peak at the middle of the decay", abs(cs[54] - 1e-3 * 0.5 * (1 + math.cos(math.pi * (46 / 92)))) < 1e-12 and abs(cs[8 + 46] - 0.5e-3) < 1e-5)
mk = make_lr(sargs(decay_shape="cosine-all", warmup=8), np.cumsum(np.full(100, 850)), 100, 80, 0, 1)
check("make_lr: a *-all shape decays from the end of the warmup whatever fork_last is", mk(8) == 1e-3 and mk(50) < 0.6e-3)

# calib cooldown over a fake corpus
class FC: pass
C = FC(); C.items = [(0 if k % 3 else 1, k, 3000 + 7 * k, 2000) for k in range(120)]; C.tags = ["xq", "calib"]
class G_:
    def scored(self, i, a, b): return np.arange(a, b), np.arange(a, b)
C.g = [G_(), G_()]
order = np.random.default_rng(0).permutation(len(C.items)); steps0 = D.plan(C.items, order, 4, 1024)
rows_of, cumx = rows_cum(C, steps0)
check("rows_cum: cumulative rows, one entry per step, = sum of the segment lengths", int(cumx[-1]) == sum(b - a for st in steps0 for segs in st for (_, a, b) in segs) and len(cumx) == len(steps0))
fl = 30
st2 = calib_cooldown(C, steps0, fl, 4, 1024, 0)
used = {it for st in steps0[:fl] for segs in st for (it, _, _) in segs}
tail_items = {it for st in st2[fl:] for segs in st for (it, _, _) in segs}
check("calib cooldown: steps before fork_last unchanged (same plan checksum), the tail reads only unused calib documents",
      st2[:fl] == steps0[:fl] and plan_sha(st2, fl) == plan_sha(steps0, fl) and tail_items and all(C.tags[C.items[it][0]] == "calib" for it in tail_items) and not (tail_items & used))
check("calib cooldown: every unused calib document is planned once", tail_items == {k for k, x in enumerate(C.items) if x[0] == 1 and k not in used})
check("plan_sha differs when the plan differs", plan_sha(st2, fl + 5) != plan_sha(steps0, fl + 5))

# ---------------------------------------------------------------- item 6: prompt weight, local MSE
ns["chunk_kl"] = __import__("qad").chunk_kl
parse_layers, scored_all, chunk_kl_rows, seg_loss, seg_loss_ext = grab("parse_layers", "scored_all", "chunk_kl_rows", "seg_loss", "seg_loss_ext", classes=("LocalMSE", "TapStore"))[:5]
LocalMSE, TapStore = ns["LocalMSE"], ns["TapStore"]
ns["TAPS"] = (5, 19, 33, 47, 61)
check("parse_layers", parse_layers("0-3") == [0, 1, 2, 3] and parse_layers("5+19+33") == [5, 19, 33] and parse_layers("0-1+4") == [0, 1, 4])

# a real data181.Gold on a synthetic golden: 2 docs, gen mask with prompt-side rows
V_, K_, d_ = 32, 4, 16
gdir = os.path.join(tmp, "gold"); os.makedirs(gdir)
nt = [40, 30]; sf = [0, 2]; rg_ = np.random.default_rng(5)
R_ = sum(n - 1 - s0 for n, s0 in zip(nt, sf))
rec = D.rec(K_); top = np.empty((R_, K_), rec); rows = np.zeros((R_, 3), np.float32)
for r in range(R_):
    pr = rg_.dirichlet(np.ones(V_) * 0.3); ix = np.argsort(-pr)[:K_]; top["id"][r] = ix; top["lp"][r] = np.log(pr[ix]); rows[r, 2] = 1 - pr[ix].sum()
top.tofile(f"{gdir}/top.bin"); rows.tofile(f"{gdir}/rows.f32")
rg_.integers(0, V_, sum(nt)).astype(np.int32).tofile(f"{gdir}/tokens.i32")
r0 = 0; docs = []
for n, s0 in zip(nt, sf): docs.append({"tokens": n, "score_from": s0, "row0": r0}); r0 += n - 1 - s0
json.dump({"docs": docs, "top": K_, "rows": R_}, open(f"{gdir}/kld.json", "w"))
gen = (np.arange(R_) % 3 != 0).astype(np.uint8); gen.tofile(f"{gdir}/gen.u8")
gold = D.Gold(gdir, "toy")
j_all, r_all, isg = scored_all(gold, 0, 0, 40); jg, rg = gold.scored(0, 0, 40)
check("scored_all: every row of the window, gen flags = the golden's gen mask, gen rows = Gold.scored", len(j_all) == 39 and np.array_equal(j_all[isg], jg)
      and np.array_equal(r_all[isg], rg) and 0 < isg.sum() < 39 and np.array_equal(isg, gen[r_all].astype(bool)))
j2, r2, g2_ = scored_all(gold, 1, 5, 20); check("scored_all: a mid-document window", np.array_equal(j2, np.arange(5, 20)) and len(r2) == 15)

class ToyStu2(ToyStu):
    def __init__(self, nl=6, seed=0):
        super().__init__(nl, d_, seed)
        torch.manual_seed(seed + 1); self.head = torch.nn.Linear(d_, V_, bias=False); self.v2row = None
        self.emb_ids = None
    def embed(self, ids): return torch.randn(1, len(ids), d_, generator=torch.Generator().manual_seed(int(ids[0]) + 7)) * 0.5
gl = [gold]
def grads_of(stu, loss):
    for p in list(stu.head.parameters()) + [q for l in stu.layers for q in l.parameters()]: p.grad = None
    loss.backward()
    return [None if q.grad is None else q.grad.clone() for l in stu.layers for q in (l.w, l.b)] + [None if stu.head.weight.grad is None else stu.head.weight.grad.clone()]
def same(x, y, tol=0.0):
    return all((u is None and v is None) or (u is not None and v is not None and (torch.equal(u, v) if tol == 0 else torch.allclose(u, v, rtol=tol, atol=tol * float(v.abs().max())))) for u, v in zip(x, y))

st_ = ToyStu2(); tot0, m0, _ = seg_loss(st_, gl, (0, 0), 0, 40, None); g0 = grads_of(st_, tot0)
st_ = ToyStu2(); loss, genkl, mg, n_all, _, ls = seg_loss_ext(st_, gl, (0, 0), 0, 40, None); g1 = grads_of(st_, loss)
check("seg_loss_ext, prompt weight 0 and no local MSE: loss, gen KL, rows and gradients identical to seg_loss",
      torch.equal(loss, tot0) and torch.equal(genkl, tot0.detach()) and mg == m0 == n_all and same(g0, g1) and float(ls) == 0.0)
# prompt weight W: loss = gen KL + W x prompt KL, from per-row KL
W = 0.3
st_ = ToyStu2(); h_, _ = run_segment(st_, gold.ids(0, 0, 40), 0, None)
tid, tlp, pt = gold.targets(r_all, torch.device("cpu"))
kl_all = ns["chunk_kl"](h_[torch.from_numpy(j_all)], st_.head, tid, tlp, pt, True)
want = kl_all[torch.from_numpy(isg)].sum() + W * kl_all[torch.from_numpy(~isg)].sum(); gw = grads_of(st_, want)
st_ = ToyStu2(); loss, genkl, mg, n_all, _, _ = seg_loss_ext(st_, gl, (0, 0), 0, 40, None, pw=W); g2 = grads_of(st_, loss)
check("--prompt-weight W: loss = gen KL sum + W x prompt KL sum (per-row recomputation), gradients match, gen KL stays gen-only, rows = gen rows",
      torch.allclose(loss, want, rtol=1e-6) and torch.allclose(genkl, kl_all[torch.from_numpy(isg)].sum(), rtol=1e-6) and mg == int(isg.sum()) and n_all == 39 and same(g2, gw, 1e-5),
      f"loss {float(loss):.5f} want {float(want):.5f}")
check("--prompt-weight changes the gradient (prompt rows contribute)", not same(g2, g1, 1e-4))

# local MSE: a store of the teacher's states at layers [1, 2, 4] of the gen rows (synthetic), taps format
sdir = os.path.join(tmp, "taps"); os.makedirs(sdir)
tl_ = [1, 2, 4]; Rg = int(gen.astype(bool)[:38].sum())
nrow_doc0 = len(jg)
mm = np.random.default_rng(3).standard_normal((nrow_doc0, len(tl_), d_)).astype(np.float32)
mm_bf = torch.from_numpy(mm).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16); mm_bf.tofile(f"{sdir}/taps.bin")
json.dump({"taps": tl_, "hidden": d_, "rows": nrow_doc0, "docs": {"toy:0": [0, 40]}}, open(f"{sdir}/index.json", "w")); open(f"{sdir}/.done", "w").close()
store = TapStore(sdir, any_layers=True)
check("TapStore(any_layers): layers from the index, rows at the gen positions", store.layers == (1, 2, 4) and torch.equal(store.rows(gold, 0, jg, "cpu")[:, 1].float(), torch.from_numpy(mm[:, 1]).to(torch.bfloat16).float()))
try: TapStore(sdir); ok = False
except AssertionError: ok = True
check("TapStore (the T arm) still requires the DFlash2 layers", ok)
try: LocalMSE("0-3,0.1", store); ok = False
except AssertionError: ok = True
check("--local-mse: layers missing from the store are an error", ok)
def run_loc(mode, lam=0.5, layers="1+2"):
    st = ToyStu2(); loc = LocalMSE(f"{layers},{lam},{mode}", store)
    loss, genkl, mg, n_all, _, ls = seg_loss_ext(st, gl, (0, 0), 0, 40, None, loc=loc)
    return st, loc, loss, genkl, ls
# KL-only reference gradients (the plain run) and MSE-only gradients on the same graph shape
st_ = ToyStu2(); tot_kl, _, _ = seg_loss(st_, gl, (0, 0), 0, 40, None); gk = grads_of(st_, tot_kl)
st, loc, loss, genkl, ls = run_loc("local"); gl_ = grads_of(st, loss)
check("local mode: the gen KL is the plain run's (forward unchanged)", torch.equal(genkl, tot_kl.detach()))
# MSE-only through a local chain: lambda x term, same forward
st_ = ToyStu2(); loc_ = LocalMSE("1+2,0.5,local", store)
h_, _, outs = run_segment(st_, gold.ids(0, 0, 40), 0, None, loc=loc_)
mse_only = loc_.lam * loc_.term(outs, store.rows(gold, 0, jg, "cpu"), torch.from_numpy(jg))
gm = grads_of(st_, mse_only)
nl = len(st.layers)
ok_loc = all(same([gl_[2 * L], gl_[2 * L + 1]], [gm[2 * L], gm[2 * L + 1]]) for L in (1, 2)) and gl_[0] is None and gl_[1] is None
ok_back = all(same([gl_[2 * L], gl_[2 * L + 1]], [gk[2 * L], gk[2 * L + 1]]) for L in (3, 4, 5))
check("local mode: listed layers (1, 2) get exactly their own MSE gradient (no KL), layer 0 none, the layers after (3-5) get exactly the plain KL gradient", ok_loc and ok_back)
check("local mode: the KL reaches no listed layer (their gradients differ from the plain KL run's)", not same([gl_[2], gl_[3]], [gk[2], gk[3]], 1e-3))
st, loc, loss, genkl, ls = run_loc("joint"); gj = grads_of(st, loss)
st_ = ToyStu2(); h_, _, outs = run_segment(st_, gold.ids(0, 0, 40), 0, None, loc=LocalMSE("1+2,0.5,joint", store))
tt = store.rows(gold, 0, jg, "cpu"); lc = LocalMSE("1+2,0.5,joint", store)
tid, tlp, pt = gold.targets(rg, torch.device("cpu"))
kl_g = ns["chunk_kl"](h_[torch.from_numpy(jg)], st_.head, tid, tlp, pt)
gjw = grads_of(st_, kl_g + 0.5 * lc.term(outs, tt, torch.from_numpy(jg)))
check("joint mode: gradient of KL + lambda MSE through the whole chain (layer 0 included)", same(gj, gjw, 1e-5) and gj[0] is not None)
check("local MSE term reported = the term's value", abs(float(ls) - float(lc.term(outs, tt, torch.from_numpy(jg)))) < 1e-5 * float(ls))
# block mode (w222): layers 2 and 4 end blocks [0-2] and [3-4]; the KL reaches only layer 5 and the head
st, loc, loss, genkl, ls = run_loc("block", 0.5, "2+4"); gb = grads_of(st, loss)
st_ = ToyStu2(); loc_ = LocalMSE("2,0.25,block", store)
h_, _, outs = run_segment(st_, gold.ids(0, 0, 40), 0, None, loc=loc_)
gr = grads_of(st_, loc_.lam * loc_.term(outs, store.rows(gold, 0, jg, "cpu"), torch.from_numpy(jg)))
check("block mode: the gen KL is the plain run's (forward unchanged)", torch.equal(genkl, tot_kl.detach()))
check("block mode: layers 0-2 get exactly the gradient of their block's MSE at layer 2 through the chain (layer 0 included)", same(gb[:6], gr[:6]) and gb[0] is not None)
check("block mode: layers 3-4 (second block, input detached) get a gradient that differs from the plain KL's; layer 5 and the head get exactly the plain KL gradient",
      gb[6] is not None and gb[8] is not None and not same(gb[6:10], gk[6:10], 1e-3) and same([gb[10], gb[11], gb[12]], [gk[10], gk[11], gk[12]]))
# w222: Adam state by name for a fork into a bigger trainable set (staged MSE -> KL)
mos, = grab("map_opt_state")[:1]
old_st = {"state": {0: {"step": torch.tensor(5.), "exp_avg": torch.ones(2)}, 1: {"step": torch.tensor(5.), "exp_avg": 2 * torch.ones(3)}}, "param_groups": [{"params": [0, 1]}]}
new_st = mos(old_st, ["a:r", "b:r"], ["x:r", "b:r", "a:r"], [{"params": [0, 1, 2]}])
check("map_opt_state: state by name (b, a kept at their new indices), fresh for the new parameter", sorted(new_st["state"]) == [1, 2] and float(new_st["state"][1]["exp_avg"][0]) == 2.0 and float(new_st["state"][2]["exp_avg"][0]) == 1.0 and new_st["param_groups"] == [{"params": [0, 1, 2]}])
lcr = LocalMSE("0-3,0.1,local", type("S", (), {"layers": (0, 1, 2, 3)})()); logs2 = []; lcr.restrict(2, logs2.append)
check("LocalMSE.restrict drops layers before the first trainable one", lcr.layers == [2, 3] and logs2)

# ---------------------------------------------------------------- item 7: --a8 composes (tied scales; the a8 forward reads r / c through su() / sv())
use_a8, = grab("use_a8", classes=("_A8",))[:1]
qa = mkq(); tie_linear(qa, G); qb = mkq()
with torch.no_grad():
    qa._parameters["r_t"].copy_(rt); qa._parameters["c_t"].copy_(ct); qb.r.copy_(rt.repeat_interleave(G)); qb.c.copy_(ct.repeat_interleave(G))
use_a8(types.SimpleNamespace(qlin={"a": qa, "b": qb}))
ya, yb = qa(x), qb(x)
check("a8 forward: tied linear bitwise = the untied one with the expanded scales", torch.equal(ya, yb) and not torch.equal(ya, (mkq().__class__.forward)(qa, x)))
(ya.float() ** 2).sum().backward(); (yb.float() ** 2).sum().backward()
check("a8 backward (STE): the tied gradient is the group sum of the untied one",
      torch.allclose(qa._parameters["r_t"].grad, qb.r.grad.view(-1, G).sum(1), rtol=1e-5, atol=1e-6 * float(qb.r.grad.abs().max())))

# ---------------------------------------------------------------- parser: every w220 flag defaults to today's behaviour
ns["__doc__"] = ast.get_docstring(TREE)
parser, = grab("parser")
d = parser().parse_args(["OUT", "--model", "m", "--level", "l", "--chunks", "c"])
check("parser defaults: no w220 flag is on", d.train_layers is None and d.train_tensors is None and d.mask is None and d.tie_groups == 1 and d.tie_layers is None
      and d.clip_gnorm == 0.0 and d.beta2 == 0.99 and d.lr_steps is None and d.decay_shape == "linear" and d.fork_from is None and d.decay_data == "mix"
      and d.save_at is None and d.prompt_weight == 0.0 and d.local_mse is None and d.local_taps is None and d.tap_layers is None
      and d.eps == 1e-8 and d.wd == 0.0 and d.accum == 1 and not d.a8 and not d.compile_scales and not d.compile_ew)
d = parser().parse_args(["OUT", "--model", "m", "--level", "l", "--chunks", "c", "--train-layers", "32-63", "--tie-groups", "128", "--tie-layers", "0-31", "--clip-gnorm", "3",
                         "--beta2", "0.95", "--lr-steps", "1e-3:2.5e5,3e-4:9e5", "--decay-shape", "sqrt", "--decay-data", "calib", "--prompt-weight", "0.2",
                         "--local-mse", "0-31,0.1", "--local-taps", "T", "--fork-from", "F", "--save-at", "6e5", "--mask", "m.npz", "--train-tensors", "q_proj", "--a8", "--compile-scales", "--compile-ew"])
check("parser: every w220 flag parses (with --a8 and the compile flags)", d.train_layers == "32-63" and d.tie_groups == 128 and d.clip_gnorm == 3.0 and d.beta2 == 0.95 and d.decay_shape == "sqrt"
      and d.decay_data == "calib" and d.prompt_weight == 0.2 and d.local_mse == "0-31,0.1" and d.a8 and d.compile_scales and d.compile_ew)
print("\n" + ("ALL PASS" if not fails else f"FAILED: {fails}"))
sys.exit(1 if fails else 0)
