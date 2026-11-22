#!/usr/bin/env python3
"""testhf220.py -- CPU integration checks of qad220 on a tiny REAL Qwen3.5 stack (transformers' decoder layers, 4 layers: 3 GDN + 1 full attention; every trunk linear a
QLinear with random int8 codes), no GPU, no 27B: the real Trainer.run_group (optimizer, gradient clip / mask, accumulation of segments with carried GDN / KV state) of qad200.py
and qad220.py on the same synthetic golden:
  1. every w220 flag at its default: qad220's Trainer reproduces qad200's bit for bit (2 optimizer steps: step records, gradient norms, all parameters);
  2. frozen prefix (--train-layers 2-3): same step records / parameters as the full-backward path (--no-frozen-prefix) on the same trainable set;
  3. each new flag runs through the Trainer (tied scales, mask, clip, prompt weight, local MSE both modes, schedule helpers) and moves what it should.
Needs transformers (box2: /data/q27/venv/bin/python, HIP_VISIBLE_DEVICES= so no card is touched). Run: python3 testhf220.py   (exit 0 = all pass)"""
import argparse, importlib.util, json, os, sys, tempfile, types
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "..", "qad"), os.path.join(HERE, "..", "qad181"), os.path.join(HERE, "..", "qad200"), HERE):
    sys.path.insert(0, p)
import qmodel, data181 as D                                    # noqa: E402
from rxq import deq                                            # noqa: E402
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402
import transformers.models.qwen3_5.modeling_qwen3_5 as q35     # noqa: E402


def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sys.modules[name] = m; sp.loader.exec_module(m); return m
Q200 = load("qad200", os.path.join(HERE, "..", "qad200", "qad200.py"))
Q220 = load("qad220", os.path.join(HERE, "qad220.py"))
fails = []
def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info else ""))
    if not ok: fails.append(name)

NL, H, V_ = 4, 256, 64
LP = "model.language_model."; L = LP + "layers"


def mkq(key, k, n, g, train_r=True, train_c=True):
    q = qmodel.QLinear.__new__(qmodel.QLinear); torch.nn.Module.__init__(q)
    q.key = key; q.k, q.n = k, n; q.in_features, q.out_features = k, n; q.exact = True; q.s = 0.05
    q.register_buffer("wq", torch.randint(-20, 20, (k, n), generator=g).to(torch.int8), persistent=False)
    q.register_buffer("suh", (torch.randint(0, 2, (k,), generator=g) * 2 - 1).float() * 0.8, persistent=False)
    q.register_buffer("svh", (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float() * 0.8, persistent=False)
    q.register_buffer("H", deq.hadamard128("cpu"), persistent=False); q.register_buffer("Hb", q.H.to(torch.bfloat16), persistent=False)
    q.r = torch.nn.Parameter(torch.zeros(k)) if train_r else None; q.c = torch.nn.Parameter(torch.zeros(n)) if train_c else None; q.la = q.lb = None
    return q


class Tiny:
    """the Student surface the trainer uses, on a tiny stack"""
    def __init__(self, seed=0):
        torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
        lt = ["linear_attention"] * 3 + ["full_attention"]
        cfg = Qwen3_5TextConfig(vocab_size=V_, hidden_size=H, intermediate_size=384, num_hidden_layers=NL, num_attention_heads=4, num_key_value_heads=2, head_dim=128,
                                linear_key_head_dim=128, linear_value_head_dim=128, linear_num_key_heads=2, linear_num_value_heads=4, layer_types=lt,
                                max_position_embeddings=512, rope_parameters={"mrope_interleaved": True, "mrope_section": [6, 5, 5], "partial_rotary_factor": 0.25,
                                                                              "rope_theta": 10000, "rope_type": "default"})
        cfg._attn_implementation = "sdpa"
        self.cfg, self.q35, self.L, self.LP = cfg, q35, L, LP
        self.devs = [torch.device("cpu")]; self.layers = []; self.P = {}; self.qlin = {}; self.norm = {}
        for i in range(NL):
            ly = q35.Qwen3_5DecoderLayer(cfg, i).to(torch.bfloat16)
            for sub in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
                        "linear_attn.out_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
                try: lin = ly.get_submodule(sub)
                except AttributeError: continue
                key = f"{L}.{i}.{sub}"
                q = mkq(key, lin.in_features, lin.out_features, g, train_r=sub != "mlp.down_proj", train_c=sub != "self_attn.v_proj")
                parent = ly.get_submodule(sub.rsplit(".", 1)[0]); setattr(parent, sub.rsplit(".", 1)[1], q)
                self.qlin[key] = q
                if q.r is not None: self.P[key + ":r"] = q.r
                if q.c is not None: self.P[key + ":c"] = q.c
            mine = {id(p) for p in self.P.values()}
            for p in ly.parameters():
                if id(p) not in mine: p.requires_grad_(False)
            self.layers.append(ly.train())
        self.fnorm = q35.Qwen3_5RMSNorm(H, eps=cfg.rms_norm_eps).to(torch.bfloat16).requires_grad_(False)
        self.head = (torch.randn(V_, H, generator=g) * 0.1).to(torch.bfloat16); self.v2row = None
        self.rotary = {self.devs[0]: q35.Qwen3_5TextRotaryEmbedding(config=cfg)}
        self.emb = (torch.randn(V_, H, generator=g) * 0.5).to(torch.bfloat16).view(torch.int16).numpy()
        self.NL = NL
    def embed(self, ids): return torch.from_numpy(np.ascontiguousarray(self.emb[ids])).view(torch.bfloat16)[None]
    def state(self): return {n: p.detach().cpu().clone() for n, p in self.P.items()}
    def load_state(self, st):
        with torch.no_grad():
            for n, v in st.items():
                if n in self.P: self.P[n].copy_(v)


# a synthetic golden of 4 documents over the tiny vocab
tmp = tempfile.mkdtemp(); gdir = os.path.join(tmp, "gold"); os.makedirs(gdir)
nt = [300, 260, 340, 280] * 4; sf = [0, 5, 0, 3] * 4; K_ = 4; rg = np.random.default_rng(5)
R_ = sum(n - 1 - s0 for n, s0 in zip(nt, sf))
top = np.empty((R_, K_), D.rec(K_)); rows = np.zeros((R_, 3), np.float32)
for r in range(R_):
    pr = rg.dirichlet(np.ones(V_) * 0.3); ix = np.argsort(-pr)[:K_]; top["id"][r] = ix; top["lp"][r] = np.log(pr[ix]); rows[r, 2] = 1 - pr[ix].sum()
top.tofile(f"{gdir}/top.bin"); rows.tofile(f"{gdir}/rows.f32"); rg.integers(0, V_, sum(nt)).astype(np.int32).tofile(f"{gdir}/tokens.i32")
r0 = 0; docs = []
for n, s0 in zip(nt, sf): docs.append({"tokens": n, "score_from": s0, "row0": r0}); r0 += n - 1 - s0
json.dump({"docs": docs, "top": K_, "rows": R_}, open(f"{gdir}/kld.json", "w"))
(np.arange(R_) % 3 != 0).astype(np.uint8).tofile(f"{gdir}/gen.u8")
gold = D.Gold(gdir, "toy")
C = types.SimpleNamespace(g=[gold], tags=["toy"], items=[(0, i, nt[i], gold.nrows(i, 8192)) for i in range(len(nt))])
steps = D.plan(C.items, np.arange(len(nt)), 2, 128)           # 2 streams x 128 positions a step: documents continue across steps (carried state)
rows_of, cum = Q220.rows_cum(C, steps)
print(f"tiny plan: {len(steps)} steps, {int(cum[-1])} gen rows")

Q220.use_seg_attention(Tiny(0)); qmodel.use_shift_conv()


def targs(**kw):
    d = dict(lr=1e-3, wd=0.0, eps=1e-8, beta2=0.99, lr_lora=1e-4, V=2, Q=128, len=2048, trust=0.3, accum=1, clip_gnorm=0.0, prompt_weight=0.0, tap_weight=0.0)
    d.update(kw); return argparse.Namespace(**d)


def run(mod, stu, a, nsteps=2, loc=None, mask=None, **kw):
    if mod is Q220:
        stu.mask = mask; Q220.set_first_train(stu, lambda s: None, a.__dict__.get("no_frozen_prefix", False))
    T = mod.Trainer(a, stu, C, steps, rows_of, mod.Dist(), lambda s: None, **({"loc": loc} if mod is Q220 and loc is not None else {}))
    out = []
    for s in range(nsteps):
        accs, w = T.run_group([s], a.lr)
        out.append((float(accs[0]), float(accs[1]), T.gnorm, getattr(T, "clip", None)))
    return T, out


def same_state(x, y): return all(torch.equal(x[k], y[k]) for k in x)
def moved(x, y, f=lambda k: True): return any(not torch.equal(x[k], y[k]) for k in x if f(k))

# 1. defaults: Trainer of qad220 == qad200's, bit for bit
s0 = Tiny(0).state()
T2, o2 = run(Q200, Tiny(0), targs()); st2 = T2.stu.state()
T3, o3 = run(Q220, Tiny(0), targs()); st3 = T3.stu.state()
check("defaults: 2 optimizer steps of qad220's Trainer = qad200's (KL sums, rows, gradient norms) bit for bit", o2 == [x[:3] + (None,) for x in o3][:2] or [x[:3] for x in o2] == [x[:3] for x in o3], f"{o2[0][:3]} {o3[0][:3]}")
check("defaults: ... and every parameter bit for bit (parameters moved from the start)", same_state(st2, st3) and moved(s0, st3))

# 2. frozen prefix: layers 0-1 frozen: the prefix path == the full-backward path on the same trainable set
def frozen(stu, upto):
    for n, p in stu.P.items():
        if Q220.layer_of(stu, n) < upto: p.requires_grad_(False)
    return stu
Ta, oa = run(Q220, frozen(Tiny(0), 2), targs()); sa = Ta.stu.state()
Tb, ob = run(Q220, frozen(Tiny(0), 2), targs(no_frozen_prefix=True)); sb = Tb.stu.state()
check("frozen prefix (layers 0-1 frozen, real GDN / attention layers, carried state): step records and all parameters bit for bit = the full-backward path", oa == ob and same_state(sa, sb),
      f"first_train {Ta.stu.first_train} vs {Tb.stu.first_train}; max diff {max(float((sa[k] - sb[k]).abs().max()) for k in sa):.2e}")
check("frozen prefix: frozen parameters did not move, trainable did", not moved(s0, sa, lambda k: Q220.layer_of(Ta.stu, k) < 2) and moved(s0, sa, lambda k: Q220.layer_of(Ta.stu, k) >= 2) and Ta.stu.first_train == 2)

# 3. flags through the Trainer
# tied scales: tie 64 channels in layers 2-3
stu = Tiny(0); a_t = argparse.Namespace(tie_groups=64, tie_layers="2-3"); Q220.apply_ties(a_t, stu, lambda s: None)
Tt, ot = run(Q220, stu, targs()); st_t = Tt.stu.state()
q = stu.qlin[f"{L}.2.mlp.up_proj"]
check("tied scales: the Trainer trains the tied parameters (a group shares one value, the export expands per channel)", isinstance(q, Q220.TiedQLinear) and q.r.numel() == H and float(q._parameters["r_t"].abs().max()) > 0
      and torch.equal(q.r.detach(), q._parameters["r_t"].detach().repeat_interleave(64)) and stu.qlin[f"{L}.0.mlp.up_proj"].r.numel() == H and "r_t" not in stu.qlin[f"{L}.0.mlp.up_proj"]._parameters)
# mask: half of the scales of layer 3 up_proj:r, everything else unmasked
stu = Tiny(0); nm = f"{L}.3.mlp.up_proj:r"; mk = torch.zeros(stu.P[nm].numel(), dtype=torch.bool); mk[::2] = True
Tm, om = run(Q220, stu, targs(), mask={nm: mk}); sm_ = Tm.stu.state()
dm = (sm_[nm] - s0[nm]).abs()
check("mask: masked scales stay at their start value, kept ones move; unlisted vectors train as usual", bool((dm[~mk] == 0).all()) and bool((dm[mk] > 0).any()) and moved(s0, sm_, lambda k: k != nm))
# clip: a huge relative clip does nothing (bit for bit the unclipped run); a tight one clips and changes the result
Tc, oc = run(Q220, Tiny(0), targs(clip_gnorm=1e9))
check("clip: X huge = the unclipped run (bit for bit)", same_state(Tc.stu.state(), st3) and [x[:3] for x in oc] == [x[:3] for x in o3])
Tk, ok_ = run(Q220, Tiny(0), targs(clip_gnorm=0.9), nsteps=10)
nc = [x[3] for x in ok_ if x[3] is not None]
check("clip: nothing before step 6; later steps above 0.9 x the median of the earlier norms are scaled to it", all(x[3] is None for x in ok_[:5]) and len(nc) > 0
      and all(abs(f * ok_[i][2] - thr) < 1e-9 * thr for i, (f, thr) in [(i, x[3]) for i, x in enumerate(ok_) if x[3] is not None]), f"{len(nc)} clipped of 10, gnorms {[round(x[2], 4) for x in ok_]}")
# beta2: another value changes the update
Tbt, obt = run(Q220, Tiny(0), targs(beta2=0.9)); check("beta2 0.9 changes the parameters, 0.99 is the default", not same_state(Tbt.stu.state(), st3))
# prompt weight
Tp, op = run(Q220, Tiny(0), targs(prompt_weight=0.3))
check("prompt weight 0.3: gen KL sums / rows logged as before (gen only), the gradient (norm) and the parameters differ", [x[1] for x in op] == [x[1] for x in o3] and op[0][2] != o3[0][2] and not same_state(Tp.stu.state(), st3), f"gnorm {op[0][2]:.4e} vs {o3[0][2]:.4e}")
Tp0, op0 = run(Q220, Tiny(0), targs(prompt_weight=1e-12))
check("prompt weight ~0 (ext path): same step records as the plain path to rounding", all(abs(x[0] / y[0] - 1) < 1e-6 for x, y in zip(op0, o3)), f"{op0[0][0]} vs {o3[0][0]}")
# local MSE with a synthetic store of the bf16 'teacher' states at layers 1 and 2
ctx = {"ok": True}
sdir = os.path.join(tmp, "taps"); os.makedirs(sdir)
def mk_store(layers, rows_per_doc):
    o = 0; idx = {}; Jd = []
    for i in range(len(nt)):
        j = gold.scored(i, 0, 8192)[0]; idx[f"toy:{i}"] = [o, nt[i]]; Jd.append(len(j)); o += len(j)
    m = (torch.randn(o, len(layers), H, generator=torch.Generator().manual_seed(11)) * 0.5).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    m.tofile(f"{sdir}/taps.bin"); json.dump({"taps": layers, "hidden": H, "rows": o, "docs": idx}, open(f"{sdir}/index.json", "w")); open(f"{sdir}/.done", "w").close()
    return Q220.TapStore(sdir, any_layers=True)
store = mk_store([1, 2], None)
for mode in ("local", "joint"):
    stu = Tiny(0); loc = Q220.LocalMSE(f"1+2,0.5,{mode}", store)
    Tl, ol = run(Q220, stu, targs(), loc=loc); sl = Tl.stu.state()
    ch0 = moved(s0, sl, lambda k: Q220.layer_of(stu, k) == 0); ch12 = moved(s0, sl, lambda k: Q220.layer_of(stu, k) in (1, 2)); ch3 = moved(s0, sl, lambda k: Q220.layer_of(stu, k) == 3)
    check(f"local MSE ({mode}) through the Trainer: KL sums as in the plain run; layers 1-2 train; layer 0 {'gets no gradient' if mode == 'local' else 'trains through the chain'}; layer 3 (KL) trains",
          abs(ol[0][0] / o3[0][0] - 1) < 1e-6 and ch12 and ch3 and (ch0 == (mode == "joint")), f"layer0 moved {ch0}, 1-2 {ch12}, 3 {ch3}")
print("\n" + ("ALL PASS" if not fails else f"FAILED: {fails}"))
sys.exit(1 if fails else 0)
