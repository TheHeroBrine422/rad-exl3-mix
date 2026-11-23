#!/usr/bin/env python3
"""testhf225.py -- CPU checks of the w225 switches of qad220.py on a tiny REAL Qwen3.5 stack (8 layers: GDN, GDN, GDN, attention, x 2; every
trunk linear a QLinear with random int8 codes; testhf220's construction), through the real Trainer.run_group (2 streams x 128 positions, documents
carried across steps). Two "cards" = two CPU device entries (the split's code paths: boundary, per-card states, the pipeline's worker thread):
  1. buffered KV (SegCacheBuf: CPU park, and kept on the card) = the default loop, bit for bit;
  2. --split-layers 4 (naive, connected graph) = the default one-card loop, bit for bit;
  3. --pipeline = the naive split, bit for bit (also with the first half frozen, and with --accum 2 at V = 1: a stream's consecutive segments);
  4. --attn-gqa: rounding-level vs the default; --depth-test runs (positions and K / V offset; KL differs).
Needs transformers (box2: HIP_VISIBLE_DEVICES= /data/q27/venv/bin/python testhf225.py with QAD_PIPELINE set). Exit 0 = all pass."""
import argparse, json, os, sys, tempfile, types
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "..", "qad"), os.path.join(HERE, "..", "qad181"), os.path.join(HERE, "..", "qad220")):
    sys.path.insert(0, p)
import qmodel, data181 as D                                    # noqa: E402
from rxq import deq                                            # noqa: E402
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402
import transformers.models.qwen3_5.modeling_qwen3_5 as q35     # noqa: E402
import qad220 as Q                                             # noqa: E402

torch.set_num_threads(4)
fails = []
def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info else ""), flush=True)
    if not ok: fails.append(name)

NL, H, V_ = 8, 256, 64
LP = "model.language_model."; L = LP + "layers"
CPU = torch.device("cpu")


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
    """the Student surface the trainer uses, on a tiny 8-layer stack; split = K puts layers K.. on the second device entry"""
    def __init__(self, seed=0, split=0):
        torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
        lt = (["linear_attention"] * 3 + ["full_attention"]) * 2
        cfg = Qwen3_5TextConfig(vocab_size=V_, hidden_size=H, intermediate_size=384, num_hidden_layers=NL, num_attention_heads=4, num_key_value_heads=2, head_dim=128,
                                linear_key_head_dim=128, linear_value_head_dim=128, linear_num_key_heads=2, linear_num_value_heads=4, layer_types=lt,
                                max_position_embeddings=4096, rope_parameters={"mrope_interleaved": True, "mrope_section": [6, 5, 5], "partial_rotary_factor": 0.25,
                                                                               "rope_theta": 10000, "rope_type": "default"})
        cfg._attn_implementation = "sdpa"
        self.cfg, self.q35, self.L, self.LP = cfg, q35, L, LP
        self.devs = [CPU, CPU] if split else [CPU]; self.dev_of = [CPU] * NL; self.split = split
        self.layers = []; self.P = {}; self.qlin = {}; self.norm = {}
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
        self.rotary = {CPU: q35.Qwen3_5TextRotaryEmbedding(config=cfg)}
        self.emb = (torch.randn(V_, H, generator=g) * 0.5).to(torch.bfloat16).view(torch.int16).numpy()
        self.NL = NL
    def embed(self, ids): return torch.from_numpy(np.ascontiguousarray(self.emb[ids])).view(torch.bfloat16)[None]
    def state(self): return {n: p.detach().cpu().clone() for n, p in self.P.items()}
    def load_state(self, st):
        with torch.no_grad():
            for n, v in st.items():
                if n in self.P: self.P[n].copy_(v)


tmp = tempfile.mkdtemp(); gdir = os.path.join(tmp, "gold"); os.makedirs(gdir)
nt = [300, 260, 340, 280, 700, 330] * 2; sf = [0, 5, 0, 3, 0, 2] * 2; K_ = 4; rg = np.random.default_rng(5)
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
Q.use_seg_attention(Tiny(0)); qmodel.use_shift_conv()


def targs(**kw):
    d = dict(lr=1e-3, wd=0.0, eps=1e-8, beta2=0.99, lr_lora=1e-4, V=2, Q=128, len=96, trust=0.3, accum=1, clip_gnorm=0.0, prompt_weight=0.0, tap_weight=0.0,
             clip_doc=0.0, lr_layers=None, pipeline=False, state_card_gb=0.0)
    d.update(kw); return argparse.Namespace(**d)


def run(stu, a, nsteps=3, kvbuf=False, pos_off=0, gqa=False, train_from=0):
    stu.kvbuf = kvbuf or bool(stu.split); stu.pos_off = pos_off; stu.kv_cap = 0; Q.ATTN_GQA[0] = gqa
    for n, p in stu.P.items():
        if Q.layer_of(stu, n) < train_from: p.requires_grad_(False)
    stu.mask = None; Q.set_first_train(stu, lambda s: None)
    steps = D.plan(C.items, np.arange(len(nt)), a.V, a.Q)
    rows_of, cum = Q.rows_cum(C, steps)
    T = Q.Trainer(a, stu, C, steps, rows_of, Q.Dist(), lambda s: None)
    out = []
    for s in range(0, nsteps * a.accum, a.accum):
        accs, w = T.run_group(list(range(s, s + a.accum)), a.lr)
        out.append((float(accs[0]), float(accs[1]), T.gnorm, float(accs[-1])))
    Q.ATTN_GQA[0] = False
    return T, out


def same(x, y): return all(torch.equal(x[k], y[k]) for k in x)


T0, o0 = run(Tiny(0), targs()); s0 = T0.stu.state()
print("default:", o0)
if os.environ.get("Q220_OLD"):                                # the trainer before w225 (e.g. the queue's installed copy): its default loop = this one's
    import importlib.util
    sp = importlib.util.spec_from_file_location("qad220_old", os.environ["Q220_OLD"]); QO = importlib.util.module_from_spec(sp); sp.loader.exec_module(QO)
    QO.use_seg_attention(Tiny(0)); stu = Tiny(0); stu.mask = None; QO.set_first_train(stu, lambda s: None)
    st_ = D.plan(C.items, np.arange(len(nt)), 2, 128); ro_, _ = QO.rows_cum(C, st_)
    TO = QO.Trainer(targs(), stu, C, st_, ro_, QO.Dist(), lambda s: None); oo = []
    for s in range(3):
        accs, w = TO.run_group([s], 1e-3); oo.append((float(accs[0]), float(accs[1]), TO.gnorm, float(accs[-1])))
    Q.use_seg_attention(Tiny(0))
    check(f"defaults: this qad220 = {os.environ['Q220_OLD']} bit for bit (3 steps: KL sums, rows, gnorm, parameters)", oo == o0 and same(stu.state(), s0), f"{oo[0]} vs {o0[0]}")
Tb, ob = run(Tiny(0), targs(), kvbuf=True)
check("buffered KV, parked on the CPU: = default loop bit for bit (KL sums, rows, gnorm, parameters)", ob == o0 and same(Tb.stu.state(), s0), f"{ob[0]} vs {o0[0]}")
Tk, ok_ = run(Tiny(0), targs(state_card_gb=1.0), kvbuf=True)
check("buffered KV, kept in place (--state-card-gb): = default bit for bit", ok_ == o0 and same(Tk.stu.state(), s0) and any(isinstance(v, Q.KVView) and v.device.type == "cpu" for _, _, st in Tk.states.values() for v in st.values()))
Ts, os_ = run(Tiny(0, split=4), targs())
check("--split-layers 4 (naive, connected graph over the two device entries): = default one-card loop bit for bit", os_ == o0 and same(Ts.stu.state(), s0), f"{os_[0]} vs {o0[0]}")
Tp, op = run(Tiny(0, split=4), targs(pipeline=True))
check("--pipeline: = naive split bit for bit", op == os_ and same(Tp.stu.state(), Ts.stu.state()), f"{op[0]} vs {os_[0]}")
Tp2, op2 = run(Tiny(0, split=4), targs(pipeline=True, state_card_gb=1.0))
check("--pipeline with states kept on the cards: = naive split bit for bit", op2 == os_ and same(Tp2.stu.state(), Ts.stu.state()))
Tf0, of0 = run(Tiny(0), targs(), train_from=5); Tf1, of1 = run(Tiny(0, split=4), targs()   , train_from=5); Tf2, of2 = run(Tiny(0, split=4), targs(pipeline=True), train_from=5)
check("first half frozen (train layers 5-7, split 4): default = naive split = pipeline bit for bit", of0 == of1 == of2 and same(Tf1.stu.state(), Tf0.stu.state()) and same(Tf2.stu.state(), Tf0.stu.state()), f"{of0[0]} {of1[0]} {of2[0]}")
Ta0, oa0 = run(Tiny(0), targs(V=1, accum=2)); Ta1, oa1 = run(Tiny(0, split=4), targs(V=1, accum=2, pipeline=True))
check("--accum 2 at V = 1 (a stream's consecutive segments in one pipeline): pipeline = default bit for bit", oa0 == oa1 and same(Ta1.stu.state(), Ta0.stu.state()), f"{oa0[0]} vs {oa1[0]}")
Tg, og = run(Tiny(0), targs(), gqa=True)
check("--attn-gqa: KL sums within 1e-3 relative of the default (rounding-level), rows equal", all(abs(x[0] / y[0] - 1) < 1e-3 and x[1] == y[1] for x, y in zip(og, o0)), f"{[x[0] for x in og]} vs {[x[0] for x in o0]}")
Tgs, ogs = run(Tiny(0, split=4), targs(pipeline=True), gqa=True)
check("--attn-gqa: pipeline = one card (same attention form) bit for bit", ogs == og, f"{ogs[0]} vs {og[0]}")
Td, od = run(Tiny(0, split=4), targs(pipeline=True), pos_off=512)
Td1, od1 = run(Tiny(0, split=4), targs(), pos_off=512)
kv = [v for _, _, st in Td.states.values() for k, v in st.items() if k[0] == "kv"]
check("--depth-test 512: runs (pipeline = naive), KL differs from the default, carried K / V include the 512 synthetic positions", od == od1 and od[0][0] != o0[0][0] and len(kv) > 0 and all(v.n > 512 for v in kv), f"{od[0]} vs {o0[0]}; kv n {[v.n for v in kv][:4]}")
print("\n" + ("ALL PASS" if not fails else f"FAILED: {fails}"))
sys.exit(1 if fails else 0)
