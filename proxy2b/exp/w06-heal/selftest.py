#!/usr/bin/env python3
"""selftest.py CKPT WORK -- CPU self-test of heal.py / fold.py / mkmix.py on the real 0.8B checkpoint, synthetic
'quantized' weights (W + 3% noise, f16) and a synthetic golden. Short windows (48 tokens), fp32 or bf16 on the CPU.
  1 identity: the wrapped student at zero parameters = the plain substituted model (evalq.py's), bit-exact (bf16)
  2 fold: student with random multipliers (fp32) vs fold.py's wq + model dir loaded as evalq loads them (fp32)
  3 gauge: up_proj c == down_proj r; v_proj c == o_proj r tiled over the GQA group (fp32)
  4 gradient: autograd directional derivative of the KL vs a central finite difference (fp32)
  5 data: Meta.window rows = evalq's scored rows; qad / held-out / golden item selection on a synthetic pool
  6 resume: 6 optimizer steps straight == 2 steps, kill, restart, 4 more (bit-exact), and the early-stop bookkeeping
  7 mkmix: levels, bits per weight, transferred parameters; fold --jitter"""
import json, os, shutil, sys, types
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
CK, WK = sys.argv[1:3]
os.environ["W06_CKPT"] = CK
import heal, fold, mkmix
sys.path.insert(0, os.environ.get("W06_P2B", heal.P2B))
from common import load_model
from safetensors.torch import load_file, save_file
torch.manual_seed(0); torch.set_num_threads(int(os.environ.get("NT", 16)))
ok_all = []
def check(name, cond, msg):
    ok_all.append(bool(cond)); print(f"[{'PASS' if cond else 'FAIL'}] {name}: {msg}", flush=True)

# ---- synthetic quantized weights for every trunk linear the quantizer touches
WQ = f"{WK}/wq"
m0, dec0, _ = load_model(CK, dev="cpu")
idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
QK = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj")
keys = [k[:-7] for k in idx if k.startswith("model.language_model.layers.") and k[:-7].endswith(QK)]
if not os.path.exists(f"{WQ}/.done"):
    os.makedirs(WQ, exist_ok=True); g = torch.Generator().manual_seed(1)
    mods = {n.split("layers.", 1)[1]: m for n, m in dec0.named_modules() if n.startswith("layers.")}
    for k in keys:
        w = mods[k.split("layers.", 1)[1]].weight.float()
        save_file({"weight": (w + 0.03 * w.std() * torch.randn(w.shape, generator=g)).half()}, f"{WQ}/{k}.safetensors")
    open(f"{WQ}/.done", "w").close()
print(f"{len(keys)} synthetic quantized linears", flush=True)
ids = np.asarray(json.load(open(f"{CK}/../ids.json")) if os.path.exists(f"{CK}/../ids.json") else
                 torch.randint(0, 200000, (48,), generator=torch.Generator().manual_seed(2)).tolist(), np.int64)
pos = np.arange(8, 47)

def fresh(dtype=torch.bfloat16, wq=WQ, **kw):
    m, _, _ = load_model(CK, dev="cpu", dtype=dtype); m.requires_grad_(False)
    return heal.Student(m, wq, **kw)

# ---- 1 identity
heal.MODE.update(fp32=False, bypass=False)
s = fresh(); s.model.eval()
plain, pdec, _ = load_model(CK, dev="cpu")
pp = {n.split("layers.", 1)[1]: p for n, p in plain.named_parameters() if "layers." in n}
for k in keys:
    with torch.no_grad(): pp[k.split("layers.", 1)[1] + ".weight"].copy_(load_file(f"{WQ}/{k}.safetensors")["weight"].to(torch.bfloat16))
with torch.no_grad():
    y1 = s.logits(s.hidden(ids)); y0 = plain.get_output_embeddings()(pdec(input_ids=torch.from_numpy(ids)[None]).last_hidden_state[0]).float()
check("identity", float((y1 - y0).abs().max()) == 0.0, f"max |dlogit| {float((y1 - y0).abs().max()):.1e}; "
      f"{len(s.P)} parameter vectors ({sum(p.numel() for p in s.P.values())} values), norm form {s.form}, "
      f"{len(s.norm)} norms; gauge-fixed: no down_proj r {not any(n.endswith('down_proj:r') for n in s.P)}, "
      f"no v_proj c {not any(n.endswith('v_proj:c') for n in s.P)}")
del plain, pdec

# ---- 2 fold (fp32 end to end, so only the f16 storage of the folded product differs)
s = fresh(torch.float32); s.model.eval()
with torch.no_grad():
    yb = s.logits(s.hidden(ids))
    g = torch.Generator().manual_seed(3)
    for p in s.P.values(): p.copy_(0.05 * torch.randn(p.shape, generator=g))
    ya = s.logits(s.hidden(ids))
st = s.state(); torch.save(st, f"{WK}/p.pt")
shutil.rmtree(f"{WK}/fold", ignore_errors=True)
nf = fold.fold_wq(st["lin"], WQ, f"{WK}/fold/wq"); fold.fold_model(st["norm"], st["norm_form"], CK, f"{WK}/fold/model")
mb, db, _ = load_model(f"{WK}/fold/model", dev="cpu", dtype=torch.float32)
pb = {n.split("layers.", 1)[1]: p for n, p in mb.named_parameters() if "layers." in n}
for f in os.listdir(f"{WK}/fold/wq"):
    if f.endswith(".safetensors"):
        with torch.no_grad(): pb[f[:-12].split("layers.", 1)[1] + ".weight"].copy_(load_file(f"{WK}/fold/wq/{f}")["weight"].float())
with torch.no_grad(): yf = mb.get_output_embeddings()(db(input_ids=torch.from_numpy(ids)[None]).last_hidden_state[0]).float()
rel = float((ya - yf).norm() / (ya - yb).norm())
check("fold", rel < 0.05, f"{nf} linears + {len(st['norm'])} norms folded; |student - folded| / |effect of the "
      f"multipliers| = {rel:.2e} (f16 storage of the product); links {sum(os.path.islink(f'{WK}/fold/model/{x}') for x in os.listdir(f'{WK}/fold/model'))}")
del mb, db

# ---- 3 gauge
s = fresh(torch.float32, gauge=False); s.model.eval()
cfg = s.model.config.get_text_config() if hasattr(s.model.config, "get_text_config") else s.model.config
L = "model.language_model.layers"
def run(sets):
    with torch.no_grad():
        for p in s.P.values(): p.zero_()
        for n, v in sets.items(): s.P[n].copy_(v)
        return s.logits(s.hidden(ids))
d = 0.1 * torch.randn(cfg.intermediate_size)
a1, a2 = run({f"{L}.5.mlp.up_proj:c": d}), run({f"{L}.5.mlp.down_proj:r": d})
nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
dv = 0.1 * torch.randn(nkv * hd)
tile = torch.cat([dv[(h // (nh // nkv)) * hd:(h // (nh // nkv) + 1) * hd] for h in range(nh)])
b1, b2 = run({f"{L}.3.self_attn.v_proj:c": dv}), run({f"{L}.3.self_attn.o_proj:r": tile})
base = run({})
check("gauge", float((a1 - a2).norm() / (a1 - base).norm()) < 1e-4 and float((b1 - b2).norm() / (b1 - base).norm()) < 1e-4,
      f"up c vs down r {float((a1 - a2).norm() / (a1 - base).norm()):.1e}, v c vs o r tiled {float((b1 - b2).norm() / (b1 - base).norm()):.1e} (relative to the effect)")

# ---- 4 gradient (fp64 model, train mode with checkpointing as in training; the GDN reference path computes in fp32)
s = fresh(torch.float64); tm, _, _ = load_model(CK, dev="cpu", dtype=torch.float64); tm.requires_grad_(False); tea = heal.Teacher(tm)
s.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); s.model.train()
with torch.no_grad():
    for p in s.P.values(): p.copy_(0.02 * torch.randn(p.shape, generator=torch.Generator().manual_seed(4)))
for p in s.P.values(): p.grad = None
heal.window_kl(s, tea, ids, pos).backward()
u = {n: torch.randn(p.shape, generator=torch.Generator().manual_seed(5)) for n, p in s.P.items()}
dd = sum(float((p.grad.double() * u[n]).sum()) for n, p in s.P.items())
P0 = {n: p.detach().clone() for n, p in s.P.items()}
def at(e):
    with torch.no_grad():
        for n, p in s.P.items(): p.copy_(P0[n] + e * u[n])
        v = float(heal.window_kl(s, tea, ids, pos, grad=False))
        for n, p in s.P.items(): p.copy_(P0[n])
    return v
fds = {e: (at(e) - at(-e)) / (2 * e) for e in (1e-3, 3e-4, 1e-4)}
best = min(abs(v - dd) / abs(v) for v in fds.values())
check("gradient", best < 1e-3, f"autograd {dd:.7e} vs central differences " + ", ".join(f"eps {e:g}: {v:.7e}" for e, v in fds.items())
      + f" (best rel {best:.1e})")
del tm, tea
tm, _, _ = load_model(CK, dev="cpu", dtype=torch.float32); tm.requires_grad_(False); tea = heal.Teacher(tm)

# ---- 5 data
D = f"{WK}/gold"; os.makedirs(D, exist_ok=True); rng = np.random.default_rng(6)
docs, toks, gen, r0 = [], [], [], 0
for i in range(12):
    n = int(rng.integers(5, 5000)); sf = int(rng.integers(0, n // 2)) if i % 3 == 0 else 0
    docs.append({"source": "x", "tokens": n, "score_from": sf, "row0": r0, "orig": i}); r0 += n - 1 - sf
    toks.append(rng.integers(0, 1000, n).astype(np.int32)); gen.append((rng.random(n - 1 - sf) < 0.6).astype(np.uint8))
json.dump({"docs": docs, "rows": r0}, open(f"{D}/kld.json", "w"))
np.concatenate(toks).tofile(f"{D}/tokens.i32"); np.concatenate(gen).tofile(f"{D}/gen.u8")
M = heal.Meta(D); bad = 0
for i, dc in enumerate(docs):                       # evalq: rows sf .. len-2, row0 + j, gen-masked
    want = [dc["score_from"] + j for j in range(dc["tokens"] - 1 - dc["score_from"]) if gen[i][j]]
    got = []
    for a0 in range(0, dc["tokens"], 1000):
        wi, wp = M.window(i, a0, min(1000, dc["tokens"] - a0)); got += list(a0 + wp)
        bad += int(not np.array_equal(wi, toks[i][a0:a0 + len(wi)]))
    bad += int(got != want)
pools = ["qad", "search", "verify", "fit"] * 3
json.dump({"docs": [{"orig": 1}]}, open(f"{WK}/g.json", "w")); os.makedirs(f"{WK}/gg", exist_ok=True); shutil.copy(f"{WK}/g.json", f"{WK}/gg/kld.json")
qi = heal.qad_items(M, pools, 100, 1000, 0, 2); hi = heal.heldout_items(M, pools, f"{WK}/gg", 100, 1000, 1)
check("data", bad == 0 and all(pools[i] == "qad" for i, _, _ in qi) and all(pools[i] == "search" and i != 1 for i, _, _ in hi)
      and all(a0 // 1000 < 2 for _, a0, _ in qi) and len(heal.golden_items(M)) == sum(1 for g_ in gen if g_.any()),
      f"window rows = evalq rows on 12 docs ({r0} rows), {len(qi)} qad windows (<= 2 a doc), held-out {[i for i, _, _ in hi]} (search, golden doc 1 excluded)")

# ---- 6 resume (tiny real windows, fp32, CPU-deterministic)
items = [(i, 0, 40) for i in range(12) if len(M.window(i, 0, 40)[1])][:3]
A = types.SimpleNamespace(lr=3e-3, epochs=2, accum=1, max_steps=6, trust=0.3, eval_every=2, patience=99, seed=0)
def tr(out, stop=0):
    s_ = fresh(torch.float32, layers=(20, 23)); s_.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); s_.model.train()
    A.stop_at = stop; st_ = heal.train(s_, tea, M, items, M, items[:2], out, A, log=lambda x: None)
    return st_, s_.state()
for d_ in ("rA", "rB"): shutil.rmtree(f"{WK}/{d_}", ignore_errors=True); os.makedirs(f"{WK}/{d_}")
sa, ca = tr(f"{WK}/rA"); tr(f"{WK}/rB", stop=2); sb, cb = tr(f"{WK}/rB")
same = all(torch.equal(ca["lin"][k][x], cb["lin"][k][x]) for k in ca["lin"] for x in "rc" if ca["lin"][k][x] is not None) \
    and all(torch.equal(ca["norm"][k], cb["norm"][k]) for k in ca["norm"])
vals = [h_["val_kl"] for h_ in sa["hist"]]
check("resume", same and sa["step"] == sb["step"] == 6 and [h_["val_kl"] for h_ in sb["hist"]] == vals
      and sa["best"] == min([sa["base_val"]] + vals) and max(float(v.abs().max()) for x in ca["lin"].values() for v in x.values() if v is not None) <= 0.3,
      f"6 steps straight == 2 + resumed 4 (bit-exact); held-out KL {sa['base_val']:.5f} -> {vals} best {sa['best']:.5f}")

# ---- 7 mkmix + jitter
w2 = f"{WK}/fold/wq"                                 # stand-in for a second level
P2, P3 = st, s.state(); torch.save(P3, f"{WK}/p3.pt")
shutil.rmtree(f"{WK}/mix", ignore_errors=True)
sys.argv = ["mkmix.py", "family", f"{WK}/mix", w2, WQ, f"{WK}/p.pt", f"{WK}/p3.pt"]; mkmix.main()
mj = json.load(open(f"{WK}/mix/mix.json")); X = torch.load(f"{WK}/mix/xfer.pt", weights_only=False)
k_up, k_q = f"{L}.4.mlp.up_proj", f"{L}.4.linear_attn.in_proj_qkv"
okm = os.path.realpath(f"{WK}/mix/wq/{k_up}.safetensors") == os.path.realpath(f"{w2}/{k_up}.safetensors") \
    and os.path.realpath(f"{WK}/mix/wq/{k_q}.safetensors") == os.path.realpath(f"{WQ}/{k_q}.safetensors") \
    and torch.equal(X["lin"][k_up]["c"], P2["lin"][k_up]["c"]) and torch.equal(X["lin"][k_q]["c"], P3["lin"][k_q]["c"]) \
    and torch.equal(X["norm"][f"{L}.3.self_attn.q_norm.weight"], P3["norm"][f"{L}.3.self_attn.q_norm.weight"]) \
    and torch.allclose(X["norm"]["model.language_model.norm.weight"], 0.5 * (P2["norm"]["model.language_model.norm.weight"] + P3["norm"]["model.language_model.norm.weight"]))
shutil.rmtree(f"{WK}/jit", ignore_errors=True)
sys.argv = ["fold.py", "-", WQ, f"{WK}/jit", CK, "--jitter", "0.003"]; fold.main()
k0 = f"{L}.0.mlp.down_proj"
r_ = (load_file(f"{WK}/jit/wq/{k0}.safetensors")["weight"].float() / load_file(f"{WQ}/{k0}.safetensors")["weight"].float()).log()
check("mkmix+jitter", okm and abs(mj["bpw"] - 2.5) < 0.1 and not os.path.exists(f"{WK}/jit/model"),
      f"family mix {mj['bpw']:.3f} bits/weight, links and transferred tensors right; jitter log-ratio SD {float(r_[r_.isfinite()].std()):.4f} (want ~0.0042)")
print(f"SELFTEST {'PASS' if all(ok_all) else 'FAIL'} ({sum(ok_all)}/{len(ok_all)})")
