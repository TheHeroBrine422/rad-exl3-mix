#!/usr/bin/env python3
"""heal.py MODE OUT --wq DIR [options] -- frozen-code healing of a fake-quantized 0.8B (ranked.md H1 / H4; README.md).

Student = the bf16 checkpoint with every trunk linear whose dequantized weight is in DIR (ourq.py --save-wq, as
evalq.py) substituted and wrapped as  y = c * (W_q (r * x))  with log-parameterised r (input side) and c (output side).
quantize.py stores W_q = diag(su) H Q H diag(sv) (in, out layout), so r and c are exactly multipliers on the stored
fp16 suh / svh: the trellis codes stay frozen. Optionally q_norm / k_norm (full-attention layers) and the final norm:
log multipliers on their effective gain (1 + w). input / post-attention / GDN gated norms are not trained: their gains
are inside the span of the r of the linears that read them. Gauge (default on): down_proj r is fixed (it equals up_proj
c: down reads silu(gate) * up), v_proj c is fixed (it is o_proj r tied across the GQA group: o reads softmax V * gate).
Teacher = the same checkpoint in bf16 (online, no_grad); loss = full-vocabulary KL(p_teacher || q_student) summed over
the scored (gen) rows of each window, the challenge's per-token weighting.

modes:
  train    heal on qad-pool windows (or --data oracle: the scoring golden's own documents), Adam on the log
           parameters, clamp |log| <= --trust, early stop on a held-out search-pool slice (documents not in the
           golden); OUT/params.pt = the best held-out parameters; OUT/state.pt makes a restart resume (bit-exact
           data order); OUT/train.jsonl the curve. --gate: identity check + 8 windows timed, then exit.
  gradcos  gradients at the unhealed point over --nseq windows, student in bf16 and in fp32 (weights the same bf16
           values, cast per op; norms / GDN / attention fp32), per layer: cosine bf16 vs fp32 and the even / odd
           split-half cosine of each -> OUT/gradcos.jsonl, OUT/gradcos.txt
Env VRAM_CAP_GB caps the torch allocator."""
import argparse, json, math, os, sys, time, types
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

P2B = os.environ.get("W06_P2B", "/workspace/rad-exl3-mix/proxy2b")
CAL = os.environ.get("W06_CAL", "/data/challenge/calib-meta")          # calib docs: tokens, gen mask, score_from
POOLS = os.environ.get("W06_POOLS", "/workspace/rad-exl3-mix/eval/pools/pools-v3.json")
GOLD = os.environ.get("W06_GOLD", "/data/g08-search1")
CKPT = os.environ.get("W06_CKPT", "/data/models/qwen35-08b")

# ------------------------------------------------------------------ data
class Meta:
    """a golden-format directory (kld.json, tokens.i32, gen.u8): row r of doc i predicts token score_from + j + 1"""
    def __init__(self, d):
        self.m = json.load(open(f"{d}/kld.json")); self.docs = self.m["docs"]
        self.tok = np.memmap(f"{d}/tokens.i32", np.int32, "r"); self.gen = np.memmap(f"{d}/gen.u8", np.uint8, "r")
        self.off = np.concatenate([[0], np.cumsum([x["tokens"] for x in self.docs])])
    def window(self, i, a, n):
        """tokens [a, a + n) of doc i and the window positions whose (next-token) row is scored"""
        d = self.docs[i]
        ids = np.asarray(self.tok[self.off[i] + a:self.off[i] + a + n], np.int64)
        ap = a + np.arange(len(ids)); ok = (ap >= d["score_from"]) & (ap <= d["tokens"] - 2)
        g = np.asarray(self.gen[d["row0"] + ap[ok] - d["score_from"]]).astype(bool)
        return ids, np.nonzero(ok)[0][g]

def qad_items(meta, pools, n, L, seed, per_doc):
    """(doc, start, len) windows of qad docs, from each doc's start in L steps (short docs whole, <= per_doc a doc),
    windows without a scored row dropped; fixed shuffle, first n"""
    out = []
    for i, d in enumerate(meta.docs):
        if pools[i] != "qad": continue
        for w, a in enumerate(range(0, d["tokens"] - 1, L)):
            if w >= per_doc: break
            m = min(L, d["tokens"] - a)
            if m >= 64: out.append((i, a, m))
    rng = np.random.default_rng(seed); rng.shuffle(out)
    keep = []
    for it in out:
        if len(meta.window(*it)[1]): keep.append(it)
        if len(keep) == n: break
    return keep

def heldout_items(meta, pools, gold, n, cap, seed):
    """search-pool docs not in the scoring golden (its kld.json 'orig'), first cap tokens, fixed shuffle, first n"""
    used = {d["orig"] for d in json.load(open(f"{gold}/kld.json"))["docs"]}
    c = [i for i, p in enumerate(pools) if p == "search" and i not in used]
    rng = np.random.default_rng(seed); rng.shuffle(c)
    out = []
    for i in c:
        it = (i, 0, min(cap, meta.docs[i]["tokens"]))
        if len(meta.window(*it)[1]): out.append(it)
        if len(out) == n: break
    return out

def golden_items(meta):
    """every document of a golden, whole (its docs are already capped), as evalq.py scores them"""
    return [(i, 0, d["tokens"]) for i, d in enumerate(meta.docs) if len(meta.window(i, 0, d["tokens"])[1])]

# ------------------------------------------------------------------ the healed student
MODE = {"fp32": False, "bypass": False}

def _lin_fwd(self, x):
    if MODE["bypass"]: return F.linear(x, self.weight)
    if MODE["fp32"]:
        x = x.float()
        if self._hr is not None: x = x * self._hr.exp()
        y = F.linear(x, self.weight.float())
        return y if self._hc is None else y * self._hc.exp()
    wd = self.weight.dtype; t = torch.promote_types(wd, torch.float32)    # multipliers in fp32 (fp64: the self-test)
    if self._hr is not None: x = (x.to(t) * self._hr.exp()).to(wd)
    y = F.linear(x, self.weight)
    return y if self._hc is None else (y.to(t) * self._hc.exp()).to(wd)

def _norm_hook(mod, inp, out):
    if MODE["bypass"]: return None
    return (out.to(torch.promote_types(out.dtype, torch.float32)) * mod._hg.exp()).to(out.dtype)

def norm_form(mod):
    """effective gain of an RMSNorm: '1+w' (Qwen3.5) or 'w', found numerically"""
    x = torch.randn(4, mod.weight.shape[0], device=mod.weight.device, dtype=mod.weight.dtype)
    w0 = mod.weight.data.clone()
    with torch.no_grad():
        mod.weight.data.zero_(); y = mod(x); mod.weight.data.copy_(w0)
    return "w" if float(y.float().abs().max()) == 0.0 else "1+w"

class Student:
    """wraps a loaded model in place; .P = {name: Parameter} (name 'key:r' / 'key:c' / 'normkey:g'), .lin key -> module"""
    def __init__(self, model, wq, layers=(0, 10**6), norms=True, gauge=True, load=True):
        from safetensors.torch import load_file
        self.model = model; self.dec = model.get_decoder(); self.P = {}; self.lin = {}; self.norm = {}
        mods = {n.split("layers.", 1)[1]: m for n, m in self.dec.named_modules() if n.startswith("layers.")}
        self.keys = []
        for f in sorted(os.listdir(wq)):
            if not f.endswith(".safetensors"): continue
            key = f[:-len(".safetensors")]; suf = key.split("layers.", 1)[1]
            m = mods[suf]
            if load:
                w = load_file(os.path.join(wq, f))["weight"]
                assert tuple(w.shape) == tuple(m.weight.shape), (key, w.shape, m.weight.shape)
                with torch.no_grad(): m.weight.copy_(w.to(m.weight.dtype))
            self.lin[key] = m; self.keys.append(key)
        dev = next(model.parameters()).device
        for n, m in self.dec.named_modules():         # every trunk linear goes through the wrapper (fp32 mode needs a/b too)
            if n.startswith("layers.") and isinstance(m, torch.nn.Linear):
                m._hr = m._hc = None; m.forward = types.MethodType(_lin_fwd, m)
        lo, hi = layers; self.nl = len(self.dec.layers)
        for key, m in self.lin.items():
            l = int(key.split("layers.", 1)[1].split(".")[0])
            if not lo <= l <= hi: continue
            if not (gauge and key.endswith("mlp.down_proj")):
                m._hr = self.P[key + ":r"] = torch.nn.Parameter(torch.zeros(m.in_features, device=dev))
            if not (gauge and key.endswith("self_attn.v_proj")):
                m._hc = self.P[key + ":c"] = torch.nn.Parameter(torch.zeros(m.out_features, device=dev))
        self.form = None
        if norms:
            pre = self.keys[0].split("layers.", 1)[0]          # e.g. "model.language_model."
            cand = [(f"{pre}layers.{n.split('layers.', 1)[1]}.weight", m) for n, m in self.dec.named_modules()
                    if n.startswith("layers.") and n.endswith(("self_attn.q_norm", "self_attn.k_norm"))
                    and lo <= int(n.split(".")[1]) <= hi]
            if hi >= self.nl - 1: cand.append((f"{pre}norm.weight", self.dec.norm))
            for k, m in cand:
                f_ = norm_form(m); assert self.form in (None, f_); self.form = f_
                m._hg = self.P[k + ":g"] = torch.nn.Parameter(torch.zeros(m.weight.shape[0], device=dev))
                m.register_forward_hook(_norm_hook); self.norm[k] = m

    def to_fp32(self):
        """fp32 student: every non-linear, non-embedding parameter fp32 (linear weights stay bf16, cast per op)"""
        MODE["fp32"] = True
        for m in self.model.modules():
            if not isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
                for _, p in m.named_parameters(recurse=False):
                    if p.dtype != torch.float32 and not any(p is q for q in self.P.values()): p.data = p.data.float()

    def hidden(self, ids):
        t = torch.from_numpy(ids)[None].to(next(self.model.parameters()).device)
        if MODE["fp32"]: return self.dec(inputs_embeds=self.dec.embed_tokens(t).float(), use_cache=False).last_hidden_state[0]
        return self.dec(input_ids=t, use_cache=False).last_hidden_state[0]

    def logits(self, h):
        W = self.model.get_output_embeddings().weight
        if h.dtype == torch.float32 and W.dtype != torch.float32:      # fp32 head without a whole fp32 copy
            return torch.cat([h @ W[i:i + 32768].float().T for i in range(0, W.shape[0], 32768)], -1)
        y = F.linear(h, W); return y if y.dtype == torch.float64 else y.float()

    def state(self):
        """{'lin': {key: {'r', 'c'}}, 'norm': {normkey: g}} on the CPU (None = fixed at 1)"""
        lin = {k: {s: (self.P[f"{k}:{s}"].detach().cpu().clone() if f"{k}:{s}" in self.P else None) for s in "rc"}
               for k in self.keys if f"{k}:r" in self.P or f"{k}:c" in self.P}
        return {"lin": lin, "norm": {k: self.P[k + ":g"].detach().cpu().clone() for k in self.norm}, "norm_form": self.form}

    def load_state(self, st):
        with torch.no_grad():
            for k, v in st["lin"].items():
                for s in "rc":
                    if v.get(s) is not None and f"{k}:{s}" in self.P: self.P[f"{k}:{s}"].copy_(v[s])
            for k, g in st["norm"].items():
                if k + ":g" in self.P: self.P[k + ":g"].copy_(g)

# ------------------------------------------------------------------ loss
class Teacher:
    def __init__(self, model):
        self.model = model.eval(); self.dec = model.get_decoder()
    @torch.no_grad()
    def hidden(self, ids):
        t = torch.from_numpy(ids)[None].to(next(self.model.parameters()).device)
        return self.dec(input_ids=t, use_cache=False).last_hidden_state[0]
    @torch.no_grad()
    def logits(self, h):
        y = F.linear(h, self.model.get_output_embeddings().weight); return y if y.dtype == torch.float64 else y.float()

def window_kl(stu, tea, ids, pos, ch=128, grad=True):
    """sum over the scored positions of KL(p_teacher || q_student), full vocabulary; chunks recomputed in backward"""
    p_ = torch.from_numpy(pos).to(next(stu.model.parameters()).device)
    ht = tea.hidden(ids)[p_]
    hs = stu.hidden(ids)[p_]
    def chunk(h, t):
        with torch.no_grad(): lp = torch.log_softmax(tea.logits(t), -1)
        lq = torch.log_softmax(stu.logits(h), -1)
        return (lp.exp() * (lp - lq)).sum()
    tot = 0.0
    for c in range(0, hs.shape[0], ch):
        tot = tot + (checkpoint(chunk, hs[c:c + ch], ht[c:c + ch], use_reentrant=False) if grad else chunk(hs[c:c + ch], ht[c:c + ch]))
    return tot

@torch.no_grad()
def eval_kl(stu, tea, meta, items):
    stu.model.eval(); s = n = 0.0
    for it in items:
        ids, pos = meta.window(*it); s += float(window_kl(stu, tea, ids, pos, grad=False)); n += len(pos)
    stu.model.train()
    return s / n

# ------------------------------------------------------------------ training
def lr_at(step, total, lr, warm=16):
    if step < warm: return lr * (step + 1) / warm
    return lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))

def train(stu, tea, meta, items, hmeta, held, out, a, log=print):
    """Adam on the log parameters; early stop on held; resumable from out/state.pt; returns the best state"""
    params = list(stu.P.values())
    opt = torch.optim.Adam(params, lr=a.lr, betas=(0.9, 0.99))
    order = np.concatenate([np.random.default_rng(a.seed + e).permutation(len(items)) for e in range(a.epochs)])
    total = len(order) // a.accum
    if a.max_steps: total = min(total, a.max_steps)
    sp = os.path.join(out, "state.pt")
    st = {"step": 0, "best": None, "best_state": None, "bad": 0, "hist": [], "base_val": None}
    if os.path.exists(sp):
        z = torch.load(sp, weights_only=False); stu.load_state(z["cur"]); opt.load_state_dict(z["opt"]); st = z["st"]
        log(f"resumed at step {st['step']}")
    if st["base_val"] is None:
        st["base_val"] = eval_kl(stu, tea, hmeta, held); st["best"] = st["base_val"]; st["best_state"] = stu.state()
        log(f"step 0 held-out KL {st['base_val']:.6f}")
    stu.model.train(); t0 = time.time(); tl = []
    while st["step"] < total and st["bad"] < a.patience:
        s = st["step"]
        for g in opt.param_groups: g["lr"] = lr_at(s, total, a.lr)
        opt.zero_grad(set_to_none=True); ls = nr = 0.0
        for j in range(a.accum):
            ids, pos = meta.window(*items[order[s * a.accum + j]])
            loss = window_kl(stu, tea, ids, pos); (loss / 1000.0).backward()
            ls += float(loss.detach()); nr += len(pos)
        opt.step()
        with torch.no_grad():
            for p in params: p.clamp_(-a.trust, a.trust)
        st["step"] = s + 1; tl.append(ls / nr)
        if st["step"] % a.eval_every == 0 or st["step"] == total:
            v = eval_kl(stu, tea, hmeta, held)
            if v < st["best"]: st["best"], st["best_state"], st["bad"] = v, stu.state(), 0
            else: st["bad"] += 1
            mx = max(float(p.detach().abs().max()) for p in params)
            rec = {"step": st["step"], "train_kl": float(np.mean(tl)), "val_kl": v, "best": st["best"],
                   "max_abs_log": mx, "s_per_window": (time.time() - t0) / (len(tl) * a.accum)}
            st["hist"].append(rec); tl = []; t0 = time.time()
            with open(os.path.join(out, "train.jsonl"), "a") as fh: fh.write(json.dumps(rec) + "\n")
            log(json.dumps(rec))
            torch.save({"cur": stu.state(), "opt": opt.state_dict(), "st": st}, sp + ".part"); os.replace(sp + ".part", sp)
            if getattr(a, "stop_at", 0) and st["step"] >= a.stop_at: break      # self-test: a killed run
    return st

def build(a, dev):
    sys.path.insert(0, P2B)
    from common import load_model
    tm, _, _ = load_model(CKPT, dev=dev); sm, _, _ = load_model(CKPT, dev=dev)
    tm.requires_grad_(False); sm.requires_grad_(False)
    lo, hi = map(int, a.layers.split("-"))
    stu = Student(sm, a.wq, (lo, hi), norms=not a.no_norms, gauge=not a.no_gauge)
    sm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); sm.train()
    return stu, Teacher(tm)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("train", "gradcos")); ap.add_argument("out")
    ap.add_argument("--wq", required=True); ap.add_argument("--layers", default="0-23")
    ap.add_argument("--data", choices=("qad", "oracle"), default="qad")
    ap.add_argument("--ntrain", type=int, default=1024); ap.add_argument("--len", type=int, default=2048)
    ap.add_argument("--per-doc", type=int, default=8); ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--trust", type=float, default=0.3); ap.add_argument("--eval-every", type=int, default=32)
    ap.add_argument("--patience", type=int, default=4); ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--nheld", type=int, default=48); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-norms", action="store_true"); ap.add_argument("--no-gauge", action="store_true")
    ap.add_argument("--init", default=None, help="params.pt to start from (warm polish)")
    ap.add_argument("--nseq", type=int, default=64, help="gradcos: windows")
    ap.add_argument("--gate", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    cap = float(os.environ.get("VRAM_CAP_GB", 0))
    if cap: torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    meta = Meta(CAL); pools = json.load(open(POOLS))["pool"]; assert len(pools) == len(meta.docs)
    held = heldout_items(meta, pools, GOLD, a.nheld, a.len, 1)
    if a.data == "oracle": gm = Meta(GOLD); items = golden_items(gm); hmeta, held = gm, items   # in-sample by design
    else: gm = meta; items = qad_items(meta, pools, a.ntrain, a.len, a.seed, a.per_doc); hmeta = meta
    assert all(pools[i] == "qad" for i, _, _ in items) if a.data == "qad" else True
    print(f"{len(items)} training windows ({a.data}), {len(held)} held-out docs", flush=True)
    stu, tea = build(a, dev)
    if a.init: stu.load_state(torch.load(a.init, weights_only=False))
    print(f"{len(stu.P)} parameter vectors, {sum(p.numel() for p in stu.P.values())} values, norm form {stu.form}", flush=True)
    if a.gate:
        ids, pos = hmeta.window(*held[0]); t = torch.from_numpy(pos).cuda()
        with torch.no_grad():
            stu.model.eval(); y1 = stu.logits(stu.hidden(ids)[t[:64]])
            MODE["bypass"] = True; y0 = stu.logits(stu.hidden(ids)[t[:64]]); MODE["bypass"] = False
        d = float((y1 - y0).abs().max()); print(f"gate identity: wrapped vs plain max |dlogit| {d:.3e}", flush=True)
        assert d == 0.0, "wrapper is not the identity at zero"
        stu.model.train(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time(); nr = 0
        for it in items[:8]:
            ids, pos = gm.window(*it); (window_kl(stu, tea, ids, pos) / 1000).backward(); nr += len(pos)
        torch.cuda.synchronize()
        g = sum(float(p.grad.norm()) ** 2 for p in stu.P.values() if p.grad is not None) ** 0.5
        print(f"GATE OK {(time.time() - t0) / 8:.2f} s/window ({nr / 8:.0f} scored rows), peak "
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, grad norm {g:.3e}", flush=True)
        return
    if a.mode == "gradcos": return gradcos(stu, tea, gm, items[:a.nseq], a.out)
    st = train(stu, tea, gm, items, hmeta, held, a.out, a, log=lambda s: print(s, flush=True))
    torch.save({**st["best_state"], "meta": {"args": vars(a), "base_val": st["base_val"], "best_val": st["best"],
                                             "steps": st["step"], "hist": st["hist"]}}, os.path.join(a.out, "params.pt"))
    print(f"HEAL DONE steps {st['step']} held-out KL {st['base_val']:.6f} -> {st['best']:.6f} "
          f"({st['best'] / st['base_val'] - 1:+.2%})", flush=True)

def layer_groups(stu):
    """parameter name -> layer index (final norm: nl)"""
    return {n: (int(n.split("layers.", 1)[1].split(".")[0]) if "layers." in n else stu.nl) for n in stu.P}

def gradcos(stu, tea, meta, items, out):
    """summed gradients at the unhealed point, bf16 then fp32 student, all and even / odd halves, per layer"""
    G = {}
    for mode in ("bf16", "fp32"):
        if mode == "fp32": stu.to_fp32()
        acc = {h: {n: torch.zeros_like(p, dtype=torch.float64) for n, p in stu.P.items()} for h in (0, 1)}
        t0 = time.time()
        for s, it in enumerate(items):
            for p in stu.P.values(): p.grad = None
            ids, pos = meta.window(*it); (window_kl(stu, tea, ids, pos) / 1000).backward()
            for n, p in stu.P.items(): acc[s % 2][n] += p.grad.double()
        G[mode] = acc; print(f"{mode}: {(time.time() - t0) / len(items):.2f} s/window", flush=True)
    lg = layer_groups(stu)
    def vec(acc, l, h=None):
        ns = [n for n in stu.P if lg[n] == l]
        return torch.cat([(acc[0][n] + acc[1][n]) if h is None else acc[h][n] for n in ns])
    cos = lambda x, y: float((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-300))
    rows = []
    for l in sorted(set(lg.values())):
        b, f = vec(G["bf16"], l), vec(G["fp32"], l)
        rows.append({"layer": l, "cos_bf16_fp32": cos(b, f), "rel_l2": float((b - f).norm() / f.norm()),
                     "halves_fp32": cos(vec(G["fp32"], l, 0), vec(G["fp32"], l, 1)),
                     "halves_bf16": cos(vec(G["bf16"], l, 0), vec(G["bf16"], l, 1))})
    with open(os.path.join(out, "gradcos.jsonl"), "w") as fh:
        for r in rows: fh.write(json.dumps(r) + "\n")
    ok = all(r["cos_bf16_fp32"] >= 0.95 or r["cos_bf16_fp32"] >= r["halves_fp32"] for r in rows)
    lines = ["layer  cos(bf16,fp32)  relL2   halves fp32  halves bf16"] + \
            [f"{r['layer']:5d}  {r['cos_bf16_fp32']:14.5f}  {r['rel_l2']:.2e}  {r['halves_fp32']:11.5f}  {r['halves_bf16']:11.5f}" for r in rows] + \
            [f"GRADCOS {'PASS' if ok else 'FAIL'} (every layer: cos >= 0.95 or >= its fp32 split-half cos)"]
    open(os.path.join(out, "gradcos.txt"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines), flush=True)

if __name__ == "__main__":
    main()
