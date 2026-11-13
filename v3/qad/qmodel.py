# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The QAD student (#165): a Qwen3.5-family model whose quantized trunk linears hold their frozen trellis codes (decoded
once at load to the codebook's integers, int8 on the card) and train only continuous parameters: log multipliers r / c
on every tensor's stored suh / svh, optionally log gains on q_norm / k_norm / the final norm (arm C). Everything else
(embedding, GDN conv / A / dt / a / b projections, layer norms, lm_head) is the bf16 checkpoint's, frozen.

  y = (had_n(had_k(x . suh e^r) @ W_hat) . s) . svh e^c       (rxq.deq's W = diag(suh) Hk W_hat Hn diag(svh), x W)
W_hat is the codebook value of each trellis state. Integer codebooks (i8, i8c: q s4; i4: (2q+1) st/2) are stored as the
int8 integers m and an f16 step s with fp16(m s) == the codebook value bit for bit (checked at load), so the bf16 GEMM
runs on m exactly and s is applied in f32 to its output (= the int8 kernel's arithmetic; the f16 LUT rounds m s once,
at most 2^-11 relative). mul1 tensors are held as bf16 W_hat (2^-9 relative). Activations bf16 around the GEMM, f32
around it (hadamards, scales).

Gauge (heal.py, #89): down_proj r fixed (= up_proj c through silu(g) u), v_proj c fixed (inside o_proj r's span).
Layers are spread over the given devices in contiguous ranges; the embedding is a memmap of the checkpoint (rows on
demand), lm_head + final norm sit on the last device. An optional bf16 teacher (arm B) puts the same layer ranges on the
same devices."""
import json, math, os, struct, sys, types

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("QAD_PIPELINE", os.path.join(_HERE, "..", "..", "pipeline")))
from rxq import deq  # noqa: E402

BF = torch.bfloat16


def _f16(bits): return float(np.array([bits], np.uint16).view(np.float16)[0])


def int_form(cb):
    """(step s as a python float exactly representable in f16, offset form) of an integer codebook, else None:
    value = fp16(m * s) with m int8"""
    if cb in ("i8", "i8c"): return float(np.float16(4.0 * _f16(0x1EEE)))
    if cb == "i4": return float(np.float16(np.float32(2.6) / np.float32(7.5) / np.float32(2.0)))
    return None


def to_int8(what, cb):
    """W_hat f32 [k, n] -> (int8 m, s) with fp16(m s) == W_hat (bit-exact), or None"""
    s = int_form(cb)
    if s is None: return None
    m = torch.round(what / s)
    if float(m.abs().max()) > 127: return None
    back = (m.half() * torch.tensor(s, dtype=torch.float16, device=m.device)).float()
    if not torch.equal(back, what): return None
    return m.to(torch.int8), s


class QLinear(torch.nn.Module):
    """a frozen-code trunk linear with trainable log multipliers on suh (r, input side) and svh (c, output side);
    lora = r > 0 adds a rank-r adapter on the layer's input, y += (x A^T) B^T (arm L: A [r, k] ~ N(0, 1/k), B [n, r]
    = 0, f32 parameters, bf16 GEMMs); hold_bf16 keeps the integers m as bf16 on the card (exact; the head, so its
    forward has no int8 -> bf16 transient per chunk)"""
    def __init__(self, key, raw, cb, table, dev, train_r=True, train_c=True, lora=0, hold_bf16=False, seed=0):
        super().__init__()
        self.key = key
        tr = raw["trellis"].to(dev)
        self.K = tr.shape[2] / 16
        w = deq.decode_hat(tr, self.K, table)                         # f32 [k, n]
        del tr
        self.k, self.n = w.shape
        ii = to_int8(w, cb)
        if ii is not None:
            self.register_buffer("wq", (ii[0].to(BF) if hold_bf16 else ii[0]).cpu(), persistent=False); self.s = ii[1]; self.exact = True
        else:
            self.register_buffer("wq", w.to(BF).cpu(), persistent=False); self.s = 1.0; self.exact = False
        del w, ii                                   # decoded on dev, held on the CPU until the layer is placed
        self.register_buffer("suh", raw["suh"].float(), persistent=False)
        self.register_buffer("svh", raw["svh"].float(), persistent=False)
        self.register_buffer("H", deq.hadamard128("cpu"), persistent=False)
        self.register_buffer("Hb", self.H.to(BF), persistent=False)
        self.r = torch.nn.Parameter(torch.zeros(self.k)) if train_r else None
        self.c = torch.nn.Parameter(torch.zeros(self.n)) if train_c else None
        self.in_features, self.out_features = self.k, self.n
        self.la = self.lb = None
        if lora:
            g = torch.Generator().manual_seed(seed)
            self.la = torch.nn.Parameter(torch.randn(lora, self.k, generator=g) / math.sqrt(self.k))
            self.lb = torch.nn.Parameter(torch.zeros(self.n, lora))

    def su(self): return self.suh * self.r.exp() if self.r is not None else self.suh

    def sv(self): return self.svh * self.c.exp() if self.c is not None else self.svh

    def forward(self, x):
        sh = x.shape
        x0 = x.reshape(-1, self.k)
        x = (x0 * self.su()).to(BF)                                    # f32 scale, bf16 out
        x = (x.view(-1, 128) @ self.Hb).view(-1, self.k)               # hadamards as one 2D bf16 GEMM (f32 accumulate)
        y = x @ self.wq.to(BF)
        y = (y.view(-1, 128) @ self.Hb).view(-1, self.n)
        y = y * (self.sv() * self.s)
        if self.la is not None:
            y = y + (x0.to(BF) @ self.la.to(BF).T) @ self.lb.to(BF).T
        return y.to(BF).view(*sh[:-1], self.n)

    @torch.no_grad()
    def export(self):
        """the trained suh / svh as stored (f16)"""
        return self.su().half().cpu(), self.sv().half().cpu()


def shift_conv1d(hidden_states, weight, bias=None, activation=None, **kwargs):
    """transformers' causal_conv1d_fn (depthwise causal conv, [B, C, L] x [C, K]) as K shifted multiply-adds in f32,
    rounded to the weight dtype like F.conv1d's output: no MIOpen (its per-shape solver search stalls the 8,192-token
    windows, arm W)"""
    from transformers.activations import ACT2FN
    L, K = hidden_states.shape[-1], weight.shape[-1]
    x = F.pad(hidden_states.to(weight.dtype).float(), (K - 1, 0)); w = weight.float()
    out = x[:, :, 0:L] * w[:, 0:1]
    for k in range(1, K): out = out + x[:, :, k:k + L] * w[:, k:k + 1]
    if bias is not None: out = out + bias.float()[:, None]
    out = out.to(weight.dtype)
    if activation is not None: out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def use_shift_conv():
    import transformers.models.qwen3_5.modeling_qwen3_5 as q35
    q35.causal_conv1d_fn = shift_conv1d


def _norm_hook(mod, inp, out):
    return (out.float() * mod._qg.exp()).to(out.dtype)


class Student:
    """layers on devices; .P = {name: Parameter}; .qlin = {key: QLinear}; .norm = {key: module with _qg}
    head_dir: a quantized head (ourq --only-head dir, `lm_head`) as a QLinear instead of the bf16 lm_head, its rows in
    the stored order; head_perm (stored row s = vocab head_perm[s], R16.npy) maps the targets' vocab ids to stored rows
    (.v2row); head_train: its r / c trainable (arm H). lora: rank of the adapters on every trunk QLinear (arm L)."""
    def __init__(self, model_dir, level_dir, devs, layers=None, norms=False, gauge=True, teacher=False, quant_layers=None, log=print,
                 head_dir=None, head_perm=None, head_train=False, lora=0):
        from safetensors import safe_open
        from transformers import AutoConfig
        import transformers.models.qwen3_5.modeling_qwen3_5 as q35
        self.q35, self.dir, self.log = q35, model_dir, log
        cfg = AutoConfig.from_pretrained(model_dir)
        cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        cfg._attn_implementation = "sdpa"
        self.cfg = cfg
        idx = json.load(open(f"{model_dir}/model.safetensors.index.json"))["weight_map"]
        self.wmap = idx
        self.LP = "model.language_model." if any(k.startswith("model.language_model.") for k in idx) else "model."
        self.L = self.LP + "layers"
        self.NL = cfg.num_hidden_layers if layers is None else layers          # a truncated model: the first `layers`
        self.devs = [torch.device(d) for d in devs]
        nd = len(self.devs)
        self.dev_of = [self.devs[min(nd - 1, i * nd // self.NL)] for i in range(self.NL)]
        self.lev = deq.Level(level_dir, "cpu")
        self.cb = self.lev.codebook
        self.tables = {d: torch.from_numpy(deq.lut(self.cb)).to(d) for d in self.devs}
        qkeys = set(self.lev.keys())
        self.quant_layers = set(range(self.NL)) if quant_layers is None else set(quant_layers)
        self.P, self.qlin, self.norm, self.layers, self.tlayers = {}, {}, {}, [], []
        self._fh = {}
        for i in range(self.NL):
            dev = self.dev_of[i]
            pre = f"{self.L}.{i}."
            keys = [k for k in idx if k.startswith(pre)]
            with torch.device("meta"): ly = q35.Qwen3_5DecoderLayer(cfg, i)
            qk = [k[:-len(".weight")] for k in keys if k.endswith(".weight") and k[:-len(".weight")] in qkeys] if i in self.quant_layers else []
            sd = self._load([k for k in keys if k[:-len(".weight")] not in qk], pre)
            if teacher:
                with torch.device("meta"): tl = q35.Qwen3_5DecoderLayer(cfg, i)
                tl.load_state_dict({**sd, **self._load([k + ".weight" for k in qk], pre)}, strict=True, assign=True)
                self.tlayers.append(tl.eval().requires_grad_(False))
            for k in qk:
                sub = k[len(pre):]
                tr_r = not (gauge and sub == "mlp.down_proj"); tr_c = not (gauge and sub == "self_attn.v_proj")
                ql = QLinear(k, self.lev.raw(k), self.cb, self.tables[dev], dev, tr_r, tr_c, lora=lora, seed=len(self.qlin))
                parent = ly.get_submodule(sub.rsplit(".", 1)[0]); setattr(parent, sub.rsplit(".", 1)[1], ql)
                self.qlin[k] = ql
                if ql.r is not None: self.P[k + ":r"] = ql.r
                if ql.c is not None: self.P[k + ":c"] = ql.c
                if ql.la is not None: self.P[k + ":la"] = ql.la; self.P[k + ":lb"] = ql.lb
            missing, unexpected = ly.load_state_dict(sd, strict=False, assign=True)
            assert not unexpected, unexpected
            assert all(m.rsplit(".", 1)[0] in [q[len(pre):] for q in qk] for m in missing), missing
            mine = {id(p) for p in self.P.values()}
            for p in ly.parameters():
                if id(p) not in mine: p.requires_grad_(False)
            assert not any(t.is_meta for t in list(ly.parameters()) + list(ly.buffers())), f"layer {i}: a tensor left on meta"
            self.layers.append(ly.train())
            if dev.type == "cuda":
                with torch.cuda.device(dev): torch.cuda.empty_cache()
        self.hq, self.v2row = None, None
        if head_dir:                                                    # decoded on the last card while it is still empty
            hl = deq.Level(head_dir, "cpu"); hcb = hl.codebook
            ht = self.tables[self.devs[-1]] if hcb == self.cb else torch.from_numpy(deq.lut(hcb)).to(self.devs[-1])
            self.hq = QLinear("lm_head", hl.raw("lm_head"), hcb, ht, self.devs[-1], head_train, head_train, hold_bf16=True)
            assert self.hq.exact and self.hq.n == cfg.vocab_size, (self.hq.exact, self.hq.n)
            self.hq.requires_grad_(False)
            for p in (self.hq.r, self.hq.c):
                if p is not None: p.requires_grad_(True)
            if head_perm is not None:
                perm = np.load(head_perm).astype(np.int64); assert sorted(perm.tolist()) == list(range(len(perm)))
                self.v2row = torch.from_numpy(np.argsort(perm)).to(self.devs[-1])
            if self.devs[-1].type == "cuda":
                with torch.cuda.device(self.devs[-1]): torch.cuda.empty_cache()
        # phase 2: every layer onto its card in one pass (persistent tensors packed, no decode transients between them);
        # self.P is rebuilt from the placed modules (whatever Module.to does with Parameter identity)
        self.P = {}
        for i, ly in enumerate(self.layers):
            ly.to(self.dev_of[i])
            for k, ql in self.qlin.items():
                if not k.startswith(f"{self.L}.{i}."): continue
                for nm in ("r", "c", "la", "lb"):
                    p = getattr(ql, nm)
                    if p is not None:
                        assert p.requires_grad and p.device.type == self.dev_of[i].type, (k, nm, p.device)
                        self.P[f"{k}:{nm}"] = p
            if self.tlayers: self.tlayers[i].to(self.dev_of[i])
            if norms and cfg.layer_types[i] == "full_attention":
                for nm in ("self_attn.q_norm", "self_attn.k_norm"): self._add_norm(f"{self.L}.{i}.{nm}.weight", ly.get_submodule(nm))
        last = self.devs[-1]
        with torch.device("meta"): fn = q35.Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        fn.load_state_dict(self._load([f"{self.LP}norm.weight"], f"{self.LP}norm."), strict=True, assign=True)
        self.fnorm = fn.to(last).requires_grad_(False)
        if norms: self._add_norm(f"{self.LP}norm.weight", self.fnorm)
        if self.hq is not None:
            self.hq.to(last); self.head = self.hq
            for nm in ("r", "c"):
                if getattr(self.hq, nm) is not None: self.P[f"lm_head:{nm}"] = getattr(self.hq, nm)
        else:
            hk = "lm_head.weight" if "lm_head.weight" in idx else f"{self.LP}embed_tokens.weight"
            self.head = self._load([hk], "")[hk].to(last)
        self.emb = self._embed_memmap()
        self.rotary = {d: q35.Qwen3_5TextRotaryEmbedding(config=cfg).to(d) for d in self.devs}
        for f in self._fh.values(): f.__exit__(None, None, None)
        self._fh = {}
        for d in self.devs:
            if d.type == "cuda":
                with torch.cuda.device(d): torch.cuda.empty_cache()     # the decode transients (f32 W_hat) fragment the pool

    def _load(self, keys, strip):
        from safetensors import safe_open
        sd = {}
        for k in keys:
            f = self.wmap[k]
            if f not in self._fh: self._fh[f] = safe_open(f"{self.dir}/{f}", "pt", device="cpu").__enter__()
            sd[k[len(strip):]] = self._fh[f].get_tensor(k).to(BF)
        return sd

    def _add_norm(self, key, mod):
        mod._qg = torch.nn.Parameter(torch.zeros(mod.weight.shape[0], device=mod.weight.device))
        mod.register_forward_hook(_norm_hook); self.norm[key] = mod; self.P[key + ":g"] = mod._qg

    def _embed_memmap(self):
        k = f"{self.LP}embed_tokens.weight"; fn = f"{self.dir}/{self.wmap[k]}"
        with open(fn, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; hd = json.loads(f.read(n))
        e = hd[k]; assert e["dtype"] == "BF16", e
        return np.memmap(fn, np.int16, "r", offset=8 + n + e["data_offsets"][0], shape=tuple(e["shape"]))

    def embed(self, ids):
        rows = torch.from_numpy(np.ascontiguousarray(self.emb[ids])).view(BF)
        return rows[None].to(self.devs[0])

    def _ctx(self, dev, x):
        """position embeddings and masks for a [1, n] document on dev (forward.Model.run_layers' calls)"""
        q35, cfg = self.q35, self.cfg
        n = x.shape[1]
        pid = torch.arange(n, device=dev).view(1, 1, -1).expand(4, 1, -1)
        tpid, pid3 = pid[0], pid[1:]
        mk = {"config": cfg, "inputs_embeds": x, "attention_mask": None, "past_key_values": None, "position_ids": tpid}
        masks = {"full_attention": q35.create_causal_mask(**mk), "linear_attention": q35.create_recurrent_attention_mask(**mk)}
        return self.rotary[dev](x, pid3), masks, tpid

    def run(self, ids, ckpt=True, teacher_mse=False):
        """final-norm hidden states [n, H] on the last device; with teacher_mse also the list of per-layer normalized
        MSE terms (student block output vs the bf16 teacher's, each on its own trajectory)"""
        from torch.utils.checkpoint import checkpoint
        x = self.embed(ids); xt = x if teacher_mse else None
        ctx = {}; mses = []
        for i, ly in enumerate(self.layers):
            dev = self.dev_of[i]
            if x.device != dev:
                x = x.to(dev)
                if xt is not None: xt = xt.to(dev)
            if dev not in ctx: ctx[dev] = self._ctx(dev, x)
            pe, masks, tpid = ctx[dev]
            kw = dict(position_embeddings=pe, attention_mask=masks[self.cfg.layer_types[i]], position_ids=tpid)
            if ckpt and torch.is_grad_enabled():
                x = checkpoint(lambda h, _l=ly, _k=kw: _l(h, **_k), x, use_reentrant=False)
            else:
                x = ly(x, **kw)
            if xt is not None:
                with torch.no_grad(): xt = self.tlayers[i](xt, **kw)
                d = (x.float() - xt.float()).pow(2).mean() / xt.float().pow(2).mean()
                mses.append(d)
        x = x.to(self.devs[-1])
        h = self.fnorm(x)[0]
        return (h, mses) if teacher_mse else h

    def logits(self, h):
        return (self.head(h) if isinstance(self.head, torch.nn.Module) else F.linear(h, self.head)).float()

    def state(self):
        return {n: p.detach().cpu().clone() for n, p in self.P.items()}

    def load_state(self, st):
        with torch.no_grad():
            for n, v in st.items():
                if n in self.P: self.P[n].copy_(v.to(self.P[n].device))

    @torch.no_grad()
    def export_scales(self):
        """{key: (suh f16, svh f16)} with the trained multipliers folded in"""
        return {k: q.export() for k, q in self.qlin.items()}

    @torch.no_grad()
    def export_norms(self):
        """{norm weight key: new weight (bf16)} for trained norm gains: (1 + w) e^g - 1 (Qwen3.5's 1 + w form)"""
        return {k: ((1.0 + m.weight.float()) * m._qg.exp() - 1.0).to(BF).cpu() for k, m in self.norm.items()}
