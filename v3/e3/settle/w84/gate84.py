#!/usr/bin/env python3
"""gate84.py ORDER CKPT META OUT {f32|f64} [NDOC=2] [LEN=384] -- gate G3 of ../w84.md (CPU): the converted model computes
the original's function. Layer-streamed forward (transformers' Qwen3.5 modules, one decoder layer in memory at a time) of
  A  the original bf16 checkpoint, upcast;
  B  the converted checkpoint: perm117.permute_tensor(name, W) for every tensor (the converter's own per-tensor
     function; applied after the exact upcast, which commutes with a permutation),
on the first LEN tokens of the first NDOC fit-pool calib documents with >= LEN tokens (META: calib tokens.i32 /
kld.json / pools-v3.json), in one dtype: f32 (the model's own casts) or f64 (every fp32 cast in the modeling file and
Tensor.float() promoted to fp64: pure float64 arithmetic, so any A / B difference is float reassociation at 1e-16).
Also: the MTP head (fc on [pre_fc_norm_embedding(embed(x_t+1)), pre_fc_norm_hidden(h_t)], its full-attention layer,
mtp.norm, lm_head) and the vision merger's linear_fc2 on a fixed random input.
Reports per part: bitwise equal, max |B - A|, max |A|, mean KL(A || B), top-1 agreement. Writes OUT/gate84-<dtype>.json.
Verdict (both dtypes' files): gate84.py verdict OUT."""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import perm117

if sys.argv[1] == "verdict":
    O = sys.argv[2]; r = {d: json.load(open(f"{O}/gate84-{d}.json")) for d in ("f32", "f64")}
    ok, lines = True, []
    for part in ("logits", "mtp", "merger"):
        a, b = r["f32"][part], r["f64"][part]
        rel64 = b["max_diff"] / max(b["max_abs"], 1e-300)
        shrink = a["max_diff"] / b["max_diff"] if b["max_diff"] > 0 else float("inf")
        p = b["bitwise"] or (rel64 <= 1e-9 and (a["bitwise"] or shrink >= 1e5) and b["top1"] == 1.0)
        ok &= p
        lines.append(f"{part}: f32 bitwise {a['bitwise']} max|d| {a['max_diff']:.3e} (max|x| {a['max_abs']:.2f}, KL {a['kl']:.2e}, "
                     f"top-1 {a['top1']:.6f}); f64 bitwise {b['bitwise']} max|d| {b['max_diff']:.3e} (rel {rel64:.1e}, "
                     f"top-1 {b['top1']:.6f}); shrink f32/f64 {shrink:.1e} -> {'PASS' if p else 'FAIL'}")
    print("\n".join(lines)); print(f"BF16 GATE {'PASS' if ok else 'FAIL'}"); sys.exit(0)

ORDER, CKPT, META, OUT, DT = sys.argv[1:6]
NDOC = int(sys.argv[6]) if len(sys.argv) > 6 else 2
LN = int(sys.argv[7]) if len(sys.argv) > 7 else 384
torch.set_num_threads(int(os.environ.get("GATE_THREADS", 8)))
os.makedirs(OUT, exist_ok=True)
so, idx = perm117.load_order(ORDER)

import transformers.models.qwen3_5.modeling_qwen3_5 as M
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
DTYPE = torch.float32 if DT == "f32" else torch.float64
if DT == "f64":
    class _T:
        float32 = torch.float64
        float = torch.float64
        def __getattr__(self, n): return getattr(torch, n)
    M.torch = _T()
    torch.Tensor.float = lambda self, *a, **k: self.to(torch.float64)

cfg = Qwen3_5Config.from_pretrained(CKPT).text_config
cfg._attn_implementation = "eager"
torch.manual_seed(0)

# ---- tokens: first NDOC fit-pool docs with >= LN tokens
m = json.load(open(f"{META}/kld.json")); pools = json.load(open(f"{META}/pools-v3.json"))["pool"]
toks = np.memmap(f"{META}/tokens.i32", np.int32, "r")
off = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
docs = [i for i, d in enumerate(m["docs"]) if pools[i] == "fit" and d["tokens"] >= LN + 1][:NDOC]
X = torch.from_numpy(np.stack([np.array(toks[off[i]:off[i] + LN + 1], np.int64) for i in docs]))   # +1: the MTP's next token
ids, nxt = X[:, :LN], X[:, 1:]

from safetensors import safe_open
wm = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
_h = {}
def get(name, rows=None):
    f = wm[name]
    if f not in _h: _h[f] = safe_open(f"{CKPT}/{f}", "pt", device="cpu")
    if rows is None: return _h[f].get_tensor(name)
    return _h[f].get_slice(name)[rows[0]:rows[1]]
def both(name, t):                                # (A, B) in DTYPE
    a = t.to(DTYPE); b = perm117.permute_tensor(name, a, idx); return a, (a if b is None else b)

def embed(tokens):
    w = get("model.language_model.embed_tokens.weight")      # bf16 [V, 5120] (2.5 GB), rows gathered
    e = w[tokens.flatten()]; del w; _h.clear()                          # [B * T, 5120]: the embedding's own layout, rows gathered
    a, b = both("model.language_model.embed_tokens.weight", e)
    return a.reshape(*tokens.shape, -1), b.reshape(*tokens.shape, -1)

def mkmod(f):                                     # a module on the meta device: load_into assigns the real tensors
    with torch.device("meta"): m_ = f()
    return m_.eval()

def load_into(mod, prefix, perm):                 # module params from the checkpoint (A or B), assigned (no extra copy)
    sd = {}
    for k, _ in mod.named_parameters():
        t = get(prefix + k).to(DTYPE)
        if perm:
            p = perm117.permute_tensor(prefix + k, t, idx); t = t if p is None else p
        sd[k] = t
    mod.load_state_dict(sd, strict=True, assign=True)
    bad = [k for k, b in mod.named_buffers() if b.is_meta]
    assert not bad, f"meta buffers left in {prefix}: {bad}"

def permute_module(mod, prefix):                  # A's module -> B's in place, one parameter at a time
    with torch.no_grad():
        for k, p in mod.named_parameters():
            q = perm117.permute_tensor(prefix + k, p.data, idx)
            if q is not None: p.data = q

t0 = time.time()
hA, hB = embed(ids)
B_, T_ = ids.shape
pos = torch.arange(T_).view(1, 1, -1).expand(3, B_, -1)
rot = M.Qwen3_5TextRotaryEmbedding(cfg).to(DTYPE)
pe = rot(hA, pos)
mk = dict(config=cfg, inputs_embeds=hA, attention_mask=None, past_key_values=None, position_ids=pos[0])
masks = {"full_attention": create_causal_mask(**mk), "linear_attention": create_recurrent_attention_mask(**mk)}
st = {"docs": docs, "len": LN, "dtype": DT, "layer_bitwise": []}
with torch.no_grad():
    for l in range(cfg.num_hidden_layers):
        lay = mkmod(lambda: M.Qwen3_5DecoderLayer(cfg, l))
        pre = f"model.language_model.layers.{l}."
        load_into(lay, pre, False)
        kw = dict(position_embeddings=pe, attention_mask=masks[cfg.layer_types[l]], position_ids=pos[0])
        hA = lay(hA, **kw)
        permute_module(lay, pre)
        hB = lay(hB, **kw)
        _h.clear()                                # unmap the shards (mapped pages otherwise pile up in RSS)
        st["layer_bitwise"].append(bool(torch.equal(hA[..., idx], hB)))
        d = float((hA[..., idx] - hB).abs().max())
        print(f"layer {l} ({cfg.layer_types[l][:4]}): max|hB - P hA| {d:.3e} max|hA| {float(hA.abs().max()):.1f} "
              f"bitwise {st['layer_bitwise'][-1]}  {time.time() - t0:.0f}s", flush=True)
        del lay
    # final norm + head, per document (logits kept in DTYPE for exact comparison)
    hfinA, hfinB = hA, hB
    nrm = mkmod(lambda: M.Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps))
    load_into(nrm, "model.language_model.norm.", False); xA = nrm(hA)
    permute_module(nrm, "model.language_model.norm."); xB = nrm(hB)
    def head(xA, xB):
        V = cfg.vocab_size; outA = torch.empty(*xA.shape[:-1], V, dtype=DTYPE); outB = torch.empty_like(outA)
        for r in range(0, V, 16384):
            w = get("lm_head.weight", (r, min(r + 16384, V)))
            wA, wB = both("lm_head.weight", w); del w
            outA[..., r:r + wA.shape[0]] = xA @ wA.T; outB[..., r:r + wB.shape[0]] = xB @ wB.T
        _h.clear()
        return outA, outB
    def stats(lA, lB):
        lA2, lB2 = lA.reshape(-1, lA.shape[-1]), lB.reshape(-1, lB.shape[-1])
        pa = torch.log_softmax(lA2.double(), -1); pb = torch.log_softmax(lB2.double(), -1)
        return {"bitwise": bool(torch.equal(lA2, lB2)), "max_diff": float((lA2 - lB2).abs().max()), "max_abs": float(lA2.abs().max()),
                "kl": float((pa.exp() * (pa - pb)).sum(-1).mean()), "top1": float((lA2.argmax(-1) == lB2.argmax(-1)).double().mean()),
                "positions": int(lA2.shape[0])}
    parts = []
    for i in range(B_):
        lA, lB = head(xA[i], xB[i]); parts.append(stats(lA, lB)); del lA, lB
    agg = lambda k, f: f(p[k] for p in parts)
    st["logits"] = {"bitwise": agg("bitwise", all), "max_diff": agg("max_diff", max), "max_abs": agg("max_abs", max),
                    "kl": float(np.mean([p["kl"] for p in parts])), "top1": float(np.mean([p["top1"] for p in parts])), "per_doc": parts}
    print(f"logits: {json.dumps({k: v for k, v in st['logits'].items() if k != 'per_doc'})}  {time.time() - t0:.0f}s", flush=True)

    # MTP: h_t (last layer output, pre-norm) and embed(x_t+1) -> fc -> full-attention layer -> mtp.norm -> lm_head
    eA, eB = embed(nxt)
    def mtp(h, e, perm):
        n1 = mkmod(lambda: M.Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)); load_into(n1, "mtp.pre_fc_norm_embedding.", perm)
        n2 = mkmod(lambda: M.Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)); load_into(n2, "mtp.pre_fc_norm_hidden.", perm)
        fc = mkmod(lambda: torch.nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False)); load_into(fc, "mtp.fc.", perm)
        x = fc(torch.cat([n1(e), n2(h)], -1))
        lay = mkmod(lambda: M.Qwen3_5DecoderLayer(cfg, 3)); load_into(lay, "mtp.layers.0.", perm)
        x = lay(x, position_embeddings=pe, attention_mask=masks["full_attention"], position_ids=pos[0])
        nn_ = mkmod(lambda: M.Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)); load_into(nn_, "mtp.norm.", perm)
        return nn_(x)
    yA, yB = mtp(hfinA, eA, False), mtp(hfinB, eB, True)
    lA, lB = head(yA[0], yB[0]); st["mtp"] = stats(lA, lB); del lA, lB
    print(f"mtp: {json.dumps(st['mtp'])}", flush=True)

    # vision merger linear_fc2 on a fixed input: outputs in the residual basis -> B == P A
    fc2 = mkmod(lambda: torch.nn.Linear(4608, cfg.hidden_size)); load_into(fc2, "model.visual.merger.linear_fc2.", False)
    xin = torch.randn(64, 4608, generator=torch.Generator().manual_seed(1)).to(DTYPE)
    ya = fc2(xin); permute_module(fc2, "model.visual.merger.linear_fc2."); yb = fc2(xin)
    ya = ya[..., idx]
    st["merger"] = {"bitwise": bool(torch.equal(ya, yb)), "max_diff": float((ya - yb).abs().max()), "max_abs": float(ya.abs().max()),
                    "kl": 0.0, "top1": 1.0}
    print(f"merger: {json.dumps(st['merger'])}", flush=True)
st["seconds"] = round(time.time() - t0)
json.dump(st, open(f"{OUT}/gate84-{DT}.json", "w"), indent=1)
print(f"GATE84 {DT} done in {st['seconds']} s", flush=True)
