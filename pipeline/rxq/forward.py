# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""The staged bf16 forward the level passes share (S1 H_in collection rxq.hin, S4 swj fit rxq.swjfit): a Qwen3.5-family
model that does not fit on one card, run in stages of decoder layers, one document at a time, with the bf16 residual
stream between stages on disk (lossless: the model's own boundary dtype). Port of the staging that rad-exl3-mix
eval/gold27/mkgold27.py introduced and mkhin27.py / swjfit27.py copy (the same calls in the same order; the passes'
checks in tests/check_levelpass.py are bit-identical against those tools).

Model(dir, dev): config (the text config, sdpa as from_pretrained picks), weight map, layer keys and shapes, stage
loading (transformers' own Qwen3_5DecoderLayer built on meta, then the checkpoint's bf16 tensors assigned), the
embedding as a memmap of its checkpoint file, run_layers (Qwen3_5TextModel.forward over a subset of layers: use_cache
False, or a fresh DynamicCache per document = the whole model's dec(input_ids)).
split_plan(): the stage boundaries (explicit, N balanced by bytes, or auto: the fewest that fit a VRAM budget).
P2B_BLOCK=causal_conv1d,fla (env, read at import) makes those packages unimportable, so transformers runs its torch
GDN path (proxy2b/common.py's switch; the CPU checks use it)."""
import json, os, struct, sys

for _b in filter(None, os.environ.get("P2B_BLOCK", "").split(",")): sys.modules[_b] = None
import numpy as np  # noqa: E402
import torch  # noqa: E402

GB = 2**30


def kernel_paths():
    """which GDN kernel packages this process can import (proxy2b/common.py's record)"""
    import importlib, importlib.metadata as md
    out = {}
    for mod, dist in (("causal_conv1d", "causal-conv1d"), ("fla", "flash-linear-attention")):
        try:
            importlib.import_module(mod); out[mod] = md.version(dist)
        except Exception:
            out[mod] = None
    out["torch"] = torch.__version__
    return out


def write_atomic(path, data):
    with open(path + ".part", "wb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(path + ".part", path)


class Model:
    def __init__(self, model, dev):
        from safetensors import safe_open
        from transformers import AutoConfig
        import transformers.models.qwen3_5.modeling_qwen3_5 as q35
        self.q35, self.safe_open, self.dir, self.dev = q35, safe_open, model, dev
        cfg = AutoConfig.from_pretrained(model)
        cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        cfg._attn_implementation = "sdpa"           # what from_pretrained picks
        self.cfg = cfg
        self.NL, self.H, self.NJ = cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size
        idx = f"{model}/model.safetensors.index.json"
        self.wmap = json.load(open(idx))["weight_map"] if os.path.exists(idx) else \
            {k: f for f in os.listdir(model) if f.endswith(".safetensors") for k in safe_open(f"{model}/{f}", "pt").keys()}
        self.LP = "model.language_model." if any(k.startswith("model.language_model.") for k in self.wmap) else "model."
        self.L = self.LP + "layers"
        self.rotary = None

    def shape(self, k):
        with self.safe_open(f"{self.dir}/{self.wmap[k]}", "pt") as f: return f.get_slice(k).get_shape()

    def layer_keys(self, i): return [k for k in self.wmap if k.startswith(f"{self.L}.{i}.")]

    def load_tensors(self, keys, strip):
        sd, byfile = {}, {}
        for k in keys: byfile.setdefault(self.wmap[k], []).append(k)
        for f, ks in byfile.items():
            with self.safe_open(f"{self.dir}/{f}", "pt", device="cpu") as fh:
                for k in ks: sd[k[len(strip):]] = fh.get_tensor(k).to(torch.bfloat16)
        return sd

    def load_layers(self, lo, hi):
        """[(i, Qwen3_5DecoderLayer on dev, eval)] for layers lo .. hi - 1"""
        mods = []
        for i in range(lo, hi):
            with torch.device("meta"): ly = self.q35.Qwen3_5DecoderLayer(self.cfg, i)
            ly.load_state_dict(self.load_tensors(self.layer_keys(i), f"{self.L}.{i}."), strict=True, assign=True)
            mods.append((i, ly.to(self.dev).eval()))
            assert not any(t.is_meta for t in list(ly.parameters()) + list(ly.buffers())), "a tensor was left on meta"
        return mods

    def load_norm(self):
        with torch.device("meta"): norm = self.q35.Qwen3_5RMSNorm(self.H, eps=self.cfg.rms_norm_eps)
        norm.load_state_dict(self.load_tensors([f"{self.LP}norm.weight"], f"{self.LP}norm."), strict=True, assign=True)
        return norm.to(self.dev).eval()

    def embed_memmap(self):
        """the embedding matrix as a read-only memmap of the checkpoint file (bf16 as int16): rows read on demand"""
        k = f"{self.LP}embed_tokens.weight"; fn = f"{self.dir}/{self.wmap[k]}"
        with open(fn, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; hd = json.loads(f.read(n))
        e = hd[k]; assert e["dtype"] == "BF16", e
        o0, _ = e["data_offsets"]
        return np.memmap(fn, np.int16, "r", offset=8 + n + o0, shape=tuple(e["shape"]))

    def make_rotary(self):
        """inv_freq made on the CPU, as from_pretrained (call where the tools build it: before or after loading)"""
        self.rotary = self.q35.Qwen3_5TextRotaryEmbedding(config=self.cfg).to(self.dev)
        return self.rotary

    def run_layers(self, mods, x, cache=False):
        """Qwen3_5TextModel.forward(inputs_embeds=x) over a subset of its layers, batch 1, positions 0 ..; cache=True:
        a fresh DynamicCache (mkgold27's: = the whole model's dec(input_ids)), else use_cache=False (mkhin27's)"""
        from transformers import DynamicCache
        q35, cfg, dev = self.q35, self.cfg, self.dev
        n = x.shape[1]
        kv = DynamicCache(config=cfg) if cache else None
        pid = torch.arange(n, device=dev).view(1, 1, -1).expand(4, 1, -1)
        tpid, pid = pid[0], pid[1:]
        mk = {"config": cfg, "inputs_embeds": x, "attention_mask": None, "past_key_values": kv, "position_ids": tpid}
        masks = {"full_attention": q35.create_causal_mask(**mk), "linear_attention": q35.create_recurrent_attention_mask(**mk)}
        pe = self.rotary(x, pid)
        for i, ly in mods:
            x = ly(x, position_embeddings=pe, attention_mask=masks[cfg.layer_types[i]], position_ids=tpid,
                   past_key_values=kv, use_cache=kv is not None)
        return x


def split_plan(cost, budget, stages="auto", split=None, head_cost=0):
    """stage boundaries [0, .., NL]: explicit split, N balanced by bytes, or auto (the fewest stages a greedy fill fits
    under budget, balanced when that still fits). head_cost: bytes the last stage also holds (mkhin27's norm + head
    accumulator; 0 = swjfit27's plan)"""
    NL = len(cost)

    def balanced(N):
        cum = np.cumsum(cost); tot = cum[-1] + head_cost
        sp = [0] + [int(np.argmin(np.abs(cum - tot * s / N))) + 1 for s in range(1, N)] + [NL]
        for s in range(1, N): sp[s] = min(max(sp[s], sp[s - 1] + 1), NL - (N - s))   # >= 1 layer a stage
        return sp

    def stage_bytes(sp): return [sum(cost[sp[s]:sp[s + 1]]) + (head_cost if s == len(sp) - 2 else 0) for s in range(len(sp) - 1)]
    if split:
        sp = [int(x) for x in split.split(",")] if isinstance(split, str) else list(split)
    elif stages == "auto":
        if budget <= 0: raise SystemExit("--stages auto needs VRAM_CAP_GB or a GPU")
        sp, used = [0], 0
        for i in range(NL):
            need = cost[i] + (head_cost if i == NL - 1 else 0)
            if used and used + need > budget: sp.append(i); used = 0
            used += need
        sp.append(NL)
        bal = balanced(len(sp) - 1)
        if max(stage_bytes(bal)) <= budget: sp = bal
    else:
        sp = balanced(int(stages))
    assert sp[0] == 0 and sp[-1] == NL and all(x < y for x, y in zip(sp, sp[1:])), f"bad stage split {sp}"
    return sp, stage_bytes(sp)


def corpus(meta, cap=None, maxdocs=0):
    """a golden-style META dir: (kld.json docs, tokens memmap, token offsets, number of docs, per-doc lengths capped)"""
    m = json.load(open(f"{meta}/kld.json")); toks = np.memmap(f"{meta}/tokens.i32", np.int32, "r")
    toff = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
    nd = len(m["docs"]) if not maxdocs else min(maxdocs, len(m["docs"]))
    dlen = [min(int(toff[i + 1] - toff[i]), cap) if cap else int(toff[i + 1] - toff[i]) for i in range(nd)]
    return m, toks, toff, nd, dlen
