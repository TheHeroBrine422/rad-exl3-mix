#!/usr/bin/env python3
"""headx94.py CKPT GOLDEN DOCS94.json OUT [--sets search,qad] -- the LM head's inputs (final norm output) at the rows of
mkdocs94.py's windows, from a bf16 forward of the checkpoint on ONE card (#124, merge 11-08 A2b).

The 27B in bf16 (54 GB) does not fit a 32 GB card, so the forward streams layer by layer: every document's hidden
state stays in host RAM (bf16, exactly what the HF forward keeps between layers), one decoder layer at a time is
loaded onto the card, applied to every document, and dropped. Same modules and kernels as an HF forward (sdpa, the
fla gated-delta path transformers picks), batch 1, no padding, documents forwarded from token 0 to the window's end.

OUT/<set>/336.x      bf16 rows (headsim's DUMPDIR layout), 336.meta "5120 1 N"
OUT/<set>/rows.i64   golden row of each dumped row;  doc.i32 calib document;  gen.u8 scored flag;  nxt.i32 next token
OUT/<set>/done.json  counts (written last: a set without it is incomplete and is redone)
Host RAM ~ 2 x 2 bytes x 5120 x the largest set's forwarded tokens (search: 114k tokens -> ~2.4 GB) + one layer."""
import argparse, json, os, sys, time
import numpy as np, torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("golden"); ap.add_argument("docs"); ap.add_argument("out")
ap.add_argument("--sets", default="search,qad"); ap.add_argument("--limit", type=int, default=0, help="test: docs per set")
ap.add_argument("--cpu", action="store_true", help="test on the CPU (torch gated-delta path instead of fla)")
a = ap.parse_args()
torch.set_grad_enabled(False)
from transformers import AutoConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as M
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda:0")
if dev.type == "cpu":                                 # fla's triton kernels are GPU-only: the decorated torch function
    M.torch_chunk_gated_delta_rule = M.torch_chunk_gated_delta_rule.__wrapped__
cfg = AutoConfig.from_pretrained(a.ckpt); tcfg = getattr(cfg, "text_config", cfg)
tcfg._attn_implementation = "sdpa"
with init_empty_weights(): tm = M.Qwen3_5TextModel(tcfg)
tm.eval()
idx = json.load(open(f"{a.ckpt}/model.safetensors.index.json"))["weight_map"]
PFX = "model.language_model."
names = {k[len(PFX):]: f for k, f in idx.items() if k.startswith(PFX)}


def load(prefix):
    """set every checkpoint tensor under `prefix` (text-model names) onto the card"""
    byf = {}
    for n, f in names.items():
        if n.startswith(prefix): byf.setdefault(f, []).append(n)
    for f, ns in byf.items():
        with safe_open(f"{a.ckpt}/{f}", "pt", device="cpu") as h:
            for n in ns: set_module_tensor_to_device(tm, n, dev, value=h.get_tensor(PFX + n), dtype=torch.bfloat16)
    left = [n for n, p in tm.named_parameters() if n.startswith(prefix) and p.device.type == "meta"]
    assert not left, f"not in the checkpoint: {left}"


m = json.load(open(f"{a.golden}/kld.json")); toks = np.fromfile(f"{a.golden}/tokens.i32", np.int32)
gen_all = np.fromfile(f"{a.golden}/gen.u8", np.uint8)
off = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])]); assert off[-1] == len(toks)
D = json.load(open(a.docs))
load("embed_tokens."); load("norm."); tm.rotary_emb.to(dev)
for s in a.sets.split(","):
    od = f"{a.out}/{s}"
    if os.path.exists(f"{od}/done.json"): print(f"{s}: done", flush=True); continue
    os.makedirs(od, exist_ok=True)
    W = D["sets"][s][:a.limit or None]; t0 = time.time()
    H = []                                            # per document: hidden state [1, L, 5120] bf16 on the host
    for w in W:
        ids = torch.from_numpy(toks[off[w["doc"]]:off[w["doc"]] + w["fwd"]].astype(np.int64))[None].to(dev)
        H.append(tm.embed_tokens(ids).cpu())
    for l in range(tcfg.num_hidden_layers):
        load(f"layers.{l}."); layer = tm.layers[l]; lt = tcfg.layer_types[l]
        for i, w in enumerate(W):
            h = H[i].to(dev); L = h.shape[1]
            pid = torch.arange(L, device=dev).view(1, 1, -1).expand(4, 1, -1)
            tpid, ppid = pid[0], pid[1:]
            mk = {"config": tcfg, "inputs_embeds": h, "attention_mask": None, "past_key_values": None, "position_ids": tpid}
            mask = M.create_causal_mask(**mk) if lt == "full_attention" else M.create_recurrent_attention_mask(**mk)
            pe = tm.rotary_emb(h, ppid)
            H[i] = layer(h, position_embeddings=pe, attention_mask=mask, position_ids=tpid, past_key_values=None).cpu()
        layer.to("meta")
        if dev.type == "cuda": torch.cuda.empty_cache()
        print(f"{s}: layer {l} ({lt}) {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    X, rows, doc, gen, nxt = [], [], [], [], []
    for i, w in enumerate(W):
        x = tm.norm(H[i].to(dev))[0, w["pos0"]:w["fwd"]].cpu(); assert x.shape[0] == w["rows"]
        X.append(x); r = np.arange(w["row0"], w["row0"] + w["rows"]); rows.append(r); doc.append(np.full(len(r), w["doc"]))
        gen.append(gen_all[r]); p = np.arange(w["pos0"], w["fwd"]); nxt.append(toks[off[w["doc"]] + p + 1])
    X = torch.cat(X).contiguous(); N = X.shape[0]
    X.view(torch.int16).numpy().tofile(f"{od}/336.x")
    open(f"{od}/336.meta", "w").write(f"{X.shape[1]} 1 {N}\n")
    np.concatenate(rows).astype(np.int64).tofile(f"{od}/rows.i64"); np.concatenate(doc).astype(np.int32).tofile(f"{od}/doc.i32")
    np.concatenate(gen).astype(np.uint8).tofile(f"{od}/gen.u8"); np.concatenate(nxt).astype(np.int32).tofile(f"{od}/nxt.i32")
    g = int(np.concatenate(gen).sum())
    json.dump({"set": s, "docs": len(W), "rows": N, "gen_rows": g, "tokens": int(sum(w["fwd"] for w in W)),
               "sec": round(time.time() - t0, 1)}, open(f"{od}/done.json", "w"))
    print(f"{s}: {len(W)} docs, {N} rows ({g} gen) in {time.time() - t0:.0f}s", flush=True)
    del H, X
