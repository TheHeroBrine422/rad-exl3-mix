#!/usr/bin/env python3
"""prectest.py MODEL META OUT.jsonl [--nseq 16] -- does the bf16 hi/lo split ("x2", v3/yaqa/bf16x.py) give fp32-quality
gradients for the diagonal H_out collector (#33-13)? 0.8B on one card.

The same NSEQ windows (META, as yaqaB.py: xcorp, fixed shuffle) and the same sampled targets (drawn once, in the fp32
run) are run three times, the trunk linears' GEMMs in mode fp32 (reference) / x2 / bf16; norms, GDN, attention and
the head stay fp32 in all three (as collect.py's fp32 mode), so only the linear GEMMs differ. Per linear: the summed
diag(G_b G_bᵀ) over the sequences (what the collector accumulates), compared with fp32: relative L2 error, cosine,
worst row's relative error; plus the per-sequence loss difference and the time per sequence of each mode.
The precision a mode needs: its error well below the estimator's own sampling noise (the diagonal's two 4K halves
agree at cos >= 0.997, #33-09, i.e. relative differences ~1e-2)."""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, "/workspace/rad-exl3-mix/v3/yaqa")
import bf16x

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("meta"); ap.add_argument("out")
ap.add_argument("--nseq", type=int, default=16); ap.add_argument("--len", type=int, default=2048)
ap.add_argument("--chunk", type=int, default=128); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--modes", default="fp32,x2,bf16")
a = ap.parse_args()
L, dev = a.len, torch.device("cuda")
m = json.load(open(f"{a.meta}/kld.json")); toks = np.memmap(f"{a.meta}/tokens.i32", np.int32, "r")
off = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
wins = [int(off[i] + s) for i, d in enumerate(m["docs"]) for s in range(0, d["tokens"] - L + 1, L)]
np.random.default_rng(a.seed).shuffle(wins)

from transformers import AutoModelForCausalLM
from torch.utils.checkpoint import checkpoint
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev)
model.requires_grad_(False)
dec, head = model.get_decoder(), model.get_output_embeddings()
MODE = ["fp32"]; ACC = {}

class LinD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, key):
        ctx.save_for_backward(x, w); ctx.key = key
        return bf16x.lin(x, w, MODE[0])
    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        g2, x2 = g.reshape(-1, g.shape[-1]).float(), x.reshape(-1, x.shape[-1]).float()
        d = bf16x.diag_gg(g2, x2, MODE[0])
        ACC[ctx.key] = ACC.get(ctx.key, 0) + d.double()
        return bf16x.lin_back(g.float(), w, MODE[0]), None, None

import types
def fwd_lin(self, x): return LinD.apply(x.float(), self.weight, self._yk)
def fwd32(self, x):
    y = torch.nn.functional.linear(x.float(), self.weight.float())
    return y if self.bias is None else y + self.bias.float()
keys = []
for l, ly in enumerate(dec.layers):
    for nm, mod in ly.named_modules():
        if not isinstance(mod, torch.nn.Linear): continue
        if nm.endswith(("in_proj_a", "in_proj_b")): mod.forward = types.MethodType(fwd32, mod); continue
        mod._yk = f"{l}.{nm}"; keys.append(mod._yk); mod.forward = types.MethodType(fwd_lin, mod)
for mod in model.modules():
    if not isinstance(mod, (torch.nn.Linear, torch.nn.Embedding)):
        for pn, p in list(mod.named_parameters(recurse=False)): p.data = p.data.float()
dec.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.train()
HW = head.weight
def logits(h): return torch.cat([h @ HW[i:i + 32768].float().T for i in range(0, HW.shape[0], 32768)], -1)
def chunk_ce(h, y): return torch.nn.functional.cross_entropy(logits(h), y, reduction="sum")

LABELS, RES = {}, {}
for mode in a.modes.split(","):
    MODE[0] = mode; ACC.clear(); losses = []; torch.cuda.synchronize(); t0 = time.time()
    for s in range(a.nseq):
        ids = torch.from_numpy(np.asarray(toks[wins[s]:wins[s] + L], np.int64))[None].to(dev)
        emb = dec.embed_tokens(ids).float().detach().requires_grad_(True)
        hid = dec(inputs_embeds=emb, use_cache=False).last_hidden_state[0]
        if s not in LABELS:                       # drawn once (first mode), reused by the others
            gen = torch.Generator(device=dev); gen.manual_seed(s); ys = []
            with torch.no_grad():
                for c in range(0, hid.shape[0], a.chunk):
                    ys.append(torch.multinomial(torch.softmax(logits(hid[c:c + a.chunk]), -1), 1, generator=gen)[:, 0])
            LABELS[s] = ys
        loss = 0
        for i, c in enumerate(range(0, hid.shape[0], a.chunk)):
            loss = loss + checkpoint(chunk_ce, hid[c:c + a.chunk], LABELS[s][i], use_reentrant=False)
        (loss / hid.shape[0]).backward(); losses.append(loss.item() / hid.shape[0])
        del hid, loss, emb
    torch.cuda.synchronize(); dt = (time.time() - t0) / a.nseq
    RES[mode] = {"acc": {k: v.cpu() for k, v in ACC.items()}, "loss": losses, "s_per_seq": dt}
    print(f"{mode}: {dt:.2f} s/seq, mean loss/tok {np.mean(losses):.5f}", flush=True)

ref = RES["fp32"]
with open(a.out, "w") as fo:
    for mode in RES:
        if mode == "fp32": continue
        rows = []
        for k in keys:
            r, x = ref["acc"][k], RES[mode]["acc"][k]
            rel = float((x - r).norm() / r.norm()); cos = float((x * r).sum() / (x.norm() * r.norm()))
            worst = float(((x - r).abs() / r.abs().clamp_min(1e-30 + float(r.abs().mean()) * 1e-3)).max())
            rows.append({"mode": mode, "key": k, "layer": int(k.split(".")[0]), "type": k.split(".", 1)[1], "rel": rel, "cos": cos, "worst_row_rel": worst})
            fo.write(json.dumps(rows[-1]) + "\n")
        dl = np.abs(np.array(RES[mode]["loss"]) - np.array(ref["loss"]))
        print(f"\n{mode} vs fp32: loss/tok |diff| max {dl.max():.2e}; time {RES[mode]['s_per_seq']:.2f} vs {ref['s_per_seq']:.2f} s/seq")
        print("  depth band   median rel   max rel   min cos   max worst-row rel")
        for lo_, hi_ in ((0, 6), (6, 12), (12, 18), (18, 24)):
            g = [r for r in rows if lo_ <= r["layer"] < hi_]
            print(f"  {lo_:2d}-{hi_ - 1:2d}       {np.median([r['rel'] for r in g]):9.2e} {max(r['rel'] for r in g):9.2e} "
                  f"{min(r['cos'] for r in g):9.6f} {max(r['worst_row_rel'] for r in g):9.2e}")
print("PRECTEST DONE")
