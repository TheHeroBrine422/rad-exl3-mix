#!/usr/bin/env python3
"""yaqaB.py MODEL META OUT NSEQ [--len 2048] [--snap 1024,2048,...] -- YAQA's Sketch B (Tseng et al. 2025), as the
reference code collects it (Cornell-RelaxML/yaqa-quantization, hessian_llama/custom_linear_B.py + get_hess_llama.py,
default power_iters 1), for a small proxy on one card:

  per sequence b (len-token windows of META documents >= len tokens, fixed shuffled order), forward the model, sample
  targets from its own logits (Categorical, a fixed seed a sequence), loss = mean cross-entropy, backward; for every
  trunk linear (weight [m, n]) the sequence's weight gradient G_b = sum_t dy_t x_t^T, and
      H_in  += G_b^T G_b      (n x n)       einsum('btm,btn,bsm,bsk->nk')
      H_out += G_b G_b^T      (m x m)       einsum('btm,btn,bsk,bsn->mk')
  fp32 throughout, divided by the sequence count when saved. Every position counts (no warm-up skip).

Differences from the reference, none in the arithmetic: one sequence a step (theirs: 2, each its own G_b); the
model's linears, norms and head compute in fp32 from bf16 weights (theirs: the model's dtype with fp32 Hessian
math -- ours is the more precise); data = xcorp (theirs: RedPajama). H_out stays on the GPU; H_in is summed in host RAM
(GPU product -> pinned copy -> CPU add in a worker thread), since both do not fit beside a desktop on 12 GB.

Output OUT/snapN/: per trunk linear <checkpoint key>.hout.safetensors ("hout", square, ourq.py --hout) and
<checkpoint key>.safetensors (input side, the challenge layout "hessian_upper", ourq.py --hin per-key file)."""
import argparse, json, os, queue, sys, threading, time
import numpy as np, torch
from safetensors.torch import save_file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import kernel_paths

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("meta"); ap.add_argument("out"); ap.add_argument("nseq", type=int)
ap.add_argument("--len", type=int, default=2048); ap.add_argument("--snap", default="1024,2048,4096,8192,16384")
ap.add_argument("--chunk", type=int, default=128); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--resume", type=int, default=0, help="continue from OUT/snap<N> (its sums x N), at sequence N")
ap.add_argument("--gate", action="store_true", help="3 sequences, no files: time, memory, and a check of the sums")
a = ap.parse_args()
L, dev = a.len, torch.device("cuda")
SNAPS = {int(x) for x in a.snap.split(",") if x} | {a.nseq}
torch.set_num_threads(8)

m = json.load(open(f"{a.meta}/kld.json")); toks = np.memmap(f"{a.meta}/tokens.i32", np.int32, "r")
off = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
wins = [int(off[i] + s) for i, d in enumerate(m["docs"]) for s in range(0, d["tokens"] - L + 1, L)]
np.random.default_rng(a.seed).shuffle(wins)
print(f"{len(wins)} windows of {L} tokens available; using {a.nseq}", flush=True)
assert len(wins) >= a.nseq

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev)
model.requires_grad_(False)
dec, head = model.get_decoder(), model.get_output_embeddings()

HOUT, HIN, PIN, NSEQ = {}, {}, {}, [0]
Q = queue.Queue()
def worker():
    while True:
        item = Q.get()
        if item is not None:
            key, ev = item; ev.synchronize(); HIN[key].add_(PIN[key])
        Q.task_done()
threading.Thread(target=worker, daemon=True).start()
CHECK = {}   # --gate: G of one linear kept for a direct check

class LinB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, key):
        ctx.save_for_backward(x, w); ctx.key = key
        return x @ w.float().T
    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        gi = g @ w.float()
        G = g.reshape(-1, g.shape[-1]).float().T @ x.reshape(-1, x.shape[-1]).float()   # [m, n], one sequence
        HOUT[ctx.key].addmm_(G, G.T)
        hi = G.T @ G
        PIN[ctx.key].copy_(hi, non_blocking=True)
        ev = torch.cuda.Event(); ev.record(); Q.put((ctx.key, ev))
        if a.gate and ctx.key in CHECK: CHECK[ctx.key] = G.clone()
        return gi, None, None

import types
def fwd_lin(self, x): return LinB.apply(x.float(), self.weight, self._yk)
def fwd32(self, x):
    y = torch.nn.functional.linear(x.float(), self.weight.float())
    return y if self.bias is None else y + self.bias.float()
for l, ly in enumerate(dec.layers):
    for nm, mod in ly.named_modules():
        if not isinstance(mod, torch.nn.Linear): continue
        if nm.endswith(("in_proj_a", "in_proj_b")):          # not quantized: fp32 compute, no Hessian
            mod.forward = types.MethodType(fwd32, mod); continue
        key = f"model.language_model.layers.{l}.{nm}"; mod._yk = key
        mod.forward = types.MethodType(fwd_lin, mod)
        HOUT[key] = torch.zeros(mod.out_features, mod.out_features, device=dev)
        HIN[key] = torch.zeros(mod.in_features, mod.in_features)
        PIN[key] = torch.empty(mod.in_features, mod.in_features).pin_memory()
for mod in model.modules():                                    # norms etc. in fp32
    if not isinstance(mod, (torch.nn.Linear, torch.nn.Embedding)):
        for pn, p in list(mod.named_parameters(recurse=False)): p.data = p.data.float()
dec.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train()
HW = head.weight
def logits(h):   # fp32 head without a whole fp32 copy of the 248k x d weight
    return torch.cat([h @ HW[i:i + 32768].float().T for i in range(0, HW.shape[0], 32768)], -1)
from torch.utils.checkpoint import checkpoint
def chunk_ce(h, y): return torch.nn.functional.cross_entropy(logits(h), y, reduction="sum")
print(f"{len(HOUT)} linears; H_out {sum(v.numel() for v in HOUT.values()) * 4 / 2**30:.2f} GiB GPU, "
      f"H_in {sum(v.numel() for v in HIN.values()) * 4 / 2**30:.2f} GiB host; GPU {torch.cuda.memory_allocated() / 2**30:.2f} GiB",
      flush=True)
if a.gate: CHECK["model.language_model.layers.5.mlp.down_proj"] = None

def save(sub):
    Q.join(); torch.cuda.synchronize()
    d = f"{a.out}/{sub}"; n = NSEQ[0]
    os.makedirs(f"{d}/hout", exist_ok=True); os.makedirs(f"{d}/hin", exist_ok=True)
    meta = {"rows": str(n), "seqs": str(n), "ctx": str(L), "sketch": "B", "model": a.model, "kernels": json.dumps(kernel_paths())}
    for k in HOUT:
        save_file({"hout": (HOUT[k] / n).cpu()}, f"{d}/hout/{k}.safetensors.part", metadata=meta)
        os.replace(f"{d}/hout/{k}.safetensors.part", f"{d}/hout/{k}.safetensors")
        H = HIN[k] / n; kk = H.shape[0]; iu = torch.triu_indices(kk, kk)
        save_file({"hessian_upper": H[iu[0], iu[1]].contiguous()}, f"{d}/hin/{k}.safetensors.part",
                  metadata=dict(meta, k=str(kk), layout="upper triangle, row-major, packed", linears=k))
        os.replace(f"{d}/hin/{k}.safetensors.part", f"{d}/hin/{k}.safetensors")
    json.dump({"seqs": n, "ctx": L}, open(f"{d}/info.json", "w"))
    print(f"saved {sub}", flush=True)

start = 0
if a.resume:
    from safetensors import safe_open
    d = f"{a.out}/snap{a.resume}"; n = a.resume
    for k in HOUT:
        with safe_open(f"{d}/hout/{k}.safetensors", "pt") as f: HOUT[k].copy_(f.get_tensor("hout").to(dev) * n)
        with safe_open(f"{d}/hin/{k}.safetensors", "pt") as f: up = f.get_tensor("hessian_upper")
        kk = HIN[k].shape[0]; iu = torch.triu_indices(kk, kk); H = torch.zeros(kk, kk); H[iu[0], iu[1]] = up
        HIN[k].copy_((H + H.triu(1).T) * n)
    NSEQ[0] = start = n; print(f"resumed from {d}", flush=True)
t0 = time.time()
for s in range(start, 3 if a.gate else a.nseq):
    ids = torch.from_numpy(np.asarray(toks[wins[s]:wins[s] + L], np.int64))[None].to(dev)
    ts = time.time()
    emb = dec.embed_tokens(ids).float().detach().requires_grad_(True)
    hid = dec(inputs_embeds=emb, use_cache=False).last_hidden_state[0]
    gen = torch.Generator(device=dev); gen.manual_seed(s)
    loss = 0
    for c in range(0, hid.shape[0], a.chunk):
        h = hid[c:c + a.chunk]
        with torch.no_grad(): y = torch.multinomial(torch.softmax(logits(h), -1), 1, generator=gen)[:, 0]
        # checkpointed: the chunk's 248k-wide logits are recomputed in backward, not kept for every chunk
        loss = loss + checkpoint(chunk_ce, h, y, use_reentrant=False)
    (loss / hid.shape[0]).backward()
    Q.join()                                     # this sequence's H_in adds done before the pinned buffers are reused
    NSEQ[0] = s + 1; n = s + 1
    if a.gate or n % 50 == 0 or n in SNAPS:
        el = time.time() - t0
        print(f"seq {n}/{a.nseq}: {time.time() - ts:.2f}s, loss/tok {loss.item() / hid.shape[0]:.3f}, {(n - start) / el:.3f} seq/s, "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    del hid, loss, emb
    if not a.gate and n in SNAPS: save(f"snap{n}")
if a.gate:
    k = "model.language_model.layers.5.mlp.down_proj"; G = CHECK[k]
    torch.cuda.synchronize(); Q.join()
    print(f"gate: last G of {k} {tuple(G.shape)}; H_out diag mean {HOUT[k].diagonal().mean().item():.3e}, "
          f"H_in diag mean {HIN[k].diagonal().mean().item():.3e}, finite {bool(torch.isfinite(HOUT[k]).all() and torch.isfinite(HIN[k]).all())}; "
          f"tr H_out = tr H_in: {HOUT[k].trace().item():.4e} vs {HIN[k].trace().item():.4e}", flush=True)
    print("GATE DONE")
print(f"YAQAB DONE {NSEQ[0]} seqs in {(time.time() - t0) / 60:.1f} min", flush=True)
