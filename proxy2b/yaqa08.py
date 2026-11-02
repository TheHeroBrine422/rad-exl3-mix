#!/usr/bin/env python3
"""yaqa08.py MODEL META OUT NSEQ [--len L] [--snap 1024,4096,16384] [--start S] -- YAQA output-side Hessians for a
small proxy (Qwen3.5-0.8B on the 4070), the single-card, GPU-resident form of v3/yaqa/collect.py's YAQA_FULL mode:

  for each L-token window of META (xcorp: kld.json docs + tokens.i32; documents >= L tokens, windows in a fixed
  shuffled order), forward the model, sample y_t ~ p(.|x_<t) from its own distribution (the true Fisher, as YAQA),
  loss = sum_t CE(y_t), backward, and for every trunk linear add  H_out += sum_t w_t dy_t dy_t^T  (dy_t = dL/d out_t,
  w_t = ||x_t||^2 -- collect.py's "x2" weighting, Sketch A's H_in ~ I first step) over positions 16..L-1.

The sums stay on the GPU in fp32 (0.8B: ~6.1 GB for every linear). The model's weights stay bf16 but every linear,
norm and the head compute in fp32 (activations and gradients fp32 end to end: the 27B gates found bf16 gradients
unusable); decoder layers are checkpointed. Snapshots OUT/hout-snapN/<checkpoint key>.safetensors ("hout",
ourq.py --hout) after N sequences, OUT/hout at the end. Resumable only from a snapshot (--start)."""
import argparse, json, os, sys, time, types
import numpy as np, torch
from safetensors.torch import save_file, load_file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import kernel_paths

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("meta"); ap.add_argument("out"); ap.add_argument("nseq", type=int)
ap.add_argument("--len", type=int, default=2048); ap.add_argument("--snap", default="1024,4096,16384")
ap.add_argument("--chunk", type=int, default=256); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--start", type=int, default=0, help="resume after this many sequences from OUT/hout-snap<start>")
ap.add_argument("--gate", action="store_true", help="2 sequences, no files: time and memory")
a = ap.parse_args()
L, dev = a.len, torch.device("cuda")
SNAPS = {int(x) for x in a.snap.split(",") if x}

# ---- data: L-token windows of documents >= L tokens
m = json.load(open(f"{a.meta}/kld.json")); toks = np.memmap(f"{a.meta}/tokens.i32", np.int32, "r")
off = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
wins = [int(off[i] + s) for i, d in enumerate(m["docs"]) for s in range(0, d["tokens"] - L + 1, L)]
np.random.default_rng(a.seed).shuffle(wins)
print(f"{len(wins)} windows of {L} tokens available; using {a.nseq}", flush=True)
assert len(wins) >= a.nseq, "not enough windows"

# ---- model: bf16 weights, fp32 compute
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev)
model.requires_grad_(False)
dec, head = model.get_decoder(), model.get_output_embeddings()
def fwd32(self, x):
    y = torch.nn.functional.linear(x.float(), self.weight.float())
    return y if self.bias is None else y + self.bias.float()
for mod in model.modules():
    if isinstance(mod, torch.nn.Linear): mod.forward = types.MethodType(fwd32, mod)
    elif not isinstance(mod, torch.nn.Embedding):
        for pn, p in list(mod.named_parameters(recurse=False)): p.data = p.data.float()
dec.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train()                                     # checkpointing is active in train mode; no dropout in this model

# ---- the trunk linears and their sums
lin = {}
for l, ly in enumerate(dec.layers):
    for nm, mod in ly.named_modules():
        if isinstance(mod, torch.nn.Linear) and not nm.endswith(("in_proj_a", "in_proj_b")):
            lin[f"model.language_model.layers.{l}.{nm}"] = mod
ACC = {k: torch.zeros(mod.out_features, mod.out_features, device=dev) for k, mod in lin.items()}
NROW = {k: 0 for k in lin}
if a.start:
    sd = f"{a.out}/hout-snap{a.start}"
    for k in lin:
        f = load_file(f"{sd}/{k}.safetensors"); ACC[k].copy_(f["hout"].to(dev))
    NROW = json.load(open(f"{sd}/rows.json"))
print(f"{len(lin)} linears; sums {sum(v.numel() for v in ACC.values()) * 4 / 2**30:.2f} GiB; "
      f"model {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated in all", flush=True)

state = {"done": set()}
def hook(key):
    def h(mod, inp, out):
        if not out.requires_grad: return
        w = inp[0][0, 16:].detach().float().pow(2).sum(1, keepdim=True)          # [T-16, 1]
        def bh(g):
            if key in state["done"]: return                                     # checkpoint recompute guard
            state["done"].add(key)
            r = g[0, 16:].float() * w.sqrt()
            ACC[key].addmm_(r.T, r); NROW[key] += r.shape[0]
        out.register_hook(bh)
    return h
for k, mod in lin.items(): mod.register_forward_hook(hook(k))

from torch.utils.checkpoint import checkpoint
def chunk_loss(h, y): return torch.nn.functional.cross_entropy(head(h), y, reduction="sum")

def save(sub):
    d = f"{a.out}/{sub}"; os.makedirs(d, exist_ok=True)
    for k, H in ACC.items():
        save_file({"hout": H.cpu()}, f"{d}/{k}.safetensors.part",
                  metadata={"rows": str(NROW[k]), "weight": "x2", "fp32_act": "True", "model": a.model,
                            "kernels": json.dumps(kernel_paths())})
        os.replace(f"{d}/{k}.safetensors.part", f"{d}/{k}.safetensors")
    json.dump(NROW, open(f"{d}/rows.json", "w"))

t0 = time.time()
for s in range(a.start, 2 if a.gate else a.nseq):
    torch.manual_seed(a.seed * 1_000_003 + s)
    ids = torch.from_numpy(np.asarray(toks[wins[s]:wins[s] + L], np.int64))[None].to(dev)
    state["done"] = set(); ts = time.time()
    emb = dec.embed_tokens(ids).float().detach().requires_grad_(True)
    hid = dec(inputs_embeds=emb, use_cache=False).last_hidden_state[0, :-1]
    loss = 0
    for c in range(0, hid.shape[0], a.chunk):
        h = hid[c:c + a.chunk]
        with torch.no_grad(): y = torch.multinomial(torch.softmax(head(h), -1), 1)[:, 0]
        loss = loss + checkpoint(chunk_loss, h, y, use_reentrant=False)
    loss.backward()
    torch.cuda.synchronize()
    n = s + 1
    if n % 25 == 0 or a.gate or n in SNAPS:
        el = time.time() - t0
        print(f"seq {n}/{a.nseq}: {time.time() - ts:.2f}s, loss/tok {loss.item() / hid.shape[0]:.3f}, "
              f"{(n - a.start) / el:.2f} seq/s, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
              f"{len(state['done'])}/{len(lin)} linears", flush=True)
    del hid, loss, emb
    if not a.gate and n in SNAPS: save(f"hout-snap{n}")
if not a.gate: save("hout")
print(f"YAQA08 DONE {a.nseq - a.start} seqs in {(time.time() - t0) / 60:.1f} min", flush=True)
