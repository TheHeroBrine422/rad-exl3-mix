#!/usr/bin/env python3
"""collect.py OUTDIR NSEQ [SEQLEN] [NPOS] -- YAQA output-side Hessian data for Qwen3.8-27B (Sketch-A style, one step).

For each sequence (qad-pool documents of the challenge's calib golden, cut into SEQLEN-token windows):
  1. forward the bf16 model (both cards, layers split), logits for every position;
  2. sample y_t ~ p(.|x_<t) from the model's own distribution (true Fisher, as YAQA), loss = -sum_t log p(y_t);
  3. backward (gradient checkpointing; weights frozen) and, at NPOS random positions shared by all linears, record
     for every trunk linear  r_t = ||x_t|| * dy_t  (dy_t = dL/d(output) of that linear, x_t its input).
The output-side Hessian of a linear is then H_out ~ sum_t r_t r_tᵀ (H_in ≈ I weighting of Sketch A's first step).
Rows go to OUTDIR/<linear>/<seq>.bf16 ([NPOS][out_features]); build.py folds them into H_out.
Env: YAQA_GATE=1 -> 2 sequences, no files, print time / memory (gate 1).
     YAQA_FP32=1 -> fp32 activations and gradients end to end (weights stay bf16, cast per op in row chunks;
                    norms / conv / gates promoted to fp32), against bf16 rounding noise in the backward pass.
     YAQA_ROWS_F32=1 -> store rows as fp32 (<seq>.f32) instead of bf16 (noise comparisons).
     YAQA_LABELS=DIR -> use the sampled labels another run saved in DIR/.labels (paired comparisons)."""
import json, math, os, sys, time
sys.path.insert(0, "/data/yaqa/pylib")
import numpy as np, torch

OUT, NSEQ = sys.argv[1], int(sys.argv[2])
L = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
NPOS = int(sys.argv[4]) if len(sys.argv) > 4 else 32
W = os.environ.get("RXM_WORK", "/root")
CK = os.environ.get("RXM_CKPT", "/data/workspace/models/qwen38-27b-bf16")
GATE = os.environ.get("YAQA_GATE") == "1"
FP32 = os.environ.get("YAQA_FP32") == "1"
ROWS_F32 = os.environ.get("YAQA_ROWS_F32") == "1"
LABELS = os.environ.get("YAQA_LABELS")
ADT = torch.float32 if FP32 else torch.bfloat16      # activation dtype
SEED = int(os.environ.get("YAQA_SEED", 0))
START = int(os.environ.get("YAQA_START", 0))   # window index to start at (shuffled order is fixed by SEED)

# ---- data: qad-pool windows from the calib golden's token stream
G = f"{W}/challenge/golden/calib"
meta = json.load(open(f"{G}/kld.json")); toks = np.fromfile(f"{G}/tokens.i32", np.int32)
pools = json.load(open(f"{W}/challenge/pools.json"))["pool"]
off = np.cumsum([0] + [d["tokens"] for d in meta["docs"]])
assert off[-1] == len(toks), (off[-1], len(toks))
wins = []
for i, d in enumerate(meta["docs"]):
    if pools[i] != "qad": continue
    for a in range(0, d["tokens"] - L + 1, L): wins.append(int(off[i] + a))
rng = np.random.default_rng(SEED); rng.shuffle(wins)
print(f"{len(wins)} qad windows of {L} tokens", flush=True)

# ---- model: text model only, built empty, every tensor streamed from the safetensors onto its card (the
# transformers loader stages too much in host RAM: OOM on the 30 GB box)
from transformers import AutoConfig, Qwen3_5ForCausalLM
from accelerate import init_empty_weights, dispatch_model
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
tcfg = AutoConfig.from_pretrained(CK).text_config
n = tcfg.num_hidden_layers; SPLIT = int(os.environ.get("YAQA_SPLIT", n // 2))
def card(name):   # model-side name -> device
    if name.startswith("model.layers."): return 0 if int(name.split(".")[2]) < SPLIT else 1
    return 1 if name.startswith(("lm_head", "model.norm")) else 0
t0 = time.time()
with init_empty_weights(): model = Qwen3_5ForCausalLM(tcfg)
idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
want = {k: v for k, v in idx.items() if k.startswith("model.language_model.") or k == "lm_head.weight"}
byfile = {}
for k, f in want.items(): byfile.setdefault(f, []).append(k)
for f, ks in byfile.items():
    with safe_open(f"{CK}/{f}", "pt", device="cpu") as h:
        for k in ks:
            name = "lm_head.weight" if k == "lm_head.weight" else "model." + k[len("model.language_model."):]
            set_module_tensor_to_device(model, name, f"cuda:{card(name)}", value=h.get_tensor(k), dtype=torch.bfloat16)
if FP32:
    class Lin32(torch.autograd.Function):   # x @ W.T in fp32 with W bf16, cast per 32k-row chunk (lm_head: 5 GB in fp32)
        @staticmethod
        def forward(ctx, x, w):
            ctx.save_for_backward(w)
            return torch.cat([x @ w[i:i + 32768].float().T for i in range(0, w.shape[0], 32768)], -1)
        @staticmethod
        def backward(ctx, g):
            (w,) = ctx.saved_tensors
            gx = 0
            for i in range(0, w.shape[0], 32768): gx = gx + g[..., i:i + 32768] @ w[i:i + 32768].float()
            return gx, None
    import types
    def fwd32(self, x):
        y = Lin32.apply(x.float(), self.weight)
        return y if self.bias is None else y + self.bias.float()
    for m in model.modules():
        if isinstance(m, torch.nn.Linear): m.forward = types.MethodType(fwd32, m)
        elif not isinstance(m, torch.nn.Embedding):
            for pn, p_ in list(m.named_parameters(recurse=False)): p_.data = p_.data.float()
dmap = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 1, "lm_head": 1}
for l in range(n): dmap[f"model.layers.{l}"] = 0 if l < SPLIT else 1
model = dispatch_model(model, device_map=dmap)
model.requires_grad_(False)
model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.model.train()          # checkpointing is active in train mode; this model has no dropout
print(f"loaded in {time.time() - t0:.0f}s; " + ", ".join(f"cuda{i} {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB" for i in range(2)), flush=True)

# ---- hooks on every trunk linear
lin = {name: m for name, m in model.model.layers.named_modules() if isinstance(m, torch.nn.Linear)}
state = {"pos": None, "rows": {}}
def fwd_hook(name):
    def h(mod, inp, out):
        if not out.requires_grad: return              # the no-grad first pass of checkpointing
        x = inp[0][0, state["pos"].to(inp[0].device)].float()   # [NPOS, in]
        nx = x.norm(dim=1, keepdim=True)
        def bh(g):
            if name in state["rows"]: return          # first gradient wins (checkpoint recompute guard)
            state["rows"][name] = (g[0, state["pos"].to(g.device)].float() * nx.to(g.device)).to(torch.float32 if ROWS_F32 else torch.bfloat16).cpu()
        out.register_hook(bh)
    return h
for name, m in lin.items(): m.register_forward_hook(fwd_hook(name))
print(f"{len(lin)} trunk linears hooked", flush=True)

dev0 = torch.device("cuda:0")
CH = int(os.environ.get("YAQA_CHUNK", 256))
from torch.utils.checkpoint import checkpoint
def chunk_loss(h, y):   # sampled-label CE of one chunk of positions; recomputed in backward (no stored logits)
    return torch.nn.functional.cross_entropy(model.lm_head(h).float(), y, reduction="sum")
done = 0; t_all = time.time()
for s in range(START, START + (NSEQ if not GATE else 2)):
    if not GATE and os.path.exists(f"{OUT}/.done/{s:05d}"): continue
    srng = np.random.default_rng((SEED, s)); torch.manual_seed(SEED * 1_000_003 + s)
    ids = torch.from_numpy(toks[wins[s]:wins[s] + L].astype(np.int64))[None].to(dev0)
    state["pos"] = torch.from_numpy(np.sort(srng.choice(np.arange(16, L - 1), NPOS, replace=False))).to(dev0)
    state["rows"] = {}
    ts = time.time()
    emb = model.model.embed_tokens(ids).to(ADT).detach().requires_grad_(True)
    hid = model.model(inputs_embeds=emb, use_cache=False).last_hidden_state[0, :-1]
    loss = 0; ys = []
    for c0 in range(0, hid.shape[0], CH):
        h = hid[c0:c0 + CH]
        if LABELS:
            y = torch.from_numpy(np.fromfile(f"{LABELS}/.labels/{s:05d}.i32", np.int32)[c0:c0 + CH].astype(np.int64)).to(h.device)
        else:
            with torch.no_grad():
                y = torch.multinomial(torch.softmax(model.lm_head(h).float(), -1), 1)[:, 0]
        ys.append(y)
        loss = loss + checkpoint(chunk_loss, h, y, use_reentrant=False)
    loss.backward()
    dt = time.time() - ts
    if not GATE:
        for name, r in state["rows"].items():
            d = f"{OUT}/{name}"; os.makedirs(d, exist_ok=True)
            if ROWS_F32: r.numpy().tofile(f"{d}/{s:05d}.f32")
            else: r.view(torch.int16).numpy().tofile(f"{d}/{s:05d}.bf16")
        os.makedirs(f"{OUT}/.labels", exist_ok=True)
        torch.cat(ys).cpu().numpy().astype(np.int32).tofile(f"{OUT}/.labels/{s:05d}.i32")
        os.makedirs(f"{OUT}/.done", exist_ok=True); np.array([wins[s]], np.int64).tofile(f"{OUT}/.done/{s:05d}")   # marker = window start
    done += 1
    print(f"seq {s}: {dt:.1f}s, {len(state['rows'])} linears recorded, hid {hid.dtype}, loss/tok {loss.item() / hid.shape[0]:.3f}, peak " +
          ", ".join(f"cuda{i} {torch.cuda.max_memory_allocated(i) / 2**30:.1f} GiB" for i in range(2)), flush=True)
    del hid, loss, emb
print(f"DONE {done} seqs in {(time.time() - t_all) / 60:.1f} min", flush=True)
