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
     YAQA_LABELS=DIR -> use the sampled labels another run saved in DIR/.labels (paired comparisons).
     YAQA_FULL=L[,L..] -> for the linears of these layers, sum H_out = Σ_t w_t dy_t dy_tᵀ over EVERY position
                    (from 16 on) instead of keeping sampled rows; w_t = ||x_t||² (YAQA_UNWEIGHTED=1: w_t = 1). The
                    sums live in fp32 on the CPU (dyᵀdy per sequence, ~2.5 s for a 17,408-wide output), written
                    as OUTDIR/hout/<checkpoint key>.safetensors ("hout", the ourq.py --hout layout) every
                    YAQA_SAVE_EVERY sequences (default 25) and at the end. Sampled rows are not written then.
                    The backward starts at the lowest full layer's input (layers below run forward only; gradients
                    at and above it are unchanged; YAQA_TRUNC=0 runs it to the embedding, for checking), and the
                    CPU sums run in a background thread, overlapping the GPU (joined at the end of each sequence).
                    YAQA_SNAP=n[,n..] also writes the sums after n sequences to OUTDIR/hout-snapN (truncation check,
                    sample-count test); YAQA_HALVES=1 keeps a second sum over the even-numbered sequences, written as
                    OUTDIR/hout-even[-snapN] (split-half agreement: even vs total - even).
     YAQA_HIN=1 -> with YAQA_FULL, also the gradient-weighted input Hessians of those layers:
                    H_in = Σ_t (Σ over the group's linears ||dy_t||²) x_t x_tᵀ per input group (q/k/v or GDN qkv/z,
                    o_proj / out_proj, gate/up, down), positions 16 on, fp32 CPU sums, written next to the snapshots as
                    OUTDIR/hin[-snapN]/<ourq input-Hessian name>.safetensors (the challenge layout: hessian_upper,
                    metadata k / rows) for ourq.py --hin.
     YAQA_HSPLIT=L[,L..] -> forward only (D2): input Hessians x xᵀ of those layers' input groups (q/k/v or GDN qkv/z,
                    o_proj / out_proj, gate/up; down_proj with HS_DOWN=1) from NSEQ windows of the NON-verify docs
                    (pools-v3), split by document parity into OUTDIR/hA and OUTDIR/hB (challenge layout, ourq.py --hin /
                    hproxy). The forward stops after the last needed module. GPU fp32 sums.
     YAQA_DIAG=1 -> the diagonal of YAQA Sketch B's output factor (#33-12): per sequence b and trunk linear (weight
                    [m, n]), G_b = sum_t dy_t x_tᵀ over every position, diag(H_out) += rowsum(G_b ∘ G_b) (= diag G_b G_bᵀ),
                    fp32 on the linear's card, G_b built in DIAG_CHUNK-row blocks (never whole). No sampled rows, no
                    H_in (the calib input Hessians stay). Written as OUTDIR/hdiag[-snapN]/<checkpoint key>.safetensors,
                    "hout_diag" (mean over sequences; ourq.py --hout reads it), every YAQA_SAVE_EVERY sequences and at
                    YAQA_SNAP counts and at the end; OUTDIR/hdiag-state.pt makes a restart continue where it stopped (same SEED);
                    OUTDIR/STOP ends the run cleanly after the current sequence. Every save also writes halves.json:
                    per linear the cosine between the even- and odd-sequence means (split-half agreement, #33-17).
                    Use with YAQA_FP32=1 (bf16 gradients are too noisy, #33-03); YAQA_PREC=x2 runs the linear GEMMs as
                    bf16 hi/lo splits (bf16x.py; check proxy2b/prectest.py first). With YAQA_GATE=1: 2 sequences, and
                    one linear's diagonal is checked against a whole fp32 G_b.
     YAQA_PROFILE=n -> torch.profiler over the n-th sequence of the run (CPU + GPU activity, both cards), table of the
                    top ops by GPU time and by CPU time -> OUTDIR/profile.txt, trace -> OUTDIR/profile.json. Every
                    sequence's log line also splits its time into forward / backward / waiting for the CPU sums."""
import json, math, os, sys, time
sys.path.insert(0, "/data/yaqa/pylib"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import bf16x                                       # near-fp32 GEMMs on bf16 cores (YAQA_PREC=x2)

OUT, NSEQ = sys.argv[1], int(sys.argv[2])
L = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
NPOS = int(sys.argv[4]) if len(sys.argv) > 4 else 32
W = os.environ.get("RXM_WORK", "/root")
CK = os.environ.get("RXM_CKPT", "/data/workspace/models/qwen38-27b-bf16")
GATE = os.environ.get("YAQA_GATE") == "1"
FP32 = os.environ.get("YAQA_FP32") == "1"
PREC = os.environ.get("YAQA_PREC", "fp32")          # with YAQA_FP32: linear GEMMs in fp32 or x2 (bf16x.py)
ROWS_F32 = os.environ.get("YAQA_ROWS_F32") == "1"
LABELS = os.environ.get("YAQA_LABELS")
ADT = torch.float32 if FP32 else torch.bfloat16      # activation dtype
SEED = int(os.environ.get("YAQA_SEED", 0))
START = int(os.environ.get("YAQA_START", 0))   # window index to start at (shuffled order is fixed by SEED)

# ---- data: qad-pool windows from the calib golden's token stream
G = f"{W}/challenge/golden/calib"
meta = json.load(open(f"{G}/kld.json")); toks = np.fromfile(f"{G}/tokens.i32", np.int32)
pools = json.load(open(os.environ.get("YAQA_POOLS", f"{W}/challenge/pools-v3.json")))["pool"]   # amended 2026-11-01
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
NG = torch.cuda.device_count()     # 1: everything on card 0 (proxy tests on one GPU)
def card(name):   # model-side name -> device
    if NG == 1: return 0
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
if "lm_head.weight" not in want:     # tied embedding (the small proxies): the head is the embedding
    model.lm_head.weight = model.model.embed_tokens.weight
if FP32:
    class Lin32(torch.autograd.Function):   # x @ W.T in fp32 with W bf16, cast per 32k-row chunk (lm_head: 5 GB in fp32)
        @staticmethod
        def forward(ctx, x, w):
            ctx.save_for_backward(w)
            return bf16x.lin(x, w, PREC)                # PREC fp32: W cast per 32k-row chunk, fp32 GEMM (as before)
        @staticmethod
        def backward(ctx, g):
            (w,) = ctx.saved_tensors
            return bf16x.lin_back(g, w, PREC), None     # 8192-row chunks: card 0 sits ~1 GB from full in fp32 mode
    import types
    def fwd32(self, x):
        y = Lin32.apply(x.float(), self.weight)
        return y if self.bias is None else y + self.bias.float()
    for m in model.modules():
        if isinstance(m, torch.nn.Linear): m.forward = types.MethodType(fwd32, m)
        elif not isinstance(m, torch.nn.Embedding):
            for pn, p_ in list(m.named_parameters(recurse=False)): p_.data = p_.data.float()
dmap = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": min(1, NG - 1), "lm_head": min(1, NG - 1)}
for l in range(n): dmap[f"model.layers.{l}"] = 0 if l < SPLIT or NG == 1 else 1
model = dispatch_model(model, device_map=dmap)
model.requires_grad_(False)
model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.model.train()          # checkpointing is active in train mode; this model has no dropout
print(f"loaded in {time.time() - t0:.0f}s; " + ", ".join(f"cuda{i} {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB" for i in range(NG)), flush=True)

# ---- YAQA_HSPLIT=L[,L..]: forward-only split-half input Hessians (D2), then exit
HSPLIT = [int(x) for x in os.environ.get("YAQA_HSPLIT", "").split(",") if x]
if HSPLIT:
    from safetensors.torch import save_file
    DOWN = os.environ.get("HS_DOWN") == "1"
    pv = json.load(open(os.environ.get("YAQA_POOLS", f"{W}/challenge/pools-v3.json")))["pool"]
    hw = []                                   # (window start, half): every non-verify doc's windows, half = doc parity
    for i, d in enumerate(meta["docs"]):
        if pv[i] == "verify": continue
        for a in range(0, d["tokens"] - L + 1, L): hw.append((int(off[i] + a), i % 2))
    np.random.default_rng(SEED + 1).shuffle(hw)
    ACC, CNT = {}, {}
    def gname_of(l, mod):
        p = f"model.language_model.layers.{l}"
        if mod in ("self_attn.q_proj", "linear_attn.in_proj_qkv"): return f"{p}.input"
        if mod in ("self_attn.o_proj", "linear_attn.out_proj"): return f"{p}.self_attn.o_proj.input" if "self_attn" in mod else f"{p}.linear_attn.out_proj.input"
        if mod == "mlp.gate_proj": return f"{p}.post_attention.input"
        if mod == "mlp.down_proj" and DOWN: return f"{p}.mlp.down_proj.input"
        return None
    cur = {"half": 0}
    class Stop(Exception): pass
    def hk(gn, last):
        def h(mod, inp):
            x = inp[0][0].float()
            key = (cur["half"], gn)
            if key not in ACC: ACC[key] = torch.zeros(x.shape[1], x.shape[1], device=x.device); CNT[key] = 0
            ACC[key].addmm_(x.T, x); CNT[key] += x.shape[0]
            if last: raise Stop()
        return h
    LMAX = max(HSPLIT)
    for l in HSPLIT:
        lay = model.model.layers[l]
        for mname, m in lay.named_modules():
            gn = gname_of(l, mname)
            if gn: m.register_forward_pre_hook(hk(gn, l == LMAX and mname == ("mlp.down_proj" if DOWN else "mlp.gate_proj")))
    print(f"hsplit: layers {HSPLIT}, {len(hw)} non-verify windows, taking {NSEQ}", flush=True)
    dev0 = torch.device("cuda:0")
    t_all = time.time()
    with torch.no_grad():
        for s in range(NSEQ):
            w0, cur["half"] = hw[s]
            ids = torch.from_numpy(toks[w0:w0 + L].astype(np.int64))[None].to(dev0)
            try: model.model(input_ids=ids, use_cache=False)
            except Stop: pass
            if s % 50 == 0: print(f"seq {s}: {(time.time() - t_all) / (s + 1):.2f} s/seq", flush=True)
    for (hf, gn), H in ACC.items():
        d = f"{OUT}/h{'AB'[hf]}"; os.makedirs(d, exist_ok=True)
        k = H.shape[0]; iu = torch.triu_indices(k, k, device=H.device)
        save_file({"hessian_upper": H[iu[0], iu[1]].cpu().contiguous()}, f"{d}/{gn}.safetensors",
                  metadata={"k": str(k), "rows": str(CNT[(hf, gn)]), "source": f"collect.py hsplit half {'AB'[hf]}"})
    print(f"DONE hsplit {NSEQ} seqs in {(time.time() - t_all) / 60:.1f} min", flush=True)
    sys.exit(0)

# ---- hooks on every trunk linear
lin = {name: m for name, m in model.model.layers.named_modules() if isinstance(m, torch.nn.Linear)}
state = {"pos": None, "rows": {}, "full_done": set()}
FULL = {int(x) for x in os.environ.get("YAQA_FULL", "").split(",") if x}
UNW = os.environ.get("YAQA_UNWEIGHTED") == "1"
HACC, HN = {}, {}          # full-mode accumulators (CPU fp32) and token counts, by linear name
HACC_E = {}                # the even-sequence half (YAQA_HALVES)
HALVES = os.environ.get("YAQA_HALVES") == "1"
HIN = os.environ.get("YAQA_HIN") == "1"
SNAPS = {int(x) for x in os.environ.get("YAQA_SNAP", "").split(",") if x}
HINACC, HINN = {}, {}      # gradient-weighted input Hessians by ourq input-Hessian name
def hin_group(name):       # "22.self_attn.q_proj" -> ourq's input-Hessian name (None: not quantized, e.g. GDN a/b)
    L, rest = name.split(".", 1); p = f"model.language_model.layers.{L}"
    if rest in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "linear_attn.in_proj_qkv", "linear_attn.in_proj_z"):
        return f"{p}.input"
    return {"self_attn.o_proj": f"{p}.self_attn.o_proj.input", "linear_attn.out_proj": f"{p}.linear_attn.out_proj.input",
            "mlp.gate_proj": f"{p}.post_attention.input", "mlp.up_proj": f"{p}.post_attention.input",
            "mlp.down_proj": f"{p}.mlp.down_proj.input"}.get(rest)
@torch.no_grad()           # worker threads start with grad mode ON (thread-local): an in-place sum would join the graph
def _acc_hin(gname, x, gn):
    xw = x * gn.sqrt()[:, None]
    if gname not in HINACC: HINACC[gname] = torch.zeros(x.shape[1], x.shape[1])
    HINACC[gname].addmm_(xw.T, xw)
torch.set_num_threads(int(os.environ.get("YAQA_CPU_THREADS", 12)))
TRUNC = bool(FULL) and os.environ.get("YAQA_TRUNC", "1") == "1"
from concurrent.futures import ThreadPoolExecutor
POOL, PEND = ThreadPoolExecutor(1), []    # one worker: the sums of one tensor never race; torch threads inside
@torch.no_grad()
def _acc(name, r, even=False):
    if name not in HACC: HACC[name] = torch.zeros(r.shape[1], r.shape[1])
    HACC[name].addmm_(r.T, r)
    if even:
        if name not in HACC_E: HACC_E[name] = torch.zeros(r.shape[1], r.shape[1])
        HACC_E[name].addmm_(r.T, r)
def join_full():
    for f in PEND: f.result()
    PEND.clear()
if TRUNC:
    def _start_grad(mod, args, kwargs):          # the backward starts here: a fresh leaf at the layer's input
        if args:
            x = args[0]
            return (x.detach().requires_grad_(True),) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs); kwargs["hidden_states"] = kwargs["hidden_states"].detach().requires_grad_(True)
        return args, kwargs
    model.model.layers[min(FULL)].register_forward_pre_hook(_start_grad, with_kwargs=True)
    print(f"backward truncated at layer {min(FULL)}", flush=True)
def fwd_hook_full(name):
    def h(mod, inp, out):
        if not out.requires_grad: return
        w = None if UNW else inp[0][0, 16:].detach().float().pow(2).sum(1, keepdim=True)   # [T, 1]
        gname = hin_group(name) if HIN else None
        if gname:                                   # the group's input, once per sequence, on the CPU
            if gname not in state["xin"]: state["xin"][gname] = inp[0][0, 16:].detach().float().cpu()   # detach: no graph link
            xin = state["xin"][gname]
        def bh(g):
            if name in state["full_done"]: return     # checkpoint recompute guard
            state["full_done"].add(name)
            r = g[0, 16:].float()
            if gname:
                gn = r.pow(2).sum(1).cpu()
                HINN[gname] = HINN.get(gname, 0) + (r.shape[0] if name.endswith(("q_proj", "in_proj_qkv", "o_proj", "out_proj", "gate_proj", "down_proj")) else 0)
                PEND.append(POOL.submit(_acc_hin, gname, xin, gn))
            if w is not None: r = r * w.to(r.device).sqrt()
            r = r.cpu()
            HN[name] = HN.get(name, 0) + r.shape[0]
            PEND.append(POOL.submit(_acc, name, r, HALVES and state["seq"] % 2 == 0))
        out.register_hook(bh)
    return h
def save_full(sub="hout"):
    join_full()
    from safetensors.torch import save_file
    _save_hout(sub, HACC)
    if HALVES: _save_hout(sub.replace("hout", "hout-even"), HACC_E)
    if HIN:
        hd = sub.replace("hout", "hin"); os.makedirs(f"{OUT}/{hd}", exist_ok=True)
        for gname, H in HINACC.items():
            k = H.shape[0]; iu = torch.triu_indices(k, k)
            save_file({"hessian_upper": H[iu[0], iu[1]].contiguous()}, f"{OUT}/{hd}/{gname}.safetensors.part",
                      metadata={"k": str(k), "rows": str(HINN.get(gname, 0)), "weight": "sum ||dy||^2 over the group"})
            os.replace(f"{OUT}/{hd}/{gname}.safetensors.part", f"{OUT}/{hd}/{gname}.safetensors")
def _save_hout(sub, ACC):
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    from safetensors.torch import save_file
    for name, H in ACC.items():
        key = "model.language_model.layers." + name
        save_file({"hout": H.contiguous()}, f"{OUT}/{sub}/{key}.safetensors.part",
                  metadata={"rows": str(HN[name]), "weight": "none" if UNW else "x2", "fp32_act": str(FP32)})
        os.replace(f"{OUT}/{sub}/{key}.safetensors.part", f"{OUT}/{sub}/{key}.safetensors")
DIAG = os.environ.get("YAQA_DIAG") == "1"
DCH = int(os.environ.get("DIAG_CHUNK", 2048))
DACC, DACC_E, DSTATE = {}, {}, {"n": 0, "next": START}     # DACC_E: sum over the even-numbered sequences (split half)
DCHECK = os.environ.get("DIAG_CHECK", "63.self_attn.o_proj")   # small enough for a whole G on the head card
if DIAG and os.path.exists(f"{OUT}/hdiag-state.pt") and not GATE:
    _st = torch.load(f"{OUT}/hdiag-state.pt")
    DSTATE.update(_st["state"]); DACC.update({k: v for k, v in _st["acc"].items()}); DACC_E.update(_st.get("acc_even", {}))
    print(f"hdiag: resuming after {DSTATE['n']} sequences at window {DSTATE['next']}", flush=True)
def fwd_hook_diag(name):
    if name.endswith(("in_proj_a", "in_proj_b")): return lambda *a_: None     # not quantized
    def h(mod, inp, out):
        if not out.requires_grad: return                 # the no-grad first pass of checkpointing
        x = inp[0][0].detach()
        def bh(g):
            if name in state["full_done"]: return        # checkpoint recompute guard
            state["full_done"].add(name)
            with torch.no_grad():
                gg, xx = g[0].float(), x.float()          # [T, m], [T, n]
                d = bf16x.diag_gg(gg, xx, PREC, DCH)      # G_b in DCH-row blocks
                if GATE and name == DCHECK:               # vs one whole fp32 G: chunking and (x2) precision
                    ref = (gg.T @ xx).pow(2).sum(1)
                    print(f"gate check {name} ({PREC}): vs whole fp32 G rel L2 {float((d - ref).norm() / ref.norm()):.2e}, "
                          f"max row {float(((d - ref).abs() / ref.abs().clamp_min(float(ref.mean()) * 1e-3)).max()):.2e}", flush=True)
                if name not in DACC: DACC[name] = torch.zeros_like(d); DACC_E[name] = torch.zeros_like(d)
                elif DACC[name].device != d.device:            # resumed sums load on the CPU: move to this linear's card
                    DACC[name] = DACC[name].to(d.device); DACC_E[name] = DACC_E[name].to(d.device)
                DACC[name] += d
                if DSTATE["n"] % 2 == 0: DACC_E[name] += d     # this sequence's index (0-based) is DSTATE["n"]
        out.register_hook(bh)
    return h
def save_diag(sub="hdiag"):
    from safetensors.torch import save_file
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    for name, v in DACC.items():
        key = "model.language_model.layers." + name
        save_file({"hout_diag": (v / DSTATE["n"]).float().cpu().contiguous()}, f"{OUT}/{sub}/{key}.safetensors.part",
                  metadata={"seqs": str(DSTATE["n"]), "ctx": str(L), "sketch": "B-diag", "fp32_act": str(FP32), "prec": PREC, "seed": str(SEED)})
        os.replace(f"{OUT}/{sub}/{key}.safetensors.part", f"{OUT}/{sub}/{key}.safetensors")
    # split-half agreement: even vs odd sequences (independent halves), cosine per linear -> halves.json (#33-17)
    ne = (DSTATE["n"] + 1) // 2; no = DSTATE["n"] // 2
    if no > 0:
        hv = {}
        for name in DACC:
            e = (DACC_E[name] / ne).double(); o_ = ((DACC[name] - DACC_E[name]) / no).double()
            hv["model.language_model.layers." + name] = float((e * o_).sum() / (e.norm() * o_.norm()))
        v = sorted(hv.values())
        json.dump({"seqs": DSTATE["n"], "min": v[0], "p5": v[max(0, len(v) // 20 - 1)], "median": v[len(v) // 2], "cos": hv},
                  open(f"{OUT}/{sub}/halves.json", "w"), indent=0)
    if sub == "hdiag":
        torch.save({"state": dict(DSTATE), "acc": {k: v.cpu() for k, v in DACC.items()},
                    "acc_even": {k: v.cpu() for k, v in DACC_E.items()}}, f"{OUT}/hdiag-state.pt.part")
        os.replace(f"{OUT}/hdiag-state.pt.part", f"{OUT}/hdiag-state.pt")
def fwd_hook(name):
    if DIAG: return fwd_hook_diag(name)
    if FULL and int(name.split(".")[0]) in FULL: return fwd_hook_full(name)
    def h(mod, inp, out):
        if FULL: return                                # full mode: no sampled rows
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
    if DIAG and s < DSTATE["next"]: continue           # resumed: already in the sums
    if DIAG and os.path.exists(f"{OUT}/STOP"): print(f"STOP file at window {s}", flush=True); break
    if not DIAG and not GATE and os.path.exists(f"{OUT}/.done/{s:05d}"): continue
    srng = np.random.default_rng((SEED, s)); torch.manual_seed(SEED * 1_000_003 + s)
    ids = torch.from_numpy(toks[wins[s]:wins[s] + L].astype(np.int64))[None].to(dev0)
    state["pos"] = torch.from_numpy(np.sort(srng.choice(np.arange(16, L - 1), NPOS, replace=False))).to(dev0)
    state["rows"] = {}; state["full_done"] = set(); state["xin"] = {}; state["seq"] = s
    PROF = int(os.environ.get("YAQA_PROFILE", 0)) == done + 1
    if PROF:
        os.makedirs(OUT, exist_ok=True)
        from torch.profiler import profile, ProfilerActivity
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True); prof.__enter__()
    for i in range(NG): torch.cuda.synchronize(i)
    ts = time.time()
    emb = model.model.embed_tokens(ids).to(ADT).detach().requires_grad_(not TRUNC)
    hid = model.model(inputs_embeds=emb, use_cache=False).last_hidden_state[0, :-1]
    loss = 0; ys = []
    for c0 in range(0, hid.shape[0], CH):
        h = hid[c0:c0 + CH]
        if LABELS and os.path.exists(f"{LABELS}/.labels/{s:05d}.i32"):   # past the labelled run's range: sample
            y = torch.from_numpy(np.fromfile(f"{LABELS}/.labels/{s:05d}.i32", np.int32)[c0:c0 + CH].astype(np.int64)).to(h.device)
        else:
            with torch.no_grad():
                y = torch.multinomial(torch.softmax(model.lm_head(h).float(), -1), 1)[:, 0]
        ys.append(y)
        loss = loss + checkpoint(chunk_loss, h, y, use_reentrant=False)
    for i in range(NG): torch.cuda.synchronize(i)
    t_fwd = time.time() - ts
    loss.backward()
    for i in range(NG): torch.cuda.synchronize(i)
    t_bwd = time.time() - ts - t_fwd
    t_j = time.time()
    if FULL: join_full()
    t_wait = time.time() - t_j
    if PROF:
        prof.__exit__(None, None, None)
        with open(f"{OUT}/profile.txt", "w") as fh:
            fh.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40) + "\n\n")
            fh.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))
        prof.export_chrome_trace(f"{OUT}/profile.json")
    dt = time.time() - ts
    if FULL: join_full()
    if FULL and not GATE and done + 1 in SNAPS: save_full(f"hout-snap{done + 1}")
    if DIAG:
        DSTATE["n"] += 1; DSTATE["next"] = s + 1
        if not GATE:
            if DSTATE["n"] in SNAPS: save_diag(f"hdiag-snap{DSTATE['n']}")
            if DSTATE["n"] % int(os.environ.get("YAQA_SAVE_EVERY", 25)) == 0: save_diag()
    elif FULL and not GATE:
        if (done + 1) % int(os.environ.get("YAQA_SAVE_EVERY", 25)) == 0: save_full()
    elif not GATE:
        for name, r in state["rows"].items():
            d = f"{OUT}/{name}"; os.makedirs(d, exist_ok=True)
            if ROWS_F32: r.numpy().tofile(f"{d}/{s:05d}.f32")
            else: r.view(torch.int16).numpy().tofile(f"{d}/{s:05d}.bf16")
        os.makedirs(f"{OUT}/.labels", exist_ok=True)
        torch.cat(ys).cpu().numpy().astype(np.int32).tofile(f"{OUT}/.labels/{s:05d}.i32")
        os.makedirs(f"{OUT}/.done", exist_ok=True); np.array([wins[s]], np.int64).tofile(f"{OUT}/.done/{s:05d}")   # marker = window start
    done += 1
    print(f"seq {s}: {dt:.1f}s (fwd {t_fwd:.1f} bwd {t_bwd:.1f} cpu-wait {t_wait:.1f}), {len(state['full_done']) if DIAG else len(state['rows'])} linears recorded, hid {hid.dtype}, loss/tok {loss.item() / hid.shape[0]:.3f}, peak " +
          ", ".join(f"cuda{i} {torch.cuda.max_memory_allocated(i) / 2**30:.1f} GiB" for i in range(NG)), flush=True)
    del hid, loss, emb
    import gc; gc.collect(); torch.cuda.empty_cache()     # hook closures can hold graph pieces in reference cycles
    print(f"  mem after seq {s}: " + ", ".join(f"cuda{i} alloc {torch.cuda.memory_allocated(i) / 2**30:.2f} GiB" for i in range(NG)), flush=True)
    for i in range(NG): torch.cuda.reset_peak_memory_stats(i)     # the next line's peak is that sequence's own
if FULL and not GATE: save_full()
if DIAG and not GATE and done: save_diag()
print(f"DONE {done} seqs in {(time.time() - t_all) / 60:.1f} min", flush=True)
