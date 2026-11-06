#!/usr/bin/env python3
"""swjfit27.py MODEL META OUT --src NAME=DIR[:wq|:level] ... --scratch DIR [--stages N|auto | --split A,B,..] -- C10 "swj"
(proxy2b/exp/w02-fixedscale/swiglufit.py: gate / up output scales fitted jointly through the SwiGLU, codes fixed) for a
model that does not fit on one card: the bf16 forward runs in staged layer passes (eval/gold27/mkgold27.py's staging:
transformers' own decoder layers, one document at a time with a fresh cache, bf16 boundaries on disk), and any number of
quantized sources are fitted in the same pass (the forward, the MLP inputs and the bf16 target are shared).

Per source and layer, as swiglufit.py (fsutil.py's swiglu_acc / swiglu_fit / swiglu_resid, the same ops in the same
order): x = the MLP input of the bf16 model, a = silu(W_g x) * (W_u x) in f32, ĝ / û = the source's gate / up;
per channel j the sums over positions of a p and p², p = silu(c ĝ) û, at every grid point c, and of a², in f64 per
document half (even / odd documents: the held-out check); then c_g = the grid point minimising the residual, c_u its
closed form, clamped to [0.5, 2].
- Source DIR:wq  -- DIR/wq/<key>.safetensors (ourq --save-wq, (out, in) f16): swiglufit.py's input, bit-exact with it.
- Source DIR:level -- an ourq.py level (q-L*.safetensors): gate / up decoded by exl3deq.py in f32 (what the kernels
  serve). The 27B arms have no --save-wq files: this is their path.
META: a golden-style dir (kld.json "docs" with "tokens", + tokens.i32), documents cut to their first --cap tokens.

OUT/<NAME>/: scales/<key>.safetensors {"r": ones (in), "c": the multiplier (out)} for every gate / up (fsutil's layout;
swjapply.py edits a level's svh with it), swiglu.jsonl (swiglufit.py's per-layer records), swjfit.json, .done.
--write-wq (0.8B tests): also OUT/<NAME>/wq/ as swiglufit.py's joint arm (gate / up rows x c; the other keys hard-linked
from DIR/wq for :wq sources).

Restartable: DIR/plan.json fixes the plan; progress.json {stage, docs} is committed after each checkpoint (every
--ckpt-min minutes: the stage's f64 sums, ~45 MB a layer and source on the 27B) and at each stage's end (the stage's
sums to DIR/sums/). A killed run resumes from the last checkpoint, bit-identical (the sums are saved exactly). Input
boundaries go once their stage is done. Rerunning a finished OUT is a no-op. VRAM_CAP_GB caps the allocator."""
import argparse, hashlib, json, os, resource, shutil, struct, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, os.path.join(REPO, "proxy2b"))
from common import kernel_paths  # noqa: E402  (P2B_BLOCK before transformers)
sys.path.insert(0, os.path.join(REPO, "proxy2b", "exp", "w02-fixedscale"))
import fsutil as U  # noqa: E402
sys.path.insert(0, HERE)
import exl3deq  # noqa: E402
import torch  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("model"); ap.add_argument("meta"); ap.add_argument("out")
ap.add_argument("--src", action="append", required=True, help="NAME=DIR[:wq|:level] (default :level)")
ap.add_argument("--scratch", required=True)
ap.add_argument("--stages", default="auto", help="N (balanced by bytes), or auto: as few as fit the VRAM budget")
ap.add_argument("--split", default=None, help="explicit layer boundaries, e.g. 0,11,22,..,64")
ap.add_argument("--reserve-gb", type=float, default=5.0, help="VRAM left for activations and the grid's temporaries")
ap.add_argument("--grid", default="0.6:1.4:0.01"); ap.add_argument("--cap", type=int, default=4096)
ap.add_argument("--maxdocs", type=int, default=0)
ap.add_argument("--ckpt-min", type=float, default=10.0, help="checkpoint the sums every this many minutes")
ap.add_argument("--write-wq", action="store_true", help="also write OUT/NAME/wq (swiglufit.py's joint arm)")
ap.add_argument("--fused", action="store_true", help="the grid sums in one Triton kernel (swjtri.py): ~50x less memory "
                "traffic, not bit-identical (checked on the first block against the torch path, abort above --fused-tol)")
ap.add_argument("--fused-tol", type=float, default=1e-5)
ap.add_argument("--no-cache", action="store_true", help="run the layers with use_cache=False (mkhin27.py's path, the one "
                "run on gpubox's ROCm) instead of a fresh DynamicCache (mkgold27.py's, = swiglufit.py's dec(input_ids))")
ap.add_argument("--plan-only", action="store_true")
a = ap.parse_args()
dev = torch.device(os.environ.get("SWJ_DEV", "cuda")); GPU = dev.type == "cuda"
torch.set_grad_enabled(False)
GB = 2**30
if GPU: U.cap_vram()


def log(*x): print(time.strftime("%H:%M:%S"), *x, file=sys.stderr, flush=True)


def write_atomic(path, data):
    with open(path + ".part", "wb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(path + ".part", path)


SRC = []
for s in a.src:
    name, _, rest = s.partition("=")
    d, mode = (rest.rsplit(":", 1) if rest.rsplit(":", 1)[-1] in ("wq", "level") else (rest, "level"))
    assert name and os.path.isdir(d), f"--src {s}: NAME=DIR[:wq|:level]"
    SRC.append((name, os.path.abspath(d), mode))
assert len({n for n, _, _ in SRC}) == len(SRC), "source names must differ"
if all(os.path.exists(f"{a.out}/{x[0]}/.done") for x in SRC): log(f"{a.out}: every source is done"); raise SystemExit(0)
todo = SRC                    # any source left: the run (and its plan) covers all of them; the fit is cheap

# ------------------------------------------------------------------ META, model config, stage plan
m = json.load(open(f"{a.meta}/kld.json")); toks = np.memmap(f"{a.meta}/tokens.i32", np.int32, "r")
toff = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
nd = len(m["docs"]) if not a.maxdocs else min(a.maxdocs, len(m["docs"]))
dlen = [min(int(toff[i + 1] - toff[i]), a.cap) for i in range(nd)]
NTOK = int(sum(dlen))

from safetensors import safe_open  # noqa: E402
from transformers import AutoConfig, DynamicCache  # noqa: E402
import transformers.models.qwen3_5.modeling_qwen3_5 as q35  # noqa: E402

cfg = AutoConfig.from_pretrained(a.model)
cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
cfg._attn_implementation = "sdpa"           # what from_pretrained picks (swiglufit.py's model)
NL, H, NJ = cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size
idx = f"{a.model}/model.safetensors.index.json"
wmap = json.load(open(idx))["weight_map"] if os.path.exists(idx) else \
    {k: f for f in os.listdir(a.model) if f.endswith(".safetensors") for k in safe_open(f"{a.model}/{f}", "pt").keys()}
LP = "model.language_model." if any(k.startswith("model.language_model.") for k in wmap) else "model."
L = LP + "layers"
def gu(l): return f"{L}.{l}.mlp.gate_proj", f"{L}.{l}.mlp.up_proj"


def shape(k):
    with safe_open(f"{a.model}/{wmap[k]}", "pt") as f: return f.get_slice(k).get_shape()


def layer_keys(i): return [k for k in wmap if k.startswith(f"{L}.{i}.")]


# the sources hold every gate / up?
LV = {}
for name, d, mode in todo:
    if mode == "level":
        LV[name] = exl3deq.Level(d, dev=dev)
        miss = [k for l in range(NL) for k in gu(l) if f"{k}.trellis" not in LV[name].ix]
    else:
        miss = [k for l in range(NL) for k in gu(l) if not os.path.exists(f"{d}/wq/{k}.safetensors")]
    assert not miss, f"source {name} ({d}, {mode}): {len(miss)} gate / up missing (first {miss[0]})"

GRID = torch.arange(*(lambda g0, g1, gs: (g0, g1 + gs / 2, gs))(*map(float, a.grid.split(":"))), device=dev)
NG = len(GRID); i1 = int(torch.argmin((GRID - 1).abs()))
assert abs(float(GRID[i1]) - 1) < 1e-6, "grid must contain 1.0"
wbytes = [sum(int(np.prod(shape(k))) * 2 for k in layer_keys(i)) for i in range(NL)]
cost = [w + len(todo) * (2 * NJ * H * 4 + 2 * (2 * NG + 1) * NJ * 8) for w in wbytes]   # + f32 gate / up, f64 sums
cap = os.environ.get("VRAM_CAP_GB")
total_vram = torch.cuda.get_device_properties(0).total_memory if GPU else 0
budget = (float(cap) * GB if cap else total_vram) - a.reserve_gb * GB


def balanced(N):
    cum = np.cumsum(cost); tot = cum[-1]
    sp = [0] + [int(np.argmin(np.abs(cum - tot * s / N))) + 1 for s in range(1, N)] + [NL]
    for s in range(1, N): sp[s] = min(max(sp[s], sp[s - 1] + 1), NL - (N - s))
    return sp


if a.split:
    split = [int(x) for x in a.split.split(",")]
elif a.stages == "auto":
    if budget <= 0: raise SystemExit("--stages auto needs VRAM_CAP_GB or a GPU")
    split, used = [0], 0
    for i in range(NL):
        if used and used + cost[i] > budget: split.append(i); used = 0
        used += cost[i]
    split.append(NL)
    bal = balanced(len(split) - 1)
    if max(sum(cost[x:y]) for x, y in zip(bal, bal[1:])) <= budget: split = bal
else:
    split = balanced(int(a.stages))
N = len(split) - 1
assert split[0] == 0 and split[-1] == NL and all(x < y for x, y in zip(split, split[1:])), f"bad split {split}"
sgb = [sum(cost[x:y]) / GB for x, y in zip(split, split[1:])]
log(f"{nd} documents, {NTOK} tokens (cap {a.cap}); {len(todo)} sources {[x[0] for x in todo]}; {N} stages {split}, "
    f"weights + sources + sums {min(sgb):.2f} .. {max(sgb):.2f} GiB a stage (budget {budget / GB:.2f} GiB)")
if a.plan_only: raise SystemExit(0)

meta_hash = hashlib.sha256(open(f"{a.meta}/kld.json", "rb").read()).hexdigest()[:16]
plan = {"model": os.path.abspath(a.model), "meta": os.path.abspath(a.meta), "meta_sha256": meta_hash, "docs": nd,
        "tokens": NTOK, "cap": a.cap, "grid": a.grid, "split": split, "sources": [list(x) for x in todo],
        "fused": a.fused, "no_cache": a.no_cache}
os.makedirs(f"{a.scratch}/sums", exist_ok=True)
pf = f"{a.scratch}/plan.json"
if os.path.exists(pf):
    old = json.load(open(pf))
    if old != plan: raise SystemExit(f"{pf} holds another plan {old}; this run is {plan}: use a fresh --scratch")
else:
    write_atomic(pf, json.dumps(plan).encode())
prf = f"{a.scratch}/progress.json"
prog = json.load(open(prf)) if os.path.exists(prf) else {"stage": 0, "docs": 0, "gen": 0}
def bpath(s, i): return f"{a.scratch}/b{s}/{i:06d}.bf16"
def ackpt(s, g): return f"{a.scratch}/acc-s{s}g{g}"      # a checkpoint generation: one file per (source, layer)
def sums_path(name, l): return f"{a.scratch}/sums/{name}.L{l:02d}.pt"


def commit(s, j, g=0):
    global prog
    prog = {"stage": s, "docs": j, "gen": g}; write_atomic(prf, json.dumps(prog).encode())
    cleanup()


def cleanup():
    """drop what the committed progress does not need: other checkpoint generations, boundaries of finished stages"""
    for x in os.listdir(a.scratch):
        if x.startswith("acc-") and x != os.path.basename(ackpt(prog["stage"], prog["gen"])):
            shutil.rmtree(f"{a.scratch}/{x}", ignore_errors=True)
        if x.startswith("b") and x[1:].isdigit() and int(x[1:]) not in (prog["stage"], prog["stage"] + 1):
            shutil.rmtree(f"{a.scratch}/{x}", ignore_errors=True)


cleanup()


# ------------------------------------------------------------------ weights (mkhin27.py / mkgold27.py)
def load_tensors(keys, strip):
    sd, byfile = {}, {}
    for k in keys: byfile.setdefault(wmap[k], []).append(k)
    for f, ks in byfile.items():
        with safe_open(f"{a.model}/{f}", "pt", device="cpu") as fh:
            for k in ks: sd[k[len(strip):]] = fh.get_tensor(k).to(torch.bfloat16)
    return sd


def load_stage(s):
    mods = []
    for i in range(split[s], split[s + 1]):
        with torch.device("meta"): ly = q35.Qwen3_5DecoderLayer(cfg, i)
        ly.load_state_dict(load_tensors(layer_keys(i), f"{L}.{i}."), strict=True, assign=True)
        mods.append((i, ly.to(dev).eval()))
        assert not any(t.is_meta for t in list(ly.parameters()) + list(ly.buffers())), "a tensor was left on meta"
    return mods


def embed_memmap():
    k = f"{LP}embed_tokens.weight"; fn = f"{a.model}/{wmap[k]}"
    with open(fn, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hd = json.loads(f.read(n))
    e = hd[k]; assert e["dtype"] == "BF16", e
    o0, _ = e["data_offsets"]
    return np.memmap(fn, np.int16, "r", offset=8 + n + o0, shape=tuple(e["shape"]))


rotary = q35.Qwen3_5TextRotaryEmbedding(config=cfg).to(dev)


def run_layers(mods, x):
    """Qwen3_5TextModel.forward(inputs_embeds=x) over a subset of its layers, one whole document with a fresh cache
    (mkgold27.py: bit-identical to the whole model's dec(input_ids=...), which swiglufit.py runs)"""
    n = x.shape[1]
    cache = None if a.no_cache else DynamicCache(config=cfg)
    pid = torch.arange(n, device=dev).view(1, 1, -1).expand(4, 1, -1)
    tpid, pid = pid[0], pid[1:]
    mk = {"config": cfg, "inputs_embeds": x, "attention_mask": None, "past_key_values": cache, "position_ids": tpid}
    masks = {"full_attention": q35.create_causal_mask(**mk), "linear_attention": q35.create_recurrent_attention_mask(**mk)}
    pe = rotary(x, pid)
    for i, ly in mods:
        x = ly(x, position_embeddings=pe, attention_mask=masks[cfg.layer_types[i]], position_ids=tpid,
               past_key_values=cache, use_cache=cache is not None)
    return x


def source_wq(name, d, mode, key):
    """(out, in) f32 contiguous on dev: swiglufit.py's WQ[l][.] for :wq (U.load_wq(..).T.contiguous())"""
    if mode == "wq": return U.load_wq(d, key, dev).T.contiguous()
    return LV[name].get(key).contiguous()


def new_sums(): return [torch.zeros(NG, NJ, dtype=torch.float64, device=dev), torch.zeros(NG, NJ, dtype=torch.float64, device=dev),
                        torch.zeros(NJ, dtype=torch.float64, device=dev)]


def save_sums(S, name, l, path):
    """one (source, layer): both halves' f64 sums, exactly (~45 MB on the 27B: one at a time in host RAM)"""
    torch.save({str(hf): [t.cpu() for t in S[name][(hf, l)]] for hf in (0, 1)}, path + ".part"); os.replace(path + ".part", path)


def load_sums(S, name, l, path):
    for hf, v in torch.load(path).items():
        for t, x in zip(S[name][(int(hf), l)], v): t.copy_(x)


# ------------------------------------------------------------------ the stages
silu = torch.nn.functional.silu
embed = None
timing = open(f"{a.scratch}/timing.jsonl", "a")
for s in range(N):
    if prog["stage"] > s: continue
    j0 = prog["docs"]
    if s > 0:
        bad = [i for i in range(j0, nd) if not os.path.exists(bpath(s, i)) or os.path.getsize(bpath(s, i)) != dlen[i] * H * 2]
        if bad: raise SystemExit(f"stage {s}: {len(bad)} input boundaries missing or torn (first {bad[0]}): scratch damaged")
    if s < N - 1: os.makedirs(f"{a.scratch}/b{s + 1}", exist_ok=True)
    if GPU: torch.cuda.reset_peak_memory_stats()
    tl = time.time(); mods = load_stage(s); layers = [i for i, _ in mods]
    WQ = {name: {l: tuple(source_wq(name, d, mode, k) for k in gu(l)) for l in layers} for name, d, mode in todo}
    S = {name: {(hf, l): new_sums() for hf in (0, 1) for l in layers} for name, _, _ in todo}
    g0 = prog["gen"]
    if j0 > 0:
        for name, _, _ in todo:
            for l in layers: load_sums(S, name, l, f"{ackpt(s, g0)}/{name}.L{l:02d}.pt")
    tl = time.time() - tl
    cur = {"half": 0}
    if a.fused:
        import swjtri
        ACC = swjtri.swiglu_acc
    else:
        ACC = U.swiglu_acc
    chk = {"done": not a.fused}

    def hook(l, mlp):
        def h(mod, inp):
            x = inp[0][0].float()                              # [T, hidden]
            g = x @ mlp.gate_proj.weight.float().T; u = x @ mlp.up_proj.weight.float().T
            aa = silu(g) * u; del g, u
            for name, _, _ in todo:
                gq = x @ WQ[name][l][0].T; uq = x @ WQ[name][l][1].T
                if not chk["done"]:                    # --fused: the kernel against the torch path, first block only
                    err = swjtri.check(U.swiglu_acc, aa, gq, uq, GRID); chk["done"] = True
                    log(f"fused check (layer {l}, source {name}, {aa.shape[0]} positions): max rel difference {err:.2e}")
                    if not err <= a.fused_tol: raise SystemExit(f"fused grid sums differ by {err:.2e} > --fused-tol {a.fused_tol}")
                ACC(S[name][(cur["half"], l)], aa, gq, uq, GRID)
                del gq, uq
        return h
    hooks = [ly.mlp.register_forward_pre_hook(hook(i, ly.mlp)) for i, ly in mods]
    if s == 0 and embed is None: embed = embed_memmap()
    rgb = torch.cuda.memory_allocated() / GB if GPU else 0.0
    log(f"stage {s}: layers {layers[0]}..{layers[-1]}, resident {rgb:.2f} GiB, loaded in {tl:.0f} s; documents {j0}..{nd - 1}")
    t0 = tck = time.time(); ntok = 0
    for i in range(j0, nd):
        n = dlen[i]; cur["half"] = i % 2
        if s == 0:
            e = embed[np.asarray(toks[toff[i]:toff[i] + n], np.int64)]
            x = torch.from_numpy(e).view(torch.bfloat16)[None].to(dev)
        else:
            x = torch.from_numpy(np.fromfile(bpath(s, i), np.int16).reshape(1, n, H)).view(torch.bfloat16).to(dev)
        x = run_layers(mods, x)
        if s < N - 1: write_atomic(bpath(s + 1, i), x[0].contiguous().view(torch.int16).cpu().numpy().tobytes())
        del x; ntok += n
        if time.time() - tck > a.ckpt_min * 60 and i < nd - 1:
            g0 += 1; os.makedirs(ackpt(s, g0), exist_ok=True)
            for name, _, _ in todo:
                for l in layers: save_sums(S, name, l, f"{ackpt(s, g0)}/{name}.L{l:02d}.pt")
            commit(s, i + 1, g0); tck = time.time()
        if (i + 1) % 20 == 0 or i == nd - 1:
            el = time.time() - t0
            print(f"\rstage {s}: {i + 1}/{nd} docs, {ntok} tokens, {ntok / max(el, 1e-9):.0f} tok/s", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    if GPU: torch.cuda.synchronize()
    el = time.time() - t0
    for h in hooks: h.remove()
    for name, _, _ in todo:
        for l in layers: save_sums(S, name, l, sums_path(name, l))
    del mods, WQ, S
    if GPU: torch.cuda.empty_cache()
    commit(s + 1, 0)                     # cleanup() drops this stage's checkpoints and input boundaries
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    peak = torch.cuda.max_memory_allocated() / GB if GPU else 0.0
    timing.write(json.dumps({"stage": s, "layers": [split[s], split[s + 1]], "sources": len(todo), "docs": nd - j0,
                             "tokens": ntok, "load_s": round(tl, 2), "run_s": round(el, 2), "resident_gib": round(rgb, 3),
                             "peak_gib": round(peak, 3), "maxrss_gib": round(rss, 2),
                             "gpu": torch.cuda.get_device_name(0) if GPU else "cpu",
                             "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n"); timing.flush()
    log(f"stage {s}: {ntok} tokens in {el:.0f} s ({ntok / max(el, 1e-9):.0f} tok/s), peak {peak:.2f} GiB, max RSS {rss:.2f} GiB")

if prog["stage"] < N:
    log(f"stage {prog['stage']} of {N} next (rerun the same command)"); raise SystemExit(0)

# ------------------------------------------------------------------ the fit (swiglufit.py's, per source)
fit, resid = U.swiglu_fit, U.swiglu_resid
CLAMP = (0.5, 2.0)
for name, d, mode in todo:
    o = f"{a.out}/{name}"
    os.makedirs(f"{o}/scales", exist_ok=True)
    if a.write_wq: os.makedirs(f"{o}/wq", exist_ok=True)
    logj = open(f"{o}/swiglu.jsonl", "w")
    done = set()
    for l in range(NL):
        blob = torch.load(sums_path(name, l))
        Sl = {hf: [t.to(dev) for t in blob[str(hf)]] for hf in (0, 1)}
        tot = [x + y for x, y in zip(Sl[0], Sl[1])]
        gi, cu, cu1 = fit(tot, i1)
        one = torch.full_like(gi, i1)
        cu = cu.clamp(*CLAMP); cu1 = cu1.clamp(*CLAMP)
        rb, aa = resid(tot, one, torch.ones_like(cu)); ru, _ = resid(tot, one, cu1); rj, _ = resid(tot, gi, cu)
        ge, cue, cu1e = fit(Sl[0], i1)                         # fitted on even docs, scored on odd
        rbo, aao = resid(Sl[1], one, torch.ones_like(cu)); ruo, _ = resid(Sl[1], one, cu1e.clamp(*CLAMP))
        rjo, _ = resid(Sl[1], ge, cue.clamp(*CLAMP))
        cg = GRID[gi]
        rec = {"layer": l, "rel_base": rb / aa, "rel_up": ru / aa, "rel_joint": rj / aa,
               "heldout_base": rbo / aao, "heldout_up": ruo / aao, "heldout_joint": rjo / aao,
               "cg_mean": float(cg.mean()), "cg_sd": float(cg.std()), "cu_mean": float(cu.mean()), "cu_sd": float(cu.std()),
               "cg_at_edge": int(((gi == 0) | (gi == NG - 1)).sum())}
        logj.write(json.dumps(rec) + "\n"); logj.flush()
        print(f"{name} L{l:02d} rel resid base {rb / aa:.4e} up {ru / aa:.4e} joint {rj / aa:.4e} | held-out {rbo / aao:.4e} "
              f"{ruo / aao:.4e} {rjo / aao:.4e} | c_g {float(cg.mean()):.3f}±{float(cg.std()):.3f} edge {rec['cg_at_edge']}", flush=True)
        for key, mult in zip(gu(l), (cg.float(), cu.float())):
            k_in = H
            U.write_scales(o, key, torch.ones(k_in), mult)
            if a.write_wq:
                wq = source_wq(name, d, mode, key)
                U.write_wq(o, key, (wq * mult.view(-1, 1)).T)  # rows of (out, in) = output channels
                del wq
            done.add(key)
    if a.write_wq and mode == "wq":
        for key in U.wq_keys(d):
            if key not in done: U.link_wq(d, o, key)
    json.dump({"tool": "v3/yaqa/swj27/swjfit27.py", "src": d, "mode": mode, "meta": os.path.abspath(a.meta), "grid": a.grid,
               "docs": nd, "tokens": NTOK, "cap": a.cap, "split": split, "fused": a.fused, "no_cache": a.no_cache, "kernels": kernel_paths(),
               "codebook": LV[name].codebook if mode == "level" else None,
               "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(f"{o}/swjfit.json", "w"), indent=1)
    open(f"{o}/.done", "w").close()
    log(f"{name}: scales -> {o}")
log(f"SWJFIT27 DONE {len(todo)} sources, {nd} docs, {NTOK} tokens")
