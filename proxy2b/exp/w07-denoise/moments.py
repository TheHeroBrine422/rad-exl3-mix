#!/usr/bin/env python3
"""moments.py MODEL META OUT --mode noise|sib [--wq DIR] [--layers a-b] [--pools qad] [--tokens N] [--win 2048]
[--gamma 0.05] [--seed 0] -- second moments of each trunk input group's clean input x and a perturbation eta of it,
H = sum x x^T, C = sum x eta^T, N = sum eta eta^T, for the denoised LDLQ target and metric (activation denoising,
arXiv 2610.07522, Algorithm 2; mkdenoise.py turns them into ourq --hin / --wpre dirs). 0.8B / 2B proxies, one card.

  --mode noise   (w07-denoise, ranked.md C1) one forward of [clean, noisy] as a batch of 2: at every group input
                 (the input / post-attention norm outputs, the o_proj / out_proj and down_proj inputs) Gaussian noise
                 xi_c ~ N(0, (gamma max_t |x_c|)^2) is added to the noisy copy and propagated downstream, so eta = noisy -
                 clean at a group = upstream + local noise. The channel maxima come from a clean pass over the same
                 windows (OUT/sigma.pt, every layer, made once). Noise draws are seeded per (window, layer, group): a
                 run split into layer ranges gives the same moments as one run.
  --mode sib     (w07-seqsib, ranked.md C6) one clean forward; every attention / GDN block and every MLP is run a second
                 time on its same clean input with the layer's quantized q/k/v (GDN qkv/z) and gate/up from --wq DIR
                 (ourq --save-wq, <key>.safetensors "weight" (out, in) f16): eta = the o_proj / out_proj / down_proj input
                 with quantized siblings - the clean one (intra-layer sequential inputs). Groups: o / out and down only.

  --mode quant   (w07-denoise, the paper's "parallel-quantized upstream error" proxy) per window one clean forward and one
                 with every trunk linear that has a file in --wq DIR swapped for its quantized weight (a whole uniform-K
                 quant, ourq --save-wq): eta = the quantized model's group input - the clean one (upstream error of all
                 earlier linears, the layer's own siblings included). Level-matched: one collection per K.

Windows: the documents of --pools (pools-v3, $POOLS to override) cut into --win-token windows (a shorter document
is one window if >= 64 tokens), in a shuffled order (--seed), until --tokens. Out: OUT/<ourq group name>.safetensors,
"H", "C", "N" (k x k f32 sums; C not symmetric) + metadata n (tokens), mode, gamma. GPU fp32 sums, flushed into fp64 on
the host every 32 windows. Env DEV (cuda), VRAM_CAP_GB (caps the torch allocator), P2B_BLOCK (common.py)."""
import argparse, json, os, sys, time
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from common import load_model, kernel_paths

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("meta"); ap.add_argument("out")
ap.add_argument("--mode", choices=("noise", "sib", "quant"), required=True)
ap.add_argument("--wq", default=None); ap.add_argument("--layers", default="0-23")
ap.add_argument("--pools", default="qad"); ap.add_argument("--tokens", type=int, default=2_000_000)
ap.add_argument("--win", type=int, default=2048); ap.add_argument("--gamma", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sigma", default=None, help="noise mode: the channel-maxima file (default OUT/sigma.pt); made when absent")
ap.add_argument("--sigma-tokens", type=int, default=None, help="noise mode: windows for the maxima, the first N tokens' worth "
                "of the same order (default --tokens; several runs at different --tokens then share one file)")
a = ap.parse_args()
L = "model.language_model.layers"
DEV = os.environ.get("DEV", "cuda")
if DEV == "cuda" and os.environ.get("VRAM_CAP_GB"):
    torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["VRAM_CAP_GB"]) * 2**30 / torch.cuda.get_device_properties(0).total_memory))
assert a.mode == "noise" or a.wq, f"--mode {a.mode} needs --wq"
lo, hi = map(int, a.layers.split("-"))

# ---- windows
m = json.load(open(f"{a.meta}/kld.json")); toks = np.memmap(f"{a.meta}/tokens.i32", np.int32, "r")
pools = json.load(open(os.environ.get("POOLS", "/workspace/rad-exl3-mix/eval/pools/pools-v3.json")))["pool"]
off = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
want = set(a.pools.split(","))
wins = []
for i, d in enumerate(m["docs"]):
    if pools[i] not in want: continue
    n = d["tokens"]
    if n < a.win:
        if n >= 64: wins.append((int(off[i]), n))
    else: wins += [(int(off[i] + s), a.win) for s in range(0, n - a.win + 1, a.win)]
np.random.default_rng(a.seed).shuffle(wins)
cum = np.cumsum([n for _, n in wins]); allw = wins
wins = allw[:int(np.searchsorted(cum, a.tokens)) + 1]
swins = allw[:int(np.searchsorted(cum, a.sigma_tokens)) + 1] if a.sigma_tokens else wins
print(f"{len(wins)} windows, {sum(n for _, n in wins)} tokens from pools {sorted(want)}", file=sys.stderr, flush=True)

model, dec, _ = load_model(a.model, dev=DEV)
model.requires_grad_(False)
def gname(l, g):
    ly = dec.layers[l]; full = hasattr(ly, "self_attn")
    return [f"{L}.{l}.input", f"{L}.{l}.self_attn.o_proj.input" if full else f"{L}.{l}.linear_attn.out_proj.input",
            f"{L}.{l}.post_attention.input", f"{L}.{l}.mlp.down_proj.input"][g]
GROUPS = (1, 3) if a.mode == "sib" else (0, 1, 2, 3)

# ---- accumulators: GPU fp32 per flush interval, host fp64 totals
ACC, HOST, NT = {}, {}, {}
def acc(name, x, eta):                       # x, eta [T, k] fp32
    if name not in ACC:
        k = x.shape[1]; ACC[name] = {s: torch.zeros(k, k, device=x.device) for s in "HCN"}; NT[name] = 0
    A = ACC[name]
    for r in range(0, x.shape[0], 4096):
        xb, eb = x[r:r + 4096], eta[r:r + 4096]
        A["H"].addmm_(xb.T, xb); A["C"].addmm_(xb.T, eb); A["N"].addmm_(eb.T, eb)
    NT[name] += x.shape[0]
def flush():
    for name, A in ACC.items():
        h = HOST.setdefault(name, {})
        for s, v in A.items():
            h[s] = h[s] + v.double().cpu() if s in h else v.double().cpu(); v.zero_()

class Stop(Exception): pass
hooks = []
cur = {"w": 0}

if a.mode == "noise":
    # ---- channel maxima of the clean inputs (all layers, so every layer-range run injects the same noise)
    sp = a.sigma or f"{a.out}/sigma.pt"
    if os.path.exists(sp): MX = torch.load(sp)
    else:
        MX = {}
        def mx_hook(name, out_side):
            def f(mod, *xs):
                x = (xs[1] if out_side else xs[0][0]).reshape(-1, (xs[1] if out_side else xs[0][0]).shape[-1])
                v = x.float().abs().amax(0)
                MX[name] = torch.maximum(MX[name], v) if name in MX else v
            return f
        hk = []
        for l in range(len(dec.layers)):
            ly = dec.layers[l]; mix = ly.self_attn.o_proj if hasattr(ly, "self_attn") else ly.linear_attn.out_proj
            hk += [ly.input_layernorm.register_forward_hook(mx_hook(gname(l, 0), True)),
                   mix.register_forward_pre_hook(mx_hook(gname(l, 1), False)),
                   ly.post_attention_layernorm.register_forward_hook(mx_hook(gname(l, 2), True)),
                   ly.mlp.down_proj.register_forward_pre_hook(mx_hook(gname(l, 3), False))]
        with torch.no_grad():
            for w0, n in swins:
                dec(input_ids=torch.from_numpy(np.asarray(toks[w0:w0 + n], np.int64))[None].to(DEV), use_cache=False)
        for h in hk: h.remove()
        MX = {k: v.cpu() for k, v in MX.items()}
        os.makedirs(os.path.dirname(os.path.abspath(sp)), exist_ok=True); torch.save(MX, sp)
        print(f"channel maxima over {sum(n for _, n in swins)} tokens: {len(MX)} groups -> {sp}", file=sys.stderr, flush=True)
    SIG = {k: (a.gamma * v).to(DEV) for k, v in MX.items()}

    def inject(name, l, g, x):               # x [2, T, k] (clean, noisy): returns the new noisy copy, records moments
        gen = torch.Generator(device=DEV); gen.manual_seed((a.seed * 1_000_003 + cur["w"]) * 1000 + l * 4 + g)
        xi = torch.randn(x.shape[1], x.shape[2], generator=gen, device=DEV) * SIG[name]
        xn = (x[1].float() + xi).to(x.dtype)
        if lo <= l <= hi:
            xc = x[0].float(); acc(name, xc, xn.float() - xc)
        return torch.stack([x[0], xn])
    def out_hook(name, l, g):                # norm outputs (input / post-attention groups)
        def f(mod, args, out): return inject(name, l, g, out)
        return f
    def pre_hook(name, l, g):                # o_proj / out_proj / down_proj inputs
        def f(mod, args): return (inject(name, l, g, args[0]),) + tuple(args[1:])
        return f
    for l in range(hi + 1):                  # noise is injected in every layer up to hi; moments kept for lo..hi
        ly = dec.layers[l]; mix = ly.self_attn.o_proj if hasattr(ly, "self_attn") else ly.linear_attn.out_proj
        hooks += [ly.input_layernorm.register_forward_hook(out_hook(gname(l, 0), l, 0)),
                  mix.register_forward_pre_hook(pre_hook(gname(l, 1), l, 1)),
                  ly.post_attention_layernorm.register_forward_hook(out_hook(gname(l, 2), l, 2)),
                  ly.mlp.down_proj.register_forward_pre_hook(pre_hook(gname(l, 3), l, 3))]
    BATCH = 2
def wq(key):                                 # a quantized weight (ourq --save-wq), bf16 on the card as evalq.py substitutes it
    with safe_open(f"{a.wq}/{key}.safetensors", "pt", device="cpu") as f: return f.get_tensor("weight")
if a.mode == "quant":
    # ---- every trunk linear with a file swapped in the second forward; group inputs of lo..hi compared
    SW = []
    for l in range(hi + 1):
        for nm, mod in dec.layers[l].named_modules():
            if isinstance(mod, torch.nn.Linear) and os.path.exists(f"{a.wq}/{L}.{l}.{nm}.safetensors"):
                q = wq(f"{L}.{l}.{nm}"); assert tuple(q.shape) == tuple(mod.weight.shape), (l, nm)
                SW.append((mod.weight, q.to(device=mod.weight.device, dtype=mod.weight.dtype)))
    assert SW, f"no quantized weights in {a.wq}"
    print(f"{len(SW)} quantized linears from {a.wq}", file=sys.stderr, flush=True)
    qs = {"q": False, "x": {}}
    def rec(name, l):
        def f(x):
            if not lo <= l <= hi: return
            x = x.reshape(-1, x.shape[-1])
            if not qs["q"]: qs["x"][name] = x.clone()
            else: xc = qs["x"].pop(name).float(); acc(name, xc, x.float() - xc)
        return f
    def qout(name, l):
        r = rec(name, l)
        def f(mod, args, out): r(out)
        return f
    def qpre(name, l):
        r = rec(name, l)
        def f(mod, args): r(args[0])
        return f
    for l in range(lo, hi + 1):
        ly = dec.layers[l]; mix = ly.self_attn.o_proj if hasattr(ly, "self_attn") else ly.linear_attn.out_proj
        hooks += [ly.input_layernorm.register_forward_hook(qout(gname(l, 0), l)),
                  mix.register_forward_pre_hook(qpre(gname(l, 1), l)),
                  ly.post_attention_layernorm.register_forward_hook(qout(gname(l, 2), l)),
                  ly.mlp.down_proj.register_forward_pre_hook(qpre(gname(l, 3), l))]
    BATCH = 1
elif a.mode == "sib":
    # ---- quantized siblings
    state = {"rerun": False, "x": {}}
    def cap_hook(name):                      # the target's input: clean in the main pass, quantized-sibling in the rerun
        def f(mod, args):
            x = args[0].reshape(-1, args[0].shape[-1])
            if not state["rerun"]: state["x"][name] = x
            else:
                xc = state["x"].pop(name).float(); acc(name, xc, x.float() - xc)
        return f
    def rerun_hook(sibs):                    # run the block again with its quantized siblings, on the same inputs
        def f(mod, args, kwargs, out):
            keep = [(p, p.data) for p, _ in sibs]
            for p, q in sibs: p.data = q
            state["rerun"] = True
            try: mod.forward(*args, **kwargs)
            finally:
                state["rerun"] = False
                for p, d in keep: p.data = d
            return out
        return f
    nsib = 0
    for l in range(lo, hi + 1):
        ly = dec.layers[l]; p = f"{L}.{l}"
        if hasattr(ly, "self_attn"):
            blk, tgt, sk = ly.self_attn, ly.self_attn.o_proj, [f"self_attn.{s}_proj" for s in "qkv"]
        else:
            blk, tgt, sk = ly.linear_attn, ly.linear_attn.out_proj, ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"]
        mlp_s = ["mlp.gate_proj", "mlp.up_proj"]
        def sib(names):
            out = []
            for s in names:
                prm = ly.get_submodule(s).weight; q = wq(f"{p}.{s}")
                assert tuple(q.shape) == tuple(prm.shape), (s, q.shape, prm.shape)
                out.append((prm, q.to(device=prm.device, dtype=prm.dtype)))
            return out
        s1, s2 = sib(sk), sib(mlp_s); nsib += len(s1) + len(s2)
        hooks += [tgt.register_forward_pre_hook(cap_hook(gname(l, 1))),
                  blk.register_forward_hook(rerun_hook(s1), with_kwargs=True),
                  ly.mlp.down_proj.register_forward_pre_hook(cap_hook(gname(l, 3))),
                  ly.mlp.register_forward_hook(rerun_hook(s2), with_kwargs=True)]
    print(f"{nsib} quantized siblings from {a.wq}", file=sys.stderr, flush=True)
    BATCH = 1

if hi < len(dec.layers) - 1:                # the forward stops after the last layer kept
    def _stop(mod, args, out): raise Stop()
    hooks.append(dec.layers[hi].register_forward_hook(_stop))
t0 = time.time(); ntok = 0
with torch.no_grad():
    for i, (w0, n) in enumerate(wins):
        cur["w"] = i
        ids = torch.from_numpy(np.asarray(toks[w0:w0 + n], np.int64))[None].to(DEV).repeat(BATCH, 1)
        try: dec(input_ids=ids, use_cache=False)
        except Stop: pass
        if a.mode == "quant":                # the same window through the quantized model
            for p_, q_ in SW: p_.data, q_.data = q_.data, p_.data
            qs["q"] = True
            try: dec(input_ids=ids, use_cache=False)
            except Stop: pass
            finally:
                qs["q"] = False
                for p_, q_ in SW: p_.data, q_.data = q_.data, p_.data
            assert not qs["x"], "group inputs without a quantized counterpart"
        ntok += n
        if (i + 1) % 32 == 0: flush()
        if (i + 1) % 50 == 0 or i + 1 == len(wins):
            print(f"\r{i + 1}/{len(wins)} windows, {ntok} tokens, {ntok / (time.time() - t0):.0f} tok/s", end="", file=sys.stderr, flush=True)
flush(); print(file=sys.stderr)
want_g = {gname(l, g) for l in range(lo, hi + 1) for g in GROUPS}
assert set(HOST) == want_g, f"groups without moments: {sorted(want_g - set(HOST))[:4]}"
os.makedirs(a.out, exist_ok=True)
for name, h in HOST.items():
    assert NT[name] == ntok, (name, NT[name], ntok)
    save_file({s: v.float().contiguous() for s, v in h.items()}, f"{a.out}/{name}.safetensors.part",
              metadata={"n": str(ntok), "mode": a.mode, "gamma": str(a.gamma), "sigma": str(a.sigma), "seed": str(a.seed), "pools": a.pools,
                        "win": str(a.win), "wq": str(a.wq), "model": a.model, "kernels": json.dumps(kernel_paths())})
    os.replace(f"{a.out}/{name}.safetensors.part", f"{a.out}/{name}.safetensors")
if DEV == "cuda": print(f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", file=sys.stderr)
print(f"MOMENTS DONE {len(HOST)} groups, {ntok} tokens, {(time.time() - t0) / 60:.1f} min")
