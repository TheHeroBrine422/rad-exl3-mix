#!/usr/bin/env python3
"""sibmom27.py MODEL META OUT (--wq DIR | --level DIR) [--layers a-b] [--tokens N] [--wins moments|long] ... --
moments.py --mode sib (proxy2b/exp/w07-denoise, #87) for a model that does not fit beside its activations: the 27B on
one 32 GB card with 30 GB of host RAM. Same statistics, file names and metadata; streamed one decoder layer at a time
(ModelOpt's layerwise calibration loop, docs/research/modelopt.md):

  for layer l = 0 .. hi: load l's bf16 weights (and its quantized siblings) onto the card; for every window, its clean
  input (layer 0: the embedding; else the boundary file OUT/work/b<l>.bf16) -> the layer -> its output appended to
  b<l+1>.bf16; meanwhile the attention / GDN block and the MLP run a second time on the same clean inputs with the
  quantized q/k/v (GDN qkv/z) and gate/up, and the o_proj / out_proj and down_proj inputs give
  H = sum x x^T, C = sum x eta^T, N = sum eta eta^T (eta = quantized-sibling input - clean input).

The residual stream between layers is bf16 in the full model too, so the boundary files hold exactly what moments.py's
forward passes from layer to layer, and each layer is called with the kwargs the model itself makes (captured from
Qwen3_5TextModel.forward with recorder modules in place of the layers). GPU f32 sums per 32 windows, flushed into f64
(on the card: the 27B's down group is 3 x 17408^2; DEV=cpu keeps everything on the host), as moments.py.
Out: OUT/<ourq group name>.safetensors "H", "C", "N" (k x k f32 sums) + moments.py's metadata (n, mode sib, gamma,
sigma, seed, pools, win, wq, model, kernels) + collector / level / codebook / wins / attn. Resumable per layer:
OUT/.done/L<l> marks layer l finished (its moments written, b<l+1> complete); b<l> is deleted after layer l
(--keep-boundaries keeps them); OUT/STOP ends the run before the next layer (exit 3). --snap-layers: those layers'
moments also at about half the windows (the nearest flush boundary, so the full sums are unchanged) in OUT/half/, for
a dose check of F.

Siblings: --wq DIR (ourq --save-wq files, <key>.safetensors "weight" (out, in) f16 -> bf16, as moments.py) or --level
DIR (an ourq export, decoded by deq.py: i8 / mul1, any rate in 1/8 steps; f32 -> f16 -> bf16, so --level = --wq on
deq.py wq's files).
Windows: --wins moments = moments.py's list (the --pools docs cut into --win windows, a shorter doc >= 64 tokens is one
window, shuffled by --seed, until --tokens); --wins long = collect.py's (whole --win windows only, the same shuffle).
Env DEV (cuda), VRAM_CAP_GB (as moments.py), P2B_BLOCK (as proxy2b/common.py), SIB_ATTN (attention implementation, default sdpa), POOLS.
v3/yaqa/asym27 copy = v3/sib27/sibmom27.py (sibmom27-lowram.patch already merged there) + sibmom27-w78.patch: no
expandable_segments on ROCm, row-block moment saves, open checkpoint shards dropped after each load (w78, 11-07; the
27B run faulted / ran out of host RAM without them; same moments bit for bit). Merge the patch, then drop the copy."""
import argparse, functools, hashlib, json, os, resource, shutil, sys, time
for _b in filter(None, os.environ.get("P2B_BLOCK", "").split(",")): sys.modules[_b] = None
import numpy as np, torch
from safetensors import safe_open
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("meta"); ap.add_argument("out")
ap.add_argument("--wq", default=None); ap.add_argument("--level", default=None)
ap.add_argument("--codebook", default=None, help="--level: expected codebook (checked against its stamp.json)")
ap.add_argument("--layers", default=None, help="moment layers a-b (default all); layers below a are run forward only")
ap.add_argument("--pools", default="qad"); ap.add_argument("--tokens", type=int, default=1_000_000)
ap.add_argument("--win", type=int, default=2048); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--wins", choices=("moments", "long"), default="moments")
ap.add_argument("--work", default=None, help="boundary files (default OUT/work)")
ap.add_argument("--keep-boundaries", action="store_true")
ap.add_argument("--snap-layers", default="", help="comma list: also write these layers' moments at ~half the windows")
ap.add_argument("--no-disk-check", action="store_true")
a = ap.parse_args()
assert (a.wq is None) != (a.level is None), "exactly one of --wq / --level"
L = "model.language_model.layers"
DEV = torch.device(os.environ.get("DEV", "cuda"))
# w78 (11-07): with the caching allocator's expandable segments, torch 2.15+rocm7.14 on the R9700 (gfx1201) faults in
# random stock kernels (HSA illegal instruction / aperture violation in GEMM, reduce, CK conv) within 30 s - 2 min
# of the full run; gone without them, and hidden by AMD_SERIALIZE_KERNEL=3. Refuse rather than die mid-layer.
if DEV.type == "cuda" and torch.version.hip and any("expandable_segments:true" in os.environ.get(v, "").lower().replace(" ", "")
        for v in ("PYTORCH_HIP_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")):
    sys.exit("sibmom27: expandable_segments is set in the allocator config: it faults on ROCm (w78); unset it")
if DEV.type == "cuda" and os.environ.get("VRAM_CAP_GB"):   # as moments.py (the 4070 shares its card with the desktop)
    torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["VRAM_CAP_GB"]) * 2**30 / torch.cuda.get_device_properties(0).total_memory))
WORK = a.work or f"{a.out}/work"
os.makedirs(a.out, exist_ok=True); os.makedirs(WORK, exist_ok=True); os.makedirs(f"{a.out}/.done", exist_ok=True)
SNAP = {int(x) for x in a.snap_layers.split(",") if x}


def kernel_paths():   # proxy2b/common.py's
    import importlib, importlib.metadata as md
    out = {}
    for mod, dist in (("causal_conv1d", "causal-conv1d"), ("fla", "flash-linear-attention")):
        try: importlib.import_module(mod); out[mod] = md.version(dist)
        except Exception: out[mod] = None
    out["torch"] = torch.__version__
    return out


# ---- windows (moments.py's list, or collect.py's long windows)
m = json.load(open(f"{a.meta}/kld.json")); toks = np.memmap(f"{a.meta}/tokens.i32", np.int32, "r")
pf = os.environ.get("POOLS", "/workspace/rad-exl3-mix/eval/pools/pools-v3.json")
pools = json.load(open(pf))["pool"]
assert len(pools) == len(m["docs"]), (len(pools), len(m["docs"]))
off = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
want = set(a.pools.split(","))
wins = []
for i, d in enumerate(m["docs"]):
    if pools[i] not in want: continue
    n = d["tokens"]
    if n < a.win:
        if n >= 64 and a.wins == "moments": wins.append((int(off[i]), n))
    else: wins += [(int(off[i] + s), a.win) for s in range(0, n - a.win + 1, a.win)]
np.random.default_rng(a.seed).shuffle(wins)
cum = np.cumsum([n for _, n in wins])
wins = wins[:int(np.searchsorted(cum, a.tokens)) + 1]
NTOK = sum(n for _, n in wins)
wkey = hashlib.sha256(json.dumps([a.meta, pf, a.pools, a.win, a.seed, a.wins, wins]).encode()).hexdigest()[:16]
wf = f"{a.out}/windows.json"
if os.path.exists(wf):
    if json.load(open(wf))["key"] != wkey: sys.exit(f"{wf}: another window list (args changed): new OUT, or remove it")
else: json.dump({"key": wkey, "n": len(wins), "tokens": NTOK, "wins": wins}, open(wf, "w"))
print(f"{len(wins)} windows, {NTOK} tokens from pools {sorted(want)} ({a.wins})", file=sys.stderr, flush=True)

# ---- the model, built empty: embedding, rotary and final norm resident, one decoder layer at a time
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
cfg = AutoConfig.from_pretrained(a.model); tcfg = getattr(cfg, "text_config", cfg)
tcfg._attn_implementation = os.environ.get("SIB_ATTN", "sdpa")
with init_empty_weights(): tm = Qwen3_5TextModel(tcfg)
tm.requires_grad_(False); tm.eval()
NL, HID = tcfg.num_hidden_layers, tcfg.hidden_size
lo, hi = map(int, a.layers.split("-")) if a.layers else (0, NL - 1)
idx = json.load(open(f"{a.model}/model.safetensors.index.json"))["weight_map"]
_open = {}
def ck(key):
    f = idx[key]
    if f not in _open: _open[f] = safe_open(f"{a.model}/{f}", "pt", device="cpu")
    return _open[f].get_tensor(key)
def load(prefix, mod):       # module prefix in the text model -> checkpoint "model.language_model.<prefix>"
    for pn, _ in list(mod.named_parameters()):
        set_module_tensor_to_device(tm, f"{prefix}.{pn}", DEV, value=ck(f"model.language_model.{prefix}.{pn}"), dtype=torch.bfloat16)
    _open.clear()            # an open shard keeps what it read in host RAM (the 2.5 GB embedding through layer 0: w78)
def unload(prefix, mod):
    for pn, _ in list(mod.named_parameters()): set_module_tensor_to_device(tm, f"{prefix}.{pn}", "meta")
load("embed_tokens", tm.embed_tokens); load("norm", tm.norm); tm.rotary_emb.to(DEV)
ATTN = tm.config._attn_implementation
print(f"model {a.model}: {NL} layers, hidden {HID}, attention {ATTN}, kernels {kernel_paths()}", file=sys.stderr, flush=True)

class Rec(torch.nn.Module):  # stands in for a decoder layer: records the kwargs the model passes it
    def __init__(self, i, rec): super().__init__(); self.i, self.rec = i, rec
    def forward(self, hidden_states, **kw): self.rec[self.i] = kw; return hidden_states
@functools.lru_cache(maxsize=4)
def layer_kwargs(n):
    rec, real = {}, tm.layers
    tm.layers = torch.nn.ModuleList([Rec(i, rec) for i in range(NL)])
    try:
        with torch.no_grad(): tm(inputs_embeds=torch.zeros(1, n, HID, dtype=torch.bfloat16, device=DEV), use_cache=False)
    finally: tm.layers = real
    assert len(rec) == NL
    return rec

# ---- siblings
if a.level:
    import deq
    LV = deq.Level(a.level, codebook=a.codebook, dev=DEV)
    def wq(key): return LV.get(key).half()   # f16 like --save-wq's files (then bf16 as moments.py): the same double rounding
    WQDESC = f"decoded:{a.level}"
else:
    LV = None
    def wq(key):
        with safe_open(f"{a.wq}/{key}.safetensors", "pt", device="cpu") as f: return f.get_tensor("weight")
    WQDESC = a.wq
def sib_keys(l):
    p = f"{L}.{l}"
    att = [f"self_attn.{s}_proj" for s in "qkv"] if tcfg.layer_types[l] == "full_attention" else ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"]
    return att, ["mlp.gate_proj", "mlp.up_proj"]
def gname(l, g):
    full = tcfg.layer_types[l] == "full_attention"
    return {1: f"{L}.{l}.self_attn.o_proj.input" if full else f"{L}.{l}.linear_attn.out_proj.input", 3: f"{L}.{l}.mlp.down_proj.input"}[g]
for l in range(lo, hi + 1):  # every sibling exists before any work
    for s in sum(sib_keys(l), []):
        k = f"{L}.{l}.{s}"
        ok = f"{k}.trellis" in LV.ix if LV else os.path.exists(f"{a.wq}/{k}.safetensors")
        if not ok: sys.exit(f"no quantized sibling {k} in {a.level or a.wq}")

# ---- disk budget: two boundaries + the moment files
kd = {1: (tcfg.num_attention_heads * tcfg.head_dim), 3: tcfg.intermediate_size}
kgdn = tcfg.linear_num_value_heads * tcfg.linear_value_head_dim
mom_bytes = sum(3 * 4 * (kd[1] if tcfg.layer_types[l] == "full_attention" else kgdn) ** 2 + 3 * 4 * kd[3] ** 2 for l in range(lo, hi + 1))
bnd_bytes = 2 * NTOK * HID * 2
free = shutil.disk_usage(a.out).free
print(f"disk: boundaries 2 x {NTOK * HID * 2 / 2**30:.1f} GiB, moments {mom_bytes / 2**30:.1f} GiB; free {free / 2**30:.0f} GiB",
      file=sys.stderr, flush=True)
if not a.no_disk_check and free < 1.05 * (bnd_bytes + mom_bytes): sys.exit("not enough disk (--no-disk-check to override)")

# ---- accumulators (moments.py's): fp32 per flush interval, fp64 totals (on DEV)
ACC, TOT, NT = {}, {}, {}
def acc(name, x, eta):
    if name not in ACC:
        k = x.shape[1]; ACC[name] = {s: torch.zeros(k, k, device=x.device) for s in "HCN"}; NT[name] = 0
    A = ACC[name]
    for r in range(0, x.shape[0], 4096):
        xb, eb = x[r:r + 4096], eta[r:r + 4096]
        A["H"].addmm_(xb.T, xb); A["C"].addmm_(xb.T, eb); A["N"].addmm_(eb.T, eb)
    NT[name] += x.shape[0]
def flush():
    for name, A in ACC.items():
        h = TOT.setdefault(name, {})
        for s, v in A.items():
            if s in h: h[s].add_(v)                      # f32 into f64 in place: = h + v.double(), no temporary
            else: h[s] = v.double()
            v.zero_()
def save_lean(path, tot, metadata, chunk_bytes=64 << 20):
    """safetensors file of tot's "H", "C", "N" as f32, written in row blocks of ~chunk_bytes, each tensor synced and
    dropped from the page cache: host RAM holds one block (64 MB), not a k x k f32 (1.2 GB at 17408) plus ~3.6 GB of
    dirty pages per down group (w78, 11-07: that pushed iceman's MemAvailable under the 3 GB memory guard at the end
    of layer 0), nor save_file's three tensors plus their serialized copies (~7 GB). Same bytes, tensors and metadata
    (safe_open reads it as save_file's; header padded with spaces to 8 bytes as the format allows)."""
    hdr, off = {}, 0
    for s in "HCN":
        k = tot[s].shape; nb = 4 * k[0] * k[1]
        hdr[s] = {"dtype": "F32", "shape": list(k), "data_offsets": [off, off + nb]}; off += nb
    hdr["__metadata__"] = metadata
    hb = json.dumps(hdr, separators=(",", ":")).encode(); hb += b" " * (-len(hb) % 8)
    with open(path, "wb") as fh:
        fh.write(len(hb).to_bytes(8, "little")); fh.write(hb)
        for s in "HCN":
            T = tot[s]; step = max(1, chunk_bytes // (4 * T.shape[1]))
            for r in range(0, T.shape[0], step): T[r:r + step].float().cpu().contiguous().numpy().tofile(fh)
            fh.flush(); os.fdatasync(fh.fileno()); os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    assert os.path.getsize(path) == 8 + len(hb) + off, path


def save(name, d, ntok):
    os.makedirs(d, exist_ok=True)
    save_lean(f"{d}/{name}.safetensors.part", TOT[name], metadata={
        "n": str(ntok), "mode": "sib", "gamma": "0.05", "sigma": "None", "seed": str(a.seed), "pools": a.pools,
        "win": str(a.win), "wq": WQDESC, "model": a.model, "kernels": json.dumps(kernel_paths()),
        "collector": "v3/sib27/sibmom27.py", "wins": a.wins, "windows": str(len(wins)), "attn": ATTN,
        "level": str(a.level), "codebook": str(LV.codebook if LV else None)})
    os.replace(f"{d}/{name}.safetensors.part", f"{d}/{name}.safetensors")

state = {"rerun": False, "x": {}}
def cap_hook(name):          # the target's input: clean in the main pass, quantized-sibling in the rerun
    def f(mod, args):
        x = args[0].reshape(-1, args[0].shape[-1])
        if not state["rerun"]: state["x"][name] = x
        else:
            xc = state["x"].pop(name).float(); acc(name, xc, x.float() - xc)
    return f
def rerun_hook(sibs):        # run the block again with its quantized siblings, on the same inputs
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

bpath = lambda l: f"{WORK}/b{l:02d}.bf16"
done = lambda l: os.path.exists(f"{a.out}/.done/L{l:02d}")
have_mom = lambda l: all(os.path.exists(f"{a.out}/{gname(l, g)}.safetensors") for g in (1, 3))
FLUSH = 32
snap_at = max(FLUSH, int(round(len(wins) / 2 / FLUSH)) * FLUSH)
t_all = time.time(); nlay = 0
for l in range(hi + 1):
    if done(l) and (l < lo or have_mom(l)): continue
    if os.path.exists(f"{a.out}/STOP"): print(f"STOP file before layer {l}: rerun to resume", file=sys.stderr); sys.exit(3)
    if l > 0 and not os.path.exists(bpath(l)):
        sys.exit(f"layer {l}: input boundary {bpath(l)} missing (deleted after a later layer?): restart in a new OUT")
    if l > 0: assert os.path.getsize(bpath(l)) == NTOK * HID * 2, (bpath(l), os.path.getsize(bpath(l)))
    t0 = time.time(); ly = tm.layers[l]; load(f"layers.{l}", ly)
    mom = lo <= l <= hi; hooks = []
    if mom:
        p = f"{L}.{l}"; sk, mk = sib_keys(l)
        blk, tgt = (ly.self_attn, ly.self_attn.o_proj) if tcfg.layer_types[l] == "full_attention" else (ly.linear_attn, ly.linear_attn.out_proj)
        def sib(names):
            out = []
            for s in names:
                prm = ly.get_submodule(s).weight; q = wq(f"{p}.{s}")
                assert tuple(q.shape) == tuple(prm.shape), (s, q.shape, prm.shape)
                out.append((prm, q.to(device=prm.device, dtype=prm.dtype)))
            return out
        s1, s2 = sib(sk), sib(mk)
        hooks = [tgt.register_forward_pre_hook(cap_hook(gname(l, 1))), blk.register_forward_hook(rerun_hook(s1), with_kwargs=True),
                 ly.mlp.down_proj.register_forward_pre_hook(cap_hook(gname(l, 3))), ly.mlp.register_forward_hook(rerun_hook(s2), with_kwargs=True)]
    src = open(bpath(l), "rb") if l > 0 else None
    wout = l < NL - 1
    dst = open(bpath(l + 1) + ".part", "wb") if wout else None
    ntok = 0
    with torch.no_grad():
        for i, (w0, n) in enumerate(wins):
            if src is None: x = tm.embed_tokens(torch.from_numpy(np.asarray(toks[w0:w0 + n], np.int64))[None].to(DEV))
            else:
                buf = np.fromfile(src, np.int16, n * HID); assert buf.size == n * HID
                x = torch.from_numpy(buf).view(torch.bfloat16).view(1, n, HID).to(DEV)
            y = ly(x, **layer_kwargs(n)[l])
            if dst: dst.write(y.contiguous().view(torch.int16).cpu().numpy().tobytes())
            ntok += n
            if mom and (i + 1) % FLUSH == 0:
                flush()
                if l in SNAP and i + 1 == snap_at:
                    for g in (1, 3): save(gname(l, g), f"{a.out}/half", ntok)
            if (i + 1) % 16 == 0:           # bound the page cache this streaming leaves behind (30 GB host)
                if dst: dst.flush(); os.fdatasync(dst.fileno()); os.posix_fadvise(dst.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                if src: os.posix_fadvise(src.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    assert not state["x"], "target inputs without a quantized counterpart"
    for h in hooks: h.remove()
    if dst: dst.flush(); os.fdatasync(dst.fileno()); dst.close(); os.replace(bpath(l + 1) + ".part", bpath(l + 1))
    if src: src.close()
    if mom:
        flush()
        want_g = {gname(l, g) for g in (1, 3)}
        assert set(TOT) == want_g, (sorted(TOT), sorted(want_g))
        for name in TOT:
            assert NT[name] == ntok == NTOK, (name, NT[name], ntok, NTOK)
            save(name, a.out, ntok)
        ACC.clear(); TOT.clear(); NT.clear()
        del s1, s2
    unload(f"layers.{l}", ly); _open.clear()
    if DEV.type == "cuda": torch.cuda.empty_cache()
    open(f"{a.out}/.done/L{l:02d}", "w").write(f"{ntok}\n")
    if l > 0 and not a.keep_boundaries: os.remove(bpath(l))
    nlay += 1; dt = time.time() - t0
    rss = next((int(x.split()[1]) >> 10 for x in open("/proc/self/status") if x.startswith("RssAnon")), -1)
    hwm = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >> 10
    pk = f", peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if DEV.type == "cuda" else ""
    print(f"layer {l}{'' if mom else ' (forward only)'}: {dt:.0f}s, {ntok / dt:.0f} tok/s{pk}, anon rss {rss} MiB (peak {hwm} MiB), "
          f"eta {(hi - l) * dt / 60:.0f} min", file=sys.stderr, flush=True)
    if DEV.type == "cuda": torch.cuda.reset_peak_memory_stats()
ng = sum(2 for l in range(lo, hi + 1) if have_mom(l))
print(f"MOMENTS DONE {ng} groups, {NTOK} tokens, {(time.time() - t_all) / 60:.1f} min ({nlay} layers this run)")
