#!/usr/bin/env python3
"""mkhin27.py MODEL META OUT --scratch DIR [--stages N|auto] -- input Hessians of a Qwen3.5-family model (Qwen3.8-27B,
Qwen3.5-0.8B / 2B) on one GPU in staged layer passes: proxy2b/hcollect.py's sums and output, mkgold27.py's staging.

OUT/<name>.safetensors in the published hessians/calib layout: "hessian_upper" f32, the packed upper triangle (row
major) of sum x x^T over every token of every META document; metadata exactly the published keys k / rows / layout /
linears (rows = tokens summed). 4 files a layer -- <L>.<l>.input (q/k/v, or GDN in_proj_qkv/z/a/b),
<L>.<l>.self_attn.o_proj.input / linear_attn.out_proj.input, <L>.<l>.post_attention.input (gate/up),
<L>.<l>.mlp.down_proj.input -- plus lm_head.input (the final norm's output) in the last stage: 257 files on the 27B.
Provenance (model, META, split, kernels, ...) goes to OUT/collect.json, written last (the completion marker).
META = a golden-style dir (kld.json "docs" with "tokens", + tokens.i32). Documents are prefilled whole, those longer
than --chunk cut into independent windows (hcollect.py's rule: no cache, positions restart at 0).

Stage s holds its decoder layers (bf16, transformers' Qwen3_5DecoderLayer, run as Qwen3_5TextModel.forward runs them
with use_cache=False, as hcollect.py's dec(input_ids)) and their fp32 accumulators on the GPU, and runs every window
through them; the hidden state after the stage goes to DIR/b<s+1>/<window>.bf16 (the residual stream is bf16: lossless).
Stage 0 reads the embedding rows from a memmap of the checkpoint (no 2.5 GB embedding in RAM or VRAM); the last stage
holds the final norm (no LM head: only its input is needed). Stage split: --split, or --stages N balanced by bytes
(weights + accumulators), or --stages auto (default): the fewest stages a greedy fill fits under VRAM_CAP_GB (else
the card) minus --reserve-gb, balanced by bytes when that still fits. 27B: a layer is 0.76 GB of weights + 1.57 GB
of fp32 accumulators (down_proj's 17,408 wide input alone 1.21 GB), so 2.33 GB a layer.

Sums (hcollect.py's arithmetic): fp32 addmm on the GPU a 4096-row block at a time, flushed into fp64 every --flush
documents (counted as hcollect counts them) and at the end of the stage. The fp64 sums live on disk, not in RAM:
DIR/acc/s<s>g<g>/<name>.f64 (packed upper), one generation per checkpoint (old + new, written whole and fsync'ed), then
DIR/progress.json {stage, units, gen} is replaced atomically -- that is the checkpoint. At a stage's end the last
generation + the GPU sums go straight to OUT (fp64 sum, rounded to fp32; .part, renamed), then the next stage is
committed. A killed run resumes from the last checkpoint (bit-identical to an uninterrupted run: the flush points are
fixed by the plan); input boundaries are deleted only once a checkpoint covers them. Rerunning a finished OUT is a
no-op; the plan (DIR/plan.json) is fixed at the first run.
Disk (27B, 1.14M tokens, 6 stages): boundaries 11.7 GB each, two at a time at most; fp64 generations 1.57 GB a layer
(16-17 GB a stage, two during a checkpoint); OUT 49 GB (as the published dir); peak ~110 GB. RAM: max RSS 2.7 GiB
(4070, 27B); the fp64 traffic goes through the page cache (fadvise DONTNEED after each file).

Gates (2026-11-05 ~5:05-6:00 PM CST, RTX 4070, transformers 5.22.1, torch 2.11, fla 0.6.0, causal-conv1d 1.7.0;
raw on the 4070 in /data/yb/w33: gputest*.sh / .log, cputest*, cmphin.py, chkblend.py, pos0.py, half.py):
 1. 0.8B equality vs hcollect.py --head (layers 0-23 + head) on the first 50 docs of d1111's mX (52,690 tokens,
    97 files): --stages 1, 2, 3 all BIT-IDENTICAL (97/97 files, k / rows equal). --flush 7 (other flush points than
    hcollect's): max |dH| / max|H| 5.9e-7 per file (summation order). CPU (torch GDN path, 3 docs in 512-token windows,
    flush 2): 3 stages = 1 stage, and 3 stages killed (SIGKILL in stage 1) + rerun = uninterrupted, bit-identical.
    Header bytes = the published files' (checked on 4 files with rows 10259331). Tried and dropped: x^T x as a bf16
    GEMM with fp32 output (addmm out_dtype): 2.5e-5 .. 8e-3 of max|H|, not exact enough.
 2. 27B: 4070 smoke (VRAM_CAP_GB=7.2 -> 32 stages of 2 layers; 20 docs, 19,835 tokens, stages 0-1): resident
    4.35 GiB a stage, peak 5.06 GiB, max RSS 2.7 GiB; ~90 us per token and layer (forward ~24 us + fp32 x^T x hooks,
    0.79 GFLOP), fp64 checkpoint 3.1 GB in 13 s and stage end 10 s (this SSD, ~240 MB/s).
    gpubox card 0 (lead; META copied to /data/hin27/mX, sha256 = the 4070's; code copied with the repo layout to
    /data/hin27/code: eval/gold27/mkhin27.py, proxy2b/common.py, proxy2b/exp/d1108/blend.py; recopy if these change):
      cd /data/hin27 && nohup flock /root/card0.lock flock -s /root/ram.lock env ROCR_VISIBLE_DEVICES=0 VRAM_CAP_GB=30 \
        TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false nice -n 5 \
        /data/venv-exl3/bin/python /data/hin27/code/eval/gold27/mkhin27.py /data/workspace/models/qwen38-27b-bf16 \
        /data/hin27/mX /data/hin27/hinX --scratch /data/hin27/scr-hinX --stages 6 --flush 400 \
        > /data/hin27/hinX.log 2>&1 < /dev/null &
    (rerun the same line after a kill). Plan: [0, 11, 21, 32, 43, 53, 64], 21.7-24.0 GiB resident a stage (+ ~0.7
    GiB working set at 2,048-token docs) under the 30 GB cap; checkpoints after docs 400 and 800 of 1,148.
    Estimate (not measured on the R9700): forward 15-24 us + hooks 30-65 us (fp32 GEMM at 12-25 TFLOPS) per token and
    layer -> 55-110 min for 1.14M tokens x 64 layers, + ~10 min of loads, fp64 checkpoints and OUT writes (~600 GB of
    disk traffic incl. boundaries, NVMe): ~1-2 h. Stage 0's tok/s in the log gives it: total ~ 6 x 1.14M / tok/s.
    Disk ~110 GB peak (3.2 TB free), RAM ~3 GiB, VRAM <= 25 GiB. fla imports there (0.6.0), causal_conv1d does not
    (torch conv fallback; kernels recorded in collect.json).
 3. blend.py hin accepts the output: 0.8B, /data/h08/calib + gate 1's 50 docs, W 0.5: 97/97 blended; 27B, the
    published layers 0 and 3 (8 files, from colo) + the smoke's: 8/8. Per file: metadata = the published (A's) and
    O / rows_A = (1 - W) A / rows_A + W B / rows_B to <= 5e-8 (fp32 rounding): W is B's share of the per-token mean,
    stored on A's rows scale, as on the 0.8B. Command (after collect.json exists -- blend.py links any file missing
    from B unchanged, so a partial OUT would blend silently):
      /data/venv-exl3/bin/python /data/hin27/code/proxy2b/exp/d1108/blend.py hin /data/challenge/hessians/calib \
        /data/hin27/hinX 0.5 /data/hin27/hi-X0.5
    blend.py reads each file whole in fp64: 5.2 GiB RSS on the 27B down_proj files; /data/yb/w33/blend-stream.patch
    (4070) slices it (2.5 GiB, identical tensors and metadata on the 8 files).
    Sanity, xcorp vs published calib per-token diagonals (27B layers 0 / 3): input, out_proj, o_proj, post_attention
    pearson 0.94-1.0; down_proj low and rising with the sample (20 docs 0.18 / 0.10, 100 docs 0.61 / 0.33; spearman
    0.82 / 0.58), not a first-position effect (skipping positions 0 or 0-3: unchanged). Split-half control (200 docs,
    even vs odd): layer 3 even vs odd 0.15 / 0.58 = even vs published 0.34 / 0.57 (heavy-tailed channels: sampling
    noise at 0.2M tokens); layer 0 even vs odd 0.92 / 0.80, vs published 0.62 / 0.83 (a real xcorp-vs-calib
    difference in a few channels, the point of the blend). No sign of a layout or method difference.
"""
import argparse, hashlib, json, os, resource, shutil, struct, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "proxy2b"))
from common import kernel_paths  # noqa: E402  (P2B_BLOCK handling before transformers is imported)
import torch  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("model"); ap.add_argument("meta"); ap.add_argument("out")
ap.add_argument("--scratch", required=True, help="boundary hidden states, fp64 sums, plan and progress")
ap.add_argument("--stages", default="auto", help="N, or auto: as few as fit VRAM_CAP_GB - --reserve-gb")
ap.add_argument("--split", default=None, help="explicit layer boundaries, e.g. 0,11,22,...,64")
ap.add_argument("--reserve-gb", type=float, default=2.5, help="VRAM left for activations and temporaries (auto plan)")
ap.add_argument("--docs", type=int, default=0, help="the first N documents only (tests)")
ap.add_argument("--chunk", type=int, default=8192, help="windows of at most this many tokens (hcollect.py's --chunk)")
ap.add_argument("--flush", type=int, default=200, help="fp32 -> fp64 flush and checkpoint every this many documents")
ap.add_argument("--only-stages", default=None, help="A:B, run stages A .. B-1 only")
ap.add_argument("--plan-only", action="store_true", help="print the stage plan and its VRAM, then exit")
a = ap.parse_args()
dev = os.environ.get("MKH27_DEV", "cuda")      # cpu: a slow dry run of the code paths (P2B_BLOCK=causal_conv1d,fla)
GPU = dev == "cuda"
torch.set_grad_enabled(False)
GB = 2**30


def log(*x): print(time.strftime("%H:%M:%S"), *x, file=sys.stderr, flush=True)


def write_atomic(path, data):
    with open(path + ".part", "wb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(path + ".part", path)


# ------------------------------------------------------------------ META and its windows
m = json.load(open(f"{a.meta}/kld.json"))
alld = m["docs"]; docs = alld[:a.docs] if a.docs else alld
toff = np.concatenate([[0], np.cumsum([d["tokens"] for d in alld])])
toks = np.memmap(f"{a.meta}/tokens.i32", np.int32, "r")
assert len(toks) >= toff[-1], f"{a.meta}: tokens.i32 has {len(toks)} tokens, kld.json's docs {toff[-1]}"
units = []                     # (doc, first token, tokens): hcollect.py's --chunk windows
for d, D in enumerate(docs):
    for s in range(0, D["tokens"], a.chunk): units.append((d, s, min(a.chunk, D["tokens"] - s)))
ROWS = int(sum(D["tokens"] for D in docs))
# checkpoints after the last window of every --flush-th document (hcollect.py flushes after document i, (i + 1) % flush == 0)
ckpt_after = {j for j, (d, s, n) in enumerate(units) if s + n == docs[d]["tokens"] and (d + 1) % a.flush == 0}
meta_hash = hashlib.sha256(open(f"{a.meta}/kld.json", "rb").read()).hexdigest()[:16]

# ------------------------------------------------------------------ the model and the stage plan
from safetensors import safe_open  # noqa: E402
from transformers import AutoConfig  # noqa: E402
import transformers.models.qwen3_5.modeling_qwen3_5 as q35  # noqa: E402

cfg = AutoConfig.from_pretrained(a.model)
cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
cfg._attn_implementation = "sdpa"           # what from_pretrained picks (hcollect.py's model)
NL, H = cfg.num_hidden_layers, cfg.hidden_size
idx = f"{a.model}/model.safetensors.index.json"
wmap = json.load(open(idx))["weight_map"] if os.path.exists(idx) else \
    {k: f for f in os.listdir(a.model) if f.endswith(".safetensors") for k in safe_open(f"{a.model}/{f}", "pt").keys()}
LP = "model.language_model." if any(k.startswith("model.language_model.") for k in wmap) else "model."
L = LP + "layers"


def shape(k):
    with safe_open(f"{a.model}/{wmap[k]}", "pt") as f: return f.get_slice(k).get_shape()


def layer_keys(i): return [k for k in wmap if k.startswith(f"{L}.{i}.")]


def groups(i):
    """(file name, hooked module path, its input width, linears) for layer i: the published hessians/calib groups"""
    p = f"{L}.{i}"
    if cfg.layer_types[i] == "full_attention":
        g = [(f"{p}.input", "self_attn.q_proj", [f"{p}.self_attn.{x}_proj" for x in "qkv"]),
             (f"{p}.self_attn.o_proj.input", "self_attn.o_proj", [f"{p}.self_attn.o_proj"])]
    else:
        g = [(f"{p}.input", "linear_attn.in_proj_qkv", [f"{p}.linear_attn.in_proj_{x}" for x in ("qkv", "z", "a", "b")]),
             (f"{p}.linear_attn.out_proj.input", "linear_attn.out_proj", [f"{p}.linear_attn.out_proj"])]
    g += [(f"{p}.post_attention.input", "mlp.gate_proj", [f"{p}.mlp.gate_proj", f"{p}.mlp.up_proj"]),
          (f"{p}.mlp.down_proj.input", "mlp.down_proj", [f"{p}.mlp.down_proj"])]
    return [(n, mp, int(shape(f"{p}.{mp}.weight")[1]), lin) for n, mp, lin in g]


G = [groups(i) for i in range(NL)]
HEAD = ("lm_head.input", None, H, ["lm_head"])
wbytes = [sum(int(np.prod(shape(k))) * 2 for k in layer_keys(i)) for i in range(NL)]
abytes = [sum(4 * k * k for _, _, k, _ in G[i]) for i in range(NL)]
cost = [w + c for w, c in zip(wbytes, abytes)]
head_cost = 4 * H * H + 2 * H

cap = os.environ.get("VRAM_CAP_GB")
total_vram = torch.cuda.get_device_properties(0).total_memory if GPU else 0
budget = (float(cap) * GB if cap else total_vram) - a.reserve_gb * GB
def stage_bytes(sp): return [sum(cost[sp[s]:sp[s + 1]]) + (head_cost if s == len(sp) - 2 else 0) for s in range(len(sp) - 1)]


def balanced(N):
    cum = np.cumsum(cost); tot = cum[-1] + head_cost
    sp = [0] + [int(np.argmin(np.abs(cum - tot * s / N))) + 1 for s in range(1, N)] + [NL]
    for s in range(1, N): sp[s] = min(max(sp[s], sp[s - 1] + 1), NL - (N - s))   # >= 1 layer a stage
    return sp


if a.split:
    split = [int(x) for x in a.split.split(",")]
elif a.stages == "auto":   # the fewest stages a greedy fill needs, then balanced by bytes if that still fits
    if budget <= 0: raise SystemExit("--stages auto needs VRAM_CAP_GB or a GPU")
    split, used = [0], 0
    for i in range(NL):
        need = cost[i] + (head_cost if i == NL - 1 else 0)
        if used and used + need > budget: split.append(i); used = 0
        used += need
    split.append(NL)
    bal = balanced(len(split) - 1)
    if max(stage_bytes(bal)) <= budget: split = bal
else:
    split = balanced(int(a.stages))
N = len(split) - 1
assert split[0] == 0 and split[-1] == NL and all(x < y for x, y in zip(split, split[1:])), f"bad stage split {split}"
sgb = [x / GB for x in stage_bytes(split)]
log(f"{len(docs)} documents, {len(units)} windows, {ROWS} tokens; {N} stages {split}, weights + accumulators "
    f"{min(sgb):.2f} .. {max(sgb):.2f} GiB a stage (budget {budget / GB:.2f} GiB)")
if a.plan_only: raise SystemExit(0)
if max(sgb) * GB > budget > 0: log(f"WARNING: a stage needs {max(sgb):.2f} GiB, over the {budget / GB:.2f} GiB budget")

plan = {"model": os.path.abspath(a.model), "meta": os.path.abspath(a.meta), "meta_sha256": meta_hash, "docs": len(docs),
        "rows": ROWS, "chunk": a.chunk, "flush": a.flush, "split": split, "out": os.path.abspath(a.out)}
if os.path.exists(f"{a.out}/collect.json"):
    if json.load(open(f"{a.out}/collect.json")).get("plan") == plan:
        log(f"{a.out} is finished (collect.json, same plan): nothing to do"); raise SystemExit(0)
    raise SystemExit(f"{a.out} holds another collection (collect.json); write into a new directory")
os.makedirs(a.out, exist_ok=True); os.makedirs(f"{a.scratch}/acc", exist_ok=True)
pf = f"{a.scratch}/plan.json"
if os.path.exists(pf):
    old = json.load(open(pf))
    if old != plan: raise SystemExit(f"{pf} holds another plan {old}; this run is {plan}: use a fresh --scratch")
else:
    write_atomic(pf, json.dumps(plan).encode())
prf = f"{a.scratch}/progress.json"
prog = json.load(open(prf)) if os.path.exists(prf) else {"stage": 0, "units": 0, "gen": 0}


def bdir(s): return f"{a.scratch}/b{s}"


def bpath(s, u): return f"{bdir(s)}/{u:07d}.bf16"


def gdir(s, g): return f"{a.scratch}/acc/s{s}g{g}"


def cleanup():
    """drop everything the checkpoint in prog does not need: other generations, boundaries of other stages, consumed
    inputs, torn .part files"""
    s, j, g = prog["stage"], prog["units"], prog["gen"]
    for x in os.listdir(f"{a.scratch}/acc"):
        if x != f"s{s}g{g}": shutil.rmtree(f"{a.scratch}/acc/{x}", ignore_errors=True)
    for x in os.listdir(a.scratch):
        if x.startswith("b") and x[1:].isdigit() and int(x[1:]) not in (s, s + 1): shutil.rmtree(f"{a.scratch}/{x}")
    for t in (s, s + 1):
        if os.path.isdir(bdir(t)):
            for x in os.listdir(bdir(t)):
                if x.endswith(".part") or (t == s and int(x.split(".")[0]) < j): os.remove(f"{bdir(t)}/{x}")
    for x in os.listdir(a.out):
        if x.endswith(".part"): os.remove(f"{a.out}/{x}")


def commit(s, j, g):
    global prog
    prog = {"stage": s, "units": j, "gen": g}; write_atomic(prf, json.dumps(prog).encode())
    cleanup()


cleanup()
if GPU and cap:
    torch.cuda.set_per_process_memory_fraction(min(1.0, float(cap) * GB / total_vram))


# ------------------------------------------------------------------ weights
def load_tensors(keys, strip):
    sd, byfile = {}, {}
    for k in keys: byfile.setdefault(wmap[k], []).append(k)
    for f, ks in byfile.items():
        with safe_open(f"{a.model}/{f}", "pt", device="cpu") as fh:
            for k in ks: sd[k[len(strip):]] = fh.get_tensor(k).to(torch.bfloat16)
    return sd


def load_stage(s):
    t0 = time.time(); mods = []
    for i in range(split[s], split[s + 1]):
        with torch.device("meta"): ly = q35.Qwen3_5DecoderLayer(cfg, i)
        ly.load_state_dict(load_tensors(layer_keys(i), f"{L}.{i}."), strict=True, assign=True)
        mods.append((i, ly.to(dev).eval()))
    norm = None
    if s == N - 1:
        with torch.device("meta"): norm = q35.Qwen3_5RMSNorm(H, eps=cfg.rms_norm_eps)
        norm.load_state_dict(load_tensors([f"{LP}norm.weight"], f"{LP}norm."), strict=True, assign=True)
        norm = norm.to(dev).eval()
    for _, ly in mods:
        assert not any(t.is_meta for t in list(ly.parameters()) + list(ly.buffers())), "a tensor was left on meta"
    return mods, norm, time.time() - t0


def embed_memmap():
    """the embedding matrix as a read-only memmap of the checkpoint file (bf16 as int16): rows read on demand"""
    k = f"{LP}embed_tokens.weight"; fn = f"{a.model}/{wmap[k]}"
    with open(fn, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hd = json.loads(f.read(n))
    e = hd[k]; assert e["dtype"] == "BF16", e
    o0, o1 = e["data_offsets"]
    return np.memmap(fn, np.int16, "r", offset=8 + n + o0, shape=tuple(e["shape"]))


rotary = q35.Qwen3_5TextRotaryEmbedding(config=cfg).to(dev)   # inv_freq made on the CPU, as from_pretrained


def run_layers(mods, x):
    """Qwen3_5TextModel.forward(inputs_embeds=x, use_cache=False) over a subset of its layers (batch 1, positions 0 ..)"""
    n = x.shape[1]
    pid = torch.arange(n, device=dev).view(1, 1, -1).expand(4, 1, -1)
    tpid, pid = pid[0], pid[1:]
    mk = {"config": cfg, "inputs_embeds": x, "attention_mask": None, "past_key_values": None, "position_ids": tpid}
    masks = {"full_attention": q35.create_causal_mask(**mk), "linear_attention": q35.create_recurrent_attention_mask(**mk)}
    pe = rotary(x, pid)
    for i, ly in mods:
        x = ly(x, position_embeddings=pe, attention_mask=masks[cfg.layer_types[i]], position_ids=tpid,
               past_key_values=None, use_cache=False)
    return x


# ------------------------------------------------------------------ accumulators
def accumulate(acc, x):
    x = x.reshape(-1, x.shape[-1])
    for r in range(0, x.shape[0], 4096):   # hcollect.py: fp32 a row block at a time, no whole-input fp32 copy
        xb = x[r:r + 4096].float(); acc.addmm_(xb.T, xb)


def combine(acc, old, path, final=None):
    """path = old (fp64 packed upper, a file, or None) + the upper triangle of acc (fp32 on the GPU), streamed in row
    blocks: fp64 (a checkpoint generation), or with final = (name, k, linears) OUT's safetensors (the published header,
    metadata k / rows / layout / linears; data rounded to fp32). Written .part, fsync'ed, renamed; acc zeroed."""
    k = acc.shape[0]; rb = max(1, (1 << 23) // k); cols = torch.arange(k, device=acc.device)
    fi = open(old, "rb") if old else None
    with open(path + ".part", "wb") as fo:
        if final:
            n = k * (k + 1) // 2
            hd = json.dumps({"__metadata__": {"k": str(k), "rows": str(ROWS), "layout": "upper triangle, row-major, packed",
                                              "linears": ",".join(final[2])},
                             "hessian_upper": {"dtype": "F32", "shape": [n], "data_offsets": [0, 4 * n]}},
                            separators=(",", ":")).encode()
            hd += b" " * (-len(hd) % 8)
            fo.write(struct.pack("<Q", len(hd))); fo.write(hd)
        for r0 in range(0, k, rb):
            r1 = min(k, r0 + rb)
            u = acc[r0:r1][cols[None, :] >= cols[r0:r1, None]].double().cpu().numpy()
            if fi: u += np.fromfile(fi, np.float64, count=u.size)
            fo.write((u.astype(np.float32) if final else u).tobytes())
        fo.flush(); os.fsync(fo.fileno())
        os.posix_fadvise(fo.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    if fi: fi.close()
    os.replace(path + ".part", path)
    acc.zero_()


# ------------------------------------------------------------------ the stages
s0, s1 = (map(int, a.only_stages.split(":")) if a.only_stages else (0, N))
embed = None
timing = open(f"{a.scratch}/timing.jsonl", "a")
for s in range(s0, s1):
    if prog["stage"] > s:
        log(f"stage {s}: done"); continue
    if prog["stage"] < s:
        log(f"stage {s}: stage {prog['stage']} is not finished; stopping"); break
    j0, g = prog["units"], prog["gen"]
    if s > 0:
        bad = [u for u in range(j0, len(units)) if not os.path.exists(bpath(s, u))
               or os.path.getsize(bpath(s, u)) != units[u][2] * H * 2]
        if bad: raise SystemExit(f"stage {s}: {len(bad)} input boundaries missing or torn (first {bad[0]}): scratch damaged")
    if s < N - 1: os.makedirs(bdir(s + 1), exist_ok=True)
    if GPU: torch.cuda.reset_peak_memory_stats()
    mods, norm, tl = load_stage(s)
    gs = [x for i in range(split[s], split[s + 1]) for x in G[i]] + ([HEAD] if norm is not None else [])
    acc = {name: torch.zeros(k, k, device=dev) for name, _, k, _ in gs}
    hooks = []
    for i, ly in mods:
        for name, mp, _, _ in G[i]:
            hooks.append(ly.get_submodule(mp).register_forward_pre_hook(
                lambda mod, args, name=name: accumulate(acc[name], args[0])))
    if s == 0 and embed is None: embed = embed_memmap()
    wgb = torch.cuda.memory_allocated() / GB if GPU else 0.0
    log(f"stage {s}: layers {split[s]}..{split[s + 1] - 1}{' + norm (lm_head.input)' if norm is not None else ''}, "
        f"weights + accumulators {wgb:.2f} GiB, loaded in {tl:.0f} s; windows {j0}..{len(units) - 1}")
    t0, ntok, tflush = time.time(), 0, 0.0
    for j in range(j0, len(units)):
        d, st, n = units[j]
        if s == 0:
            e = embed[np.asarray(toks[toff[d] + st:toff[d] + st + n], np.int64)]
            x = torch.from_numpy(e).view(torch.bfloat16)[None].to(dev)
        else:
            x = torch.from_numpy(np.fromfile(bpath(s, j), np.int16).reshape(1, n, H)).view(torch.bfloat16).to(dev)
        x = run_layers(mods, x)
        if norm is not None:
            accumulate(acc[HEAD[0]], norm(x))
        else:
            write_atomic(bpath(s + 1, j), x[0].contiguous().view(torch.int16).cpu().numpy().tobytes())
        del x; ntok += n
        if j in ckpt_after and j < len(units) - 1:     # the stage's end goes straight to OUT (below)
            tf = time.time(); os.makedirs(gdir(s, g + 1), exist_ok=True)
            for name in acc:
                old = f"{gdir(s, g)}/{name}.f64"
                combine(acc[name], old if os.path.exists(old) else None, f"{gdir(s, g + 1)}/{name}.f64")
            g += 1; commit(s, j + 1, g); tflush += time.time() - tf
        if (j + 1) % 50 == 0 or j == len(units) - 1:
            el = time.time() - t0
            print(f"\rstage {s}: {j + 1}/{len(units)} windows, {ntok} tokens, {ntok / el:.0f} tok/s, flushes {tflush:.0f} s",
                  end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    if GPU: torch.cuda.synchronize()
    el = time.time() - t0
    for h in hooks: h.remove()
    del mods, norm
    tw = time.time()
    for name, _, k, lin in gs:      # a kill in here resumes from the last checkpoint (its inputs are still there)
        old = f"{gdir(s, g)}/{name}.f64"
        combine(acc[name], old if os.path.exists(old) else None, f"{a.out}/{name}.safetensors", (name, k, lin))
    tw = time.time() - tw
    del acc
    if GPU: torch.cuda.empty_cache()
    commit(s + 1, 0, 0)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    peak = torch.cuda.max_memory_allocated() / GB if GPU else 0.0
    timing.write(json.dumps({"stage": s, "layers": [split[s], split[s + 1]], "head": s == N - 1, "windows": len(units) - j0,
                             "tokens": ntok, "load_s": round(tl, 2), "run_s": round(el, 2), "flush_s": round(tflush, 2),
                             "write_s": round(tw, 2), "resident_gib": round(wgb, 3), "peak_gib": round(peak, 3), "maxrss_gib": round(rss, 2),
                             "gpu": torch.cuda.get_device_name(0) if GPU else "cpu",
                             "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    timing.flush()
    log(f"stage {s}: {ntok} tokens in {el:.0f} s ({ntok / max(el, 1e-9):.0f} tok/s; flushes {tflush:.0f} s), "
        f"OUT files {tw:.0f} s, peak {peak:.2f} GiB, max RSS {rss:.2f} GiB")

# ------------------------------------------------------------------ the manifest, once every stage is in
if prog["stage"] < N:
    log(f"stage {prog['stage']} of {N} next; collect.json not written (rerun the same command)"); raise SystemExit(0)
names = [x[0] for i in range(NL) for x in G[i]] + [HEAD[0]]
miss = [x for x in names if not os.path.exists(f"{a.out}/{x}.safetensors")]
if miss: raise SystemExit(f"{len(miss)} OUT files missing (first {miss[0]}) although every stage is done")
write_atomic(f"{a.out}/collect.json", json.dumps(
    {"tool": "eval/gold27/mkhin27.py", "plan": plan, "files": len(names), "windows": len(units),
     "kernels": kernel_paths(), "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1).encode())
shutil.rmtree(f"{a.scratch}/acc", ignore_errors=True)
log(f"MKHIN27 DONE {len(names)} Hessians, {ROWS} rows -> {a.out}")
