#!/usr/bin/env python3
"""mkgold27.py MODEL CORPUS.jsonl OUT --scratch DIR [--stages N] -- a bf16 reference golden of a Qwen3.5-family model
(Qwen3.8-27B, Qwen3.5-0.8B / 2B) on one GPU, in staged layer passes: proxy2b/mkgold.py's rows for a radiance
--kld-corpus (xcorp/kldcorpus.py's {prompt, source, score_from} lines), without holding the whole model on a card.

Stage s loads only its decoder layers (bf16, transformers' own module classes) onto the GPU and runs every document
through them, one document at a time, whole, with a fresh per-document cache and positions 0 .. n-1; the hidden state
after its last layer goes to DIR/b<s+1>/<doc>.{f32,bf16} and the stage's input file is deleted once consumed. Stage 0
takes the embedding rows on the CPU; the last stage holds the final norm and the LM head and writes the rows. The
residual stream is bf16 throughout, so a bf16 scratch (--scratch-dtype bf16) is lossless and half the disk of f32.

Output OUT/{kld.json, tokens.i32, gen.u8, top.bin, rows.f32}: radiance's --kld-record --kld-top 512 layout
(format radiance-kld-top-1; top.bin [rows, 512] {i32 id, f32 log p}, rows.f32 [rows, 3] = nll of the true token,
entropy, tail mass summed outside the 512), the arithmetic of proxy2b/common.py (log-softmax in f32 a chunk of rows at
a time, sums in f64). Tokenised as radiance tokenises a corpus line: the raw prompt, control tokens parsed, nothing
prepended. Positions score_from .. n-2 are scored; gen.u8 is all 1 (a corpus line carrying "gen" is refused).

Resumable per (stage, document): a document's progress is the highest boundary file it has (files are written as
.part and renamed, checked by size), its rows are fsync'ed before DIR/done/<doc> marks it, and kld.json is written
last, once every document is marked. --docs LO:HI bounds the documents a run touches (to bound the scratch, or to
split one corpus over two cards: disjoint ranges, same OUT and DIR); --only-stages A:B runs a subset of the stages.
The stage split (DIR/plan.json) is fixed at the first run. VRAM_CAP_GB caps this process's allocator."""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "proxy2b"))
from common import ref_rows, kernel_paths, TOP  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("model"); ap.add_argument("corpus"); ap.add_argument("out")
ap.add_argument("--scratch", required=True, help="boundary hidden states, plan and progress markers")
ap.add_argument("--stages", type=int, default=2)
ap.add_argument("--split", default=None, help="explicit layer boundaries, e.g. 0,22,44,64 (default: by weight bytes)")
ap.add_argument("--scratch-dtype", choices=("f32", "bf16"), default="f32")
ap.add_argument("--only-stages", default=None, help="A:B, run stages A .. B-1 only")
ap.add_argument("--docs", default=None, help="LO:HI, only these documents (corpus order)")
ap.add_argument("--chunk", type=int, default=int(os.environ.get("P2B_CHUNK", 256)), help="head rows at a time")
ap.add_argument("--model-name", default=None, help="kld.json 'model' (default: MODEL as given)")
ap.add_argument("--keep-scratch", action="store_true", help="keep DIR's plan and markers after kld.json")
ap.add_argument("--attn-chunk", type=int, default=0, help="full attention in query blocks of this many rows (0 = "
                "transformers' sdpa over the whole document): for a backend without a flash / memory-efficient sdpa "
                "kernel, where the math path would hold [heads, n, n] scores")
a = ap.parse_args()
dev = os.environ.get("MKG27_DEV", "cuda")      # cpu: a slow dry run of the code paths
GPU = dev == "cuda"
torch.set_grad_enabled(False)


def log(*x): print(time.strftime("%H:%M:%S"), *x, file=sys.stderr, flush=True)


def fnv1a(b, h):
    for x in b: h = ((h ^ x) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def write_atomic(path, data):
    with open(path + ".part", "wb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(path + ".part", path)


# ------------------------------------------------------------------ the corpus
raw = open(a.corpus, "rb").read()
lines = raw.split(b"\n")
if lines and lines[-1] == b"": lines.pop()
h = 1469598103934665603
for l in lines: h = fnv1a(l, h)          # radiance hashes each getline() line, without its newline
corpus_hash = f"{h:016x}"
if os.path.exists(f"{a.out}/kld.json"):
    if json.load(open(f"{a.out}/kld.json")).get("corpus_fnv1a64") == corpus_hash:
        print(f"{a.out} is finished (kld.json, same corpus): nothing to do", file=sys.stderr); raise SystemExit(0)
    raise SystemExit(f"{a.out} already holds a reference of another corpus; record into a new directory")
os.makedirs(a.out, exist_ok=True); os.makedirs(a.scratch, exist_ok=True)
dj = f"{a.scratch}/docs.json"
if os.path.exists(dj) and os.path.exists(f"{a.out}/tokens.i32") and json.load(open(dj))["corpus_fnv1a64"] == corpus_hash:
    D = json.load(open(dj)); docs = D["docs"]
    toks = np.fromfile(f"{a.out}/tokens.i32", np.int32)
else:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    docs, T, r = [], [], 0
    for ln, l in enumerate(lines, 1):
        if not l: continue
        j = json.loads(l)
        if "gen" in j: raise SystemExit(f"{a.corpus} line {ln}: a 'gen' mask is not supported (gen.u8 is all 1)")
        ids = tok(j["prompt"], add_special_tokens=False)["input_ids"]
        sf = int(j.get("score_from", 0)); n = len(ids)
        if sf < 0 or n - 1 - sf < 1:
            raise SystemExit(f"{a.corpus} line {ln}: {n} tokens leave nothing to score from position {sf}")
        docs.append({"source": j.get("source", ""), "tokens": n, "score_from": sf, "row0": r}); r += n - 1 - sf
        T.append(np.asarray(ids, np.int32))
    toks = np.concatenate(T)
    write_atomic(f"{a.out}/tokens.i32", toks.tobytes())
    write_atomic(f"{a.out}/gen.u8", np.ones(r, np.uint8).tobytes())
    write_atomic(dj, json.dumps({"corpus_fnv1a64": corpus_hash, "docs": docs}).encode())
R = docs[-1]["row0"] + docs[-1]["tokens"] - 1 - docs[-1]["score_from"]
toff = np.concatenate([[0], np.cumsum([d["tokens"] for d in docs])])
log(f"corpus {a.corpus}: {len(docs)} documents, {len(toks)} tokens, {R} scored rows")

# ------------------------------------------------------------------ the model and the stage plan
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
import transformers.models.qwen3_5.modeling_qwen3_5 as q35

cfg = AutoConfig.from_pretrained(a.model)
cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
cfg._attn_implementation = "sdpa"           # what from_pretrained picks (mkgold.py's model)
L, H = cfg.num_hidden_layers, cfg.hidden_size
idx = f"{a.model}/model.safetensors.index.json"
wmap = json.load(open(idx))["weight_map"] if os.path.exists(idx) else \
    {k: f for f in os.listdir(a.model) if f.endswith(".safetensors") for k in safe_open(f"{a.model}/{f}", "pt").keys()}
LP = "model.language_model." if any(k.startswith("model.language_model.") for k in wmap) else "model."
head_key = "lm_head.weight" if "lm_head.weight" in wmap and not getattr(cfg, "tie_word_embeddings", False) \
    else f"{LP}embed_tokens.weight"


def nbytes(keys):
    out = 0
    for k in keys:
        with safe_open(f"{a.model}/{wmap[k]}", "pt") as f: out += int(np.prod(f.get_slice(k).get_shape())) * 2
    return out


def layer_keys(i): return [k for k in wmap if k.startswith(f"{LP}layers.{i}.")]


N = a.stages
if a.split:
    split = [int(x) for x in a.split.split(",")]; N = len(split) - 1
else:
    lb = [nbytes(layer_keys(i)) for i in range(L)]; hb = nbytes([head_key])
    cum = np.cumsum(lb); tot = cum[-1] + hb
    split = [0] + [int(np.argmin(np.abs(cum - tot * s / N))) + 1 for s in range(1, N)] + [L]
    for s in range(1, N): split[s] = min(max(split[s], split[s - 1] + 1), L - (N - s))   # >= 1 layer a stage
assert split[0] == 0 and split[-1] == L and all(x < y for x, y in zip(split, split[1:])), \
    f"stage split {split}: every stage needs at least one layer (fewer --stages)"
plan = {"model": os.path.abspath(a.model), "corpus_fnv1a64": corpus_hash, "split": split, "dtype": a.scratch_dtype,
        "out": os.path.abspath(a.out), "docs": len(docs), "attn_chunk": a.attn_chunk}
pf = f"{a.scratch}/plan.json"
if os.path.exists(pf):
    old = json.load(open(pf))
    if old != plan: raise SystemExit(f"{pf} holds another plan {old}; this run is {plan}: use a fresh --scratch")
else:
    write_atomic(pf, json.dumps(plan).encode())
log(f"{N} stages, layer split {split}, scratch {a.scratch_dtype}")
sdt, sel = (np.float32, 4) if a.scratch_dtype == "f32" else (np.int16, 2)
done_dir = f"{a.scratch}/done"; os.makedirs(done_dir, exist_ok=True)
for s in range(1, N): os.makedirs(f"{a.scratch}/b{s}", exist_ok=True)


def bpath(s, d): return f"{a.scratch}/b{s}/{d:06d}.{a.scratch_dtype}"


def state(d):
    """the stage document d needs next (N = its rows are written); cleans stale lower boundaries and torn files"""
    if os.path.exists(f"{done_dir}/{d:06d}"):
        st = N
    else:
        st = 0
        for s in range(N - 1, 0, -1):
            p = bpath(s, d)
            if os.path.exists(p):
                if os.path.getsize(p) == docs[d]["tokens"] * H * sel: st = s; break
                os.remove(p)
    for s in range(1, min(st, N)):
        if os.path.exists(bpath(s, d)): os.remove(bpath(s, d))
    if st == N and N > 1 and os.path.exists(bpath(N - 1, d)): os.remove(bpath(N - 1, d))
    return st


lo, hi = (map(int, a.docs.split(":")) if a.docs else (0, len(docs)))
hi = min(hi, len(docs)); mine = range(lo, hi)
s0, s1 = (map(int, a.only_stages.split(":")) if a.only_stages else (0, N))

cap = os.environ.get("VRAM_CAP_GB")
if cap and GPU:
    total_vram = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, float(cap) * 2**30 / total_vram))


def load_tensors(keys, strip):
    sd, byfile = {}, {}
    for k in keys: byfile.setdefault(wmap[k], []).append(k)
    for f, ks in byfile.items():
        with safe_open(f"{a.model}/{f}", "pt", device="cpu") as fh:
            for k in ks: sd[k[len(strip):]] = fh.get_tensor(k).to(torch.bfloat16)
    return sd


cfg_attn = cfg
if a.attn_chunk:
    import copy
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    def sdpa_qchunk(module, query, key, value, attention_mask, **kw):
        """sdpa a block of queries at a time, each against the keys up to its last row with an explicit causal mask"""
        n, C = query.shape[2], a.attn_chunk
        if n <= C or attention_mask is not None: return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)
        out = []
        for q0 in range(0, n, C):
            q1 = min(n, q0 + C)
            mask = torch.ones(q1 - q0, q1, dtype=torch.bool, device=query.device).tril(q0)[None, None]
            out.append(sdpa_attention_forward(module, query[:, :, q0:q1], key[:, :, :q1], value[:, :, :q1], mask, **kw)[0])
        return torch.cat(out, 1), None

    ALL_ATTENTION_FUNCTIONS.register("sdpa_qchunk", sdpa_qchunk)
    cfg_attn = copy.deepcopy(cfg); cfg_attn._attn_implementation = "sdpa_qchunk"


def load_stage(s):
    t0 = time.time(); mods = []
    for i in range(split[s], split[s + 1]):
        with torch.device("meta"): m = q35.Qwen3_5DecoderLayer(cfg, i)
        if cfg.layer_types[i] == "full_attention": m.self_attn.config = cfg_attn
        m.load_state_dict(load_tensors(layer_keys(i), f"{LP}layers.{i}."), strict=True, assign=True)
        mods.append((i, m.to(dev).eval()))
    norm = head = None
    if s == N - 1:
        with torch.device("meta"): norm = q35.Qwen3_5RMSNorm(H, eps=cfg.rms_norm_eps)
        norm.load_state_dict(load_tensors([f"{LP}norm.weight"], f"{LP}norm."), strict=True, assign=True)
        norm = norm.to(dev).eval()
        head = load_tensors([head_key], head_key[:-len("weight")])["weight"].to(dev)
    for _, m in mods:
        assert not any(t.is_meta for t in list(m.parameters()) + list(m.buffers())), "a tensor was left on meta"
    return mods, norm, head, time.time() - t0


rotary = q35.Qwen3_5TextRotaryEmbedding(config=cfg).to(dev)   # inv_freq made on the CPU, as from_pretrained
embed = None
REC = np.dtype([("id", "<i4"), ("lp", "<f4")])


def run_layers(mods, x):
    """Qwen3_5TextModel.forward over a subset of its layers, for one whole document (batch 1, positions 0 .. n-1)"""
    n = x.shape[1]
    cache = DynamicCache(config=cfg)
    pid = torch.arange(n, device=dev).view(1, 1, -1).expand(4, 1, -1)
    tpid, pid = pid[0], pid[1:]
    mk = {"config": cfg, "inputs_embeds": x, "attention_mask": None, "past_key_values": cache, "position_ids": tpid}
    masks = {"full_attention": q35.create_causal_mask(**mk), "linear_attention": q35.create_recurrent_attention_mask(**mk)}
    pe = rotary(x, pid)
    for i, m in mods:
        x = m(x, position_embeddings=pe, attention_mask=masks[cfg.layer_types[i]], position_ids=tpid,
              past_key_values=cache, use_cache=True)
    return x


fd_top = fd_rows = None
if s1 == N:
    for name, size in (("top.bin", R * TOP * REC.itemsize), ("rows.f32", R * 12)):
        p = f"{a.out}/{name}"
        if not os.path.exists(p) or os.path.getsize(p) != size:
            with open(p, "wb") as f: f.truncate(size)
    fd_top, fd_rows = os.open(f"{a.out}/top.bin", os.O_RDWR), os.open(f"{a.out}/rows.f32", os.O_RDWR)

timing = open(f"{a.scratch}/timing.jsonl", "a")
for s in range(s0, s1):
    todo = [d for d in mine if state(d) == s]
    if not todo:
        log(f"stage {s}: nothing to do"); continue
    blocked = [d for d in mine if state(d) < s]
    if blocked: log(f"stage {s}: {len(blocked)} documents have not reached it yet (first {blocked[0]}), left for later")
    if GPU: torch.cuda.reset_peak_memory_stats()
    mods, norm, head, tl = load_stage(s)
    wgb = torch.cuda.memory_allocated() / 2**30 if GPU else 0.0
    if s == 0 and embed is None:
        embed = load_tensors([f"{LP}embed_tokens.weight"], f"{LP}embed_tokens.")["weight"]
    log(f"stage {s}: layers {split[s]}..{split[s + 1] - 1}{' + norm + head' if head is not None else ''}, "
        f"{wgb:.2f} GiB, loaded in {tl:.0f} s; {len(todo)} documents")
    t0, ntok, nrow, nll_sum = time.time(), 0, 0, 0.0
    for k, d in enumerate(todo):
        D = docs[d]; n, sf = D["tokens"], D["score_from"]
        tk = toks[toff[d]:toff[d + 1]]
        if s == 0:
            x = torch.nn.functional.embedding(torch.from_numpy(tk.astype(np.int64)), embed)[None].to(dev)
        else:
            y = torch.from_numpy(np.fromfile(bpath(s, d), sdt).reshape(1, n, H))
            x = (y if a.scratch_dtype == "f32" else y.view(torch.bfloat16)).to(dev).to(torch.bfloat16)
        x = run_layers(mods, x)
        if s < N - 1:
            y = x[0].float() if a.scratch_dtype == "f32" else x[0].contiguous().view(torch.int16)
            write_atomic(bpath(s + 1, d), y.cpu().numpy().tobytes())
        else:
            hs = norm(x)[0, sf:n - 1]; nr = n - 1 - sf
            nxt = torch.from_numpy(tk[sf + 1:n].astype(np.int64)).to(dev)
            for j in range(0, nr, a.chunk):
                lq = torch.log_softmax(torch.nn.functional.linear(hs[j:j + a.chunk], head).to(torch.float32), dim=1)
                ids, lp, ent, tail = ref_rows(lq)
                nll = -lq.gather(1, nxt[j:j + lq.shape[0], None])[:, 0]
                rec = np.empty((lq.shape[0], TOP), REC)
                rec["id"], rec["lp"] = ids.cpu().numpy(), lp.cpu().numpy()
                r0 = D["row0"] + j
                os.pwrite(fd_top, rec.tobytes(), r0 * TOP * REC.itemsize)
                rw = np.stack([nll.float().cpu().numpy(), ent.float().cpu().numpy(), tail.float().cpu().numpy()], 1)
                os.pwrite(fd_rows, rw.astype(np.float32).tobytes(), r0 * 12)
                nll_sum += float(nll.double().sum())
            os.fsync(fd_top); os.fsync(fd_rows)
            write_atomic(f"{done_dir}/{d:06d}", b""); nrow += nr
        if s > 0: os.remove(bpath(s, d))
        ntok += n; del x
        el = time.time() - t0
        print(f"\rstage {s}: {k + 1}/{len(todo)} docs, {ntok} tokens, {ntok / el:.0f} tok/s"
              + (f", {nrow} rows, ppl {np.exp(nll_sum / max(nrow, 1)):.3f}" if head is not None else ""),
              end="", file=sys.stderr, flush=True)
    if GPU: torch.cuda.synchronize()
    el = time.time() - t0
    print(file=sys.stderr)
    peak = torch.cuda.max_memory_allocated() / 2**30 if GPU else 0.0
    timing.write(json.dumps({"stage": s, "layers": [split[s], split[s + 1]], "head": head is not None, "docs": len(todo),
                             "tokens": ntok, "rows": nrow, "load_s": round(tl, 2), "run_s": round(el, 2),
                             "weights_gib": round(wgb, 3), "peak_gib": round(peak, 3), "dtype": a.scratch_dtype,
                             "gpu": torch.cuda.get_device_name(0) if GPU else "cpu", "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    timing.flush()
    log(f"stage {s}: {ntok} tokens in {el:.0f} s ({ntok / el:.0f} tok/s), peak {peak:.2f} GiB")
    del mods, norm, head
    if GPU: torch.cuda.empty_cache()

# ------------------------------------------------------------------ the manifest, once every document is in
missing = [d for d in range(len(docs)) if not os.path.exists(f"{done_dir}/{d:06d}")]
if missing:
    log(f"{len(missing)} of {len(docs)} documents not finished yet (first {missing[0]}); kld.json not written"); raise SystemExit(0)
for fd in (fd_top, fd_rows):
    if fd is not None: os.close(fd)
rows = np.memmap(f"{a.out}/rows.f32", np.float32, "r", shape=(R, 3))
m = {"format": "radiance-kld-top-1", "model": a.model_name or a.model, "corpus": a.corpus, "corpus_fnv1a64": corpus_hash,
     "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_vocab": int(cfg.vocab_size), "top": TOP, "rows": R,
     "kernels": kernel_paths(), "maker": {"tool": "eval/gold27/mkgold27.py", "split": split, "scratch_dtype": a.scratch_dtype,
                                         "attn_chunk": a.attn_chunk},
     "docs": docs}
write_atomic(f"{a.out}/kld.json", json.dumps(m, indent=1).encode())
nll = rows[:, 0].astype(np.float64)
log(f"DONE {a.out}: {R} rows, mean nll {nll.mean():.4f} (ppl {np.exp(nll.mean()):.3f}), mean entropy "
    f"{rows[:, 1].astype(np.float64).mean():.4f}, mean tail {rows[:, 2].astype(np.float64).mean():.2e}")
if not a.keep_scratch:       # tolerant: a second process on another --docs range may finish at the same time
    import shutil
    for p in ["done"] + [f"b{s}" for s in range(1, N)]: shutil.rmtree(f"{a.scratch}/{p}", ignore_errors=True)
    for p in ("plan.json", "docs.json"):
        if os.path.exists(f"{a.scratch}/{p}"): os.remove(f"{a.scratch}/{p}")
