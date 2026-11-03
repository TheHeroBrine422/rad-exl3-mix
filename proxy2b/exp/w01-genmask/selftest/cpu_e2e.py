#!/usr/bin/env python3
"""cpu_e2e.py WORK ORIG_COLLECT BF16X GENMASK_PATCH CHECK_PATCH -- CPU end-to-end self-test of the two collect.py patches
(w01: genmask.patch = ranked.md A1, collect-check.patch = A8) on a tiny random Qwen3.5 (4 layers, width 64, vocab 512)
and a synthetic golden. Every collect.py copy gets the same CUDA -> CPU text shim (device strings, dispatch, the
torch.cuda calls); nothing else is changed. Checks:
  1. no-op: patched, YAQA_GENMASK unset == original: every tensor bit-exact, metadata equal, halves.json byte-identical
     (hdiag, hdiag-snap, -xw / -dy; file bytes differ run to run: safetensors orders metadata keys at random)
  2. all-ones gen mask == unmasked (to fp32 rounding: CE sum vs (CE * 1).sum)
  3. mixed mask: windows without a scored position are skipped and counted (log), snapshot "seqs" unchanged, result
     differs from unmasked, "genmask" in the metadata
  4. A8: a linear without a hook -> the first save exits naming it; NSEQ past the pool -> exits before any work
Needs torch + transformers (Qwen3_5) + accelerate."""
import json, os, re, shutil, subprocess, sys
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file

W, ORIG, BF16X, PGM, PCK = sys.argv[1:6]
L, NS = 64, 8
shutil.rmtree(W, ignore_errors=True); os.makedirs(W)

# ---- tiny checkpoint in the 27B layout (model.language_model.*, tied head)
from transformers import AutoConfig, Qwen3_5ForCausalLM
tc = dict(model_type="qwen3_5_text", hidden_size=64, intermediate_size=128, num_hidden_layers=4,
          layer_types=["linear_attention"] * 3 + ["full_attention"], full_attention_interval=4, head_dim=32,
          num_attention_heads=2, num_key_value_heads=1, linear_key_head_dim=16, linear_value_head_dim=16,
          linear_num_key_heads=2, linear_num_value_heads=2, linear_conv_kernel_dim=4, vocab_size=512,
          attn_output_gate=True, hidden_act="silu", rms_norm_eps=1e-6, tie_word_embeddings=True, max_position_embeddings=4096,
          rope_parameters={"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                           "mrope_interleaved": True, "mrope_section": [2, 1, 1]})
CK = f"{W}/ckpt"; os.makedirs(CK)
json.dump({"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": tc,
           "tie_word_embeddings": True}, open(f"{CK}/config.json", "w"))
torch.manual_seed(0)
m = Qwen3_5ForCausalLM(AutoConfig.from_pretrained(CK).text_config).to(torch.bfloat16)
sd = {"model.language_model." + k[len("model."):]: v.contiguous() for k, v in m.state_dict().items() if k.startswith("model.")}
save_file(sd, f"{CK}/model.safetensors")
json.dump({"metadata": {}, "weight_map": {k: "model.safetensors" for k in sd}}, open(f"{CK}/model.safetensors.index.json", "w"))
NLIN = sum(1 for n, x in m.model.layers.named_modules() if isinstance(x, torch.nn.Linear) and not n.endswith(("in_proj_a", "in_proj_b")))
del m

# ---- synthetic golden: (tokens, score_from, gen pattern); windows of L = 64 from docs >= 64 tokens
rng = np.random.default_rng(1)
spec = [(200, 0, "all"), (130, 0, "none"), (70, 0, "none"), (40, 0, "all"), (260, 10, "blocks"), (150, 0, "tail"), (64, 0, "all")]
def golden(d, allones):
    os.makedirs(d); docs, gen, toks, r = [], [], [], 0
    for i, (n, sf, pat) in enumerate(spec):
        k = n - 1 - sf; g = np.ones(k, np.uint8)
        if not allones:
            if pat == "none": g[:] = 0
            if pat == "blocks": g = ((np.arange(k) // 20) % 2).astype(np.uint8)
            if pat == "tail": g[:100] = 0
        docs.append({"source": f"s{i}", "tokens": n, "score_from": sf, "row0": r}); r += k; gen.append(g)
        toks.append(rng.integers(0, 512, n).astype(np.int32))
    json.dump({"format": "radiance-kld-top-1", "rows": r, "docs": docs}, open(f"{d}/kld.json", "w"))
    np.concatenate(gen).tofile(f"{d}/gen.u8"); np.concatenate(toks).tofile(f"{d}/tokens.i32")
for tag, ones in (("mixed", False), ("ones", True)):
    rng = np.random.default_rng(1); golden(f"{W}/work-{tag}/challenge/golden/calib", ones)    # same tokens in both
json.dump({"pool": ["qad"] * len(spec)}, open(f"{W}/pools.json", "w"))
nwin = sum(len(range(0, n - L + 1, L)) for n, _, _ in spec)

# ---- collect.py copies: original, patched; the same shim on each
SHIM = [('f"cuda:{card(name)}"', '"cpu"'), ('model = dispatch_model(model, device_map=dmap)', 'model = model'),
        ('dev0 = torch.device("cuda:0")', 'dev0 = torch.device("cpu")')]
HEAD = ("import torch as _t\n_t.cuda.device_count = lambda: 1\n_t.cuda.synchronize = lambda *a: None\n"
        "_t.cuda.memory_allocated = _t.cuda.max_memory_allocated = lambda *a: 0\n"
        "_t.cuda.reset_peak_memory_stats = lambda *a: None\n_t.cuda.empty_cache = lambda: None\n")
def make(name, src, extra=()):
    d = f"{W}/{name}"; os.makedirs(d); s = open(src).read()
    for a, b in list(SHIM) + list(extra):
        assert s.count(a) >= 1, (name, a); s = s.replace(a, b)
    open(f"{d}/collect.py", "w").write(HEAD + s); shutil.copy(BF16X, d); return f"{d}/collect.py"
pd = f"{W}/patched/v3/yaqa"; os.makedirs(pd); shutil.copy(ORIG, pd)
for p in (PGM, PCK): subprocess.run(["git", "apply", p], cwd=f"{W}/patched", check=True)
orig, pat = make("c-orig", ORIG), make("c-pat", f"{pd}/collect.py")
nohook = make("c-nohook", f"{pd}/collect.py", [("for name, m in lin.items(): m.register_forward_hook(fwd_hook(name))",
                                                "for name, m in lin.items(): name != '0.mlp.down_proj' and m.register_forward_hook(fwd_hook(name))")])

def run(tag, script, out, nseq=NS, golden_tag="mixed", **env):
    e = dict(os.environ, RXM_WORK=f"{W}/work-{golden_tag}", RXM_CKPT=CK, YAQA_POOLS=f"{W}/pools.json", YAQA_FP32="1",
             YAQA_DIAG="1", YAQA_DIAG_EXTRA="1", YAQA_SEED="0", YAQA_SNAP="4", YAQA_SAVE_EVERY="4", YAQA_CHUNK="16",
             DIAG_CHUNK="32", OMP_NUM_THREADS="4", **env)
    r = subprocess.run([sys.executable, script, out, str(nseq), str(L)], env=e, capture_output=True, text=True)
    open(f"{W}/{tag}.log", "w").write(r.stdout + r.stderr)
    return r
def load(d):
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".safetensors"):
            with safe_open(f"{d}/{f}", "pt") as h: out[f] = (h.get_tensor("hout_diag"), h.metadata())
    return out
def same_bytes(a, b):   # tensors bit-exact and metadata equal (the safetensors header orders metadata keys at random), json bytes
    fa = sorted(os.listdir(a)); fb = sorted(os.listdir(b))
    if fa != fb: return False
    A, B = load(a), load(b)
    return all(torch.equal(A[f][0], B[f][0]) and A[f][1] == B[f][1] for f in A) and \
        all(open(f"{a}/{f}", "rb").read() == open(f"{b}/{f}", "rb").read() for f in fa if f.endswith(".json"))
res, ok = {}, True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); res[name] = bool(cond); print(f"{'PASS' if cond else 'FAIL'}  {name}  {info}", flush=True)

print(f"tiny model: {NLIN} quantized trunk linears; {nwin} windows of {L}; {NS} sequences, snapshot at 4", flush=True)
r0 = run("orig", orig, f"{W}/o-orig"); r1 = run("pat-off", pat, f"{W}/o-pat")
check("original collector runs", r0.returncode == 0 and "DONE" in r0.stdout, r0.stderr[-300:])
check("patched, GENMASK off: runs", r1.returncode == 0, r1.stderr[-300:])
for sub in ("hdiag", "hdiag-snap4", "hdiag-xw", "hdiag-dy", "hdiag-snap4-xw"):
    check(f"no-op: {sub} bit-exact", same_bytes(f"{W}/o-orig/{sub}", f"{W}/o-pat/{sub}"),
          f"{len(os.listdir(f'{W}/o-orig/{sub}'))} files")
check("no-op: one file per quantized trunk linear", len([f for f in os.listdir(f"{W}/o-pat/hdiag") if f.endswith(".safetensors")]) == NLIN)

r2 = run("gm-ones", pat, f"{W}/o-ones", golden_tag="ones", YAQA_GENMASK="1")
r2u = run("orig-ones", orig, f"{W}/o-orig-ones", golden_tag="ones")
A, B = load(f"{W}/o-ones/hdiag"), load(f"{W}/o-orig-ones/hdiag")
rel = max(float((A[k][0] - B[k][0]).norm() / B[k][0].norm()) for k in B)
check("all-ones mask == unmasked", r2.returncode == 0 and A.keys() == B.keys() and rel < 1e-5, f"max rel L2 {rel:.2e}")

r3 = run("gm-mixed", pat, f"{W}/o-mask", YAQA_GENMASK="1")
skipped = len(re.findall(r"no scored position, skipped", r3.stdout))
# expected skips: windows (collect.py order, first NS) with no gen position
docs = json.load(open(f"{W}/work-mixed/challenge/golden/calib/kld.json"))["docs"]
g = np.fromfile(f"{W}/work-mixed/challenge/golden/calib/gen.u8", np.uint8)
off = np.cumsum([0] + [d["tokens"] for d in docs]); gpos = np.zeros(off[-1], np.uint8)
for i, d in enumerate(docs): gpos[off[i] + d["score_from"]:off[i] + d["tokens"] - 1] = g[d["row0"]:d["row0"] + d["tokens"] - 1 - d["score_from"]]
wins = [int(off[i] + a) for i, d in enumerate(docs) for a in range(0, d["tokens"] - L + 1, L)]
np.random.default_rng(0).shuffle(wins)
exp_skip = sum(int(gpos[w:w + L - 1].sum() == 0) for w in wins[:NS])
exp_cnt = [int(gpos[w:w + L - 1].sum()) for w in wins[:NS] if gpos[w:w + L - 1].sum() > 0]
got_cnt = [int(x) for x in re.findall(r"seq \d+: (\d+) scored positions", r3.stdout)]
check("mixed mask: runs", r3.returncode == 0 and "DONE" in r3.stdout, r3.stderr[-300:])
check("mixed mask: zero-gen windows skipped", skipped == exp_skip and exp_skip > 0, f"{skipped} skipped, expected {exp_skip}")
check("mixed mask: scored positions per window match the golden", got_cnt == exp_cnt, f"{got_cnt} vs {exp_cnt}")
C = load(f"{W}/o-mask/hdiag"); M = load(f"{W}/o-orig/hdiag")
dif = min(float((C[k][0] - M[k][0]).norm() / M[k][0].norm()) for k in M)
check("mixed mask: differs from unmasked on every linear", C.keys() == M.keys() and dif > 1e-3, f"min rel L2 {dif:.2e}")
md = next(iter(C.values()))[1]
check("mixed mask: metadata seqs = NSEQ (skips counted), genmask flag", md.get("seqs") == str(NS) and md.get("genmask") == "1", str(md))
check("mixed mask: snapshot + halves written", os.path.exists(f"{W}/o-mask/hdiag-snap4/halves.json") and
      len(load(f"{W}/o-mask/hdiag-snap4")) == NLIN)
check("unmasked metadata has no genmask key", "genmask" not in next(iter(M.values()))[1])

r4 = run("nohook", nohook, f"{W}/o-nohook")
check("A8: a linear without sums -> exit at the first save, named", r4.returncode != 0 and "0.mlp.down_proj" in r4.stderr
      and not os.path.exists(f"{W}/o-nohook/hdiag-snap4"), r4.stderr.strip().splitlines()[-1][:200])
r5 = run("toomany", pat, f"{W}/o-toomany", nseq=nwin + 1)
check("A8: NSEQ past the window list -> exit before work", r5.returncode != 0 and "the pool has" in r5.stderr
      and "seq 0" not in r5.stdout, r5.stderr.strip().splitlines()[-1][:200])
r6 = run("toomany-orig", orig, f"{W}/o-toomany-orig", nseq=nwin + 1)
check("(control) original with NSEQ past the list: IndexError after the in-range windows", r6.returncode != 0 and "IndexError" in r6.stderr)
json.dump(res, open(f"{W}/result.json", "w"), indent=1)
print("ALL PASS" if ok else "SOME FAILED")
