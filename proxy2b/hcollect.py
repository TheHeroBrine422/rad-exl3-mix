#!/usr/bin/env python3
"""hcollect.py MODEL META OUT [--layers a-b] [--docs N] [--chunk T] -- input Hessians of the 2B in the challenge's
layout (OUT/<name>.safetensors: "hessian_upper" f32 packed upper triangle row-major, metadata k / rows / layout /
linears), summed over every token of every META document (META = the calib golden's kld.json + tokens.i32; the
published 27B Hessians are the same sum over all of calib). Names as ourq.py reads them: <L>.<l>.input (q/k/v or GDN
qkv/z), <L>.<l>.self_attn.o_proj.input / linear_attn.out_proj.input, <L>.<l>.post_attention.input (gate/up),
<L>.<l>.mlp.down_proj.input, lm_head.input (with --head). Documents longer than --chunk (default 8192) are prefilled
in chunks through the model's cache. Sums: fp32 on the GPU, flushed into fp64 on the host every --flush documents."""
import argparse, json, os, sys, time
import numpy as np, torch
from safetensors.torch import save_file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model, kernel_paths
ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("meta"); ap.add_argument("out")
ap.add_argument("--layers", default="0-23"); ap.add_argument("--head", action="store_true")
ap.add_argument("--docs", type=int, default=0); ap.add_argument("--chunk", type=int, default=8192)
ap.add_argument("--flush", type=int, default=200)
ap.add_argument("--maxtok", type=int, default=1 << 30, help="skip documents longer than this (recorded in the metadata)")
ap.add_argument("--only", default="", help="comma-separated document indices (tests)")
a = ap.parse_args()
L = "model.language_model.layers"
model, dec, head = load_model(a.model)
lo, hi = map(int, a.layers.split("-"))
m = json.load(open(f"{a.meta}/kld.json")); toks = np.memmap(f"{a.meta}/tokens.i32", np.int32, "r")
alld = m["docs"][:a.docs] if a.docs else m["docs"]
docs = [(i, d) for i, d in enumerate(alld) if d["tokens"] <= a.maxtok]
if a.only: docs = [(i, alld[i]) for i in map(int, a.only.split(","))]
skipped = [(i, d["tokens"]) for i, d in enumerate(alld) if d["tokens"] > a.maxtok]
print(f"{len(docs)} documents, {sum(d['tokens'] for _, d in docs)} tokens; skipped {len(skipped)} longer than "
      f"{a.maxtok} ({sum(t for _, t in skipped)} tokens)", file=sys.stderr, flush=True)
# documents longer than --chunk are cut into independent --chunk windows (no cache: a cached chunk materialises a
# chunk x past attention mask, 512 MB at 8k x 64k). Each window loses the context before it -- accepted for the proxy
# (user, 2026-11-02); an input Hessian is a sum of x x^T, so the shape-dependent rounding does not matter here.
toff = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
acc, host, linears = {}, {}, {}
def hook_for(name):
    def f(mod, args):
        x = args[0].reshape(-1, args[0].shape[-1])
        if name not in acc: acc[name] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
        for r in range(0, x.shape[0], 4096):   # fp32 a row block at a time: no whole-input fp32 copy
            xb = x[r:r + 4096].float(); acc[name].addmm_(xb.T, xb)
    return f
hooks = []
for l in range(lo, hi + 1):
    ly = dec.layers[l]; full = hasattr(ly, "self_attn")
    groups = [(f"{L}.{l}.input", ly.self_attn.q_proj if full else ly.linear_attn.in_proj_qkv,
               [f"{L}.{l}.self_attn.{p}_proj" for p in "qkv"] if full else [f"{L}.{l}.linear_attn.in_proj_qkv", f"{L}.{l}.linear_attn.in_proj_z"]),
              (f"{L}.{l}.self_attn.o_proj.input" if full else f"{L}.{l}.linear_attn.out_proj.input",
               ly.self_attn.o_proj if full else ly.linear_attn.out_proj,
               [f"{L}.{l}.self_attn.o_proj" if full else f"{L}.{l}.linear_attn.out_proj"]),
              (f"{L}.{l}.post_attention.input", ly.mlp.gate_proj, [f"{L}.{l}.mlp.gate_proj", f"{L}.{l}.mlp.up_proj"]),
              (f"{L}.{l}.mlp.down_proj.input", ly.mlp.down_proj, [f"{L}.{l}.mlp.down_proj"])]
    for name, mod, lin in groups:
        hooks.append(mod.register_forward_pre_hook(hook_for(name))); linears[name] = lin
if a.head:
    head_acc = hook_for("lm_head.input"); linears["lm_head.input"] = ["lm_head"]   # fed the final hidden states directly
# stop the forward after the last layer we need (and run the final norm + head on the hidden states only for --head)
stop_after = hi
class Stop(Exception): pass
if not a.head and stop_after < len(dec.layers) - 1:   # only when the later layers are not wanted
    def stopper(mod, args, out): raise Stop()
    hooks.append(dec.layers[stop_after].register_forward_hook(stopper))
def flush():
    for k, v in acc.items():
        h = v.double().cpu()
        host[k] = host[k] + h if k in host else h
        v.zero_()
rows, t0 = 0, time.time()
with torch.no_grad():
    for i, (di, d) in enumerate(docs):
        tk = torch.from_numpy(np.asarray(toks[toff[di]:toff[di] + d["tokens"]], np.int64))[None].cuda()
        for s in range(0, tk.shape[1], a.chunk):
            try:
                o = dec(input_ids=tk[:, s:s + a.chunk], use_cache=False)
                if a.head: head_acc(None, (o.last_hidden_state,))   # the head's input; no 248k-wide logits
            except Stop:
                pass
        rows += tk.shape[1]
        if (i + 1) % a.flush == 0: flush()
        if (i + 1) % 50 == 0 or i + 1 == len(docs):
            print(f"\r{i + 1}/{len(docs)} docs, {rows} tokens, {rows / (time.time() - t0):.0f} tok/s", end="", file=sys.stderr, flush=True)
flush(); print(file=sys.stderr)
# a new OUT is filled as OUT.part and renamed when complete (callers skip on [ -d OUT ]); into an existing OUT, each
# file is written .part + renamed (those callers mark OUT/.done themselves after exit 0)
dst = a.out if os.path.isdir(a.out) else a.out.rstrip("/") + ".part"
os.makedirs(dst, exist_ok=True)
for k, H in host.items():
    n = H.shape[0]; iu = torch.triu_indices(n, n)
    save_file({"hessian_upper": H[iu[0], iu[1]].float().contiguous()}, f"{dst}/{k}.safetensors.part",
              metadata={"k": str(n), "rows": str(rows), "layout": "upper triangle, row-major, packed",
                        "linears": ",".join(linears[k]), "model": a.model, "kernels": json.dumps(kernel_paths()),
                        "skipped_docs": json.dumps(skipped), "maxtok": str(a.maxtok)})
    fd = os.open(f"{dst}/{k}.safetensors.part", os.O_RDONLY); os.fsync(fd); os.close(fd)
    os.replace(f"{dst}/{k}.safetensors.part", f"{dst}/{k}.safetensors")
if dst != a.out: os.rename(dst, a.out)
print(f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB allocated, {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB reserved")
print(f"HCOLLECT DONE {len(host)} Hessians, {rows} rows, {(time.time() - t0) / 60:.1f} min")
