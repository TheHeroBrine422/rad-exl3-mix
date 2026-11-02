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
a = ap.parse_args()
L = "model.language_model.layers"
model, dec, head = load_model(a.model)
lo, hi = map(int, a.layers.split("-"))
m = json.load(open(f"{a.meta}/kld.json")); toks = np.memmap(f"{a.meta}/tokens.i32", np.int32, "r")
docs = m["docs"][:a.docs] if a.docs else m["docs"]
toff = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
acc, host, linears = {}, {}, {}
def hook_for(name):
    def f(mod, args):
        x = args[0].reshape(-1, args[0].shape[-1]).float()
        if name not in acc: acc[name] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
        acc[name].addmm_(x.T, x)
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
    hooks.append(head.register_forward_pre_hook(hook_for("lm_head.input"))); linears["lm_head.input"] = ["lm_head"]
# stop the forward after the last layer we need (and run the final norm + head on the hidden states only for --head)
stop_after = hi
class Stop(Exception): pass
if not a.head:
    def stopper(mod, args, out): raise Stop()
    hooks.append(dec.layers[stop_after].register_forward_hook(stopper))
def flush():
    for k, v in acc.items():
        h = v.double().cpu()
        host[k] = host[k] + h if k in host else h
        v.zero_()
rows, t0 = 0, time.time()
with torch.no_grad():
    for i, d in enumerate(docs):
        di = m["docs"].index(d) if a.docs else i
        tk = torch.from_numpy(np.asarray(toks[toff[di]:toff[di] + d["tokens"]], np.int64))[None].cuda()
        cache = None
        try:
            for s in range(0, tk.shape[1], a.chunk):
                o = dec(input_ids=tk[:, s:s + a.chunk], past_key_values=cache, use_cache=tk.shape[1] > a.chunk)
                cache = o.past_key_values
                if a.head: head(o.last_hidden_state)
        except Stop:
            assert tk.shape[1] <= a.chunk, "early stop with a chunked document: use --head or a shorter doc"
        rows += tk.shape[1]; del cache
        if (i + 1) % a.flush == 0: flush()
        if (i + 1) % 50 == 0 or i + 1 == len(docs):
            print(f"\r{i + 1}/{len(docs)} docs, {rows} tokens, {rows / (time.time() - t0):.0f} tok/s", end="", file=sys.stderr, flush=True)
flush(); print(file=sys.stderr)
os.makedirs(a.out, exist_ok=True)
for k, H in host.items():
    n = H.shape[0]; iu = torch.triu_indices(n, n)
    save_file({"hessian_upper": H[iu[0], iu[1]].float().contiguous()}, f"{a.out}/{k}.safetensors",
              metadata={"k": str(n), "rows": str(rows), "layout": "upper triangle, row-major, packed",
                        "linears": ",".join(linears[k]), "model": a.model, "kernels": json.dumps(kernel_paths())})
print(f"HCOLLECT DONE {len(host)} Hessians, {rows} rows, {(time.time() - t0) / 60:.1f} min")
