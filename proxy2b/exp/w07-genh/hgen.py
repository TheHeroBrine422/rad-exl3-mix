#!/usr/bin/env python3
"""hgen.py MODEL META OUT [--weight ng|gen|all] [--pools fit,search,qad] [--exclude GOLDEN,..] [--layers a-b] -- input
Hessians of the 0.8B (hcollect.py's groups, layout and arithmetic) summed over a subset of positions: --weight ng = the
positions the challenge does not score (gen.u8 = 0: prompts, tool output, system text, everything before score_from,
each document's last position), gen = the scored ones, all = every position. Documents: those of --pools (pools-v3,
$POOLS to override) that are in none of the --exclude goldens (mkgold.py dirs; their docs' "orig" indices). A position t
of document d is scored when t >= score_from and gen[row0 + t - score_from] (row j predicts token sf + j + 1 from position
sf + j). Documents run whole in --chunk windows without context (as hcollect.py, so the sums add up to calib's).
OUT/<group>.safetensors: "hessian_upper" f32, metadata k, rows (= positions summed), weight, docs (w07-genh, C13)."""
import argparse, json, os, sys, time
import numpy as np, torch
from safetensors.torch import save_file
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from common import load_model, kernel_paths
ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("meta"); ap.add_argument("out")
ap.add_argument("--weight", choices=("ng", "gen", "all"), default="ng")
ap.add_argument("--pools", default="fit,search,qad"); ap.add_argument("--exclude", default="")
ap.add_argument("--layers", default="0-23"); ap.add_argument("--chunk", type=int, default=8192)
ap.add_argument("--flush", type=int, default=200); ap.add_argument("--docs", type=int, default=0, help="first N selected docs (tests)")
a = ap.parse_args()
L = "model.language_model.layers"
DEV = os.environ.get("DEV", "cuda")
if DEV == "cuda" and os.environ.get("VRAM_CAP_GB"):
    torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["VRAM_CAP_GB"]) * 2**30 / torch.cuda.get_device_properties(0).total_memory))
model, dec, head = load_model(a.model, dev=DEV)
lo, hi = map(int, a.layers.split("-"))
m = json.load(open(f"{a.meta}/kld.json")); toks = np.memmap(f"{a.meta}/tokens.i32", np.int32, "r")
gen = np.fromfile(f"{a.meta}/gen.u8", np.uint8).astype(bool); assert len(gen) == m["rows"]
pools = json.load(open(os.environ.get("POOLS", "/workspace/rad-exl3-mix/eval/pools/pools-v3.json")))["pool"]
assert len(pools) == len(m["docs"])
excl = set()
for g in filter(None, a.exclude.split(",")): excl |= {d["orig"] for d in json.load(open(f"{g}/kld.json"))["docs"]}
want = set(a.pools.split(","))
docs = [(i, d) for i, d in enumerate(m["docs"]) if pools[i] in want and i not in excl]
if a.docs: docs = docs[:a.docs]
def scored(d):                                       # per position: is its prediction scored
    s = np.zeros(d["tokens"], bool); sf = d["score_from"]; k = d["tokens"] - 1 - sf
    s[sf:sf + k] = gen[d["row0"]:d["row0"] + k]
    return s
print(f"{len(docs)} documents ({sum(d['tokens'] for _, d in docs)} tokens) of pools {sorted(want)}, {len(excl)} excluded; "
      f"weight {a.weight}", file=sys.stderr, flush=True)
toff = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
acc, host, linears, cur = {}, {}, {}, {"m": None}
def hook_for(name):
    def f(mod, args):
        x = args[0].reshape(-1, args[0].shape[-1])
        if cur["m"] is not None: x = x[cur["m"]]
        if name not in acc: acc[name] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
        for r in range(0, x.shape[0], 4096):   # fp32 a row block at a time, as hcollect.py
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
class Stop(Exception): pass
if hi < len(dec.layers) - 1:
    def stopper(mod, args, out): raise Stop()
    hooks.append(dec.layers[hi].register_forward_hook(stopper))
def flush():
    for k, v in acc.items():
        h = v.double().cpu(); host[k] = host[k] + h if k in host else h; v.zero_()
rows, t0, done = 0, time.time(), 0
with torch.no_grad():
    for i, (di, d) in enumerate(docs):
        sc = scored(d); keep = ~sc if a.weight == "ng" else sc if a.weight == "gen" else np.ones_like(sc)
        tk = torch.from_numpy(np.asarray(toks[toff[di]:toff[di] + d["tokens"]], np.int64))[None].to(DEV)
        for s in range(0, tk.shape[1], a.chunk):
            km = keep[s:s + a.chunk]
            if not km.any(): continue
            cur["m"] = None if km.all() else torch.from_numpy(km).to(DEV)
            try: dec(input_ids=tk[:, s:s + a.chunk], use_cache=False)
            except Stop: pass
            rows += int(km.sum())
        done += 1
        if done % a.flush == 0: flush()
        if done % 50 == 0 or done == len(docs):
            print(f"\r{done}/{len(docs)} docs, {rows} positions, {rows / (time.time() - t0):.0f} pos/s", end="", file=sys.stderr, flush=True)
flush(); print(file=sys.stderr)
os.makedirs(a.out, exist_ok=True)
for k, H in host.items():
    n = H.shape[0]; iu = torch.triu_indices(n, n)
    save_file({"hessian_upper": H[iu[0], iu[1]].float().contiguous()}, f"{a.out}/{k}.safetensors",
              metadata={"k": str(n), "rows": str(rows), "layout": "upper triangle, row-major, packed", "linears": ",".join(linears[k]),
                        "model": a.model, "kernels": json.dumps(kernel_paths()), "weight": a.weight, "pools": a.pools,
                        "exclude": a.exclude, "docs": str(len(docs))})
print(f"HGEN DONE {len(host)} Hessians, {rows} positions ({a.weight}), {(time.time() - t0) / 60:.1f} min")
