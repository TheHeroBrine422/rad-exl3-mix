#!/usr/bin/env python3
"""swiglu_diag.py MODEL META POOLS OUT NSEQ [--len 2048] [--seed 0] [--snap 128,256,512,1024] [--device cuda] -- A9
(ranked.md): a forward-only stand-in for the Sketch B diagonal of H_out on the MLP gate / up projections.

With h = SiLU(g) * u and the down projection y = W_d h, the gradient on gate / up output channel r is
    dy_gate_r = (dL/dh_r) SiLU'(g_r) u_r,      dy_up_r = (dL/dh_r) SiLU(g_r),      dL/dh_r = W_d[:, r]ᵀ dL/dy.
If dL/dy is isotropic and independent across tokens (E[δ δᵀ] = σ² I), Sketch B's E‖Σ_t dy_t[r] x_t‖² becomes
    w_gate[r] = ‖W_d[:, r]‖² Σ_t (SiLU'(g_t,r) u_t,r)² ‖x_t‖²,      w_up[r] = ‖W_d[:, r]‖² Σ_t SiLU(g_t,r)² ‖x_t‖²
(variant a9x); a9d drops the ‖x_t‖² weight (collect.py's "dy" estimator analogue). σ² and the sequence count drop out
(prepare_H_out normalises H_out by its mean diagonal). No backward, no labels: gate/up would need a forward pass only.

Windows: the calib golden's qad-pool documents cut into LEN-token windows, shuffled with SEED, exactly collect.py's
order (the c18 collection, #33-18, uses YAQA_SEED=0): the same first NSEQ windows. Every position counts (as YAQA_DIAG).
Writes at each snapshot n: OUT/a9x-snap<n>/ and OUT/a9d-snap<n>/ (<checkpoint key>.safetensors, "hout_diag" = the mean,
gate / up only; ourq.py --hout layout) and OUT/raw-snap<n>/L<l>.safetensors (the four sums / n, without the W_d factor,
plus wd2 = ‖W_d[:, r]‖²) for a9cmp.py. fp32 activations (the model is cast to fp32)."""
import argparse, json, os, time
import numpy as np, torch
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("meta"); ap.add_argument("pools"); ap.add_argument("out"); ap.add_argument("nseq", type=int)
ap.add_argument("--len", type=int, default=2048); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--snap", default="128,256,512,1024"); ap.add_argument("--device", default="cuda")
ap.add_argument("--threads", type=int, default=0)
a = ap.parse_args()
if a.threads: torch.set_num_threads(a.threads)
L, dev = a.len, torch.device(a.device)
SNAPS = {int(x) for x in a.snap.split(",") if x} | {a.nseq}

meta = json.load(open(f"{a.meta}/kld.json")); toks = np.fromfile(f"{a.meta}/tokens.i32", np.int32)
pools = json.load(open(a.pools))["pool"]
off = np.cumsum([0] + [d["tokens"] for d in meta["docs"]])
wins = []
for i, d in enumerate(meta["docs"]):                 # collect.py's window list and shuffle, verbatim
    if pools[i] != "qad": continue
    for s in range(0, d["tokens"] - L + 1, L): wins.append(int(off[i] + s))
np.random.default_rng(a.seed).shuffle(wins)
print(f"{len(wins)} qad windows of {L} tokens; using {a.nseq}", flush=True)
assert len(wins) >= a.nseq

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
dec = model.get_decoder()
S, WD2, cur = {}, {}, {}
def silu_d(g):                                       # d/dg g sigmoid(g)
    s = torch.sigmoid(g); return s * (1 + g * (1 - s))
for l, ly in enumerate(dec.layers):
    mlp = ly.mlp
    WD2[l] = mlp.down_proj.weight.detach().float().pow(2).sum(0).cpu()       # ‖W_d[:, r]‖², [intermediate]
    S[l] = {k: torch.zeros(mlp.gate_proj.out_features, dtype=torch.float64, device=dev) for k in ("gx", "gd", "ux", "ud")}
    def h_gate(mod, inp, out, l=l): cur[l] = (inp[0][0].float(), out[0].float())
    def h_up(mod, inp, out, l=l):
        x, g = cur.pop(l); u = out[0].float()
        x2 = x.pow(2).sum(1, keepdim=True)                                   # [T, 1]
        ag = (silu_d(g) * u).pow(2); au = torch.nn.functional.silu(g).pow(2)  # [T, m]
        S[l]["gx"] += (ag * x2).sum(0).double(); S[l]["gd"] += ag.sum(0).double()
        S[l]["ux"] += (au * x2).sum(0).double(); S[l]["ud"] += au.sum(0).double()
    mlp.gate_proj.register_forward_hook(h_gate); mlp.up_proj.register_forward_hook(h_up)

def save(n):
    for v in ("a9x", "a9d"): os.makedirs(f"{a.out}/{v}-snap{n}", exist_ok=True)
    os.makedirs(f"{a.out}/raw-snap{n}", exist_ok=True)
    md = {"seqs": str(n), "ctx": str(L), "seed": str(a.seed), "sketch": "A9 analytic SwiGLU"}
    for l in S:
        r = {k: (v / n).float().cpu() for k, v in S[l].items()}
        save_file(dict(r, wd2=WD2[l]), f"{a.out}/raw-snap{n}/L{l}.safetensors", metadata=md)
        p = f"model.language_model.layers.{l}.mlp"
        for v, sfx in (("a9x", "x"), ("a9d", "d")):
            for proj, k in (("gate_proj", "g"), ("up_proj", "u")):
                save_file({"hout_diag": (WD2[l] * r[k + sfx]).contiguous()}, f"{a.out}/{v}-snap{n}/{p}.{proj}.safetensors", metadata=md)
    print(f"saved snapshot {n}", flush=True)

t0 = time.time()
with torch.no_grad():
    for s in range(a.nseq):
        ids = torch.from_numpy(toks[wins[s]:wins[s] + L].astype(np.int64))[None].to(dev)
        dec(input_ids=ids, use_cache=False)
        if (s + 1) % 32 == 0: print(f"seq {s + 1}: {(time.time() - t0) / (s + 1):.2f} s/seq", flush=True)
        if s + 1 in SNAPS: save(s + 1)
print(f"A9 DONE {a.nseq} seqs in {(time.time() - t0) / 60:.1f} min", flush=True)
