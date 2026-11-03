#!/usr/bin/env python3
"""a9oracle.py MODEL META POOLS OUTDIR [--win 16] [--threads 16] -- why does the analytic A9 diagonal miss Sketch B (CPU)?
The closed form assumes the down projection's output gradient δ_t is isotropic AND the same size at every token. This
runs a real backward (sampled-label CE over every position, fp32, the first TOK tokens of collect.py's first WIN windows; TOK 1024 by
default to stay inside the shared box's memory) and writes, for the
gate / up projections of every layer, per-token sums over the windows (raw-snap<WIN>/L<l>.safetensors, a9cmp.py layout):
  gx / ux        Σ a² ‖x‖²                    (= swiglu_diag.py's, the a9x form)
  gxo / uxo      Σ a² ‖x‖² ‖δ_t‖²/d           (oracle token weight: the true per-token gradient size, isotropy kept)
  gxf / uxf      Σ a² ‖x‖² f_t, f_t = 1 - Σ_v p_t,v²   (a forward-only token weight: the softmax Fisher trace)
  gxm / uxm      Σ (W_d[:, r]ᵀδ_t)² a² ‖x‖²    (the measured per-token form = collect.py's xw: sanity vs c18)
  C              Σ δ_t δ_tᵀ (hidden²): variant fullC = (W_d[:, r]ᵀ C W_d[:, r]) Σ a² ‖x‖² (isotropy replaced by the mean
                 residual-gradient covariance; token independence kept)
with wd2 = ‖W_d[:, r]‖². a9cmp-style table: cosine of each with the Sketch B diagonal BDIR, printed at the end."""
import argparse, json, os, sys, time
import numpy as np, torch
from safetensors.torch import save_file
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("meta"); ap.add_argument("pools"); ap.add_argument("out"); ap.add_argument("bdir")
ap.add_argument("--win", type=int, default=16); ap.add_argument("--tok", type=int, default=1024); ap.add_argument("--threads", type=int, default=16)
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(0)
meta = json.load(open(f"{a.meta}/kld.json")); toks = np.fromfile(f"{a.meta}/tokens.i32", np.int32)
pools = json.load(open(a.pools))["pool"]; off = np.cumsum([0] + [d["tokens"] for d in meta["docs"]]); wins = []
for i, d in enumerate(meta["docs"]):
    if pools[i] != "qad": continue
    for s in range(0, d["tokens"] - 2048 + 1, 2048): wins.append(int(off[i] + s))
np.random.default_rng(0).shuffle(wins)
from transformers import AutoModelForCausalLM
from torch.utils.checkpoint import checkpoint
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
model.requires_grad_(False); dec, head = model.get_decoder(), model.get_output_embeddings()
NL = len(dec.layers); hd = dec.layers[0].mlp.down_proj.out_features
S = {l: {k: torch.zeros(dec.layers[l].mlp.gate_proj.out_features, dtype=torch.float64) for k in
         ("gx", "ux", "gxo", "uxo", "gxf", "uxf", "gxm", "uxm")} | {"C": torch.zeros(hd, hd, dtype=torch.float64)} for l in range(NL)}
cur = {}; ft = {}
def silu_d(g): s = torch.sigmoid(g); return s * (1 + g * (1 - s))
for l, ly in enumerate(dec.layers):
    mlp = ly.mlp
    def hg(m, i, o, l=l): cur[(l, "x")] = i[0][0].detach(); cur[(l, "g")] = o[0].detach()
    def hu(m, i, o, l=l): cur[(l, "u")] = o[0].detach()
    def hdn(m, i, o, l=l, mlp=mlp):
        def bh(grad):
            d = grad[0]; x, g, u = cur.pop((l, "x")), cur.pop((l, "g")), cur.pop((l, "u"))
            x2 = x.pow(2).sum(1, keepdim=True); ag = (silu_d(g) * u).pow(2); au = torch.nn.functional.silu(g).pow(2)
            S[l]["C"] += (d.T @ d).double(); dn = d.pow(2).sum(1, keepdim=True) / hd; f = ft["f"][:, None]; hb = (d @ mlp.down_proj.weight).pow(2)
            for p, A in (("g", ag), ("u", au)):
                S[l][p + "x"] += (A * x2).sum(0).double(); S[l][p + "xo"] += (A * x2 * dn).sum(0).double()
                S[l][p + "xf"] += (A * x2 * f).sum(0).double(); S[l][p + "xm"] += (A * x2 * hb).sum(0).double()
        o.register_hook(bh)
    mlp.gate_proj.register_forward_hook(hg); mlp.up_proj.register_forward_hook(hu); mlp.down_proj.register_forward_hook(hdn)
t0 = time.time()
for w in range(a.win):
    ids = torch.from_numpy(toks[wins[w]:wins[w] + a.tok].astype(np.int64))[None]
    emb = dec.embed_tokens(ids).detach().requires_grad_(True)
    hid = dec(inputs_embeds=emb, use_cache=False).last_hidden_state[0, :-1]
    loss, fs = 0, []
    for c in range(0, hid.shape[0], 256):
        with torch.no_grad():
            z = head(hid[c:c + 256])
            p = torch.softmax(z, -1); y = torch.multinomial(p, 1)[:, 0]; fs.append(1 - p.pow(2).sum(1))
        loss = loss + checkpoint(lambda h_, y_: torch.nn.functional.cross_entropy(head(h_), y_, reduction="sum"),
                                 hid[c:c + 256], y, use_reentrant=False)   # logits not kept per chunk (host memory)
    ft["f"] = torch.cat(fs + [fs[-1][-1:]])       # the last position has no label: reuse its neighbour's f (one token)
    loss.backward(); del loss, hid, emb
    print(f"window {w + 1}/{a.win}: {time.time() - t0:.0f}s, rss {int(open('/proc/self/statm').read().split()[1]) * 4096 >> 30} GiB", flush=True)
    if (w + 1) % 8 == 0 or w + 1 == a.win:      # partial sums survive a killed run
        od = f"{a.out}/raw-snap{w + 1}"; os.makedirs(od, exist_ok=True)
        for l in range(NL):
            wd2 = dec.layers[l].mlp.down_proj.weight.pow(2).sum(0)
            save_file({k: (v / (w + 1)).float() for k, v in S[l].items()} | {"wd2": wd2.float().contiguous()}, f"{od}/L{l}.safetensors")
cos = lambda u, v: float((u * v).sum() / (u.norm() * v.norm()))
rows = []
for l in range(NL):
    for proj, p in (("gate_proj", "g"), ("up_proj", "u")):
        with safe_open(f"{a.bdir}/model.language_model.layers.{l}.mlp.{proj}.safetensors", "pt") as f: B = f.get_tensor("hout_diag").double()
        wd2 = dec.layers[l].mlp.down_proj.weight.pow(2).sum(0).double(); s = S[l]
        Wd = dec.layers[l].mlp.down_proj.weight.double(); wc = (Wd * (s["C"] @ Wd)).sum(0)
        rows.append({"layer": l, "proj": proj, **{nm: cos(v, B) for nm, v in (("a9x", wd2 * s[p + "x"]), ("oracle_tok", wd2 * s[p + "xo"]),
                     ("fisher_tok", wd2 * s[p + "xf"]), ("fullC", wc * s[p + "x"]), ("measured_xw", s[p + "xm"]))}})
json.dump(rows, open(f"{a.out}/a9oracle.json", "w"), indent=0)
print("cosine with Sketch B, per layer (gate | up):  a9x  oracle_tok  fisher_tok  fullC  measured_xw")
for l in range(NL):
    g, u = rows[2 * l], rows[2 * l + 1]
    print(f"  {l:2d} " + "  ".join(f"{g[k]:.3f}|{u[k]:.3f}" for k in ("a9x", "oracle_tok", "fisher_tok", "fullC", "measured_xw")))
for proj in ("gate_proj", "up_proj"):
    r = [x for x in rows if x["proj"] == proj]
    print(f"{proj}: median / min " + "  ".join(f"{k} {np.median([x[k] for x in r]):.3f}/{min(x[k] for x in r):.3f}"
                                            for k in ("a9x", "oracle_tok", "fisher_tok", "fullC", "measured_xw")))
