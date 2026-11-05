#!/usr/bin/env python3
"""headx.py CKPT GOLDEN POOLS OUT [--hpools qad] [--dpools fit] [--tok 2e6] [--cap 8192] [--dump 65536] [--per-doc 1024]
-- the LM head's input (the final norm's output) from a bf16 forward of the checkpoint over calib documents, aligned
to the golden's rows (E4 of ranked.md: head statistics from scored rows only). gpubox (27B: both cards, ~30-60 min).

GOLDEN: the calib golden's kld.json / tokens.i32 / gen.u8 (only these are read). Documents in a fixed shuffled order
(--seed), each forwarded whole up to --cap tokens; position p predicts token p + 1 = golden row row0 + p - score_from
(p >= score_from), scored if gen.u8 says so.
  --hpools docs (default qad, the pool the head Fisher ranking uses) until --tok context tokens:
      OUT/hin-gen/lm_head.input.safetensors   sum over gen rows of x xᵀ  (challenge layout: hessian_upper, k, rows)
      OUT/hin-all/lm_head.input.safetensors   sum over every position (as the published calib Hessian, same docs)
  --dpools docs (default fit: held out from the ranking and from these Hessians) until --dump rows:
      OUT/dump/336.x (bf16 rows, headsim's DUMPDIR layout), 336.meta, rows.i64 (golden row), doc.i32 (document index),
      nxt.i32 (the true next token): gen rows only, at most --per-doc a document (evenly spaced)
  OUT/headx.json: counts, and how far the gen-row Hessian is from the all-position one (and from --calib-hess).
Without a GPU everything runs on the CPU (self-test on a tiny checkpoint)."""
import argparse, json, os, sys, time
sys.path.insert(0, os.environ.get("HX_PYLIB", "/data/yaqa/pylib"))       # gpubox's transformers / accelerate (collect.py)
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("golden"); ap.add_argument("pools"); ap.add_argument("out")
ap.add_argument("--hpools", default="qad"); ap.add_argument("--dpools", default="fit")
ap.add_argument("--tok", type=float, default=2e6); ap.add_argument("--cap", type=int, default=8192)
ap.add_argument("--dump", type=int, default=65536); ap.add_argument("--per-doc", type=int, default=1024)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--calib-hess", default=None)
a = ap.parse_args()
HP, DP = set(a.hpools.split(",")), set(a.dpools.split(","))
assert not HP & DP, "Hessian and dump pools must be disjoint"

# ---- the bf16 text model, streamed onto the cards (collect.py's loader)
from transformers import AutoConfig, Qwen3_5ForCausalLM
from accelerate import init_empty_weights, dispatch_model
from accelerate.utils import set_module_tensor_to_device
cfg = AutoConfig.from_pretrained(a.ckpt); tcfg = getattr(cfg, "text_config", cfg)
n = tcfg.num_hidden_layers; NG = torch.cuda.device_count()
def dev_of(name):
    if NG == 0: return "cpu"
    if NG == 1 or not name.startswith("model.layers."): return "cuda:0" if not name.startswith("model.norm") or NG == 1 else f"cuda:{NG - 1}"
    return f"cuda:{min(NG - 1, int(name.split('.')[2]) * NG // n)}"
t0 = time.time()
with init_empty_weights(): model = Qwen3_5ForCausalLM(tcfg)
idx = json.load(open(f"{a.ckpt}/model.safetensors.index.json"))["weight_map"]
byfile = {}
for k, f in idx.items():
    if k.startswith("model.language_model."): byfile.setdefault(f, []).append(k)
for f, ks in byfile.items():
    with safe_open(f"{a.ckpt}/{f}", "pt", device="cpu") as h:
        for k in ks:
            name = "model." + k[len("model.language_model."):]
            set_module_tensor_to_device(model, name, dev_of(name), value=h.get_tensor(k), dtype=torch.bfloat16)
del model.lm_head                                   # not needed: the head's input is the decoder's output
dec = model.model
if NG > 0:
    dmap = {"embed_tokens": 0, "rotary_emb": 0, "norm": NG - 1}
    for l in range(n): dmap[f"layers.{l}"] = int(dev_of(f"model.layers.{l}").split(":")[1])
    dec = dispatch_model(dec, device_map=dmap)
dec.eval(); D = tcfg.hidden_size
hdev = torch.device(f"cuda:{NG - 1}" if NG else "cpu")
print(f"loaded in {time.time() - t0:.0f}s on {NG} GPU(s)", flush=True)

# ---- documents
m = json.load(open(f"{a.golden}/kld.json")); toks = np.fromfile(f"{a.golden}/tokens.i32", np.int32)
gen = np.fromfile(f"{a.golden}/gen.u8", np.uint8).astype(bool); pools = json.load(open(a.pools))["pool"]
assert len(pools) == len(m["docs"]) and len(gen) == m["rows"]
off = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])]); assert off[-1] == len(toks)
order = np.random.default_rng(a.seed).permutation(len(m["docs"]))
Hg = torch.zeros(D, D, dtype=torch.float64, device=hdev); Ha = torch.zeros_like(Hg)
cnt = {"h_docs": 0, "h_tok": 0, "h_gen": 0, "d_docs": 0, "d_rows": 0}
os.makedirs(f"{a.out}/dump", exist_ok=True)
fx = open(f"{a.out}/dump/336.x", "wb"); R, Dc, NX = [], [], []
t1 = time.time()
with torch.no_grad():
    for i in order:
        pl, d = pools[i], m["docs"][i]
        want_h = pl in HP and cnt["h_tok"] < a.tok
        want_d = pl in DP and cnt["d_rows"] < a.dump
        if not (want_h or want_d): 
            if cnt["h_tok"] >= a.tok and cnt["d_rows"] >= a.dump: break
            continue
        L = min(d["tokens"], a.cap); sf = d["score_from"]
        if want_d and L - 1 <= sf: continue
        ids = torch.from_numpy(toks[off[i]:off[i] + L].astype(np.int64))[None].to("cuda:0" if NG else "cpu")
        x = dec(input_ids=ids, use_cache=False).last_hidden_state[0].to(hdev)          # (L, D) bf16
        p = np.arange(sf, L - 1); r = d["row0"] + p - sf; g = gen[r]                    # rows this prefix scores
        if want_h:
            xf = x.double(); Ha.addmm_(xf.T, xf)
            xg = xf[torch.from_numpy(p[g]).to(hdev)]; Hg.addmm_(xg.T, xg)
            cnt["h_docs"] += 1; cnt["h_tok"] += L; cnt["h_gen"] += int(g.sum())
        if want_d and g.any():
            pg, rg = p[g], r[g]
            take = min(len(pg), a.per_doc, a.dump - cnt["d_rows"])
            s = np.linspace(0, len(pg) - 1, take).round().astype(int)
            fx.write(x[torch.from_numpy(pg[s]).to(hdev)].contiguous().view(torch.int16).cpu().numpy().tobytes())
            R.append(rg[s]); Dc.append(np.full(take, i, np.int32)); NX.append(toks[off[i] + pg[s] + 1])
            cnt["d_docs"] += 1; cnt["d_rows"] += take
        if (cnt["h_docs"] + cnt["d_docs"]) % 25 == 0:
            print(f"{cnt} {time.time() - t1:.0f}s", flush=True)
fx.close()
open(f"{a.out}/dump/336.meta", "w").write(f"{D} 1 1\n")
np.concatenate(R).astype(np.int64).tofile(f"{a.out}/dump/rows.i64")
np.concatenate(Dc).tofile(f"{a.out}/dump/doc.i32"); np.concatenate(NX).astype(np.int32).tofile(f"{a.out}/dump/nxt.i32")
iu = torch.triu_indices(D, D, device=hdev)
for tag, H, rows in (("gen", Hg, cnt["h_gen"]), ("all", Ha, cnt["h_tok"])):
    os.makedirs(f"{a.out}/hin-{tag}", exist_ok=True)
    save_file({"hessian_upper": H[iu[0], iu[1]].float().cpu().contiguous()}, f"{a.out}/hin-{tag}/lm_head.input.safetensors",
              metadata={"k": str(D), "rows": str(rows), "source": f"headx.py {tag} rows, pools {a.hpools}"})
def rel(A, B):                                       # per-row-normalised matrices: Frobenius cosine, diagonal spread
    A, B = A / A.trace(), B / B.trace()
    return {"cos": float((A * B).sum() / (A.norm() * B.norm())), "diag_ratio_q": [float(q) for q in
            torch.quantile((A.diagonal() / B.diagonal()).float().cpu(), torch.tensor([0.01, 0.5, 0.99]))]}
cnt.update(gen_vs_all=rel(Hg, Ha), sec=round(time.time() - t1))
if a.calib_hess:
    with safe_open(a.calib_hess, "pt") as f: up = f.get_tensor("hessian_upper").double().to(hdev)
    C = torch.zeros(D, D, dtype=torch.float64, device=hdev); C[iu[0], iu[1]] = up; C = C + C.triu(1).T
    cnt.update(all_vs_calib=rel(Ha, C), gen_vs_calib=rel(Hg, C))
json.dump(cnt, open(f"{a.out}/headx.json", "w"), indent=1)
print("HEADX DONE", json.dumps(cnt), flush=True)
