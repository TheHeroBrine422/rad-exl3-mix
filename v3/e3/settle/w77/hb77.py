#!/usr/bin/env python3
"""hb77.py -- held-out input Hessians for #114 (../w77.md): H_B = sum of x x^T over a set B of calib documents,
computed on the CPU with the bf16 checkpoint (upcast to fp32, HF transformers' Qwen3_5 decoder layers, one layer in
memory at a time), so that the quantizer can use the complement H_A = H_pub - H_B (the published calib Hessian minus
B: ~10.0M of the 10.26M tokens, about the ladder's sample size) and the proxy can score on H_B (tokens the quantizer
never saw).

  hb77.py plan OUT TOKENS_PER_HALF JOBS_PER_HALF   -> OUT/plan.json: fit-pool docs (pools-v3), whole documents of
        256..4096 tokens, shuffled (seed 77), alternately assigned to halves B1 / B2 until each has TOKENS_PER_HALF;
        each half split into JOBS_PER_HALF jobs of about equal tokens.
  hb77.py job OUT JOBID       -> runs one job: its documents (each a whole sequence, as the published H counts every
        calib token) layer by layer through layers 0..LMAX; at each needed layer the needed groups' x x^T (fp32 sums)
        are written to OUT/part/<job>/<group>.safetensors (challenge layout: hessian_upper, metadata k / rows).
        Hidden states are checkpointed every 8 layers (OUT/part/<job>/ckpt.pt): a killed job resumes there. Done
        marker OUT/part/<job>/DONE (written last).
  hb77.py reduce OUT H        -> sums the parts of half H (1 or 2) into OUT/hB<H> (B1 runs on colo, B2 on Thor; the
        two sums are exchanged, each box then runs finish).
  hb77.py finish OUT HPUB     -> OUT/hA = HPUB - hB1 - hB2 (float64 arithmetic, stored fp32 in the challenge layout),
        a check line per group (rows, min diag, diag-mean ratio of H_B / rows_B to H_pub / rows_pub, B1 vs B2).
  hb77.py time                -> seconds per layer for one 2048-token document, layers 0-3 (sizing).
Env: HB_CKPT (bf16 checkpoint), HB_META (dir with kld.json, tokens.i32, pools-v3.json), HB_LAYERS (groups needed:
default the 27 tensors of w43 / w71)."""
import json, os, sys, time
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

torch.set_grad_enabled(False)
CK = os.environ.get("HB_CKPT", "/data/models/qwen38-27b-bf16")
META = os.environ.get("HB_META", "/data/work/w03/meta")
P = "model.language_model.layers."
# (layer, group) pairs: group = the published Hessian's name after "layers.L." (e3.ROUTE's second field)
TENSORS = {"ssm_qkv": (2, 30, 50), "ssm_z": (10, 34, 62), "ssm_out": (10, 30, 50), "attn_qg": (3, 19, 43),
           "attn_k": (11, 35, 63), "attn_v": (7, 43, 59), "attn_output": (19, 43, 63), "ffn_gate_up": (10, 43, 62),
           "ffn_down": (2, 30, 63)}
GROUP = {"ssm_qkv": "input", "ssm_z": "input", "attn_qg": "input", "attn_k": "input", "attn_v": "input",
         "ssm_out": "linear_attn.out_proj.input", "attn_output": "self_attn.o_proj.input",
         "ffn_gate_up": "post_attention.input", "ffn_down": "mlp.down_proj.input"}
NEED = sorted({(l, GROUP[t]) for t, ls in TENSORS.items() for l in ls})
# group -> the module whose input it is (forward pre-hook)
HOOK = {"input": ("linear_attn.in_proj_qkv", "self_attn.q_proj"), "linear_attn.out_proj.input": ("linear_attn.out_proj",),
        "self_attn.o_proj.input": ("self_attn.o_proj",), "post_attention.input": ("mlp.gate_proj",),
        "mlp.down_proj.input": ("mlp.down_proj",)}


def text_config():
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(CK).text_config
    c._attn_implementation = "sdpa"
    return c


_IDX = None
def tensor(name):
    global _IDX
    if _IDX is None: _IDX = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{CK}/{_IDX[name]}", "pt") as f: return f.get_tensor(name)


def build_layer(cfg, l):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    with torch.device("meta"): m = Qwen3_5DecoderLayer(cfg, l)
    sd = {k: tensor(f"{P}{l}.{k}").float() for k in m.state_dict().keys()}
    m.load_state_dict(sd, assign=True); del sd
    return m.float().eval()


def rope(cfg):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    return Qwen3_5TextRotaryEmbedding(cfg)


def pos_emb(r, h):
    T = h.shape[1]
    pid = torch.arange(T).view(1, 1, T).expand(3, 1, T)
    return r(h, pid)


def docs_meta():
    meta = json.load(open(f"{META}/kld.json")); pools = json.load(open(f"{META}/pools-v3.json"))["pool"]
    off = np.cumsum([0] + [d["tokens"] for d in meta["docs"]])
    return meta["docs"], pools, off


def plan(out, per_half, jobs):
    docs, pools, off = docs_meta()
    cand = [i for i, d in enumerate(docs) if pools[i] == "fit" and 256 <= d["tokens"] <= 4096]
    rng = np.random.default_rng(77); rng.shuffle(cand)
    halves = {1: [], 2: []}; tok = {1: 0, 2: 0}
    for i in cand:                                         # the lighter half takes the next document
        h = 1 if tok[1] <= tok[2] else 2
        if tok[h] >= per_half: break                       # the lighter half is full: both are
        halves[h].append(int(i)); tok[h] += docs[i]["tokens"]
    J = []
    for h in (1, 2):
        bins = [[] for _ in range(jobs)]; bt = [0] * jobs
        for i in sorted(halves[h], key=lambda i: -docs[i]["tokens"]):   # greedy: longest first into the lightest bin
            b = int(np.argmin(bt)); bins[b].append(i); bt[b] += docs[i]["tokens"]
        for b in range(jobs): J.append({"job": f"b{h}j{b:02d}", "half": h, "docs": sorted(bins[b]), "tokens": bt[b]})
    os.makedirs(out, exist_ok=True)
    p = {"per_half": per_half, "tokens": tok, "docs": {h: sorted(halves[h]) for h in (1, 2)}, "jobs": J,
         "need": NEED, "source": f"{META}/kld.json + tokens.i32, pools-v3 fit pool"}
    json.dump(p, open(f"{out}/plan.json.part", "w"), indent=1); os.replace(f"{out}/plan.json.part", f"{out}/plan.json")
    print(f"halves: B1 {len(halves[1])} docs {tok[1]} tokens, B2 {len(halves[2])} docs {tok[2]} tokens; "
          f"{len(J)} jobs, tokens per job {min(j['tokens'] for j in J)}..{max(j['tokens'] for j in J)}")


def save_group(path, H, rows):
    k = H.shape[0]; iu = torch.triu_indices(k, k)
    save_file({"hessian_upper": H[iu[0], iu[1]].float().contiguous()}, path + ".part",
              metadata={"k": str(k), "rows": str(rows), "layout": "upper triangle, row-major, packed", "source": "hb77.py"})
    os.replace(path + ".part", path)


def job(out, jid):
    p = json.load(open(f"{out}/plan.json")); J = next(j for j in p["jobs"] if j["job"] == jid)
    need = {(l, g) for l, g in map(tuple, p["need"])}; lmax = max(l for l, _ in need)
    d = f"{out}/part/{jid}"; os.makedirs(d, exist_ok=True)
    if os.path.exists(f"{d}/DONE"): print("done already"); return
    docs, _, off = docs_meta(); toks = np.fromfile(f"{META}/tokens.i32", np.int32)
    ids = [torch.from_numpy(toks[off[i]:off[i] + docs[i]["tokens"]].astype(np.int64)) for i in J["docs"]]
    cfg = text_config(); r = rope(cfg)
    ck = f"{d}/ckpt.pt"; l0 = 0
    if os.path.exists(ck):
        c = torch.load(ck); l0, hs = c["next_layer"], c["hidden"]; print(f"resume at layer {l0}", flush=True)
    else:
        E = tensor("model.language_model.embed_tokens.weight")
        hs = [E[x].float()[None] for x in ids]; del E
    t_all = time.time()
    for l in range(l0, lmax + 1):
        t0 = time.time()
        groups = [g for (ll, g) in need if ll == l and not os.path.exists(f"{d}/{P}{l}.{g}.safetensors")]
        m = build_layer(cfg, l); acc, rows, hooks = {}, {}, []
        for g in groups:
            for mn in HOOK[g]:
                try: mod = m.get_submodule(mn)
                except AttributeError: continue
                def hk(mod, inp, g=g):
                    x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                    if g not in acc: acc[g] = torch.zeros(x.shape[1], x.shape[1]); rows[g] = 0
                    acc[g].addmm_(x.T, x); rows[g] += x.shape[0]
                hooks.append(mod.register_forward_pre_hook(hk))
        for n, h in enumerate(hs):
            hs[n] = m(h, position_embeddings=pos_emb(r, h), attention_mask=None)
        for hk_ in hooks: hk_.remove()
        for g in groups:
            assert rows[g] == J["tokens"], (g, rows[g], J["tokens"])
            save_group(f"{d}/{P}{l}.{g}.safetensors", acc[g], rows[g])
        del m, acc
        if l % 8 == 7 and l < lmax:
            torch.save({"next_layer": l + 1, "hidden": hs}, ck + ".part"); os.replace(ck + ".part", ck)
        print(f"layer {l}: {time.time() - t0:.0f}s, groups {groups}, total {(time.time() - t_all) / 60:.1f} min", flush=True)
    open(f"{d}/DONE.part", "w").write(json.dumps({"tokens": J["tokens"], "docs": len(ids)}) + "\n"); os.replace(f"{d}/DONE.part", f"{d}/DONE")
    if os.path.exists(ck): os.remove(ck)


def load_up(path):
    with safe_open(path, "pt") as f:
        md = f.metadata(); k = int(md["k"]); up = f.get_tensor("hessian_upper").double()
    H = torch.zeros(k, k, dtype=torch.float64); iu = torch.triu_indices(k, k)
    H[iu[0], iu[1]] = up; H = H + H.T - torch.diag(H.diagonal())
    return H, int(md["rows"])


def reduce_half(out, h):
    """sum the parts of half h (all its jobs DONE) into OUT/hB<h>"""
    p = json.load(open(f"{out}/plan.json")); js = [j for j in p["jobs"] if j["half"] == h]
    for j in js: assert os.path.exists(f"{out}/part/{j['job']}/DONE"), j["job"]
    os.makedirs(f"{out}/hB{h}", exist_ok=True)
    for l, g in map(tuple, p["need"]):
        name = f"{P}{l}.{g}.safetensors"
        if os.path.exists(f"{out}/hB{h}/{name}"): continue
        S, R = None, 0
        for j in js:
            H, r = load_up(f"{out}/part/{j['job']}/{name}"); S = H if S is None else S + H; R += r; del H
        assert R == sum(j["tokens"] for j in js), (name, R)
        save_group(f"{out}/hB{h}/{name}", S, R); del S
    print(f"REDUCE B{h} DONE", flush=True)


def finish(out, hpub):
    """OUT/hA = HPUB - hB1 - hB2 (float64), with the checks of ../w77.md"""
    p = json.load(open(f"{out}/plan.json")); os.makedirs(f"{out}/hA", exist_ok=True)
    for l, g in map(tuple, p["need"]):
        name = f"{P}{l}.{g}.safetensors"; tgt = f"{out}/hA/{name}"
        HB = {h: load_up(f"{out}/hB{h}/{name}") for h in (1, 2)}
        Hp, Rp = load_up(f"{hpub}/{name}")
        HA = Hp - HB[1][0] - HB[2][0]; RA = Rp - HB[1][1] - HB[2][1]
        dg = HA.diagonal(); nb = HB[1][1] + HB[2][1]
        mb = (HB[1][0].diagonal().mean() + HB[2][0].diagonal().mean()) / nb
        print(json.dumps({"group": f"{l}.{g}", "rows_pub": Rp, "rows_B": [HB[1][1], HB[2][1]], "rows_A": RA,
                          "min_diag_A": float(dg.min()), "neg_diag_A": int((dg <= 0).sum()),
                          "diagmean_B_over_pub": float(mb / (Hp.diagonal().mean() / Rp)),
                          "diagmean_B1_over_B2": float((HB[1][0].diagonal().mean() / HB[1][1]) / (HB[2][0].diagonal().mean() / HB[2][1]))}), flush=True)
        if not os.path.exists(tgt): save_group(tgt, HA, RA)
        del HA, Hp, HB
    print("FINISH DONE", flush=True)


def timing():
    cfg = text_config(); r = rope(cfg)
    docs, _, off = docs_meta(); toks = np.fromfile(f"{META}/tokens.i32", np.int32)
    i = next(i for i, d in enumerate(docs) if 2000 <= d["tokens"] <= 2100)
    x = torch.from_numpy(toks[off[i]:off[i] + docs[i]["tokens"]].astype(np.int64))
    t0 = time.time(); E = tensor("model.language_model.embed_tokens.weight"); h = E[x].float()[None]; del E
    print(f"doc {i} {len(x)} tokens; embed {time.time() - t0:.1f}s", flush=True)
    for l in range(4):
        t0 = time.time(); m = build_layer(cfg, l); t1 = time.time()
        h = m(h, position_embeddings=pos_emb(r, h), attention_mask=None)
        print(f"layer {l} ({cfg.layer_types[l]}): load {t1 - t0:.1f}s, forward {time.time() - t1:.1f}s, "
              f"rms {float(h.pow(2).mean().sqrt()):.3f}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "plan": plan(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "job": job(sys.argv[2], sys.argv[3])
    elif cmd == "reduce": reduce_half(sys.argv[2], int(sys.argv[3]))
    elif cmd == "finish": finish(sys.argv[2], sys.argv[3])
    elif cmd == "time": timing()
    else: raise SystemExit(__doc__)
