#!/usr/bin/env python3
"""perm06.py -- w75 #106-01 (../w75.md): o_proj head act order folded into the 0.8B's stored tensors.

  perm06.py make CKPT HESS OUTCKPT OUTWORK   per attention layer (3, 7, .., 23; 8 q heads x 256, 2 KV heads, groups of
      4): head trace = trace of the calib H_in (HESS/<layer>.self_attn.o_proj.input) diagonal over the head's 256
      columns; stored order = w64's `rev` (KV groups by ascending total trace, heads inside a group by ascending
      trace: heaviest stored last). OUTCKPT = CKPT with q_proj rows (per head 512 = q 256 | gate 256), k_proj / v_proj
      rows (per KV head 256) and o_proj columns permuted (every other tensor and file as CKPT's; q_norm / k_norm are
      per head_dim, shared). OUTWORK/challenge/hessians/calib = links to HESS, except the 6 o_proj.input Hessians,
      permuted (same packed-upper format and metadata). OUTCKPT/perm06.json: per layer the traces, the group / head
      orders, `cols` (stored o_proj column p = natural column cols[p]), `qrows`, `kvrows`.
  perm06.py unperm PERMJSON IN OUT           IN: <key>.safetensors "weight" (out, in) of o_proj in stored column order
      (deq.py wq of the permuted quant) -> OUT: the same in natural order (W_nat[:, cols] = W_stored).
  perm06.py permwq PERMJSON IN OUT           IN: a full natural wq set (150 files, links followed) -> OUT: the stored
      order for the permuted checkpoint (q / k / v rows, o_proj columns permuted; every other file linked)."""
import json, os, shutil, struct, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file

L = "model.language_model.layers"
HD, NQ, NKV = 256, 8, 2
GS = NQ // NKV
ATTN = range(3, 24, 4)


def hdr(p):
    with open(p, "rb") as fh: n = struct.unpack("<Q", fh.read(8))[0]; return json.loads(fh.read(n))


def hess_full(p):
    meta = hdr(p)["__metadata__"]; k = int(meta["k"])
    with safe_open(p, "pt", device="cpu") as f: up = f.get_tensor("hessian_upper")
    iu = torch.triu_indices(k, k)
    H = torch.zeros(k, k, dtype=up.dtype); H[iu[0], iu[1]] = up; H = H + H.triu(1).T
    return H, meta, up.dtype


def order(tr):
    ht = [float(tr[h * HD:(h + 1) * HD].sum()) for h in range(NQ)]
    groups = [list(range(g * GS, (g + 1) * GS)) for g in range(NKV)]
    gord = sorted(range(NKV), key=lambda g: sum(ht[h] for h in groups[g]))          # ascending: heaviest group last
    heads = [h for g in gord for h in sorted(groups[g], key=lambda h: ht[h])]         # ascending inside the group
    cols = [c for h in heads for c in range(h * HD, (h + 1) * HD)]
    qrows = [r for h in heads for r in range(h * 2 * HD, (h + 1) * 2 * HD)]
    kvrows = [r for g in gord for r in range(g * HD, (g + 1) * HD)]
    return dict(head_trace=ht, group_order=gord, head_order=heads, cols=cols, qrows=qrows, kvrows=kvrows)


def make(ck, hs, oc, ow):
    os.makedirs(oc, exist_ok=True)
    hd = f"{ow}/challenge/hessians/calib"; os.makedirs(hd, exist_ok=True)
    P = {}
    for l in ATTN:
        H, _, _ = hess_full(f"{hs}/{L}.{l}.self_attn.o_proj.input.safetensors")
        assert H.shape == (NQ * HD, NQ * HD), H.shape
        P[l] = order(H.diagonal())
    ix = json.load(open(f"{ck}/model.safetensors.index.json"))["weight_map"]
    for f in os.listdir(ck):                                                  # every non-weight file: linked
        if not f.endswith(".safetensors") and not os.path.lexists(f"{oc}/{f}"): os.symlink(os.path.realpath(f"{ck}/{f}"), f"{oc}/{f}")
    nperm = 0
    for wf in sorted(set(ix.values())):
        with safe_open(f"{ck}/{wf}", "pt", device="cpu") as f:
            md = f.metadata(); T = {k: f.get_tensor(k) for k in f.keys()}
        for l, p in P.items():
            a = f"{L}.{l}.self_attn"
            if f"{a}.q_proj.weight" not in T: continue
            T[f"{a}.q_proj.weight"] = T[f"{a}.q_proj.weight"][p["qrows"]].contiguous()
            T[f"{a}.k_proj.weight"] = T[f"{a}.k_proj.weight"][p["kvrows"]].contiguous()
            T[f"{a}.v_proj.weight"] = T[f"{a}.v_proj.weight"][p["kvrows"]].contiguous()
            T[f"{a}.o_proj.weight"] = T[f"{a}.o_proj.weight"][:, p["cols"]].contiguous()
            assert not any(k.startswith(f"{a}.") and k.endswith(".bias") for k in T), "attention biases: not handled"
            nperm += 1
        save_file(T, f"{oc}/{wf}.part", metadata=md); os.replace(f"{oc}/{wf}.part", f"{oc}/{wf}")
    assert nperm == len(P), (nperm, len(P))
    for f in os.listdir(hs):
        if not os.path.lexists(f"{hd}/{f}"): os.symlink(os.path.realpath(f"{hs}/{f}"), f"{hd}/{f}")
    for l, p in P.items():
        n = f"{L}.{l}.self_attn.o_proj.input.safetensors"
        H, meta, dt = hess_full(f"{hs}/{n}")
        c = torch.tensor(p["cols"]); Hp = H[c][:, c]
        iu = torch.triu_indices(Hp.shape[0], Hp.shape[0])
        os.remove(f"{hd}/{n}")
        save_file({"hessian_upper": Hp[iu[0], iu[1]].to(dt).contiguous()}, f"{hd}/{n}.part", metadata=meta)
        os.replace(f"{hd}/{n}.part", f"{hd}/{n}")
    json.dump({"ckpt": os.path.abspath(ck), "hess": os.path.abspath(hs), "order": "rev (ascending; heaviest stored last)",
               "layers": {str(l): p for l, p in P.items()}}, open(f"{oc}/perm06.json", "w"))
    for l, p in P.items():
        print(f"L{l}: head traces {[round(x, 1) for x in p['head_trace']]} -> groups {p['group_order']}, heads {p['head_order']}")
    print(f"PERM06 MAKE DONE {len(P)} layers -> {oc}, {hd}")


def unperm(pj, i, o):
    P = json.load(open(pj))["layers"]; os.makedirs(o + ".part", exist_ok=True); n = 0
    for l, p in P.items():
        k = f"{L}.{l}.self_attn.o_proj"
        w = load_file(f"{i}/{k}.safetensors")["weight"]
        out = torch.empty_like(w); out[:, p["cols"]] = w
        save_file({"weight": out.contiguous()}, f"{o}.part/{k}.safetensors"); n += 1
    shutil.rmtree(o, ignore_errors=True); os.replace(o + ".part", o)
    print(f"PERM06 UNPERM {n} o_proj -> {o}")


def permwq(pj, i, o):
    P = json.load(open(pj))["layers"]; os.makedirs(o + ".part", exist_ok=True); n = 0
    for f in sorted(os.listdir(i)):
        if not f.endswith(".safetensors"): continue
        key = f[:-len(".safetensors")]; l = key.split(".")[3] if key.startswith(L) else None
        if l in P and ".self_attn." in key:
            p = P[l]; w = load_file(f"{i}/{f}")["weight"]; t = key.rsplit(".", 1)[1]
            w = {"q_proj": lambda: w[p["qrows"]], "k_proj": lambda: w[p["kvrows"]], "v_proj": lambda: w[p["kvrows"]],
                 "o_proj": lambda: w[:, p["cols"]]}[t]()
            save_file({"weight": w.contiguous()}, f"{o}.part/{f}"); n += 1
        else: os.symlink(os.path.realpath(f"{i}/{f}"), f"{o}.part/{f}")
    shutil.rmtree(o, ignore_errors=True); os.replace(o + ".part", o)
    print(f"PERM06 PERMWQ {n} attention tensors permuted -> {o}")


if __name__ == "__main__":
    c = sys.argv[1]
    {"make": make, "unperm": unperm, "permwq": permwq}[c](*sys.argv[2:])
