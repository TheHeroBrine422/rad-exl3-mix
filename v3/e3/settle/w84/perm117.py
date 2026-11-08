#!/usr/bin/env python3
"""perm117.py -- #117 (../w84.md): one permutation P of the 40 residual 128-blocks of Qwen3.8-27B, folded into the
stored tensors (an exact transform: RMSNorm is permutation-equivariant, whole 128-blocks keep the Hadamard / A8
groups). ORDER.json (w82 `order82.json`): "stored_order"[p] = the natural block stored at position p (heaviest last).
Index form: idx[p * 128 + i] = stored_order[p] * 128 + i, so permuted = natural[..., idx] along a residual axis.

  perm117.py plan CKPT                       classify every tensor of CKPT (permuted axes / untouched); refuses a
                                             tensor with a 5120 dimension that no rule names
  perm117.py ckpt ORDER CKPT OUT             the converted checkpoint (same shards, names, dtypes, metadata; every
                                             non-safetensors file copied; OUT/perm117.json = order + idx)
  perm117.py hess ORDER HESS OUT             calib input Hessians: residual inputs (layers.N.input,
                                             layers.N.post_attention.input, lm_head.input) re-indexed (rows and columns
                                             of the packed upper triangle, same dtype / metadata), the rest symlinked
  perm117.py hout ORDER HOUT OUT             H_out files (<key>.safetensors, "hout_diag" or "hout"): the writers'
                                             (residual rows) re-indexed, the readers' symlinked
Not in the checkpoint: DFlash2 (a separate draft model sharing embedding / head) -- its fc input columns per tapped
hidden state (5 taps) and its whole residual dimension take the same rules (or it keeps unpermuted private copies of
the embedding and head)."""
import json, os, re, shutil, struct, sys
import torch

D = 5120                       # hidden size: the residual stream
B = 128                        # block


def load_order(path):
    so = json.load(open(path))["stored_order"]
    assert sorted(so) == list(range(D // B)), "stored_order is not a permutation of the 40 blocks"
    idx = torch.cat([torch.arange(b * B, (b + 1) * B) for b in so])
    return so, idx


# ---- which axes of which tensor are residual. (regex, axes): axis -1 = input columns of a linear (out, in), 0 = rows.
#      "fc_in" = both 5120 halves of mtp.fc's input (embedding half | hidden half), each permuted the same way.
L = r"(model\.language_model\.layers|mtp\.layers)\.\d+\."
RULES = [
    (L + r"(input_layernorm|post_attention_layernorm)\.weight$", (0,)),
    (L + r"linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b)\.weight$", (1,)),
    (L + r"self_attn\.(q_proj|k_proj|v_proj)\.weight$", (1,)),
    (L + r"mlp\.(gate_proj|up_proj)\.weight$", (1,)),
    (L + r"(linear_attn\.out_proj|self_attn\.o_proj|mlp\.down_proj)\.weight$", (0,)),
    (r"model\.language_model\.embed_tokens\.weight$", (1,)),
    (r"model\.language_model\.norm\.weight$", (0,)),
    (r"lm_head\.weight$", (1,)),
    (r"model\.visual\.merger\.linear_fc2\.(weight|bias)$", (0,)),
    (r"mtp\.(pre_fc_norm_embedding|pre_fc_norm_hidden|norm)\.weight$", (0,)),
    (r"mtp\.fc\.weight$", (0, "fc_in")),
]
# tensors with a 5120-sized dimension that is NOT the residual stream (none in this checkpoint; listed when found)
NOT_RESIDUAL = []


def rule(name):
    for pat, ax in RULES:
        if re.match(pat, name): return ax
    return None


def plan_one(name, shape):
    ax = rule(name)
    if ax is None:
        if D in shape and not any(re.match(p, name) for p in NOT_RESIDUAL):
            raise SystemExit(f"perm117: {name} {list(shape)} has a {D} dimension and no rule (residual or not?)")
        return None
    for a in ax:
        if a == "fc_in": assert shape[1] == 2 * D, (name, shape)
        else: assert shape[a] == D, (name, shape, a)
    return ax


def permute_tensor(name, t, idx):
    """the converted tensor (a new tensor; t untouched); None = unchanged"""
    ax = plan_one(name, tuple(t.shape))
    if ax is None: return None
    for a in ax:
        if a == "fc_in": t = t.index_select(1, torch.cat([idx, idx + D]))
        else: t = t.index_select(a, idx)
    return t.contiguous()


def _hdr(p):
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; return json.loads(fh.read(n))


def ckpt(order, src, out):
    from safetensors import safe_open
    from safetensors.torch import save_file
    so, idx = load_order(order)
    os.makedirs(out, exist_ok=True)
    n_perm = n_same = 0
    for f in sorted(os.listdir(src)):
        if not f.endswith(".safetensors"):
            if os.path.isfile(f"{src}/{f}"): shutil.copy2(f"{src}/{f}", f"{out}/{f}")
            continue
        if os.path.exists(f"{out}/{f}"): continue                 # resumable: shards are written whole
        meta = _hdr(f"{src}/{f}").get("__metadata__")
        ts = {}
        with safe_open(f"{src}/{f}", "pt", device="cpu") as fh:
            for k in fh.keys():
                t = fh.get_tensor(k); p = permute_tensor(k, t, idx)
                if p is None: ts[k] = t; n_same += 1
                else: ts[k] = p; n_perm += 1
        save_file(ts, f"{out}/{f}.part", metadata=meta); os.replace(f"{out}/{f}.part", f"{out}/{f}")
        print(f"{f}: {len(ts)} tensors", flush=True)
    json.dump({"stored_order": so, "idx": idx.tolist(), "source": os.path.abspath(src)}, open(f"{out}/perm117.json", "w"))
    print(f"ckpt done: {n_perm} permuted, {n_same} unchanged (this run's shards)")


# ---- Hessians (challenge layout: "hessian_upper" = packed upper triangle row-major, metadata k / rows)
def hess_permuted(name):
    return bool(re.search(r"\.layers\.\d+\.input$", name) or name.endswith(".post_attention.input") or name == "lm_head.input")


def upper_permute(up, k, idx):
    """packed upper triangle of H -> packed upper triangle of H[idx][:, idx] (same dtype; values moved, not recomputed)"""
    iu = torch.triu_indices(k, k)
    H = torch.empty(k, k, dtype=up.dtype)
    H[iu[0], iu[1]] = up; H[iu[1], iu[0]] = up
    Hp = H.index_select(0, idx).index_select(1, idx)
    return Hp[iu[0], iu[1]].contiguous()


def hess(order, src, out):
    from safetensors import safe_open
    from safetensors.torch import save_file
    so, idx = load_order(order)
    os.makedirs(out, exist_ok=True); n = 0
    for f in sorted(os.listdir(src)):
        name = f[:-len(".safetensors")] if f.endswith(".safetensors") else None
        dst = f"{out}/{f}"
        if os.path.lexists(dst): continue
        if name is None or not hess_permuted(name):
            os.symlink(os.path.abspath(f"{src}/{f}"), dst); continue
        meta = _hdr(f"{src}/{f}")["__metadata__"]; k = int(meta["k"]); assert k == D, (f, k)
        with safe_open(f"{src}/{f}", "pt", device="cpu") as fh: up = fh.get_tensor("hessian_upper")
        save_file({"hessian_upper": upper_permute(up, k, idx)}, dst + ".part", metadata=meta); os.replace(dst + ".part", dst)
        n += 1
    json.dump({"stored_order": so, "source": os.path.abspath(src)}, open(f"{out}/perm117.json", "w"))
    print(f"hess done: {n} permuted this run")


def hout(order, src, out):
    from safetensors import safe_open
    from safetensors.torch import save_file
    so, idx = load_order(order)
    os.makedirs(out, exist_ok=True); n = 0
    for f in sorted(os.listdir(src)):
        dst = f"{out}/{f}"
        if os.path.lexists(dst): continue
        key = f[:-len(".safetensors")] if f.endswith(".safetensors") else ""
        if not re.search(r"(linear_attn\.out_proj|self_attn\.o_proj|mlp\.down_proj)$", key):
            os.symlink(os.path.abspath(f"{src}/{f}"), dst); continue
        meta = _hdr(f"{src}/{f}").get("__metadata__")
        with safe_open(f"{src}/{f}", "pt", device="cpu") as fh:
            ts = {k: fh.get_tensor(k) for k in fh.keys()}
        for k, t in ts.items():
            if k == "hout_diag": assert t.shape == (D,); ts[k] = t.index_select(0, idx).contiguous()
            elif k == "hout": assert t.shape == (D, D); ts[k] = t.index_select(0, idx).index_select(1, idx).contiguous()
            else: raise SystemExit(f"{f}: unknown tensor {k}")
        save_file(ts, dst + ".part", metadata=meta); os.replace(dst + ".part", dst); n += 1
    print(f"hout done: {n} permuted this run")


def plan(src):
    wm = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
    groups = {}
    for f in sorted(set(wm.values())):
        for k, v in _hdr(f"{src}/{f}").items():
            if k == "__metadata__": continue
            ax = plan_one(k, tuple(v["shape"]))
            g = re.sub(r"\.\d+\.", ".N.", k)
            e = groups.setdefault(g, [v["shape"], ax, 0]); e[2] += 1
    for g, (s, ax, c) in groups.items(): print(f"{'PERM ' + str(ax) if ax else 'same':16s} {c:3d} x {g} {s}")
    print(f"{sum(c for _, ax, c in groups.values() if ax)} tensors permuted, "
          f"{sum(c for _, ax, c in groups.values() if not ax)} unchanged")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "plan": plan(sys.argv[2])
    elif cmd == "ckpt": ckpt(*sys.argv[2:5])
    elif cmd == "hess": hess(*sys.argv[2:5])
    elif cmd == "hout": hout(*sys.argv[2:5])
    else: raise SystemExit(__doc__)
