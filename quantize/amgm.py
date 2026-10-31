#!/usr/bin/env python3
"""amgm.py CKPT_DIR OUT.jsonl [LAYERS] -- would finer scales help EXL3? (O2 diagnostic, research-quant-sota.md)

EXL3 quantizes W' = had_n(sv * had_k(su * W)) (blockwise 128 Hadamard on both dims, random signs) with one scale per
row (svh) and per column (suh) and no finer gain. For Gaussian sources at high rate, a separate scale per group saves
0.5 * log2(AM/GM of the group variances) bits per weight against one shared scale (reverse water-filling). This
measures that, per tensor, in the rotated domain after a per-row and per-column RMS normalisation (Sinkhorn-style, 4
rounds: what svh/suh can absorb), for two group shapes:
  tile16  16x16 tiles (EXL3's code unit)
  row128  1x128 runs along the input dimension (IQR-V2's E4M3 group scales)
Also reported: the same before row/col normalisation (rotation only). Unweighted (no Hessian): an upper-bound-ish
screen; low-rate (K<=2) gains are smaller than the high-rate formula.
LAYERS: python range expression, default range(0, 64, 3) + [63]. Reads bf16 safetensors directly (no torch)."""
import json, os, struct, sys, time
import numpy as np

D, OUT = sys.argv[1], sys.argv[2]
LAYERS = eval(sys.argv[3]) if len(sys.argv) > 3 else list(range(0, 64, 3)) + [63]
wm = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
hdr = {}
def tensor(name):
    f = f"{D}/{wm[name]}"
    if f not in hdr:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr[f] = (8 + n, json.loads(fh.read(n)))
    off, h = hdr[f]; t = h[name]; assert t["dtype"] == "BF16", t["dtype"]
    a, b = t["data_offsets"]
    u = np.memmap(f, np.uint16, "r", offset=off + a, shape=tuple(t["shape"]))
    return (np.asarray(u).astype(np.uint32) << 16).view(np.float32)

H = np.array([[1.0]], np.float32)
while H.shape[0] < 128: H = np.block([[H, H], [H, -H]])
H /= np.sqrt(128)
rng = np.random.default_rng(0)
def rotate(W):                        # W (out, in): random signs + blockwise Hadamard on both dims
    o, i = W.shape
    W = W * rng.choice([-1.0, 1.0], i).astype(np.float32)[None, :]
    W = (W.reshape(o, i // 128, 128) @ H).reshape(o, i)
    W = W * rng.choice([-1.0, 1.0], o).astype(np.float32)[:, None]
    W = (H @ W.reshape(o // 128, 128, i)).reshape(o, i) if o % 128 == 0 else W
    return W
def rcnorm(W, rounds=4):
    for _ in range(rounds):
        W = W / np.sqrt((W * W).mean(1, keepdims=True) + 1e-30)
        W = W / np.sqrt((W * W).mean(0, keepdims=True) + 1e-30)
    return W
def amgm_bits(v, n):                  # 0.5 log2(AM/GM) of group variances, minus the sampling bias of n-sample
    v = v[v > 0].astype(np.float64)      # variance estimates (E ln chi2_n/n ~ -1/n -> 0.5/(n ln2) bits): 0 = uniform
    return 0.5 * (np.log2(v.mean()) - np.log2(v).mean()) - 0.5 / (n * np.log(2))
def groups(W):
    o, i = W.shape; S = W * W
    t16 = S[: o // 16 * 16].reshape(o // 16, 16, i // 16, 16).mean((1, 3)).ravel()
    r128 = S.reshape(o, i // 128, 128).mean(2).ravel()
    return t16, r128

L = "model.language_model.layers"
def plan(l):
    full = l % 4 == 3
    p = f"{L}.{l}"
    if full: names = [("attn_q", f"{p}.self_attn.q_proj"), ("attn_k", f"{p}.self_attn.k_proj"), ("attn_v", f"{p}.self_attn.v_proj"),
                      ("attn_out", f"{p}.self_attn.o_proj")]
    else: names = [("gdn_qkv", f"{p}.linear_attn.in_proj_qkv"), ("gdn_z", f"{p}.linear_attn.in_proj_z"),
                   ("gdn_out", f"{p}.linear_attn.out_proj")]
    return names + [("mlp_gate", f"{p}.mlp.gate_proj"), ("mlp_up", f"{p}.mlp.up_proj"), ("mlp_down", f"{p}.mlp.down_proj")]
jobs = [(l, g, n) for l in LAYERS for g, n in plan(l)] + [(64, "head", "lm_head")]
done = set()
if os.path.exists(OUT):
    for x in open(OUT): r = json.loads(x); done.add((r["layer"], r["group"]))
out = open(OUT, "a")
for l, g, n in jobs:
    if (l, g) in done: continue
    t0 = time.time()
    W = rotate(tensor(n + ".weight"))
    a16, a128 = groups(W)
    Wn = rcnorm(W); b16, b128 = groups(Wn)
    r = {"layer": l, "group": g, "shape": list(W.shape),
         "rot_tile16": amgm_bits(a16, 256), "rot_row128": amgm_bits(a128, 128),
         "rc_tile16": amgm_bits(b16, 256), "rc_row128": amgm_bits(b128, 128), "s": round(time.time() - t0, 1)}
    out.write(json.dumps(r) + "\n"); out.flush()
    print(json.dumps(r), flush=True)
