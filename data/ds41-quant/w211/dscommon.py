"""dscommon.py -- loaders + chunked LDLQ for the DS4.1-F routed-expert quantizer harness (w211).

Weights: MXFP4 experts of /data/ds41ck-full (e2m1 nibbles, low nibble = even K index, one e8m0 scale per 1x32 along K),
dequantized exactly (verify_dequant.py checks it against the reference model's kernel).
Hessians: Hero422/deepseek-v4.1-flash-1017-quant-calib (experts/L##-s?.safetensors, dense/L##.ffn_in.safetensors).
Quantizer: v3/e3's trellis (`E3DIR`): 128-block Hadamard rotation with seeded signs (e3.had_blocks), unit-RMS rows,
LDLQ with relative damping, trellis.vit (viterbi3.c with E3_V2=3, E3_TB=1), metric e3.rel (H-weighted relative error).
"""
import json, os, sys
import numpy as np
import torch

E3DIR = os.environ.get("W211_E3DIR", "/data/q27/w211/code/e3")
CKPT = os.environ.get("W211_CKPT", "/data/ds41ck-full")
CALIB = os.environ.get("W211_CALIB", "/data/q27/w211/calib")
HFSCRIPTS = os.environ.get("W211_HFSCRIPTS", "/data/q27/w211/hf/scripts")
sys.path.insert(0, E3DIR); sys.path.insert(0, HFSCRIPTS)
os.environ.setdefault("E3_V2", "3"); os.environ.setdefault("E3_TB", "1")
torch.set_grad_enabled(False)

FP4_TABLE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
_WMAP = None


def _weight_map():
    global _WMAP
    if _WMAP is None:
        _WMAP = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    return _WMAP


def _get(key):
    from safetensors import safe_open
    with safe_open(f"{CKPT}/{_weight_map()[key]}", "pt") as f:
        return f.get_tensor(key)


def dequant_fp4(packed_i8, scale_e8m0):
    """[N, K/2] packed e2m1 (int8 / uint8 view) + [N, K/32] e8m0 scale -> float32 [N, K], exact."""
    u = packed_i8.view(torch.uint8).numpy()
    tab = FP4_TABLE
    w = np.stack([tab[u & 0x0F], tab[u >> 4]], axis=-1).reshape(u.shape[0], -1)       # low nibble = even index
    sb = scale_e8m0.view(torch.uint8).numpy().astype(np.int32)
    s = np.ldexp(np.float32(1.0), sb - 127).astype(np.float32)                         # e8m0: 2^(b-127) (255 = NaN, absent)
    assert (sb != 255).all()
    return (w.reshape(w.shape[0], -1, 32) * s[:, :, None]).reshape(w.shape[0], -1)


def load_expert_w(layer, e, kind):
    """kind gate_up: w1 rows then w3 rows [4608, 5120]; down: w2 [5120, 2304]. float64 torch."""
    names = ["w1", "w3"] if kind == "gate_up" else ["w2"]
    out = []
    for n in names:
        k = f"layers.{layer}.ffn.experts.{e}.{n}"
        out.append(dequant_fp4(_get(k + ".weight"), _get(k + ".scale")))
    return torch.from_numpy(np.concatenate(out, 0)).double()


def load_expert_h(layer, e, kind):
    """(H [k,k] float64, stats dict) of one expert from the layer shard (h13 for gate_up, h2 for down)."""
    from safetensors import safe_open
    from hesstri import unpack
    idx = json.load(open(f"{CALIB}/experts/index.json"))
    sh = idx[f"L{layer:02d}.e{e:03d}"]
    with safe_open(f"{CALIB}/experts/{sh}" if "/" not in sh else f"{CALIB}/{sh}", "np") as f:
        t = f.get_tensor(f"e{e:03d}.h13_triu" if kind == "gate_up" else f"e{e:03d}.h2_triu")
        n, n_routed, p, wsum = f.get_tensor(f"e{e:03d}.stats")
    return torch.from_numpy(unpack(t)).double(), {"n": int(n), "n_routed": int(n_routed), "p": float(p), "wsum": float(wsum)}


def load_pooled_h(layer):
    from safetensors import safe_open
    from hesstri import unpack
    with safe_open(f"{CALIB}/dense/L{layer:02d}.ffn_in.safetensors", "np") as f:
        k = int(f.metadata()["k"]); t = f.get_tensor("hessian_upper")
    return torch.from_numpy(unpack(t, k)).double()


def prep_ldlq(H, damp):
    """trellis.ldlq_trellis' setup (dead-column fix, relative damping, U = chol(H^-1) upper) done once; per-block U^-1."""
    H = H.clone(); dead = H.diagonal() == 0
    H[dead, dead] = 1
    H.diagonal().add_(damp * H.diagonal().mean())
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Kd = H.shape[0]
    Uinv = [torch.linalg.inv(U[a:a + 16, a:a + 16]) for a in range(0, Kd, 16)]
    return {"U": U, "Uinv": Uinv, "dead": dead}


def ldlq_prepared(W, pre, cb, K):
    """trellis.ldlq_trellis on a row chunk W [R, Kd] (R % 16 == 0) with a prepared (U, Uinv): same operations."""
    import trellis
    W = W.clone(); R, Kd = W.shape
    W[:, pre["dead"]] = 0
    U = pre["U"]; Q = torch.zeros_like(W)
    for bi, a in enumerate(range(0, Kd, 16)):
        blk = W[:, a:a + 16]
        tiles = blk.reshape(R // 16, 16, 16).reshape(R // 16, 256).numpy()
        q = torch.from_numpy(trellis.vit(tiles, cb, K).astype(np.float64)).reshape(R, 16)
        Q[:, a:a + 16] = q
        if a + 16 < Kd:
            E = (blk - q) @ pre["Uinv"][bi]
            W[:, a + 16:] -= E @ U[a:a + 16, a + 16:]
    return Q


def rel_err(W, Q, H):
    D = W - Q
    return float(torch.einsum("ij,jk,ik->", D, H, D) / torch.einsum("ij,jk,ik->", W, H, W))
