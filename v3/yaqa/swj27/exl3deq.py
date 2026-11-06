#!/usr/bin/env python3
"""exl3deq.py -- an ourq.py level (q-L*.safetensors + model.safetensors.index.json + stamp.json) back to the weights it
holds, in torch on the GPU or CPU: volunteer/checker/exl3dec.py (numpy; fractional K, the 0-bit salt, mul1 / i8 / i8c /
i4) vectorised for torch. v3/sib27/deq.py does the same for integer K only; the 27B swap arms are K2.5 (qtf2), so this
one follows exl3dec's ring plan:

  W (k = in, n = out) = diag(suh) . Hb . W_hat . Hb . diag(svh)   (exllamav3's order: had over k, x suh, had over n, x svh)
  W_hat: 16x16 tiles, each a tail-biting ring of 16 K bits in 32-bit little-endian words read MSB first; ring position i
  reads D(i) = KA + bit (i mod 16) of MASK new bits, its state is the 16-bit window ending at S(i) = D(0) + .. + D(i).

The 64-bit word pair of exl3dec is split in two int64 shifts (no value above 2^47), so the states are exact; the codebook
table is exl3dec's own (imported). W is formed in f32 (no f16 rounding: what the kernels compute with fp16 suh / svh).

CLI:
  exl3deq.py vsnumpy LEVEL [N]       W_hat and W of N tensors (default 6) against exl3dec.py on the CPU: W_hat bit-exact
  exl3deq.py check LEVEL WQDIR [N]   against ourq --save-wq files (differ by f16 suh / svh and storage: ~1e-3 relative)
  exl3deq.py wq LEVEL OUT [KEYFILE]  OUT/<key>.safetensors "weight" (out, in) f16, --save-wq's layout (all keys by default)
Env DEV (default cuda if available)."""
import json, os, sys
import numpy as np, torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "volunteer", "checker"))
import exl3dec as X  # noqa: E402


def plan(K, dev):
    """exl3dec.ring_plan as torch tensors: hi / lo word index, the two shifts, salt, element -> ring position"""
    nw, w0, w1, sh, salt, elem = X.ring_plan(K)
    sh = sh.astype(np.int64)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(dev)
    return nw, t(w0), t(w1), t(sh), t(salt.astype(np.int64)), t(np.argsort(elem))


def decode_hat(tr, K, table, chunk=1 << 16):
    """trellis int16 [kt, nt, 16 K] -> W_hat f32 [16 kt, 16 nt] on tr's device; table = the codebook's f32 [65536]"""
    kt, nt, words = tr.shape
    assert words == round(16 * K), (tuple(tr.shape), K)
    dev = tr.device
    nw, w0, w1, sh, salt, inv = plan(K, dev)
    hi_s = (sh - 32).clamp(min=0); lo_l = (32 - sh).clamp(min=0); lo_r = sh.clamp(max=31); big = sh >= 32
    u = tr.reshape(kt * nt, words).to(torch.int64) & 0xFFFF
    out = torch.empty(kt * nt, 256, dtype=torch.float32, device=dev)
    for c in range(0, kt * nt, chunk):
        t = u[c:c + chunk]
        w = t[:, 0::2] | (t[:, 1::2] << 16)                                  # little-endian u32 words, < 2^32
        a, b = w[:, w0], w[:, w1]                                            # the window's first and next word
        v = torch.where(big, a >> hi_s, (a << lo_l) | (b >> lo_r))           # = ((a << 32) | b) >> sh, low 16 bits
        st = (v & 0xFFFF) ^ salt
        out[c:c + chunk] = table[st][:, inv]                                 # tile elements r * 16 + c (r = k)
    return out.reshape(kt, nt, 16, 16).permute(0, 2, 1, 3).reshape(kt * 16, nt * 16)


def hadamard128(dev):
    return torch.from_numpy(X.had128()).to(dev)


def weight(tr, suh, svh, K, table):
    """W f32 [k, n] = (in, out), exl3dec.decode's order"""
    w = decode_hat(tr, K, table)
    k, n = w.shape; H = hadamard128(w.device)
    w = (H @ w.reshape(k // 128, 128, n)).reshape(k, n)
    w *= suh.float()[:, None]
    w = (w.reshape(k, n // 128, 128) @ H).reshape(k, n)
    w *= svh.float()[None, :]
    return w


def stamp_codebook(d):
    """the level's codebook from ourq's stamp.json (last run), as the exl3i importer reads it"""
    st = f"{d}/stamp.json"
    if not os.path.exists(st): sys.exit(f"{d}: no stamp.json (which codebook?)")
    return json.load(open(st))[-1]["args"].get("codebook") or "mul1"


class Level:
    """an ourq.py export; .get(key) -> W (out, in) f32 on dev (the checkpoint's orientation)"""
    def __init__(self, d, dev="cpu", codebook=None):
        self.d, self.dev = d, torch.device(dev)
        self.ix = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
        self.codebook = codebook or stamp_codebook(d)
        self.table = torch.from_numpy(X.lut(self.codebook)).to(self.dev)
    def keys(self): return sorted(k[:-len(".trellis")] for k in self.ix if k.endswith(".trellis"))
    def raw(self, key):
        assert f"{key}.mul1" in self.ix, f"{key}: no mul1 marker (mcg / 3INST): not decodable here"
        out = {}
        for s in ("trellis", "suh", "svh"):
            with safe_open(f"{self.d}/{self.ix[f'{key}.{s}']}", "pt", device="cpu") as f: out[s] = f.get_tensor(f"{key}.{s}")
        return out
    def K(self, key):
        with safe_open(f"{self.d}/{self.ix[f'{key}.trellis']}", "pt", device="cpu") as f:
            return f.get_slice(f"{key}.trellis").get_shape()[2] / 16
    def get(self, key):
        t = self.raw(key)
        K = t["trellis"].shape[2] / 16
        return weight(t["trellis"].to(self.dev), t["suh"].to(self.dev), t["svh"].to(self.dev), K, self.table).T


if __name__ == "__main__":
    from safetensors.torch import save_file
    dev = os.environ.get("DEV", "cuda" if torch.cuda.is_available() else "cpu")
    cmd = sys.argv[1]
    if cmd == "vsnumpy":
        lv = Level(sys.argv[2], dev="cpu"); N = int(sys.argv[3]) if len(sys.argv) > 3 else 6
        keys = lv.keys(); keys = [keys[i] for i in np.linspace(0, len(keys) - 1, min(N, len(keys))).astype(int)]
        bad = 0
        for k in keys:
            t = lv.raw(k); K = t["trellis"].shape[2] / 16
            ref_hat = X.decode_what(t["trellis"].numpy(), K, lv.codebook)
            hat = decode_hat(t["trellis"], K, lv.table).numpy()
            ref_w, _ = X.decode({s: v.numpy() for s, v in t.items()}, K, lv.codebook)
            w = lv.get(k).T.numpy()
            nh = int((hat != ref_hat).sum()); rw = float(np.abs(w - ref_w).max() / np.abs(ref_w).max())
            bad += nh != 0 or rw > 1e-5
            print(f"{k:60s} K{K:g} {lv.codebook}: W_hat differing {nh}, W max rel {rw:.1e}", flush=True)
        print("EXL3DEQ VSNUMPY: PASS" if bad == 0 else f"EXL3DEQ VSNUMPY FAIL: {bad} tensors")
    elif cmd == "check":
        lv, wqd = Level(sys.argv[2], dev=dev), sys.argv[3]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
        keys = [k for k in lv.keys() if os.path.exists(f"{wqd}/{k}.safetensors")][:N]
        worst = 0.0
        for k in keys:
            with safe_open(f"{wqd}/{k}.safetensors", "pt", device="cpu") as f: q = f.get_tensor("weight").float().to(lv.dev)
            r = float((lv.get(k) - q).norm() / q.norm()); worst = max(worst, r)
            print(f"{k:60s} K{lv.K(k):g} rel {r:.2e}", flush=True)
        print(f"EXL3DEQ CHECK {len(keys)} tensors, worst rel {worst:.2e} ({lv.codebook}): {'PASS' if keys and worst < 2e-3 else 'FAIL'}")
    elif cmd == "wq":
        lv, out = Level(sys.argv[2], dev=dev), sys.argv[3]
        keys = [l.strip() for l in open(sys.argv[4]) if l.strip() and not l.startswith("#")] if len(sys.argv) > 4 else lv.keys()
        os.makedirs(out, exist_ok=True)
        for key in keys:
            save_file({"weight": lv.get(key).half().contiguous().cpu()}, f"{out}/{key}.safetensors.part",
                      metadata={"source": f"exl3deq.py {lv.d} codebook {lv.codebook}"})
            os.replace(f"{out}/{key}.safetensors.part", f"{out}/{key}.safetensors")
        print(f"EXL3DEQ WQ {len(keys)} tensors -> {out}")
    else:
        sys.exit(__doc__)
