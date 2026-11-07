#!/usr/bin/env python3
"""hashdiag.py NTILES K[,K..] OUT.jsonl -- w71 (#113) synthetic check: does the trellis error fall with the state
window L for a given computed code? iid N(0,1) tiles (seed 0), exllamav3's tail-biting heuristic, MSE at the best
of a small scale grid (per code, L and search). Codes (each a function of the L-bit state s, standardised):
  mul1     today's: bytesum(s * 0x83DCD12D mod 2^32), on the int8 grid (mul1_i8), generalised by the window only
  mul1xs   mul1 with one xorshift after the multiply: h = s * 0x83DCD12D; h ^= h >> 16; bytesum(h), int8 grid
           (+2 VALU a value: a shift and an xor)
  mad      IQR's computed code (radiance libquant tcq_code=mad, default A = 0xB5297B, B = 0x2545F491):
           bytesum(((s & 0xFFFFFF) * A + B) & 0x3F3F3F3F) - 126
  rand     an iid Gaussian table on the int8 grid (no cheap decoder; the ideal a hash could reach)
Searches (env CONFIGS, L:M list, M = 0 full Viterbi): default full Viterbi at L = 16 / 18, the beam (M = 4096) at
L = 16 / 18 / 20. SCALES: the scale grid (default 0.8 .. 1.0)."""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trellisL as T


def i8(v):
    v = (v - v.mean()) / v.std(); st = 3.5 / 127
    return np.clip(np.round(v / st), -127, 127) * st


def code(name, L):
    s = np.arange(1 << L, dtype=np.uint64)
    bsum = lambda h: sum(((h >> (8 * i)) & 0xFF) for i in range(4)).astype(np.float64)
    if name == "mul1": return T.codebook_L("mul1_i8", L)
    if name == "mul1xs":
        h = (s * T.MUL) & 0xFFFFFFFF; h ^= h >> 16
        v = (bsum(h) - T.bytesum(16).mean()) / T.bytesum(16).std(); st = 3.5 / 127
        return np.clip(np.round(v / st), -127, 127).astype(np.float32) * np.float32(st)
    if name == "mad":
        h = (((s & 0xFFFFFF) * 0xB5297B) + 0x2545F491) & 0xFFFFFFFF & 0x3F3F3F3F
        v = bsum(h) - 126; return (v / 37.0).astype(np.float32)
    if name == "rand": return i8(np.random.default_rng(7).standard_normal(1 << L)).astype(np.float32)
    raise SystemExit(name)


def main():
    n, Ks, out = int(sys.argv[1]), [int(k) for k in sys.argv[2].split(",")], sys.argv[3]
    X = np.random.default_rng(0).standard_normal((n, 256)).astype(np.float32)
    codes = os.environ.get("CODES", "mul1,mul1xs,mad,rand").split(",")
    for K in Ks:
        for name in codes:
            cf = os.environ.get("CONFIGS", "16:0,18:0,16:4096,18:4096,20:4096")
            for L, M in [tuple(int(v) for v in c.split(":")) for c in cf.split(",")]:
                cb = np.ascontiguousarray(code(name, L), dtype=np.float32)
                cb = cb / np.float32(np.sqrt((cb.astype(np.float64) ** 2).mean()))
                t0 = time.time(); best = None
                scs = [float(v) for v in os.environ.get("SCALES", "0.8,0.85,0.9,0.95,1.0").split(",")]
                for sc in scs:
                    q = T.search(X * sc, cb, L, K, M, 1) / sc; e = float(((q - X) ** 2).mean())
                    if best is None or e < best[0]: best = (e, sc)
                o = {"K": K, "code": name, "L": L, "M": M, "mse": best[0], "scale": best[1], "n": n,
                     "sec_tile": (time.time() - t0) / (len(scs) * n)}
                print(json.dumps(o), flush=True)
                with open(out, "a") as f: f.write(json.dumps(o) + "\n")


if __name__ == "__main__":
    main()
