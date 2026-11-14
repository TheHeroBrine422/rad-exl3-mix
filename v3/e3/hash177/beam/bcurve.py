#!/usr/bin/env python3
"""bcurve.py K CODE ARMS OUT [NT] -- #177 (w139) synthetic search curve: NT (default 64) iid N(0,1) 16x16 tiles (seed 0,
the same tiles for every arm and code), tail-biting, MSE and seconds a tile per arm. CODE: mad (IQR's tcq_code=mad, A
0xB5297B) / hash (A 0xd0474b) / mul1i8; values standardised over the 16-bit states (mul1i8 on its int8 grid) times a gain
g picked per (code, K) by a V16 sweep (0.85 .. 1.15, same for every L / arm: the gain is a per-tensor scale in use).
ARMS: V<L> exact Viterbi, B<L>-<M> beam (beamL.c), P<L>-<M>-<NB> beam then segment refine (beamS.c mode 6).
Libraries next to this file (libbeamL.so, libbeamS.so). One JSON line an arm appended to OUT; arms already there are
skipped (restartable). (w139, #177)"""
import ctypes, json, os, re, sys, time
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
BL = ctypes.CDLL(os.path.join(H, "libbeamL.so")); BS = ctypes.CDLL(os.path.join(H, "libbeamS.so"))
BL.trellis_tiles.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_void_p] + [ctypes.c_int] * 4
BS.trellis_tiles_seeded.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p] + [ctypes.c_int] * 5


def table(code, L):
    s = np.arange(1 << L, dtype=np.uint64); s16 = np.arange(1 << 16, dtype=np.uint64)
    if code in ("mad", "hash"):
        A = 0xB5297B if code == "mad" else 0xd0474b
        f = lambda t: sum((((t * np.uint64(A) + np.uint64(0x2545F491)) & np.uint64(0x3F3F3F3F)) >> np.uint64(8 * i)) & np.uint64(0xFF) for i in range(4)).astype(np.float64)
        v16 = f(s16); return (f(s) - v16.mean()) / v16.std()
    f = lambda t: sum((((t * np.uint64(0x83DCD12D)) & np.uint64(0xFFFFFFFF)) >> np.uint64(8 * i)) & np.uint64(0xFF) for i in range(4)).astype(np.float64)
    v16 = f(s16); v = (f(s) - v16.mean()) / v16.std(); st = 3.5 / 127
    return np.clip(np.round(v / st), -127, 127) * st


def run(x, cb, L, K, kind, M=0, NB=0):
    x = np.ascontiguousarray(x, dtype=np.float32); out = np.empty_like(x); c = np.empty(len(x), np.float32)
    cb = np.ascontiguousarray(cb, dtype=np.float32); t = time.time()
    if kind in "VB": BL.trellis_tiles(x.ctypes.data, out.ctypes.data, None, c.ctypes.data, len(x), cb.ctypes.data, L, K, M, 1)
    else:
        sc = np.empty_like(c)
        BS.trellis_tiles_seeded(x.ctypes.data, out.ctypes.data, c.ctypes.data, sc.ctypes.data, len(x), cb.ctypes.data, L,
                                cb.ctypes.data, L, K, 6, M, NB)
    return float(((x - out) ** 2).mean()), (time.time() - t) / len(x)


def main():
    K, code, arms, out = int(sys.argv[1]), sys.argv[2], sys.argv[3].split(","), sys.argv[4]
    nt = int(sys.argv[5]) if len(sys.argv) > 5 else 64
    x = np.random.default_rng(0).standard_normal((nt, 256)).astype(np.float32)
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try: done.add(json.loads(l)["arm"])
            except ValueError: pass
    gf = out + ".gain"
    if os.path.exists(gf): g = float(open(gf).read())
    else:
        cb16 = table(code, 16); best = None
        for gg in np.arange(0.85, 1.151, 0.025):
            m, _ = run(x[:16], cb16 * gg, 16, K, "V")
            if best is None or m < best[0]: best = (m, gg)
        g = float(best[1]); open(gf, "w").write(str(g))
    tabs = {}
    for arm in arms:
        if arm in done: continue
        m_ = re.match(r"^([VBP])(\d+)(?:-(\d+))?(?:-(\d+))?$", arm); kind, L = m_.group(1), int(m_.group(2))
        M = int(m_.group(3) or 0); NB = int(m_.group(4) or 0)
        if L not in tabs: tabs[L] = table(code, L) * g
        mse, sec = run(x, tabs[L], L, K, kind, M, NB)
        o = {"code": code, "K": K, "arm": arm, "L": L, "M": M, "NB": NB, "gain": g, "mse": mse, "sec_tile": sec, "tiles": nt}
        with open(out, "a") as f: f.write(json.dumps(o) + "\n")
        print(json.dumps(o), flush=True)


if __name__ == "__main__":
    main()
