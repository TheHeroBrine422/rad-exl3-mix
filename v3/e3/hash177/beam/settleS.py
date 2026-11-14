#!/usr/bin/env python3
"""settleS.py LAYER TENSOR K OUT [ROWS] -- settle177.py (w71's settle71.py + #177's hash codebooks) with the seeded
L-bit searches of beamS.c (lead notes 11-13 1:45 / 2:00 PM CST). New arm names (W71_ARMS), all at V16's g_scale
("@m" multiplier as settle71's), tail-biting:
  S<L>-<M>v<Ls>        seeded beam: exact Viterbi at Ls -> beam at L with M survivors keeping the seed path
  R<L>-<NB>v<Ls>       segment refine of the exact Ls path at L, NB free bits a segment
  Q<L>-<M>-<NB>v<Ls>   seeded beam, then segment refine
settle71's own arms (V<L>, B<L>-<M>, ...) unchanged. Codebook names as settle177.py (mad = hash:0xB5297B:0x2545F491:
0x3F3F3F3F). Env as settle71 plus W139_LIBS (libbeamS.so). (w139, #177)"""
import ctypes, os, re, sys
import numpy as np

_h = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_h))          # settle177.py's directory (it sets up the w71 paths)
import settle177                                  # noqa: F401  (patches the codebooks)
import settle71, trellisL as TL

_lib = None
def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(os.environ.get("W139_LIBS") or os.path.join(_h, "libbeamS.so"))
        _lib.trellis_tiles_seeded.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                              ctypes.c_void_p] + [ctypes.c_int] * 5
    return _lib

_parse0, _search0 = settle71.parse, TL.search
RX = re.compile(r"^([SRQ])(\d+)-(\d+)(?:-(\d+))?v(\d+)$")


def parse(arm):
    a, m = (arm.split("@") + [None])[:2]
    g = RX.match(a)
    if not g: return _parse0(arm)
    k, L, p1, p2, Ls = g.group(1), int(g.group(2)), int(g.group(3)), g.group(4), int(g.group(5))
    M, NB = {"S": (p1, 0), "R": (0, p1), "Q": (p1, int(p2 or 0))}[k]
    return "B", L, ("seed", {"S": 1, "R": 2, "Q": 3}[k], M, NB, Ls), float(m) if m else 1.0


def search(x, cb, L, K, M=0, tb=1, want_cost=False):
    if not isinstance(M, tuple): return _search0(x, cb, L, K, M, tb, want_cost)
    _, mode, Mb, NB, Ls = M
    cbn = os.environ.get("W71_CB") or ("mul1_i8" if K < 6 else "mul1")
    cbs = np.ascontiguousarray(TL.codebook_L(cbn, Ls), dtype=np.float32)
    x = np.ascontiguousarray(x, dtype=np.float32); out = np.empty_like(x)
    cost = np.empty(x.shape[0], dtype=np.float32); sc = np.empty(x.shape[0], dtype=np.float32)
    cb = np.ascontiguousarray(cb, dtype=np.float32)
    lib().trellis_tiles_seeded(x.ctypes.data, out.ctypes.data, cost.ctypes.data, sc.ctypes.data, x.shape[0],
                               cb.ctypes.data, L, cbs.ctypes.data, Ls, int(K), mode, int(Mb), int(NB))
    SEED["sse"] += float(sc.astype(np.float64).sum()); SEED["n"] += x.size
    return (out, cost) if want_cost else out


SEED = {"sse": 0.0, "n": 0}
settle71.parse = parse; TL.search = search

if __name__ == "__main__":
    settle71.main()
    if SEED["n"]: print(f"seed mse under the arm's L table (last arm): {SEED['sse'] / SEED['n']:.6g}")
