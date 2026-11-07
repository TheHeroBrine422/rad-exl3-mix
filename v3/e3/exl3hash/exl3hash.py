"""exl3hash -- exllamav3's trellis search with IQR-V4's hash codebook (#100c), on the prebuilt exllamav3 1.6.3.

Value of a 16-bit state s: q = bytesum(((s * A + B) mod 2^32) & 0x3F3F3F3F) - 126, times mul1_i8's int8 step
4 k_inv (cb_hash.cuh). Default A = 0xd0474b (w27's searched global A, r100.md), B = 0x2545F491 (IQR's); env
EXL3HASH_A / EXL3HASH_B override (hex). Quantizer only: packing and storage are unchanged (ordinary EXL3 indices;
the shards still carry the mul1 marker), so only a decoder with the same map reads them as intended -- score through
the dequantized weights (ourq --save-wq + proxy2b/evalq.py).

How: a copy of the fork's exllamav3_ext (SRC) whose quant/codebook.cuh routes the mul1 decodes (decode_3inst<2>,
decode_mul1_product_2: every value the search, the traceback, the LUTs and the LDLQ feedback read) to the hash, built
through exl3cb (integer K; its i8 variant plus EXL3_CB_HASH, which bypasses the int8 rounding) and exl3frac2 (qtf2:
fractional K and integer K1-4 / 6 / 7; its mul1 variant plus EXL3_CB_HASH). The shared loaders and SRC are not
edited, so the code stamps of the other codebooks don't change. The builds go to TORCH_EXTENSIONS_DIR/exl3hash-A-B.
Use: import exl3hash; exl3hash.install(SRC, qtf2=True)   (after exl3frac.install(SRC), as ourq does)."""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in ("exl3frac", "exl3cb", "exl3frac2"):
    sys.path.insert(0, os.path.join(HERE, "..", _d))
A_DEFAULT, B_DEFAULT = 0xD0474B, 0x2545F491


def ab():
    a = int(os.environ.get("EXL3HASH_A", hex(A_DEFAULT)), 16)
    b = int(os.environ.get("EXL3HASH_B", hex(B_DEFAULT)), 16)
    assert 0 < a < 1 << 24 and 0 <= b < 1 << 32, (a, b)
    return a, b


def values(a=None, b=None):
    """numpy reference: the 65536 states' values (float32, the f16 the kernels produce)"""
    import numpy as np
    if a is None: a, b = ab()
    s = np.arange(65536, dtype=np.uint64)
    h = ((s * a + b) % (1 << 32)).astype(np.uint32) & np.uint32(0x3F3F3F3F)
    q = sum(((h >> (8 * k)) & 0xFF).astype(np.int64) for k in range(4)) - 126
    s4 = np.float32(4.0) * np.uint16(0x1EEE).view(np.float16).astype(np.float32)
    return (q.astype(np.float32) * s4).astype(np.float16).astype(np.float32)


# codebook.cuh hooks: (anchor, text inserted after it); each anchor must occur exactly once
_HOOKS = [
    ('#include "../ptx_portable.cuh"\n',
     '#ifdef EXL3_CB_HASH\n#include "cb_hash.cuh"   // exl3hash (#100c)\n#endif\n'),
    ('__device__ inline half2 decode_mul1_product_2(uint32_t x0, uint32_t x1)\n{\n',
     '#ifdef EXL3_CB_HASH\n    return exl3_hash_product_2(x0, x1);\n#endif\n'),
    ('    if constexpr (cb == 2)\n    {\n        x *= 0x83DCD12Du;\n',
     None),   # decode_3inst<2>: the hash goes before the multiply (special-cased below)
]


def _patch_codebook(cb: str) -> str:
    for anchor, text in _HOOKS:
        assert cb.count(anchor) == 1, f"codebook.cuh changed: anchor not unique / missing: {anchor!r}"
        if text is None:
            head = "    if constexpr (cb == 2)\n    {\n"
            cb = cb.replace(anchor, head + "#ifdef EXL3_CB_HASH\n        return exl3_hash_value(x);\n#endif\n" + anchor[len(head):])
        else:
            cb = cb.replace(anchor, anchor + text)
    return cb


def _put(path, data: bytes):
    """write only if different (keeps mtimes, so ninja doesn't rebuild)"""
    if os.path.exists(path) and open(path, "rb").read() == data: return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path + ".part", "wb").write(data); os.replace(path + ".part", path)


def make_src(src: str, root: str) -> str:
    """root/exllamav3_ext: SRC with the hooked codebook.cuh and cb_hash.cuh (A, B baked in); returns its path"""
    src = os.path.abspath(src); dst = os.path.join(root, "exllamav3_ext")
    for d, _, fs in os.walk(src):
        if "__pycache__" in d: continue
        for f in fs:
            p = os.path.join(d, f); rel = os.path.relpath(p, src)
            data = open(p, "rb").read()
            if rel == os.path.join("quant", "codebook.cuh"): data = _patch_codebook(data.decode()).encode()
            _put(os.path.join(dst, rel), data)
    a, b = ab()
    hdr = f"#define EXL3_HASH_A 0x{a:06X}u\n#define EXL3_HASH_B 0x{b:08X}u\n" + open(os.path.join(HERE, "cb_hash.cuh")).read()
    _put(os.path.join(dst, "quant", "cb_hash.cuh"), hdr.encode())
    return dst


class _Env:
    """the hash builds' environment: TORCH_EXTENSIONS_DIR/exl3hash-A-B, EXL3CB_DEFS / QTF2_DEFS += EXL3_CB_HASH"""
    KEYS = ("TORCH_EXTENSIONS_DIR", "EXL3CB_DEFS", "QTF2_DEFS")
    def __init__(self, src):
        a, b = ab()
        self.saved = {k: os.environ.get(k) for k in self.KEYS}
        self.root = os.path.join(self.saved["TORCH_EXTENSIONS_DIR"] or os.path.join(HERE, "build"), f"exl3hash-{a:06x}-{b:08x}")
        self.src = make_src(src, os.path.join(self.root, "src"))
    def __enter__(self):
        os.environ["TORCH_EXTENSIONS_DIR"] = self.root
        for k in ("EXL3CB_DEFS", "QTF2_DEFS"): os.environ[k] = " ".join(["EXL3_CB_HASH"] + (self.saved[k] or "").split())
        return self
    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def build(src: str, qtf2: bool = True, verbose: bool = False):
    """compile (and load) both extensions without installing them (a CPU-only prebuild)"""
    import exl3cb, exl3frac2
    with _Env(src) as e:
        exl3cb.load(e.src, "i8", verbose)
        if qtf2: exl3frac2.load(e.src, "mul1", verbose)
    return e.root


def install(src: str, qtf2: bool = True, verbose: bool = False):
    import exl3cb, exl3frac2
    with _Env(src) as e:
        Q = exl3cb.install(e.src, "i8", verbose)
        if qtf2: Q = exl3frac2.install(e.src, "mul1", verbose)
    print("exl3hash: A 0x%06x B 0x%08x, builds in %s" % (*ab(), e.root), flush=True)
    return Q
