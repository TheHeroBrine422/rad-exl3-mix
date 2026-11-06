"""cbx.py -- exllamav3's exact mul1 and int8-rounded mul1 codebooks (fork eb5a2139, quant/codebook.cuh), beside
trellis2.codebook's names (w45, #94 rerun).

  mul1x     decode_mul1_product_2: h = fp16(1024 + bytesum), v = fp16(fma(h, k_inv, k_bias)), k_inv = fp16 0x1eee
            (0.006767 ~ 1/147.7), k_bias = fp16 0xc931 (-10.39): sd 1.0003, on a lattice of one bytesum unit
  mul1x_i8  mul1_i8_round (EXL3_MUL1_I8): s4 = 4 k_inv (fp32), q = clamp(rint(v * (1 / s4)), -127, 127), value
            fp16(q * s4): one int8 step = 4 bytesum units = 0.02707 (trellis.py's mul1_i8: 3.5 / 127 = 0.02756 sd on
            the standardised values, not aligned to the bytesum lattice)
Values are not restandardised (exllamav3 does not); the g_scale search absorbs the 0.03% scale."""
import numpy as np
import trellis2

_base = trellis2.codebook                                   # settle.py's names (kept when settlex.py patches trellis2)


def _f16(bits): return np.frombuffer(np.uint16(bits).tobytes(), np.float16)[0]


def codebook(name):
    if name not in ("mul1x", "mul1x_i8"): return _base(name)
    s = np.arange(65536, dtype=np.uint64)
    h = (s * 0x83DCD12D) & 0xFFFFFFFF
    bs = sum(((h >> (8 * i)) & 0xFF) for i in range(4)).astype(np.float64)
    k_inv, k_bias = float(_f16(0x1eee)), float(_f16(0xc931))
    v = (np.float16(1024 + bs).astype(np.float64) * k_inv + k_bias).astype(np.float16)   # fma: exact in f64, one rounding
    if name == "mul1x": return v.astype(np.float64)
    s4 = np.float32(4.0) * np.float32(k_inv); inv = np.float32(1.0) / s4
    q = np.clip(np.rint(v.astype(np.float32) * inv), -127, 127).astype(np.float32)
    return (q * s4).astype(np.float16).astype(np.float64)


if __name__ == "__main__":
    for n in ("mul1", "mul1_i8", "mul1x", "mul1x_i8"):
        c = codebook(n); u = np.unique(c)
        print(f"{n:9s} mean {c.mean():+.5f} sd {c.std():.5f} levels {len(u):5d} min step {np.diff(u).min():.5f} "
              f"range {c.min():+.4f} .. {c.max():+.4f}")
