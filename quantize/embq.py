#!/usr/bin/env python3
"""embq.py CONTAINER OFFSET ROWS COLS FORMAT [--write] -- fake-quantise a container's bf16 token table in place.

Reads the bf16 table [ROWS, COLS] at byte OFFSET of CONTAINER, quantises every row in FORMAT, decodes it
again (f32) and, with --write, stores the decoded table back as bf16 (round to nearest even) at the same
place. The engine then reads exactly the values a gather kernel for FORMAT would produce, so the KLD of the
patched container is the KLD of FORMAT; only the bytes on disk (and the speed) are not FORMAT's. Prints the
table's bits per value and its relative RMS error.

FORMAT, one of:
  fp8t          E4M3 codes, ONE scale for the whole table (radiance's embed_lookup_q form), absmax / 448
  fp8r          E4M3 codes, a bf16 scale per row, absmax / 448
  i<B>g<G>[s][h]  signed B-bit codes (-2^(B-1) .. 2^(B-1)-1), a bf16 scale per G columns of a row;
                's' searches the scale per group for least squared error (else absmax / (2^(B-1)-1));
                'h' rotates each 128 columns by the Hadamard first (fwht128: v = w H / 128, w = v H)
  u<B>g<G>[h]   asymmetric: unsigned B-bit codes and a B-bit zero point per G columns, a bf16 scale (min/max range)
  l<B>g<G>[h]   a 2^B-level codebook (Lloyd-Max, fitted to the table's own amax-normalised values) per value, a bf16
                scale per G columns, searched for least squared error
e.g. i4g128, i4g128s, i4g128sh, i4g32s, i3g128sh, i2g128sh, i8g128, u6g128h, l4g128h, l5g128h.

Run it on a COPY of the container (--write modifies it).
"""
import re, sys, time
import numpy as np

path, off, rows, cols, fmt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
write = "--write" in sys.argv[6:]
CH = 4096  # rows a chunk


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x):  # round to nearest even
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def round_bf16(x):
    return bf16_to_f32(f32_to_bf16(x))


# E4M3 (OCP, finite, max 448): the grid of every non-negative value, for round to nearest
_e4m3 = []
for e in range(16):
    for m in range(8):
        if e == 15 and m == 7:
            continue  # NaN
        _e4m3.append((m / 8) * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))
E4M3 = np.array(sorted(set(_e4m3)), np.float64)
E4M3_MID = (E4M3[1:] + E4M3[:-1]) / 2


def to_e4m3(x):  # nearest E4M3 value (ties to the even code: within the midpoint, the lower index wins on ties)
    a = np.minimum(np.abs(x), 448.0)
    i = np.searchsorted(E4M3_MID, a, side="left")
    return np.sign(x) * E4M3[i]


H128 = np.array([[1.0]])
while H128.shape[0] < 128:
    H128 = np.block([[H128, H128], [H128, -H128]])


m = re.fullmatch(r"i(\d)g(\d+)(s?)(h?)", fmt)
mu = re.fullmatch(r"([ul])(\d)g(\d+)(h?)", fmt)
if mu:
    KIND, B, G, HAD = mu.group(1), int(mu.group(2)), int(mu.group(3)), bool(mu.group(4))
    assert cols % G == 0 and (not HAD or cols % 128 == 0)
    bits = B + (16 + B) / G if KIND == "u" else B + 16 / G
elif m:
    B, G, SEARCH, HAD = int(m.group(1)), int(m.group(2)), bool(m.group(3)), bool(m.group(4))
    assert cols % G == 0 and (not HAD or cols % 128 == 0)
    bits = B + 16 / G
elif fmt == "fp8t":
    bits = 8.0
elif fmt == "fp8r":
    bits = 8 + 16 / cols
else:
    sys.exit(f"unknown format {fmt}")


def rot(w):
    n = w.shape[0]
    return (w.reshape(n, cols // 128, 128) @ H128 / 128).reshape(n, cols) if HAD else w


def unrot(v):
    n = v.shape[0]
    return (v.reshape(n, cols // 128, 128) @ H128).reshape(n, cols) if HAD else v


def quant_u(w):  # asymmetric: q = clip(round(w / s) + z), w = (q - z) s
    n = w.shape[0]
    g = rot(w).reshape(n, cols // G, G)
    lo, hi = g.min(2, keepdims=True), g.max(2, keepdims=True)
    top = 2 ** B - 1
    s = round_bf16((hi - lo) / top)
    s = np.where(s > 0, s, 1.0)
    z = np.clip(np.rint(-lo / s), 0, top)
    v = (np.clip(np.rint(g / s) + z, 0, top) - z) * s
    return unrot(v.reshape(n, cols))


LEVELS = None


def fit_levels(sample):  # Lloyd-Max on amax-normalised values (each group / its amax): 2^B levels in [-1, 1]
    x = np.sort(sample)
    L = np.quantile(x, (np.arange(2 ** B) + 0.5) / 2 ** B)
    for _ in range(60):
        edges = (L[1:] + L[:-1]) / 2
        idx = np.searchsorted(edges, x)
        sums = np.bincount(idx, x, 2 ** B); cnt = np.bincount(idx, None, 2 ** B)
        L = np.where(cnt > 0, sums / np.maximum(cnt, 1), L)
    return L


def quant_l(w):
    n = w.shape[0]
    g = rot(w).reshape(n, cols // G, G)
    amax = np.abs(g).max(2, keepdims=True)
    edges = (LEVELS[1:] + LEVELS[:-1]) / 2
    best_e = best_v = None
    for t in np.linspace(0.6, 1.2, 25):   # the scale's search: s = amax * t (bf16), values / s onto the levels
        s = round_bf16(amax * t)
        s = np.where(s > 0, s, 1.0)
        v = LEVELS[np.searchsorted(edges, g / s)] * s
        e = ((v - g) ** 2).sum(2, keepdims=True)
        if best_e is None:
            best_e, best_v = e, v
        else:
            k = e < best_e
            best_e = np.where(k, e, best_e); best_v = np.where(k, v, best_v)
    return unrot(best_v.reshape(n, cols))


def quant_int(w):  # w [n, cols] f64 -> decoded [n, cols]
    n = w.shape[0]
    if HAD:
        w = (w.reshape(n, cols // 128, 128) @ H128 / 128).reshape(n, cols)
    g = w.reshape(n, cols // G, G)
    amax = np.abs(g).max(2, keepdims=True)
    lo, hi = -(2 ** (B - 1)), 2 ** (B - 1) - 1
    base = amax / hi
    best_s = round_bf16(base)
    if SEARCH:
        best_e = None
        for t in np.linspace(0.30, 1.10, 33):
            s = round_bf16(base * t)
            s = np.where(s > 0, s, 1.0)
            q = np.clip(np.rint(g / s), lo, hi) * s
            e = ((q - g) ** 2).sum(2, keepdims=True)
            if best_e is None:
                best_e, best_s = e, s
            else:
                k = e < best_e
                best_e = np.where(k, e, best_e)
                best_s = np.where(k, s, best_s)
    s = np.where(best_s > 0, best_s, 1.0)
    v = (np.clip(np.rint(g / s), lo, hi) * s).reshape(n, cols)
    if HAD:
        v = (v.reshape(n, cols // 128, 128) @ H128).reshape(n, cols)
    return v


mm = np.memmap(path, np.uint16, "r+" if write else "r", offset=off, shape=(rows, cols))
t0 = time.time()
if mu and KIND == "l":  # the codebook: fitted on every 16th row (1/16 of the table)
    smp = []
    for r0 in range(0, rows, CH * 16):
        w = rot(bf16_to_f32(np.asarray(mm[r0:r0 + CH:16])).astype(np.float64))
        g = w.reshape(w.shape[0], cols // G, G)
        smp.append((g / np.maximum(np.abs(g).max(2, keepdims=True), 1e-30)).ravel())
    LEVELS = fit_levels(np.concatenate(smp))
if fmt == "fp8t":  # one scale: the whole table's absmax first
    amax = 0.0
    for r0 in range(0, rows, CH):
        amax = max(amax, float(np.abs(bf16_to_f32(mm[r0:r0 + CH])).max()))
    tscale = float(round_bf16(np.array([amax / 448.0], np.float32))[0])
se = sq = 0.0
for r0 in range(0, rows, CH):
    w = bf16_to_f32(np.asarray(mm[r0:r0 + CH])).astype(np.float64)
    if fmt == "fp8t":
        d = to_e4m3(w / tscale) * tscale
    elif fmt == "fp8r":
        s = round_bf16(np.abs(w).max(1, keepdims=True) / 448.0)
        s = np.where(s > 0, s, 1.0)
        d = to_e4m3(w / s) * s
    elif mu:
        d = quant_u(w) if KIND == "u" else quant_l(w)
    else:
        d = quant_int(w)
    d32 = d.astype(np.float32)
    se += float(((round_bf16(d32) - w) ** 2).sum())
    sq += float((w ** 2).sum())
    if write:
        mm[r0:r0 + CH] = f32_to_bf16(d32)
if write:
    mm.flush()
print(f"{fmt}: {bits:.4f} bits a value, relative RMS error {np.sqrt(se / sq):.5f}, {time.time() - t0:.0f} s"
      + (" (written)" if write else ""))
