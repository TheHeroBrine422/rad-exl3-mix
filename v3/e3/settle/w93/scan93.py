#!/usr/bin/env python3
"""scan93.py -- w93 (#123, ../w93.md item 1): integrity scan of the ladder's published inputs (CPU, one file a call).

  scan93.py hin FILE CKPT OUT.json      one calib H_in (packed upper fp32): header / shapes against the checkpoint,
                                        finite, diagonal > 0, Cauchy-Schwarz on EVERY packed entry, damped fp64
                                        Cholesky (lambda = 1e-3 x mean diag), exact-zero runs, sha256 of the file
  scan93.py hout DIR KEYS CKPT OUT.json every H_out diagonal of KEYS (one key a line): header, length = the weight's
                                        out_features, finite, > 0, sha256 of each file
Outputs are written as OUT.json.part then renamed (a rerun skips finished files: scan93.sh)."""
import hashlib, json, os, struct, sys, time
import numpy as np

CS_FAIL = 1e-3          # |H_ij|^2 / (H_ii H_jj) > 1 + CS_FAIL fails (fp32 storage of a fp64-ish sum: ~1e-6 expected)
CS_NOTE = 1e-5
CHOL_LAMBDA = 1e-3      # x mean diag; production damping is >= 0.005 x mean diag


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b: break
            h.update(b)
    return h.hexdigest()


def ckpt_shapes(ckpt):
    idx = json.load(open(f"{ckpt}/model.safetensors.index.json"))["weight_map"]
    shapes = {}
    for fn in sorted(set(idx.values())):
        h, _ = header(f"{ckpt}/{fn}")
        for k, v in h.items():
            if k != "__metadata__": shapes[k] = v["shape"]
    return shapes


def zero_runs(x):
    """number of exact zeros and the longest run of consecutive exact zeros in x (1-D)"""
    z = (x == 0)
    nz = int(z.sum())
    if nz == 0: return 0, 0
    d = np.diff(np.concatenate(([0], z.view(np.int8), [0])))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return nz, int((e - s).max())


def write(out, o):
    with open(out + ".part", "w") as f: json.dump(o, f, indent=1)
    os.replace(out + ".part", out)


def hin(path, ckpt, out):
    t0 = time.time()
    h, off = header(path)
    meta = h.get("__metadata__", {}); fails = []; notes = []
    k = int(meta["k"]); t = h["hessian_upper"]
    n = k * (k + 1) // 2
    if t["dtype"] != "F32" or t["shape"] != [n]: fails.append(f"tensor {t['dtype']} {t['shape']} != F32 [{n}]")
    if t["data_offsets"] != [0, 4 * n]: fails.append(f"offsets {t['data_offsets']}")
    if os.path.getsize(path) != off + 4 * n: fails.append(f"file size {os.path.getsize(path)} != {off + 4 * n}")
    if "upper triangle, row-major, packed" not in meta.get("layout", ""): fails.append(f"layout {meta.get('layout')!r}")
    sh = ckpt_shapes(ckpt) if ckpt != "-" else {}
    lin = [x for x in meta.get("linears", "").split(",") if x]
    if os.path.basename(path).startswith("lm_head"): lin = lin or ["lm_head"]
    for x in lin:
        s = sh.get(x + ".weight")
        if sh and s is None: fails.append(f"linear {x}: not in the checkpoint")
        elif sh and s[1] != k: fails.append(f"linear {x}: in_features {s[1]} != k {k}")
    A = np.memmap(path, dtype=np.float32, mode="r", offset=off, shape=(n,))
    starts = np.concatenate(([0], np.cumsum(np.arange(k, 0, -1))))[:k]
    d = np.asarray(A[starts], dtype=np.float64)
    nonfinite = 0; cs_max = 0.0; cs_arg = None; cs_fail = 0; cs_note = 0; nz = 0; zrun = 0; amax = 0.0
    for i in range(k):
        seg = np.asarray(A[starts[i]:starts[i] + k - i], dtype=np.float64)
        fin = np.isfinite(seg)
        if not fin.all():
            nonfinite += int((~fin).sum()); seg = np.where(fin, seg, 0.0)
        if not seg.all():
            z, r = zero_runs(seg); nz += z; zrun = max(zrun, r)
        amax = max(amax, float(np.abs(seg).max()))
        if i + 1 < k:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                r2 = seg[1:] ** 2 / np.abs(d[i] * d[i + 1:])  # 0 / 0 (dead pair) -> nan -> 0; x / 0 -> inf (fails)
            r2[np.isnan(r2)] = 0.0
            j = int(np.argmax(r2)); m = float(r2[j])
            if m > cs_max: cs_max, cs_arg = m, (i, i + 1 + j)
            cs_fail += int((r2 > 1 + CS_FAIL).sum()); cs_note += int((r2 > 1 + CS_NOTE).sum())
    if nonfinite: fails.append(f"{nonfinite} non-finite entries")
    if not np.isfinite(d).all() or (d <= 0).any(): fails.append(f"diagonal: {int((~(d > 0)).sum())} entries not > 0")
    if cs_fail: fails.append(f"Cauchy-Schwarz: {cs_fail} entries above 1 + {CS_FAIL:g} (max {cs_max:.6g} at {cs_arg})")
    if cs_note and not cs_fail: notes.append(f"Cauchy-Schwarz: {cs_note} entries in (1 + {CS_NOTE:g}, 1 + {CS_FAIL:g}]")
    if zrun > 64: notes.append(f"longest exact-zero run {zrun} entries (review)")
    t1 = time.time()
    # damped fp64 Cholesky (an indefinite matrix with all 2x2 minors fine is still caught here)
    chol = None
    if not fails:
        import scipy.linalg
        H = np.zeros((k, k), dtype=np.float64)
        for i in range(k): H[i, i:] = A[starts[i]:starts[i] + k - i]
        lam = CHOL_LAMBDA * float(d.mean())
        H[np.diag_indices(k)] += lam
        try:
            L = scipy.linalg.cholesky(H, lower=False, overwrite_a=True, check_finite=False)
            ld = np.abs(np.diag(L)); chol = {"ok": True, "lambda": lam, "min_diag_L2_over_lambda": float(ld.min() ** 2 / lam)}
        except np.linalg.LinAlgError as e:
            chol = {"ok": False, "lambda": lam, "err": str(e)}; fails.append(f"damped Cholesky failed: {e}")
        del H
    t2 = time.time()
    o = {"file": os.path.basename(path), "kind": "hin", "k": k, "rows": meta.get("rows"), "linears": lin,
         "sha256": sha256(path), "size": os.path.getsize(path),
         "diag_min": float(d.min()), "diag_max": float(d.max()), "diag_mean": float(d.mean()),
         "absmax_over_diagmax": amax / float(d.max()) if d.max() > 0 else None,
         "cs_max": cs_max, "cs_arg": cs_arg, "cs_gt_1e-5": cs_note, "cs_gt_1e-3": cs_fail,
         "zeros": nz, "zero_run_max": zrun, "nonfinite": nonfinite, "cholesky": chol, "exhaustive": True,
         "sec_scan": round(t1 - t0, 1), "sec_chol": round(t2 - t1, 1), "sec": round(time.time() - t0, 1),
         "fails": fails, "notes": notes, "PASS": not fails}
    write(out, o); print(json.dumps({x: o[x] for x in ("file", "PASS", "cs_max", "sec")}), flush=True)


def hout(dr, keys, ckpt, out):
    sh = ckpt_shapes(ckpt)
    rows = []
    want = [x.strip() for x in open(keys) if x.strip()]
    files = sorted(f for f in os.listdir(dr) if f.endswith(".safetensors"))
    extra = sorted(set(files) - {w + ".safetensors" for w in want})
    for key in want:
        p = f"{dr}/{key}.safetensors"; fails = []
        if not os.path.exists(p):
            rows.append({"key": key, "fails": ["missing"], "PASS": False}); continue
        h, off = header(p); meta = h.get("__metadata__", {}); t = h.get("hout_diag")
        N = sh[key + ".weight"][0]
        if t is None or t["dtype"] != "F32" or t["shape"] != [N]: fails.append(f"hout_diag {t} != F32 [{N}]")
        if (meta.get("seqs"), meta.get("blocks"), meta.get("blocknorm")) != ("1920", "15", "1"): fails.append(f"metadata {meta}")
        x = np.fromfile(p, dtype=np.float32, offset=off, count=N).astype(np.float64)
        if os.path.getsize(p) != off + 4 * N: fails.append("file size")
        nf = int((~np.isfinite(x)).sum())
        if nf: fails.append(f"{nf} non-finite")
        npos = int((~(x > 0)).sum())
        if npos: fails.append(f"{npos} entries not > 0")
        xf = x[np.isfinite(x) & (x > 0)]
        rows.append({"key": key, "N": N, "sha256": sha256(p), "min": float(xf.min()), "max": float(xf.max()),
                     "max_over_min": float(xf.max() / xf.min()), "mean": float(xf.mean()), "meta": meta,
                     "fails": fails, "PASS": not fails})
    o = {"kind": "hout", "dir": dr, "n_keys": len(want), "n_files": len(files), "extra_files": extra,
         "n_pass": sum(r["PASS"] for r in rows), "PASS": all(r["PASS"] for r in rows) and len(rows) == len(want),
         "rows": rows}
    write(out, o); print(json.dumps({x: o[x] for x in ("n_keys", "n_files", "n_pass", "PASS")}), flush=True)


def selftest(dr):
    """the scan must pass a clean packed Gram matrix and fail each planted fault (11-06's corrupt span, a NaN, a zero
    diagonal, an indefinite matrix whose 2x2 minors all pass Cauchy-Schwarz)"""
    os.makedirs(dr, exist_ok=True)
    rng = np.random.default_rng(0); k = 256
    X = rng.standard_normal((4096, k)) * np.exp(rng.standard_normal(k)); G = X.T @ X

    def save(H, name):
        iu = np.triu_indices(k); a = H[iu].astype(np.float32)
        hd = json.dumps({"__metadata__": {"k": str(k), "rows": "4096", "layout": "upper triangle, row-major, packed", "linears": ""},
                         "hessian_upper": {"dtype": "F32", "shape": [a.size], "data_offsets": [0, 4 * a.size]}}).encode()
        p = f"{dr}/{name}.safetensors"
        with open(p, "wb") as f: f.write(struct.pack("<Q", len(hd))); f.write(hd); f.write(a.tobytes())
        return p

    def packed_poke(H, flat_from, vals):      # write vals into the packed upper storage from position flat_from on
        iu = np.triu_indices(k); a = H[iu].copy(); a[flat_from:flat_from + len(vals)] = vals
        B = np.zeros((k, k)); B[iu] = a; return np.triu(B) + np.triu(B, 1).T
    span = rng.uniform(18, 29, 500); span = 10 ** span * rng.choice([-1, 1], 500)
    S = np.eye(k) * np.diag(G).mean(); S[:3, :3] = np.array([[1, .9, .9], [.9, 1, -.9], [.9, -.9, 1]]) * np.diag(G).mean()
    cases = {"clean": (G, True), "span": (packed_poke(G, 20000, span), False), "nan": (packed_poke(G, 777, [np.nan]), False),
             "zerodiag": (np.where(np.eye(k) * (np.arange(k) == 5)[:, None] > 0, 0.0, G), False), "indefinite": (S, False)}
    ok = True; res = {}
    for name, (H, want) in cases.items():
        p = save(H, name); o = f"{dr}/{name}.json"
        hin(p, "-", o); got = json.load(open(o))["PASS"]
        res[name] = {"PASS": got, "expected": want, "fails": json.load(open(o))["fails"]}; ok &= (got == want)
    write(f"{dr}/selftest.json", {"cases": res, "PASS": ok}); print("selftest", "PASS" if ok else "FAIL", flush=True)
    if not ok: sys.exit(1)


if __name__ == "__main__":
    if sys.argv[1] == "selftest": selftest(sys.argv[2])
    elif sys.argv[1] == "hin": hin(*sys.argv[2:5])
    elif sys.argv[1] == "hout": hout(*sys.argv[2:6])
    else: raise SystemExit(__doc__)
