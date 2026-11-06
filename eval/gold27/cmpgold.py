#!/usr/bin/env python3
"""cmpgold.py A B -- how far golden B is from golden A (same corpus, same format radiance-kld-top-1): documents,
tokens and gen.u8 must match exactly; then per scored row the top-512 overlap (|ids_A & ids_B| / 512), the largest
|log p_A - log p_B| over the ids both keep, top-1 agreement, and the largest differences of rows.f32 (nll, entropy,
tail). Prints one JSON line; exit 1 if the layouts differ."""
import json, sys
import numpy as np

A, B = sys.argv[1:3]
ma, mb = json.load(open(f"{A}/kld.json")), json.load(open(f"{B}/kld.json"))
key = lambda m: [(d["tokens"], d["score_from"], d["row0"]) for d in m["docs"]]
ok = ma["format"] == mb["format"] == "radiance-kld-top-1" and ma["top"] == mb["top"] and ma["rows"] == mb["rows"] \
    and key(ma) == key(mb)
ta, tb = np.fromfile(f"{A}/tokens.i32", np.int32), np.fromfile(f"{B}/tokens.i32", np.int32)
ga, gb = np.fromfile(f"{A}/gen.u8", np.uint8), np.fromfile(f"{B}/gen.u8", np.uint8)
ok = ok and np.array_equal(ta, tb) and np.array_equal(ga, gb)
if not ok:
    print(json.dumps({"layout_equal": False})); raise SystemExit(1)
R, K = ma["rows"], ma["top"]
rec = np.dtype([("id", "<i4"), ("lp", "<f4")])
xa = np.memmap(f"{A}/top.bin", rec, "r", shape=(R, K)); xb = np.memmap(f"{B}/top.bin", rec, "r", shape=(R, K))
ra = np.memmap(f"{A}/rows.f32", np.float32, "r", shape=(R, 3)); rb = np.memmap(f"{B}/rows.f32", np.float32, "r", shape=(R, 3))
ov_min, ov_sum, dlp, top1, same_rows = 1.0, 0.0, 0.0, 0, 0
for c in range(0, R, 4096):
    a, b = np.asarray(xa[c:c + 4096]), np.asarray(xb[c:c + 4096])
    same_rows += int((a.view(np.uint64) == b.view(np.uint64)).all(1).sum())
    ids = np.concatenate([a["id"], b["id"]], 1); lp = np.concatenate([a["lp"], b["lp"]], 1).astype(np.float64)
    o = np.argsort(ids, 1, kind="stable"); ids = np.take_along_axis(ids, o, 1); lp = np.take_along_axis(lp, o, 1)
    eq = ids[:, 1:] == ids[:, :-1]
    ov = eq.sum(1) / K; ov_min = min(ov_min, float(ov.min())); ov_sum += float(ov.sum())
    d = np.abs(lp[:, 1:] - lp[:, :-1])[eq]
    if d.size: dlp = max(dlp, float(d.max()))
    top1 += int((a["id"][:, 0] == b["id"][:, 0]).sum())
dr = np.abs(ra.astype(np.float64) - rb.astype(np.float64)).max(0)
print(json.dumps({"layout_equal": True, "rows": R, "rows_bitwise_equal_top": same_rows,
                  "rows_f32_bitwise_equal": bool(np.array_equal(np.asarray(ra), np.asarray(rb))),
                  "top_overlap_min": ov_min, "top_overlap_mean": ov_sum / R, "max_abs_dlogp_shared": dlp,
                  "top1_agree": top1 / R, "max_abs_d_nll": float(dr[0]), "max_abs_d_entropy": float(dr[1]),
                  "max_abs_d_tail": float(dr[2]), "mean_nll_A": float(ra[:, 0].astype(np.float64).mean()),
                  "mean_nll_B": float(rb[:, 0].astype(np.float64).mean())}))
