#!/usr/bin/env python3
"""kld512_to_16.py -- adapt a challenge-format golden (radiance-kld-top-1: 512 top cells,
3-float rows.f32, gen.u8) to the radiance 1.13.0 format (radiance-kld-1: kKldTop 16,
2-float rows.f32, no gen mask -- rows are scored from score_from).

  kld.json  format -> radiance-kld-1, top -> 16 (rows/docs unchanged)
  top.bin   [rows, 512] -> [rows, 16] (the 16 most likely, layout {i32 id, f32 log p})
  rows.f32  [rows, 3]  -> [rows, 2] (keep NLL, the second float; the third is dropped)
  tokens.i32 unchanged; gen.u8 dropped

The KL is then the coarsened lower bound over 16 cells + tail: a bit below the
challenge's 512-cell metric, but the signal for a broken vs healthy stack is the same,
and it is consistent across points (which is what the data collection needs).

Usage: kld512_to_16.py IN_DIR OUT_DIR
"""
import json, os, sys

def main(gdir, odir):
    os.makedirs(odir, exist_ok=True)
    meta = json.load(open(f"{gdir}/kld.json"))
    top_n, out_n = meta["top"], 16
    n_rows = meta["rows"]
    src = open(f"{gdir}/top.bin", "rb").read()
    e, oe = top_n * 8, out_n * 8
    out_top = bytearray()
    for i in range(n_rows):
        out_top += src[i*e:i*e + oe]
    rows = open(f"{gdir}/rows.f32", "rb").read()
    out_rows = bytearray()
    for i in range(n_rows):
        out_rows += rows[i*12:i*12 + 8]
    base = {k: v for k, v in meta.items() if k in ("model", "corpus", "n_vocab", "docs")}
    base["format"] = "radiance-kld-1"
    base["top"] = out_n
    base["rows"] = n_rows
    json.dump(base, open(f"{odir}/kld.json", "w"))
    open(f"{odir}/top.bin", "wb").write(out_top)
    open(f"{odir}/rows.f32", "wb").write(out_rows)
    import shutil
    shutil.copyfile(f"{gdir}/tokens.i32", f"{odir}/tokens.i32")
    print(f"converted {n_rows} rows: {gdir} -> {odir}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
