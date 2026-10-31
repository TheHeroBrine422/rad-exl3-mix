#!/usr/bin/env python3
"""slice2k.py -- cut a small self-consistent mini-golden from golden/public for a
fast KLD smoke test.

Picks leading documents of the public split (whole row ranges) until ~MAX_ROWS
scored rows are covered, and writes a directory in the same golden layout:

  kld.json   the same schema, the subset's docs (row0 re-based to the subset)
  tokens.i32 the subset's tokens, concatenated in doc order
  top.bin    [rows, 512] of {int32 id, f32 log p} -- the sliced rows
  rows.f32   [rows, 3] float32
  gen.u8     [rows]

Tokens: kld.json's docs carry {source, tokens, score_from, row0}; tokens.i32 is the
concatenation over docs in file order, so a doc's token offset is the prefix sum
of the earlier docs' `tokens`. A doc's row range is [row0, row0 + tokens - 1 -
score_from).

Usage: slice2k.py GOLDEN_DIR OUT_DIR MAX_ROWS
"""
import json, os, struct, sys

def main(gdir, odir, max_rows):
    meta = json.load(open(f"{gdir}/kld.json"))
    toks = open(f"{gdir}/tokens.i32", "rb").read()
    top = open(f"{gdir}/top.bin", "rb")
    rows = open(f"{gdir}/rows.f32", "rb")
    gen = open(f"{gdir}/gen.u8", "rb")
    top_n, row3 = meta["top"], 12

    picked, total = [], 0
    for d in meta["docs"]:
        n = d["tokens"] - 1 - d.get("score_from", 0)
        if picked and total + n > max_rows:
            break
        n = min(n, max_rows)          # prefix cut: a long doc contributes its first n rows
        picked.append((d, n))
        total += n

    os.makedirs(odir, exist_ok=True)
    out_top, out_rows, out_gen, new_docs = [], [], [], []
    tok_off = 0
    row_cursor = 0
    for d, n in picked:
        top.seek(d["row0"] * top_n * 8)
        out_top.append(top.read(n * top_n * 8))
        rows.seek(d["row0"] * row3)
        out_rows.append(rows.read(n * row3))
        gen.seek(d["row0"])
        out_gen.append(gen.read(n))
        out_tok = toks[tok_off * 4:(tok_off + d.get("score_from", 0) + n + 1) * 4]
        tok_off += d["score_from"] + n + 1
        new_docs.append({
            "source": d["source"],
            "tokens": d.get("score_from", 0) + n + 1,
            "score_from": d.get("score_from", 0),
            "row0": row_cursor,
        })
        row_cursor += n
        open(f"{odir}/_tok.bin", "ab").write(out_tok)

    base = {k: v for k, v in meta.items() if k != "docs"}
    base["docs"] = new_docs
    base["rows"] = row_cursor
    with open(f"{odir}/kld.json", "w") as f:
        json.dump(base, f)
    os.replace(f"{odir}/_tok.bin", f"{odir}/tokens.i32")
    open(f"{odir}/top.bin", "wb").write(b"".join(out_top))
    open(f"{odir}/rows.f32", "wb").write(b"".join(out_rows))
    open(f"{odir}/gen.u8", "wb").write(b"".join(out_gen))
    print(f"sliced {len(new_docs)} docs -> {total} rows: " +
          ", ".join(f"{d['source']}:{d['tokens']}" for d in new_docs))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
