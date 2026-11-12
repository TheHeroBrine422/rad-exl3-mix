#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""goldcat.py OUT NAME=GOLDEN... -- one golden holding several goldens' documents in order (w123, #161), so one radiance
run scores every part of a slice (one model load, one process start instead of one per part).

OUT/{kld.json, tokens.i32, gen.u8, top.bin, rows.f32}: the parts' files concatenated, documents' row0 shifted; each
document keeps its own fields and gets "part": NAME. OUT/parts.json: [{"name", "golden", "doc0", "docs", "row0", "rows"}]
-- the slice of the combined golden (and of its score rows) that is each part. Refuses goldens of different form
(format / model / top / n_vocab). Radiance scores every document on its own (max-num-seqs 1, prefill from the
document's start), so a part's rows inside the combined run are byte for byte the part's own run (checked by w123 on
golden-search27 halves; `split` below cuts a combined .rows file back into per-part files).

  goldcat.py split OUT ROWS PREFIX   -> PREFIX-<NAME>.json.rows per part (the rows each part's own run writes)
  goldcat.py stages GOLDEN OUTPREFIX SEED F1,F2,..,1   -> OUTPREFIX-s1, -s2, ...: the stages of a sequential decision,
             disjoint document subsets (a seeded permutation; stage k = documents F_{k-1} n .. F_k n of it, kept in the
             golden's order); stages.json beside them names each stage's documents ("orig" when the golden has it)

In rxrun.hostd (patch search/perf/hostd-evalperf.patch): cached() builds a slice's combined golden once per host,
report() writes each part's rows and a per-part report (positions and top1 over the part's rows: the numbers its own
run reports in "all", checked equal on golden-search27 and h2)."""
import hashlib, json, os, shutil, sys

import numpy as np

KEYS = ("format", "model", "top", "n_vocab")


def cat(out, parts):
    os.makedirs(out + ".part", exist_ok=True); o = out + ".part"
    ms = [(n, g, json.load(open(f"{g}/kld.json"))) for n, g in parts]
    for k in KEYS:
        if len({json.dumps(m.get(k)) for _, _, m in ms}) != 1: raise SystemExit(f"goldens differ in {k}")
    docs, plist, r, d0 = [], [], 0, 0
    files = {f: open(f"{o}/{f}", "wb") for f in ("tokens.i32", "gen.u8", "top.bin", "rows.f32")}
    for n, g, m in ms:
        R = m["rows"]; tok = sum(d["tokens"] for d in m["docs"])
        sizes = {"tokens.i32": tok * 4, "gen.u8": R, "top.bin": R * m["top"] * 8, "rows.f32": R * 12}
        for f, fh in files.items():
            if os.path.getsize(f"{g}/{f}") != sizes[f]: raise SystemExit(f"{g}/{f}: {os.path.getsize(f'{g}/{f}')} bytes, expected {sizes[f]}")
            with open(f"{g}/{f}", "rb") as src: shutil.copyfileobj(src, fh, 1 << 24)
        for d in m["docs"]:
            e = dict(d); e["row0"] = d["row0"] + r; e["part"] = n; docs.append(e)
        plist.append({"name": n, "golden": os.path.abspath(g), "doc0": d0, "docs": len(m["docs"]), "row0": r, "rows": R})
        r += R; d0 += len(m["docs"])
    for fh in files.values(): fh.close()
    mm = dict(ms[0][2]); mm["rows"] = r; mm["docs"] = docs
    mm["corpus"] = "+".join(str(m.get("corpus", "?")) for _, _, m in ms); mm.pop("corpus_fnv1a64", None)
    json.dump(mm, open(f"{o}/kld.json", "w")); json.dump(plist, open(f"{o}/parts.json", "w"), indent=1)
    os.replace(o, out)
    print(f"{out}: {len(docs)} documents, {r} rows from {len(parts)} goldens", file=sys.stderr)   # stdout: hostd's protocol


def split(out, rows, prefix):
    P = json.load(open(f"{out}/parts.json")); K = np.fromfile(rows, np.float32).reshape(-1, 4)
    if len(K) != sum(p["rows"] for p in P): raise SystemExit(f"{rows}: {len(K)} rows, {out} has {sum(p['rows'] for p in P)}")
    for p in P: K[p["row0"]:p["row0"] + p["rows"]].tofile(f"{prefix}-{p['name']}.json.rows")


def sub(golden, out, idx):
    """a golden of the documents idx (indices into GOLDEN's docs, any order: kept in GOLDEN's order), rows renumbered"""
    m = json.load(open(f"{golden}/kld.json")); idx = sorted(set(int(i) for i in idx)); o = out + ".part"
    os.makedirs(o, exist_ok=True)
    tok_off = [0]
    for d in m["docs"]: tok_off.append(tok_off[-1] + d["tokens"])
    per_row = {"gen.u8": 1, "top.bin": m["top"] * 8, "rows.f32": 12}
    src = {f: open(f"{golden}/{f}", "rb") for f in ("tokens.i32",) + tuple(per_row)}
    dst = {f: open(f"{o}/{f}", "wb") for f in src}
    docs, r = [], 0
    for i in idx:
        d = m["docs"][i]; n = d["tokens"] - 1 - d["score_from"]
        src["tokens.i32"].seek(tok_off[i] * 4); dst["tokens.i32"].write(src["tokens.i32"].read(d["tokens"] * 4))
        for f, b in per_row.items():
            src[f].seek(d["row0"] * b); dst[f].write(src[f].read(n * b))
        e = dict(d); e["row0"] = r; docs.append(e); r += n
    for fh in list(src.values()) + list(dst.values()): fh.close()
    mm = dict(m); mm["rows"] = r; mm["docs"] = docs; mm["subset_of"] = os.path.abspath(golden); mm.pop("corpus_fnv1a64", None)
    json.dump(mm, open(f"{o}/kld.json", "w")); os.replace(o, out)
    return len(docs), r


def stages(golden, prefix, seed, fr):
    import numpy as np
    m = json.load(open(f"{golden}/kld.json")); n = len(m["docs"]); perm = np.random.default_rng(int(seed)).permutation(n)
    F = [0.0] + [float(x) for x in fr.split(",")]
    if F[-1] != 1.0 or sorted(F) != F: raise SystemExit(f"stage fractions {fr}: increasing, ending at 1")
    rec = []
    for k in range(1, len(F)):
        idx = perm[round(F[k - 1] * n):round(F[k] * n)]; out = f"{prefix}-s{k}"
        if not os.path.exists(f"{out}/kld.json"): sub(golden, out, idx)
        rec.append({"stage": k, "golden": os.path.abspath(out), "f": F[k], "docs": sorted(int(i) for i in idx),
                    "orig": sorted(m["docs"][i].get("orig", i) for i in idx)})
        print(f"{out}: {len(idx)} documents (cumulative fraction {F[k]:g})")
    json.dump({"golden": os.path.abspath(golden), "seed": int(seed), "fractions": F[1:], "stages": rec}, open(f"{prefix}-stages.json", "w"), indent=1)


def cached(root, parts):
    """the combined golden of parts [(name, golden), ...] under root, built once (key: the part names and paths)"""
    key = hashlib.sha256(json.dumps([[n, os.path.abspath(g)] for n, g in parts]).encode()).hexdigest()[:16]
    out = f"{root}/cat-{key}"
    if not os.path.exists(f"{out}/parts.json"):
        os.makedirs(root, exist_ok=True)
        if os.path.exists(out + ".part"): shutil.rmtree(out + ".part")
        cat(out, parts)
    return out


def report(out, rows, prefix):
    """split(out, rows, prefix) plus PREFIX-<NAME>.json per part: {"all": {"positions", "top1_agreement"}, "goldcat"}"""
    import numpy as np
    split(out, rows, prefix); K = np.fromfile(rows, np.float32).reshape(-1, 4)
    for p in json.load(open(f"{out}/parts.json")):
        m = json.load(open(f"{p['golden']}/kld.json")); sel = np.zeros(p["rows"], bool)
        for d in m["docs"]: sel[d["row0"]:d["row0"] + d["tokens"] - 1 - d["score_from"]] = True
        k = K[p["row0"]:p["row0"] + p["rows"]][sel]
        json.dump({"all": {"positions": int(len(k)), "top1_agreement": float(k[:, 3].astype(np.float64).mean()) if len(k) else None},
                   "goldcat": os.path.abspath(out)}, open(f"{prefix}-{p['name']}.json", "w"))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "split": split(*a[1:4])
    elif a[0] == "stages": stages(*a[1:5])
    else: cat(a[0], [x.split("=", 1) for x in a[1:]])
