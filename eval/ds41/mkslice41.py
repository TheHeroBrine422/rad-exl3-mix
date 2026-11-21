#!/usr/bin/env python3
"""mkslice41.py -- build the DS4.1-F verify slice `ds41-verify-v1` (w212) from the published golden (HF dataset
Hero422/deepseek-v4.1-flash-1017-quant-calib, golden/main/part-*), documents of pool "verify" ONLY (eval/ds41/RULES-pools.md).

  mkslice41.py plan  POOLS.json MAIN100B.jsonl WORKDIR MANIFEST.json [--target 150000] [--cap 8192] [--seed ds41-verify-v1]
  mkslice41.py build MANIFEST.json OUTDIR

plan:  downloads the 42 parts' kld.json + gen.u8 (small), keeps the documents whose pool is "verify", and per category (quota =
       target x the category's share of the verify pool's tokens) takes documents in sha256(seed + name) order until the
       category's quota of scored positions (gen.u8 = 1 rows) is reached. A document is cut to its first CAP tokens (rows are
       causal, so the prefix's rows are exactly the golden's). Writes MANIFEST.json (documents, parts, positions).
       Never reads h1 / h2 / fit / qad / reserve documents' rows (their names are only skipped; the part tables list every doc).
build: for each selected document reads only its rows from the parts by HTTP range (top.bin, rows.f32, gen.u8, tokens.i32)
       and writes OUTDIR in the golden format (kld.json, tokens.i32, top.bin, rows.f32, gen.u8) for `radiance --kld-ref`.
"""
import hashlib, json, os, sys
import numpy as np

REPO = "Hero422/deepseek-v4.1-flash-1017-quant-calib"
NPARTS = 42
TOP = 512


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def plan(pools_p, main_p, work, out):
    from huggingface_hub import hf_hub_download
    target, cap, seed = arg("--target", 150000, int), arg("--cap", 8192, int), arg("--seed", "ds41-verify-v1")
    pools = json.load(open(pools_p))
    pool = pools["pool"]
    cat = {}
    for l in open(main_p):
        r = json.loads(l)
        if pool.get(r["name"]) == "verify":      # only verify documents' metadata is kept
            cat[r["name"]] = (r["category"], r["tokens"])
    share = {c: v["verify"] for c, v in pools["tokens_by_category"].items()}
    tot = sum(share.values())
    cands = {c: [] for c in share}
    for p in range(NPARTS):
        d = f"golden/main/part-{p:05d}"
        kj = json.load(open(hf_hub_download(REPO, d + "/kld.json", local_dir=work, repo_type="dataset")))
        gen = np.fromfile(hf_hub_download(REPO, d + "/gen.u8", local_dir=work, repo_type="dataset"), np.uint8)
        assert len(gen) == kj["rows"]
        for i, doc in enumerate(kj["docs"]):
            if pool.get(doc["source"]) != "verify":
                continue
            c, tk = cat[doc["source"]]
            assert tk == doc["tokens"], (doc, tk)
            n = min(doc["tokens"], cap)
            g = int(gen[doc["row0"]:doc["row0"] + n - 1].sum())
            if g == 0:
                continue
            h = hashlib.sha256((seed + ":" + doc["source"]).encode()).hexdigest()
            cands[c].append(dict(name=doc["source"], category=c, part=p, doc=i, tokens_full=doc["tokens"], tokens=n,
                                 row0_part=doc["row0"], scored=g, h=h,
                                 _gen=gen[doc["row0"]:doc["row0"] + n - 1].copy()))
    chosen, summary = [], {}
    for c in sorted(cands):
        quota = target * share[c] / tot
        acc = 0
        cs = sorted(cands[c], key=lambda x: x["h"])
        for x in cs:
            if acc >= quota:
                break
            if acc + x["scored"] > quota:                  # the last document is cut so the category lands on its quota
                need = max(int(quota - acc), 1)
                n = int(np.searchsorted(np.cumsum(x["_gen"]), need)) + 2      # rows 0..n-2 hold `need` scored rows
                x["tokens"] = n; x["scored"] = int(x["_gen"][:n - 1].sum())
            chosen.append(x); acc += x["scored"]
        summary[c] = dict(quota=round(quota), scored=acc, docs=sum(1 for x in chosen if x["category"] == c),
                          candidates=len(cs))
    for x in chosen:
        del x["_gen"]
    chosen.sort(key=lambda x: (x["category"], x["h"]))
    json.dump(dict(golden="ds41-verify-v1", seed=seed, target=target, cap=cap, pool="verify", repo=REPO,
                   scored=sum(x["scored"] for x in chosen), rows=sum(x["tokens"] - 1 for x in chosen),
                   by_category=summary, docs=chosen), open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1), "scored", sum(x["scored"] for x in chosen), "docs", len(chosen))


def build(man_p, outd):
    from huggingface_hub import HfFileSystem
    m = json.load(open(man_p))
    fs = HfFileSystem()
    os.makedirs(outd, exist_ok=True)
    from huggingface_hub import hf_hub_download
    kld = {}
    toks, tops, rowsf, gens, docs = [], [], [], [], []
    rows0 = 0
    for x in m["docs"]:
        p = x["part"]
        if p not in kld:
            kld[p] = json.load(open(hf_hub_download(REPO, f"golden/main/part-{p:05d}/kld.json", repo_type="dataset")))
        kj = kld[p]
        d = kj["docs"][x["doc"]]
        assert d["source"] == x["name"] and d["tokens"] == x["tokens_full"] and d["row0"] == x["row0_part"]
        tok0 = sum(dd["tokens"] for dd in kj["docs"][:x["doc"]])      # tokens.i32 is the docs' tokens concatenated
        n, r0 = x["tokens"], d["row0"]
        base = f"datasets/{REPO}/golden/main/part-{p:05d}/"
        def rd(fn, off, nbytes):
            with fs.open(base + fn, "rb") as f:
                f.seek(off)
                b = f.read(nbytes)
            assert len(b) == nbytes, (fn, len(b), nbytes)
            return b
        t = np.frombuffer(rd("tokens.i32", tok0 * 4, n * 4), "<i4")
        top = rd("top.bin", r0 * TOP * 8, (n - 1) * TOP * 8)
        rw = rd("rows.f32", r0 * 12, (n - 1) * 12)
        g = rd("gen.u8", r0, n - 1)
        assert int(np.frombuffer(g, np.uint8).sum()) == x["scored"], x["name"]
        toks.append(t.tobytes()); tops.append(top); rowsf.append(rw); gens.append(g)
        docs.append(dict(source=x["name"], tokens=n, score_from=0, row0=rows0))
        rows0 += n - 1
        print(x["name"], x["category"], n, flush=True)
    for fn, bl in (("tokens.i32", toks), ("top.bin", tops), ("rows.f32", rowsf), ("gen.u8", gens)):
        with open(os.path.join(outd, fn), "wb") as f:
            for b in bl:
                f.write(b)
    model = next(iter(kld.values()))["model"]
    # kld.json last: an interrupted build leaves a directory --kld-ref does not read
    json.dump(dict(format="radiance-kld-top-1", n_vocab=129280, top=TOP, rows=rows0, model=model,
                   corpus="ds41-verify-v1 (verify-pool subset of packs_main100b.npz, golden/main)", docs=docs),
              open(os.path.join(outd, "kld.json"), "w"))
    print("rows", rows0, "scored", int(sum(np.frombuffer(g, np.uint8).sum() for g in gens)))


if __name__ == "__main__":
    if sys.argv[1] == "plan":
        plan(*sys.argv[2:6])
    elif sys.argv[1] == "build":
        build(*sys.argv[2:4])
    else:
        sys.exit(__doc__)
