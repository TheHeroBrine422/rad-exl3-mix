#!/usr/bin/env python3
"""agent32.py CMD ... -- golden-side helpers for the chunked agentic slices (eval/traces/README.md). Low RSS: rows are
streamed in pieces, never the whole golden in memory (gpubox's RAM belongs to the GPU jobs).

  traces CALIB_GOLDEN OUTDIR DOC...      t<doc>.i32 (tokens) and g<doc>.u8 (gen rows) of whole calib documents, for
                                         chunk.py --gold
  whole CALIB_GOLDEN CHUNKDIR OUT        the slice agent-verify-whole (the routine verify traces, whole) cut from the
                                         published calib golden, as golden_pick.py cuts a whole document (kld.json docs
                                         carry "orig"); tokens and gen must equal CHUNKDIR's agent-verify-whole files
  finalize GOLDEN CHUNKDIR NAME          after mkgold27.py: check tokens.i32 = CHUNKDIR/NAME.tokens.i32, write gen.u8 =
                                         NAME.gen.u8 (mkgold27 writes all 1), add orig / chunk / from / to /
                                         header_tokens to kld.json's docs and "slice" / "manifest_sha256" to its header;
                                         sanity checks on the rows. Idempotent.
  refgap GOLDEN CALIB_GOLDEN [--json F]  chunk 0 of every trace has the published golden's context exactly (the trace's
                                         own prefix), so the two references can be compared there: top-1 agreement,
                                         mean |d nll| of the true token, KL(published || ours) over the published head
                                         (ours' log p where it has the id, else its tail spread evenly), on gen rows.
  gatecorpus CHUNKDIR CORPUS OUT [SHORT] the correctness gate's corpus: a ~SHORT-token prefix (default 2048) of the first
                                         search trace + its chunk 0, both with the published golden's context
  gate GOLDEN CALIB_GOLDEN CORPUS JSON   refgap of mkgold27's output on the gate corpus; exit 0 if every doc has
                                         KL(published || ours) <= 0.005 and top-1 agreement >= 0.95, else 3"""
import hashlib, json, os, sys, time
import numpy as np

REC = np.dtype([("id", "<i4"), ("lp", "<f4")])
PIECE = 8192                                       # rows per copy (4 MB of top.bin at top 512)


def load(g):
    m = json.load(open(f"{g}/kld.json"))
    toff = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
    return m, toff


def atomic(path, data):
    with open(path + ".part", "wb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(path + ".part", path)


def sha(b): return hashlib.sha256(b).hexdigest()


def traces(g, od, docs):
    m, toff = load(g); os.makedirs(od, exist_ok=True)
    toks = np.memmap(f"{g}/tokens.i32", np.int32, "r"); gen = np.memmap(f"{g}/gen.u8", np.uint8, "r")
    for i in map(int, docs):
        d = m["docs"][i]; assert d["score_from"] == 0, (i, d)
        atomic(f"{od}/t{i}.i32", np.asarray(toks[toff[i]:toff[i + 1]]).tobytes())
        atomic(f"{od}/g{i}.u8", np.asarray(gen[d["row0"]:d["row0"] + d["tokens"] - 1]).tobytes())
        print(f"doc {i} {d['source']}: {d['tokens']} tokens")


def whole(g, cd, out):
    m, toff = load(g); top = m["top"]
    if os.path.exists(f"{out}/kld.json"): print(f"{out} exists: nothing to do"); return
    lines = [json.loads(l) for l in open(f"{cd}/agent-verify-whole.jsonl")]
    man = json.load(open(f"{cd}/manifest.json"))
    idx = [d for d, _ in man["slices"]["agent-verify-whole"]["members"]]
    toks = np.memmap(f"{g}/tokens.i32", np.int32, "r"); gen = np.memmap(f"{g}/gen.u8", np.uint8, "r")
    tb = np.memmap(f"{g}/top.bin", np.uint8, "r").reshape(m["rows"], top * 8)
    rw = np.memmap(f"{g}/rows.f32", np.float32, "r").reshape(m["rows"], 3)
    tmp = out + ".part"; os.makedirs(tmp, exist_ok=True)
    docs, r = [], 0
    with open(f"{tmp}/tokens.i32", "wb") as ft, open(f"{tmp}/top.bin", "wb") as fb, open(f"{tmp}/rows.f32", "wb") as fr, \
            open(f"{tmp}/gen.u8", "wb") as fg:
        for i, l in zip(idx, lines):
            d = m["docs"][i]; assert d["score_from"] == 0 and d["tokens"] == l["tokens"], (i, d)
            n = d["tokens"] - 1
            ft.write(np.asarray(toks[toff[i]:toff[i + 1]]).tobytes())
            for j in range(0, n, PIECE):
                k = min(n, j + PIECE)
                fb.write(np.asarray(tb[d["row0"] + j:d["row0"] + k]).tobytes())
                fr.write(np.asarray(rw[d["row0"] + j:d["row0"] + k]).tobytes())
                fg.write(np.asarray(gen[d["row0"] + j:d["row0"] + k]).tobytes())
            docs.append({"source": d["source"], "tokens": d["tokens"], "score_from": 0, "row0": r, "orig": i}); r += n
        for f in (ft, fb, fr, fg): f.flush(); os.fsync(f.fileno())
    want_t = open(f"{cd}/agent-verify-whole.tokens.i32", "rb").read(); want_g = open(f"{cd}/agent-verify-whole.gen.u8", "rb").read()
    assert open(f"{tmp}/tokens.i32", "rb").read() == want_t, "whole: tokens differ from chunk.py's"
    assert open(f"{tmp}/gen.u8", "rb").read() == want_g, "whole: gen differs from chunk.py's rule"
    mm = dict(m); mm["rows"] = r; mm["docs"] = docs; mm["slice"] = "agent-verify-whole"
    mm["ref_created"] = m["created"]; mm["created"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())   # the cut's date
    mm["manifest_sha256"] = sha(open(f"{cd}/manifest.json", "rb").read())
    atomic(f"{tmp}/kld.json", json.dumps(mm).encode())
    os.replace(tmp, out)
    print(f"whole: {len(docs)} docs ({', '.join(d['source'] for d in docs)}), {r} rows, {int(np.frombuffer(want_g, np.uint8).sum())} gen -> {out}")


def finalize(g, cd, name):
    m = json.load(open(f"{g}/kld.json"))
    lines = [json.loads(l) for l in open(f"{cd}/{name}.jsonl")]
    msha = sha(open(f"{cd}/manifest.json", "rb").read())
    want_g = np.fromfile(f"{cd}/{name}.gen.u8", np.uint8)
    assert len(m["docs"]) == len(lines), (len(m["docs"]), len(lines))
    assert open(f"{g}/tokens.i32", "rb").read() == open(f"{cd}/{name}.tokens.i32", "rb").read(), "tokens.i32 differs from chunk.py's"
    assert m["rows"] == len(want_g), (m["rows"], len(want_g))
    for d, l in zip(m["docs"], lines):
        assert d["tokens"] == l["tokens"] and d["score_from"] == l["score_from"] and d["source"] == l["source"], (d, l)
        assert d["tokens"] - 1 - d["score_from"] == l["to"] - l["from"]
    # rows sanity: finite nll / entropy, tail in [0, 1), head log p sorted, head + tail ~ 1
    rw = np.memmap(f"{g}/rows.f32", np.float32, "r").reshape(m["rows"], 3)
    tb = np.memmap(f"{g}/top.bin", REC, "r").reshape(m["rows"], m["top"])
    bad = 0
    for j in range(0, m["rows"], PIECE):
        x = np.asarray(rw[j:j + PIECE]); h = np.asarray(tb[j:j + PIECE])
        bad += int((~np.isfinite(x).all(1)).sum() + (x[:, 2] < 0).sum() + (x[:, 2] >= 1).sum())
        bad += int((np.diff(h["lp"], axis=1) > 1e-6).any(1).sum())
        bad += int((np.abs(np.exp(h["lp"].astype(np.float64)).sum(1) + x[:, 2] - 1) > 1e-3).sum())
    assert bad == 0, f"{g}: {bad} rows fail the sanity checks"
    if open(f"{g}/gen.u8", "rb").read() != want_g.tobytes(): atomic(f"{g}/gen.u8", want_g.tobytes())
    for d, l in zip(m["docs"], lines):
        for k in ("orig", "chunk", "from", "to"): d[k] = l[k]
        d["header_tokens"] = l["score_from"] + 1
    m["slice"] = name; m["manifest_sha256"] = msha
    m["gen_rule"] = "assistant message tokens after <|im_start|>assistant\\n through <|im_end|> (the calib golden's)"
    atomic(f"{g}/kld.json", json.dumps(m, indent=1).encode())
    print(f"finalize {name}: {len(lines)} docs, {m['rows']} rows, {int(want_g.sum())} gen rows; rows sane; manifest {msha[:12]}")


def refgap(g, cg, jout=None, corpus=None):
    m = json.load(open(f"{g}/kld.json")); c, _ = load(cg)
    if corpus:                                         # an unfinalized golden: chunk fields from its corpus lines
        for d, l in zip(m["docs"], open(corpus)): d.update({k: json.loads(l)[k] for k in ("orig", "chunk", "from", "to")})
    cgen = np.memmap(f"{cg}/gen.u8", np.uint8, "r")
    tb = np.memmap(f"{g}/top.bin", REC, "r").reshape(m["rows"], m["top"]); rw = np.memmap(f"{g}/rows.f32", np.float32, "r").reshape(m["rows"], 3)
    ctb = np.memmap(f"{cg}/top.bin", REC, "r").reshape(c["rows"], c["top"]); crw = np.memmap(f"{cg}/rows.f32", np.float32, "r").reshape(c["rows"], 3)
    V = m["n_vocab"]; res = []
    for d in m["docs"]:
        if d.get("chunk") != 0: continue
        cd = c["docs"][d["orig"]]; assert cd["score_from"] == 0
        n = d["tokens"] - 1 - d["score_from"]          # rows predicting body tokens from .. to-1 = calib rows from-1 ..
        a0, c0 = d["row0"], cd["row0"] + d["from"] - 1
        assert n == d["to"] - d["from"], (d, n)
        kl, t1, dn, dns, ng = 0.0, 0, 0.0, 0.0, 0
        for j in range(0, n, 2048):
            k = min(n, j + 2048); mk = np.asarray(cgen[c0 + j:c0 + k]).astype(bool)
            if not mk.any(): continue
            P = np.asarray(ctb[c0 + j:c0 + k])[mk]; Q = np.asarray(tb[a0 + j:a0 + k])[mk]
            pr, qr = np.asarray(crw[c0 + j:c0 + k])[mk].astype(np.float64), np.asarray(rw[a0 + j:a0 + k])[mk].astype(np.float64)
            t1 += int((P["id"][:, 0] == Q["id"][:, 0]).sum())
            dn += float(np.abs(pr[:, 0] - qr[:, 0]).sum()); dns += float((qr[:, 0] - pr[:, 0]).sum()); ng += int(mk.sum())
            for x in range(len(P)):
                qd = dict(zip(Q["id"][x].tolist(), Q["lp"][x].astype(np.float64).tolist()))
                qt = np.log(max(qr[x, 2], 1e-30) / (V - m["top"]))
                lq = np.array([qd.get(i, qt) for i in P["id"][x].tolist()])
                lp = P["lp"][x].astype(np.float64); p = np.exp(lp)
                kl += float((p * (lp - lq)).sum() + (pr[x, 2] * (np.log(max(pr[x, 2], 1e-30)) - np.log(max(qr[x, 2], 1e-30))) if pr[x, 2] > 0 else 0.0))
        res.append({"orig": d["orig"], "source": d["source"], "tokens": d["tokens"], "gen_rows": ng, "top1_agree": t1 / ng, "mean_abs_dnll": dn / ng,
                    "mean_dnll_ours_minus_published": dns / ng, "kl_published_ours": kl / ng})
        r = res[-1]
        print(f"refgap {r['source']:22} chunk 0 ({d['tokens']} tok): {ng} gen rows, top-1 agree {r['top1_agree']:.4f}, |d nll| {r['mean_abs_dnll']:.4f}, "
              f"d nll {r['mean_dnll_ours_minus_published']:+.5f}, KL(pub||ours) {r['kl_published_ours']:.2e}")
    if jout: atomic(jout, json.dumps(res, indent=1).encode())
    return res


def gatecorpus(cd, corpus, out, short=2048):
    """2 lines with the published golden's context: the first trace's prefix up to the first message start at or after
    `short` tokens, and the first search chunk (chunk 0 = the trace's own prefix, ~27k tokens)"""
    man = json.load(open(f"{cd}/manifest.json")); X = round(man["max_tokens"] / 1024)
    first = open(f"{cd}/agent-search{X}.jsonl").readline(); j = json.loads(first)
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.environ.get("A32_TOKENIZER", "/data/workspace/models/qwen38-27b-bf16/tokenizer.json"))
    src = None
    for k, l in enumerate(open(corpus)):
        if k == j["orig"]: src = json.loads(l); break
    p = src["prompt"]; c = p.find("<|im_start|>", 1)
    while c >= 0 and len(tk.encode(p[:c], add_special_tokens=False).ids) < short: c = p.find("<|im_start|>", c + 1)
    n = len(tk.encode(p[:c], add_special_tokens=False).ids); H = j["score_from"] + 1
    sh = {k: j[k] for k in ("source", "category", "kind")}
    sh.update({"tokens": n, "orig": j["orig"], "chunk": 0, "from": H, "to": n, "score_from": H - 1, "prompt": p[:c]})
    atomic(out, (json.dumps(sh, ensure_ascii=False) + "\n" + first).encode())
    print(f"gate corpus: {j['source']} prefix {n} tokens + chunk 0 {j['tokens']} tokens -> {out}")


def gate(g, cg, corpus, jout, kl_max=0.005, top1_min=0.95):
    """refgap on the gate corpus; PASS if every doc has KL(published || ours) <= kl_max and top-1 agreement >= top1_min"""
    res = refgap(g, cg, jout, corpus)
    ok = all(r["kl_published_ours"] <= kl_max and r["top1_agree"] >= top1_min for r in res)
    print(f"GATE {'PASS' if ok else 'FAIL'} (KL <= {kl_max}, top-1 >= {top1_min}): " +
          ", ".join(f"{r['tokens']} tok KL {r['kl_published_ours']:.2e} top-1 {r['top1_agree']:.3f}" for r in res))
    return ok


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "traces": traces(args[0], args[1], args[2:])
    elif cmd == "whole": whole(*args)
    elif cmd == "finalize": finalize(*args)
    elif cmd == "refgap": refgap(args[0], args[1], args[3] if len(args) > 3 and args[2] == "--json" else None)
    elif cmd == "gatecorpus": gatecorpus(args[0], args[1], args[2], *(int(x) for x in args[3:4]))
    elif cmd == "gate": sys.exit(0 if gate(*args[:4]) else 3)
    else: raise SystemExit(__doc__)
