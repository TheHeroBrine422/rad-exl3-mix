#!/usr/bin/env python3
"""drift91.py dump MODEL GOLDEN WQ OUT.d16 ROWS.f32 [--bias FITGOLDEN NFIT]
   drift91.py split GOLDEN OUT.json D16 [D16 ...]
../w91.md item 2 (0.8B, 4070). `dump`: evalq.py's fake-quant eval (bf16 model, WQ's fp16 weights cast to bf16; the
challenge's 513-cell KL against the golden's top-512 + tail, common.score_rows) that also writes, per golden row, the
log-ratio on the 513 cells, Delta = log q - log p~ (p~ the golden's cells renormalised; f32 .npy [rows, 513]: 512 head cells
then the tail), and ROWS.f32 = evalq's [kl, top1, gen, 0]. --bias: item 2b's oracle, every quantized linear l gets the
output bias beta_l = (W_l - What_l) mu_l, mu_l = its mean input in the quantized model over the gen rows' positions of
the first NFIT rows of FITGOLDEN (one pass; all biases applied at once); no Delta dump then (OUT.d16 may be '-').
`split`: the drift / variance decomposition over S seeds' dumps (../w91.md item 2), document bootstrap."""
import json, os, sys, time
import numpy as np, torch

P2B = os.environ.get("P2B", "/workspace/rad-exl3-mix/proxy2b")
sys.path.insert(0, P2B)


def dump(a):
    from safetensors.torch import load_file
    from common import load_model, Golden, score_rows, doc_logsoftmax
    import glob
    bias = None
    if "--bias" in a: i = a.index("--bias"); bias = (a[i + 1], int(a[i + 2])); del a[i:i + 3]
    model_path, gd, wq, outd, outr = a[:5]
    cap = float(os.environ.get("VRAM_CAP_GB", "7.2"))
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory))
    model, dec, head = load_model(model_path)
    params = {n.split("layers.", 1)[1]: p for n, p in model.named_parameters() if "layers." in n}
    mods = {n.split("layers.", 1)[1]: m for n, m in model.named_modules() if "layers." in n and isinstance(m, torch.nn.Linear)}
    orig, nsw = {}, 0
    for f in sorted(glob.glob(f"{wq}/*.safetensors")):
        key = os.path.basename(f)[:-len(".safetensors")]; k2 = key.split("layers.", 1)[1]
        tgt = params[k2 + ".weight"]; w = load_file(f)["weight"]
        assert tuple(w.shape) == tuple(tgt.shape), (key, w.shape, tgt.shape)
        if bias: orig[k2] = tgt.detach().clone()
        with torch.no_grad(): tgt.copy_(w.to(tgt.dtype))
        nsw += 1
    nb = 0
    if bias:   # mean inputs of every swapped linear in the quantized model, on the fit golden's gen positions
        fg = Golden(bias[0]); acc = {k: [torch.zeros(mods[k].in_features, dtype=torch.float64, device="cuda"), 0] for k in orig}
        sel = {"m": None}
        def pre(mod, inp, k):
            x = inp[0].reshape(-1, inp[0].shape[-1]); m = sel["m"]
            if m is not None and m.numel() == x.shape[0]: x = x[m]
            acc[k][0] += x.double().sum(0); acc[k][1] += x.shape[0]
        hs = [mods[k].register_forward_pre_hook(lambda mod, inp, k=k: pre(mod, inp, k)) for k in orig]
        with torch.no_grad():
            for i, d in enumerate(fg.docs):
                if d["row0"] >= bias[1]: break
                tk = fg.doc_tokens(i); sf = d["score_from"]; k = len(tk) - 1 - sf
                g = np.zeros(len(tk), bool); g[sf:sf + k] = fg.gen[d["row0"]:d["row0"] + k]
                sel["m"] = torch.from_numpy(g).cuda()
                dec(input_ids=torch.from_numpy(tk.astype(np.int64))[None].cuda())
        for h in hs: h.remove()
        betas = {}
        for k, (s, n) in acc.items():
            mu = (s / max(n, 1)).float()
            betas[k] = ((orig[k].float() - params[k + ".weight"].float()) @ mu).to(torch.bfloat16)
        del orig
        for k, b in betas.items():
            mods[k].register_forward_hook(lambda mod, inp, out, b=b: out + b); nb += 1
        print(f"bias oracle: {nb} linears, mean input over {next(iter(acc.values()))[1]} positions", file=sys.stderr)
    g = Golden(gd); R = g.meta["rows"]
    D = None if outd == "-" else np.lib.format.open_memmap(outd + ".part.npy", "w+", np.float32, (R, 513))
    KL, T1, t0 = [], [], time.time()
    nd = int(os.environ.get("DRIFT_MAXDOCS", 0))   # test mode: the first N docs only (rows file short, no assert)
    with torch.no_grad():
        for i, d in enumerate(g.docs):
            if nd and i >= nd: break
            tk, sf = g.doc_tokens(i), d["score_from"]
            for j, lq in doc_logsoftmax(dec, head, tk, sf):
                r0 = d["row0"] + j; n = lq.shape[0]; top = g.top[r0:r0 + n]
                ids = torch.from_numpy(np.ascontiguousarray(top["id"])).cuda(); lp = torch.from_numpy(np.ascontiguousarray(top["lp"])).cuda()
                pt = torch.from_numpy(np.asarray(g.rows[r0:r0 + n, 2], np.float64)).cuda()
                kl, t1 = score_rows(lq, ids, lp, pt)
                KL.append(kl.cpu().numpy()); T1.append(t1.cpu().numpy())
                if D is not None:   # the 513 cells, as score_rows: p renormalised over head + tail
                    lpd = lp.double(); S = lpd.exp().sum(1) + pt.clamp_min(0)
                    lqh = lq.gather(1, ids.long()).double()
                    e = lq.exp(); e.scatter_(1, ids.long(), 0.0); qt = e.sum(1, dtype=torch.float64)
                    tiny = torch.finfo(torch.float64).tiny
                    dh = lqh - (lpd - S.log()[:, None])
                    dt = qt.clamp_min(tiny).log() - (pt.clamp_min(tiny).log() - S.log())
                    dt = torch.where(pt > 0, dt, torch.zeros_like(dt))
                    D[r0:r0 + n] = torch.cat([dh, dt[:, None]], 1).clamp(-80, 80).float().cpu().numpy()
    kl, t1 = np.concatenate(KL), np.concatenate(T1); gen = g.gen[:len(kl)]
    assert nd or len(kl) == R
    with open(outr + ".part", "wb") as fh:
        np.stack([kl, t1, gen, np.zeros_like(kl)], 1).astype(np.float32).tofile(fh); fh.flush(); os.fsync(fh.fileno())
    os.replace(outr + ".part", outr)
    if D is not None: D.flush(); del D; os.replace(outd + ".part.npy", outd)
    print(json.dumps({"golden": gd, "wq": wq, "n_swapped": nsw, "n_bias": nb, "positions": int(gen.sum()),
                      "kld": float(kl[gen].mean()), "top1": float(t1[gen].mean()), "sec": round(time.time() - t0, 1)}))


def split(a):
    """per gen row t: KL_s (exact on the 513 cells), KL_avg = KL(p~ || p~ exp(bhat) / Z), the variance terms under p~"""
    gd, outj, files = a[0], a[1], a[2:]
    S = len(files); assert S >= 2
    m = json.load(open(f"{gd}/kld.json")); gen = np.fromfile(f"{gd}/gen.u8", np.uint8).astype(bool); R = m["rows"]
    top = np.memmap(f"{gd}/top.bin", np.dtype([("id", "<i4"), ("lp", "<f4")]), "r", shape=(R, 512))
    rows = np.memmap(f"{gd}/rows.f32", np.float32, "r", shape=(R, 3))
    Ds = [np.load(f, mmap_mode="r") for f in files]
    kls = np.zeros((S, R)); kla = np.zeros(R); vb = np.zeros(R); vx = np.zeros(R); vs = np.zeros((S, R))
    CH = 4096
    for r0 in range(0, R, CH):
        r1 = min(R, r0 + CH)
        lp = top["lp"][r0:r1].astype(np.float64); pt = np.clip(rows[r0:r1, 2].astype(np.float64), 0, None)
        p = np.concatenate([np.exp(lp), pt[:, None]], 1); p /= p.sum(1, keepdims=True)
        D = np.stack([np.asarray(d[r0:r1], np.float64) for d in Ds])          # [S, n, 513]
        D[:, :, 512] = np.where(pt[None] > 0, D[:, :, 512], 0.0)
        # exact KL per seed: q = p exp(D) (each seed's cells sum to 1 up to f16 rounding: renormalise)
        for s in range(S):
            q = p * np.exp(D[s]); Z = q.sum(1)
            kls[s, r0:r1] = (p * (-(D[s]) + np.log(Z)[:, None])).sum(1)
        b = D.mean(0); qa = p * np.exp(b); Za = qa.sum(1)
        kla[r0:r1] = (p * (-b + np.log(Za)[:, None])).sum(1)
        var = lambda X: (p * X ** 2).sum(-1) - (p * X).sum(-1) ** 2
        vb[r0:r1] = var(b); vx[r0:r1] = np.mean([var(D[s] - b) for s in range(S)], 0)
        for s in range(S): vs[s, r0:r1] = var(D[s])
    docs = m["docs"]; di = np.zeros(R, int)
    for j, d in enumerate(docs):
        k = d["tokens"] - 1 - d["score_from"]; di[d["row0"]:d["row0"] + k] = j
    g = gen
    def stats(sel):
        ks, ka = kls[:, sel].mean(0), kla[sel]
        A_exact = ka.sum() - (ks.sum() - ka.sum()) / (S - 1)
        A_var = (vb[sel] - vx[sel] / (S - 1)).sum() / 2; tot_var = vs[:, sel].mean(0).sum() / 2
        return {"rows": int(sel.sum()), "kl_single": float(ks.mean()), "kl_avg": float(ka.mean()),
                "ratio_avg_single": float(ka.sum() / ks.sum()), "c_exact": float(A_exact / ks.sum()),
                "c_var": float(A_var / tot_var), "quad_over_exact": float(tot_var / ks.sum())}
    out = {"seeds": files, "S": S, "all": stats(g)}
    # document bootstrap of c_exact / ratio
    nd = len(docs); ks_d = np.bincount(di[g], kls[:, g].mean(0), nd); ka_d = np.bincount(di[g], kla[g], nd)
    rng = np.random.default_rng(0); cs, rs = [], []
    for _ in range(2000):
        i = rng.integers(0, nd, nd); KS, KA = ks_d[i].sum(), ka_d[i].sum()
        cs.append((KA - (KS - KA) / (S - 1)) / KS); rs.append(KA / KS)
    out["all"]["c_exact_ci95"] = [float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5))]
    out["all"]["ratio_ci95"] = [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))]
    ksm = kls.mean(0); thr = np.percentile(ksm[g], 99)
    out["top1pct"] = stats(g & (ksm >= thr))
    cats = {}
    for j, d in enumerate(docs):
        c = d.get("category") or d.get("source") or "?"; cats.setdefault(c, []).append(j)
    out["per_category"] = {c: stats(g & np.isin(di, js)) for c, js in sorted(cats.items()) if len(js) >= 3}
    out["per_seed_kl"] = [float(kls[s, g].mean()) for s in range(S)]
    rf = [f[:-len(".npy")] + ".f32" if f.endswith(".npy") else None for f in files]   # dump's ROWS.f32 beside it, when named so
    if all(x and os.path.exists(x) for x in rf):
        rk = np.stack([np.fromfile(x, np.float32).reshape(-1, 4)[:, 0] for x in rf])
        out["max_abs_kl_vs_rows"] = float(np.abs(rk[:, g] - kls[:, g]).max())
    json.dump(out, open(outj, "w"), indent=1)
    a_ = out["all"]
    print(f"SPLIT {os.path.basename(outj)}: S {S} KL_single {a_['kl_single']:.5f} KL_avg {a_['kl_avg']:.5f} ratio {a_['ratio_avg_single']:.4f} "
          f"c_exact {a_['c_exact']:.4f} {a_['c_exact_ci95']} c_var {a_['c_var']:.4f} (quad/exact {a_['quad_over_exact']:.3f})")


if __name__ == "__main__":
    {"dump": dump, "split": split}[sys.argv[1]](sys.argv[2:])
