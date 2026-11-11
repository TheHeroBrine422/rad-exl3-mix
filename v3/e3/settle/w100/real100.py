#!/usr/bin/env python3
"""real100.py PIPELINE TDIR CKPT HESS HOUT OUT.jsonl -- w100 (#128, ../w100.md): error of REAL 27B sub-1 codes (w92's
K0.125 / K0.875 rxq jobs, layers 2-3, 13 tensors, i4, settings ed97adb7) against a zero tensor. Each t-file (TDIR/*/t-*.safetensors)
is decoded by rxq.deq (the plugin's arithmetic, f16 suh / svh: the weight the level ships, before C9's svh fold),
then per output row d_n = q_n - w_n under the calib H_in of its group (undamped, raw sum / rows = post.hessian):
  rel1 = Σ d M d^T / Σ w M w^T (one-sided; Q = 0 gives 1), rel2 = the same weighted by the bn1920 h (two-sided),
  rel1_opt = rel1 after the best per-row scalar (what a free svh could still do; Q = 0 is its own optimum, so <= 1),
  C9's e_n statistics (writers: e_mean must equal the job's own c9 record: gate G3, 1e-6 relative).
One JSON line per tensor. CPU, one thread, f64."""
import glob, json, os, sys
import torch
from safetensors import safe_open

PIPE, TD, CK, HS, HO, OUT = sys.argv[1:7]
sys.path.insert(0, PIPE)
from rxq import deq, post, qwen35  # noqa: E402
torch.set_num_threads(1)

ix = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
done = {json.loads(l)["key"] + "|" + json.loads(l)["K"] for l in open(OUT)} if os.path.exists(OUT) else set()
with open(OUT, "a") as fo:
    for f in sorted(glob.glob(f"{TD}/*/t-*.safetensors")):
        with safe_open(f, "pt") as h:
            md = h.metadata(); key = md["key"]; K = md["K"]
            if key + "|" + K in done: continue
            tr, suh, svh = (h.get_tensor(f"{key}.{t}") for t in ("trellis", "suh", "svh"))
        Q = deq.weight(tr, suh, svh, tr.shape[2] / 16, torch.from_numpy(deq.lut(md["codebook"]))).T.double()
        with safe_open(f"{CK}/{ix[key + '.weight']}", "pt") as h: W = h.get_tensor(key + ".weight").double()
        assert Q.shape == W.shape
        l, gi, _, hn, _ = qwen35.locate(key)
        M = post.hessian(f"{HS}/{hn}.safetensors", "cpu")
        with safe_open(f"{HO}/{key}.safetensors", "pt") as h: hw = h.get_tensor("hout_diag").double()
        D = Q - W
        QM = Q @ M; WM = W @ M
        qq = (QM * Q).sum(1); qw = (QM * W).sum(1); ww = (WM * W).sum(1); del QM, WM
        dd = qq - 2 * qw + ww
        c = qw / qq.clamp(min=1e-300); dopt = ww - qw * c                 # best per-row scalar: ww - qw^2 / qq
        en = dd / ww.clamp(min=1e-300)
        fam = qwen35.family(key)
        rec = {"key": key, "layer": l, "family": fam, "K": K, "codebook": md["codebook"], "level": md.get("level"),
               "rel1": float(dd.sum() / ww.sum()), "rel2": float((hw * dd).sum() / (hw * ww).sum()),
               "rel1_opt": float(dopt.sum() / ww.sum()), "rel2_opt": float((hw * dopt).sum() / (hw * ww).sum()),
               "rows_rel_gt1": int((en > 1).sum()), "rows_rel_gt097": int((en >= 0.97).sum()), "rows": int(en.numel()),
               "e_mean": float(en.clamp(0, 0.5).mean()), "e_raw_mean": float(en.mean()), "e_q50": float(en.median()),
               "row_scale_opt_mean": float(c.mean()), "suh_zero": int((suh == 0).sum()), "svh_zero": int((svh == 0).sum())}
        cf = os.path.join(os.path.dirname(f), "log.jsonl")
        if md.get("c9"):
            al = float(md["c9"]); cc, st = post.c9_rows(Q, W, M, al)
            logged = next((json.loads(x).get("c9") for x in open(cf) if json.loads(x)["key"] == key), None)
            rec["c9"] = st
            rec["G3_rel"] = abs(st["e_mean"] / logged["e_mean"] - 1) if logged else None
        print(json.dumps(rec), flush=True); fo.write(json.dumps(rec) + "\n"); fo.flush()
        del Q, W, M, D
