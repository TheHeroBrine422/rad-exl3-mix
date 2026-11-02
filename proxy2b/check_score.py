#!/usr/bin/env python3
"""check_score.py MODEL GOLDEN SCORE_PY [NDOCS] -- gate for common.score_rows: on the first NDOCS documents of a
mkgold.py golden, the model's own logits plus noise (so KL is not ~0) scored by common.score_rows (GPU, float64) and by
the challenge's tools/score.py Reference.score_rows (numpy, the leaderboard's arithmetic). Pass (fp32 log-softmax,
fp64 sums): max per-row |dKL| < 1e-5 and |mean dKL| < 1e-4 of the mean KL; top1 differs on < 0.1% of rows (exact
near-ties may flip). Also the unperturbed model against its own golden: mean KL < 1e-4 (bf16
model vs itself through the top-512 format: only float noise)."""
import importlib.util, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model, Golden, score_rows, doc_logsoftmax
mp, gd, sp = sys.argv[1:4]; nd = int(sys.argv[4]) if len(sys.argv) > 4 else 4
spec = importlib.util.spec_from_file_location("score", sp); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
ref = S.Reference(gd); g = Golden(gd)
_, dec, head = load_model(mp)
torch.manual_seed(0); worst = 0.0; klsum = 0.0; n = 0; t1bad = 0; self_kl = []
for i in range(min(nd, len(g.docs))):
    d = g.docs[i]
    for j, lq in doc_logsoftmax(dec, head, g.doc_tokens(i), d["score_from"]):
        r0 = d["row0"] + j; m = lq.shape[0]; top = g.top[r0:r0 + m]
        args = (torch.from_numpy(np.ascontiguousarray(top["id"])).cuda(), torch.from_numpy(np.ascontiguousarray(top["lp"])).cuda(),
                torch.from_numpy(np.asarray(g.rows[r0:r0 + m, 2], np.float64)).cuda())
        self_kl.append(score_rows(lq, *args)[0].cpu().numpy())
        lqn = torch.log_softmax(lq + 0.5 * torch.randn_like(lq), dim=1)   # float32, as the pipeline
        kl, t1 = score_rows(lqn, *args)
        kr, tr, *_ = ref.score_rows(i, j, lqn.cpu().numpy())        # score.py: float64 from the same logits
        dk = kl.cpu().numpy() - kr
        worst = max(worst, float(np.abs(dk).max())); dsum = globals().get("dsum", 0.0) + float(dk.sum()); globals()["dsum"] = dsum
        klsum += float(kr.sum()); n += m
        t1bad += int((t1.cpu().numpy() != tr).sum())
mk, sk = klsum / n, float(np.concatenate(self_kl).mean())
md = dsum / n
ok = worst < 1e-5 and abs(md) < 1e-4 * mk and t1bad < 1e-3 * n and sk < 1e-4
print(f"{n} rows: mean KL (noisy) {mk:.5f}, max |GPU - score.py| {worst:.3e}, mean diff {md:.2e} ({md / mk:.1e} rel), top1 mismatches "
      f"{t1bad}; model vs its own golden {sk:.2e}\nSCORE CHECK {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
