"""kldpair.py GOLDEN RES A B [A B ...] -- KLD rows of pack pairs on one golden (radiance --kld-out .rows: float32 kl,
nll_q, nll_p, top1 per position): mean KLD of each, |d mean|, per-document |d mean| (max, and that document), per-position
max / mean |d|, top-1 agreement and flips, byte-identity. Documents from GOLDEN/kld.json's docs (row0). (w136, #164-02)"""
import ast, json, sys
import numpy as np

G, R = sys.argv[1], sys.argv[2]
g = json.load(open(f"{G}/kld.json")); docs = g["docs"]
docs = ast.literal_eval(docs) if isinstance(docs, str) else docs
row0 = [int(d["row0"]) for d in docs]


def load(t):
    j = json.load(open(f"{R}/kld-{t}.json"))["all"]
    return j, np.fromfile(f"{R}/kld-{t}.json.rows", np.float32).reshape(-1, 4)


out = []
for a, b in zip(sys.argv[3::2], sys.argv[4::2]):
    (ja, ra), (jb, rb) = load(a), load(b)
    n = len(ra); assert rb.shape == ra.shape, (ra.shape, rb.shape)
    bounds = row0 + [n]
    da = ra[:, 0].astype(np.float64); db = rb[:, 0].astype(np.float64); d = np.abs(da - db)
    per = [(abs(da[s:e].mean() - db[s:e].mean()), i, e - s) for i, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])) if e > s]
    worst = max(per)
    r = {"A": a, "B": b, "positions": n, "docs": len(per), "kld_A": ja["kld"]["mean"], "kld_B": jb["kld"]["mean"],
         "abs_dmean": abs(da.mean() - db.mean()), "doc_abs_dmean_max": worst[0], "doc_worst": worst[1], "doc_worst_rows": worst[2],
         "pos_abs_d_max": float(d.max()), "pos_abs_d_mean": float(d.mean()), "pos_d0_share": float(np.mean(d == 0)),
         "top1_A": ja["top1_agreement"], "top1_B": jb["top1_agreement"], "top1_flips": int((ra[:, 3] != rb[:, 3]).sum()),
         "rows_identical": bool(np.array_equal(ra.view(np.uint32), rb.view(np.uint32)))}
    out.append(r)
    print(json.dumps(r))
