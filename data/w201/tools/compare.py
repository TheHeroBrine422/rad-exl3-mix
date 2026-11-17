#!/usr/bin/env python3
# compare.py GOLDEN REF_BASE NEW_BASE -- equivalence of two radiance score runs on one golden (w201).
# BASE = a score's path without suffix (reads BASE.json and BASE.json.rows). Run with -I.
# Metrics: the coordinator's slice KLD (hostd.py doc_sums: per-document KL over the golden's gen.u8 positions, the
# slice mean), radiance's all-position mean (BASE.json all.kld.mean), per-document diffs, per-position and top-1 diffs,
# and whether the two .rows files are byte-identical.
import json, sys
import numpy as np

gold, ref, new = sys.argv[1:4]
m = json.load(open(f"{gold}/kld.json"))
gen = np.fromfile(f"{gold}/gen.u8", np.uint8).astype(bool)


def load(base):
    j = json.load(open(base + ".json"))
    K = np.fromfile(base + ".json.rows", np.float32).reshape(-1, 4)
    return j, K


def docs(K):
    out, allk = [], []
    for d in m["docs"]:
        s = slice(d["row0"], d["row0"] + d["tokens"] - 1 - d["score_from"])
        g = gen[s]
        k = K[s, 0][g].astype(np.float64)
        out.append((round(float(k.sum()), 6), int(g.sum())))
        allk.append(k)
    return out, np.concatenate(allk)


jr, Kr = load(ref)
jn, Kn = load(new)
dr, kr = docs(Kr)
dn, kn = docs(Kn)
per_doc = []
for i, (a, b) in enumerate(zip(dr, dn)):
    if a[1] and b[1]:
        per_doc.append((abs(a[0] / a[1] - b[0] / b[1]), i))
    else:
        per_doc.append((abs(a[0] - b[0]), i))
per_doc.sort(reverse=True)
top_docs = [{"doc": i, "ref_mean": (dr[i][0] / dr[i][1] if dr[i][1] else None),
             "new_mean": (dn[i][0] / dn[i][1] if dn[i][1] else None), "abs_diff": v}
            for v, i in per_doc[:5]]
dpos = np.abs(Kr[:, 0].astype(np.float64) - Kn[:, 0].astype(np.float64))
res = {
    "golden": gold,
    "ref": ref, "new": new,
    "coord_kld_ref": float(kr.mean()), "coord_kld_new": float(kn.mean()),
    "coord_kld_diff": float(kn.mean() - kr.mean()),
    "coord_positions": int(len(kr)),
    "all_mean_ref": jr["all"]["kld"]["mean"], "all_mean_new": jn["all"]["kld"]["mean"],
    "all_mean_diff": jn["all"]["kld"]["mean"] - jr["all"]["kld"]["mean"],
    "all_positions": int(Kr.shape[0]),
    "top1_ref": jr["all"]["top1_agreement"], "top1_new": jn["all"]["top1_agreement"],
    "rows_byte_identical": bool(open(ref + ".json.rows", "rb").read() == open(new + ".json.rows", "rb").read()),
    "max_abs_pos_kl_diff": float(dpos.max()),
    "pos_kl_differ": int((dpos > 0).sum()),
    "top1_mismatch_positions": int((Kr[:, 3] != Kn[:, 3]).sum()),
    "docs": len(dr),
    "max_doc_mean_abs_diff": top_docs[0]["abs_diff"] if top_docs else 0.0,
    "top_doc_diffs": top_docs,
    "docs_sum_differ": int(sum(1 for a, b in zip(dr, dn) if a != b)),
}
print(json.dumps(res, indent=1))
