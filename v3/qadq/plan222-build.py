#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""plan222-build.py META_DIR COUNT_DIR OUT_PLAN.json [--calib-share 0.2] [--seed 1] -- the extended QAD data plan (w222, ledger use qad214-ext, role train). Works on metadata only (kld.json / tokens / gen masks), with the trainer's own
`data181.plan`. Base = w214's plan exactly (chunks xc8k-b2-c0, xc8k-b2-c1, tr32k-b2-c0 + calib pool train214, the 22 fixed-eval documents removed, order = permutation(n, seed 0): reproduced and checked against w214's printed
order sha1 b9c122bbd7e2 and 2,519,265 rows). Extension = the documents of COUNT_DIR/ext-docs.json (new xcorp / trace goldens deduplicated against everything, + calib `qad`-pool documents of variant V2 that are not
train214: a random subset sized so that calib is --calib-share of the whole plan), appended after the base items; their order = a seeded shuffle appended to w214's order, so the stream queues order[s::V] begin with w214's.
Writes OUT_PLAN.json (the trainer's `--plan-ext` input: new goldens, items in order, expected rows) and prints the plan's size and where it departs from w214's."""
import hashlib, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "qad181"))
import data181 as D
meta, cnt, outp = sys.argv[1:4]; a = sys.argv[4:]
share = float(a[a.index("--calib-share") + 1]) if "--calib-share" in a else 0.2; seed = int(a[a.index("--seed") + 1]) if "--seed" in a else 1
CAP, MINR = 8192, 16; V, Q = 24, 1024
ext = json.load(open(f"{cnt}/ext-docs.json")); fe = json.load(open(f"{meta}/../fe_and_md5.txt").readline() if False else open(f"{meta}/../fe_and_md5.txt")) if False else None
fe = json.loads(open(f"{meta}/../fe_and_md5.txt").readline())


class G:                                    # metadata-only Gold
    def __init__(self, name, d, gen):
        self.name = name; m = json.load(open(f"{d}/kld.json")); self.docs = m["docs"]; self.gen = gen
    def ntok(self, i): return self.docs[i]["tokens"]
    def nrows(self, i, cap):
        d = self.docs[i]; b = min(cap, d["tokens"]); j = np.arange(max(0, d["score_from"]), min(b, d["tokens"] - 1)); r = d["row0"] + j - d["score_from"]
        return int(self.gen[r].sum()) if len(r) else 0
base_chunks = ["xc8k-b2-c0", "xc8k-b2-c1", "tr32k-b2-c0"]
gens = [G(c, f"{meta}/{c}", np.fromfile(f"{cnt}/genmask/{c}.u8", np.uint8)) for c in base_chunks]
cal = G("calib", f"{meta}/calib", np.fromfile(f"{meta}/calib/gen.u8", np.uint8)); gens.append(cal)
pl = json.load(open(f"{meta}/../pools214.json"))["pool"]
items = []
for gi, g in enumerate(gens):
    for i in range(len(g.docs)):
        if g.name == "calib" and pl[i] != "train214": continue
        nr = g.nrows(i, CAP)
        if nr >= MINR: items.append((gi, i, min(CAP, g.ntok(i)), nr))
fes = set(fe); items = [x for x in items if not (x[0] == len(gens) - 1 and x[1] in fes)]
order0 = np.random.default_rng(0).permutation(len(items)); sha = hashlib.sha1(np.asarray(order0, np.int64).tobytes()).hexdigest()[:12]
steps0 = D.plan(items, order0, V, Q); rows0 = lambda steps, it: np.cumsum([sum(it[s_[0]][3] if False else 0 for s_ in []) for _ in steps])
def cum_rows(steps, its):
    out = []; t = 0
    for st in steps:
        for segs in st:
            for (it, sa, sb) in segs:
                gi, i, n, nr = its[it]; g = gens_all[gi]
                j = np.arange(max(sa, g.docs[i]["score_from"]), min(sb, g.docs[i]["tokens"] - 1)); r = g.docs[i]["row0"] + j - g.docs[i]["score_from"]; t += int(g.gen[r].sum()) if len(r) else 0
        out.append(t)
    return np.array(out)
gens_all = list(gens)
c0 = cum_rows(steps0, items)
print(f"base: {len(items)} items, order sha1 {sha} (w214: b9c122bbd7e2), {len(steps0)} steps, rows {c0[-1]} (w214: 2519265)")
assert sha == "b9c122bbd7e2" and c0[-1] == 2519265, "the base plan is not w214's"
# ---------------- extension items
newg = []
for e in ext["chunks"]:
    if e["golden"] not in [x.name for x in newg]:
        newg.append(G(e["golden"], f"{meta}/{e['golden']}", np.fromfile(f"{cnt}/genmask/{e['golden']}.u8", np.uint8)))
gens_all += newg; gi_of = {g.name: k for k, g in enumerate(gens_all)}
xitems = [(gi_of[e["golden"]], e["doc"]) for e in ext["chunks"]]
xrows = sum(e["rows"] for e in ext["chunks"])
qadc = [c for c in ext["calib_docs"] if c["pool"] == "qad"]; rng = np.random.default_rng(seed + 100); rng.shuffle(qadc)
base_cal = sum(x[3] for x in items if x[0] == len(base_chunks)); base_x = sum(x[3] for x in items if x[0] != len(base_chunks))
target = share / (1 - share) * (base_x + xrows) - base_cal           # calib rows to add so that calib = share of the whole plan
pick = []; t = 0
for c in qadc:
    if t >= target: break
    pick.append(c); t += c["rows"]
citems = [(len(base_chunks), c["doc"]) for c in pick]
new = [(gi, i) for gi, i in xitems + citems]
perm = np.random.default_rng(seed).permutation(len(new)); new = [new[k] for k in perm]
n_old = len(items); full = list(items)
for gi, i in new:
    g = gens_all[gi]; nr = g.nrows(i, CAP); assert nr >= MINR; full.append((gi, i, min(CAP, g.ntok(i)), nr))
order = np.concatenate([order0, n_old + np.arange(len(new))]); assert (order[:n_old] == order0).all()
steps = D.plan(full, order, V, Q); c1 = cum_rows(steps, full)
# first step where the new plan departs from w214's
dep = next((k for k in range(min(len(steps), len(steps0))) if steps[k] != steps0[k]), None)
print(f"new: {len(full)} items ({len(new)} new: {len(xitems)} xcorp / trace docs, {len(citems)} calib qad docs), {len(steps)} steps, rows {c1[-1]}")
print(f"identical to w214's plan for the first {dep} steps ({int(c0[dep - 1])} rows); w214's plan has {len(steps0)} steps, its tail from step ~{int(np.searchsorted(c0, 0.8 * c0[-1]))}")
tag = lambda gi: "calib" if gens_all[gi].name == "calib" else ("tr32k" if gens_all[gi].name.startswith("tr32k") else "xc8k")
def mix(steps_, upto):
    rows = {}
    for st in steps_[:upto]:
        for segs in st:
            for (it, sa, sb) in segs:
                gi, i, n, nr = full[it]; g = gens_all[gi]; j = np.arange(max(sa, g.docs[i]["score_from"]), min(sb, g.docs[i]["tokens"] - 1)); r = g.docs[i]["row0"] + j - g.docs[i]["score_from"]
                rows[tag(gi)] = rows.get(tag(gi), 0) + (int(g.gen[r].sum()) if len(r) else 0)
    s = sum(rows.values()); return {k: f"{v / s * 100:.1f} %" for k, v in sorted(rows.items())}, s
for B in (2e6, 3e6, 4e6, 6e6, 8e6, 9e6, 1e7):
    k = int(np.searchsorted(c1, B)) + 1
    if k <= len(steps): print(f"budget {B:.0e}: main end at step {k} ({int(c1[k - 1])} rows), mix {mix(steps, k)[0]}")
act = [sum(1 for segs in st if segs) for st in steps]; dense = next((k for k, x in enumerate(act) if x < V), len(steps))
print(f"all {V} streams busy for the first {dense} steps ({int(c1[dense - 1])} rows); the last stream ends at step {len(steps)} ({int(c1[-1])} rows)")
print("whole plan mix:", mix(steps, len(steps)))
json.dump({"ledger_use": "qad214-ext", "role": "train", "cap": CAP, "min_rows": MINR, "base": "w214 plan (chunks xc8k-b2-c0, xc8k-b2-c1, tr32k-b2-c0 + calib pool train214 minus the fixed-eval docs; order seed 0)",
           "new_goldens": [g.name for g in newg], "items": [[gens_all[gi].name, int(i), int(gens_all[gi].nrows(i, CAP))] for gi, i in new], "order_seed": seed, "calib_share": share,
           "n_base_items": n_old, "rows_total": int(c1[-1]), "steps": len(steps), "dense_steps": int(dense), "departs_from_w214_at_step": dep}, open(outp, "w"))
