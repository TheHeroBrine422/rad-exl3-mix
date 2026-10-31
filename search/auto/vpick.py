#!/usr/bin/env python3
"""vpick.py FINAL.q OUT.q [REPORT.md] -- (EXL3-V2: only the picks and OUT.q.pref, the long-context queue, were used;
the public-corpus queue OUT.q was not run.) Pick each target's winner(s) from the ver2 final by a paired per-document
bootstrap, and write the full-public queue (one "TAG MIX" line each, in run order) for pubverify.sh.

Per target T (the v2-knap-*-T candidates of FINAL.q with ver2 scores): the winner is the lowest ver2 KLD; a candidate
is a FINALIST if P(it beats the winner) >= VP_OVERLAP (0.1) over VP_BOOT (2000) document resamples (the same documents
for every candidate). Max 3 finalists a target. Tie rule (2026-10-30): among finalists, an all-ours mix (every
weight from our quants) is listed as preferred. Targets in VP_WINNER_ONLY (1.5, 1.75: research points) send only the
winner to public. Run order: one reference container first (VP_REFCHECK, sc4.00-e4: confirms the eval build matches
the V1 public scores), then 2.0-5.0, 5.5-7.0, 1.5/1.75, then the TP2 sizes (>= 8). Report: per target, every
candidate's ver2 KLD, delta vs the winner with a 95% interval, P(beats winner), and the nearest reference's KLD."""
import json, os, re, sys
import numpy as np
W = os.environ.get("RXM_WORK", "/root")
OURS = os.environ.get("RXM_OURS", "/data/ours"); MODELS = os.environ.get("RXM_MODELS", f"{W}/models/final")
FQ, OUTQ = sys.argv[1:3]; REP = sys.argv[3] if len(sys.argv) > 3 else None
NB = int(os.environ.get("VP_BOOT", 2000)); OV = float(os.environ.get("VP_OVERLAP", 0.1))
WO = {float(x) for x in os.environ.get("VP_WINNER_ONLY", "1.5 1.75").split()}
G = os.environ.get("VP_GOLDEN", f"{W}/golden-ver2"); PX = os.environ.get("VP_PREFIX", "v2-"); P = 26895998464
m = json.load(open(f"{G}/kld.json")); gen = np.fromfile(f"{G}/gen.u8", np.uint8).astype(bool)
spans = [(d["row0"], d["row0"] + d["tokens"] - 1 - d["score_from"]) for d in m["docs"]]
nd = np.array([gen[a:b].sum() for a, b in spans], float)
res = {}
for l in open(f"{W}/evq/results.jsonl"):
    r = json.loads(l)
    if "kld" in r: res[r["tag"]] = r


def perdoc(tag):
    p = f"{W}/evq/scores/{tag}.f32"
    if not os.path.exists(p): return None
    s = np.fromfile(p, np.float32).reshape(-1, 4)[:, 0].astype(np.float64)
    return np.array([s[a:b][gen[a:b]].sum() for a, b in spans])


lines = [l.split() for l in open(FQ) if l.strip()]
mixes = {l[0]: l[2] for l in lines}
cands, refs = {}, {}
for t in mixes:
    mm = re.fullmatch(re.escape(PX) + r"(knap-.+-sc4\.00-e4-([\d.]+))", t)
    if mm: cands.setdefault(mm.group(2), []).append(t)
    elif t.startswith(PX + "ref-"): refs[t] = None
rng = np.random.default_rng(0); IX = rng.integers(0, len(nd), (NB, len(nd)))
den = nd[IX].sum(1)


def ours(path):
    w = [l.split() for l in open(path) if l.strip() and not l.startswith("token_embd")]
    return len(w) >= 337 and all(OURS + "/" in q or q.endswith("q38-27b-bf16.rad") for _, q in w)


refk = {}
for t in refs:
    if t in res: refk[t] = (res[t]["bytes"] * 8 / P, res[t]["kld"])
out, rep, order, prefs = [], ["# ver2 final picks (paired per-document bootstrap)\n"], [], []
SKIP = {float(x) for x in os.environ.get("VP_SKIP", "10.0").split()}   # dropped sizes (2026-10-31: 10.0)
for T in sorted(cands, key=float):
    if float(T) in SKIP: continue
    rows = []
    for t in cands[T]:
        d = perdoc(t)
        if d is None: continue
        rows.append((d.sum() / nd.sum(), t, d))
    if not rows: continue
    rows.sort(key=lambda x: x[0]); kw, tw, dw = rows[0]
    bw = dw[IX].sum(1) / den
    fin = []
    rep.append(f"\n## {T} bpw\n\n| candidate | real bpw | ver2 KLD | vs winner (95% CI) | P(beats winner) | ours |\n|---|---|---|---|---|---|")
    for k, t, d in rows:
        bc = d[IX].sum(1) / den; rel = bc / bw - 1
        pb = float((bc < bw).mean()) if t != tw else 1.0
        ow = ours(mixes[t]); b = res[t]["bytes"] * 8 / P
        if t == tw or pb >= OV: fin.append((k, t, ow))
        rep.append(f"| {t[len(PX):]} | {b:.3f} | {k:.5f} | {'winner' if t == tw else f'{rel.mean():+.1%} ({np.percentile(rel, 2.5):+.1%} .. {np.percentile(rel, 97.5):+.1%})'} | "
                   f"{'' if t == tw else f'{pb:.2f}'} | {'yes' if ow else ''} |")
    fin = fin[:1] if float(T) in WO else fin[:3]
    pref = next((f for f in fin if f[2]), fin[0])
    near = min(refk.items(), key=lambda x: abs(x[1][0] - float(T)), default=None)
    rep.append(f"\nfinalists: {', '.join(f[1][len(PX):] for f in fin)}; preferred (tie rule: ours): {pref[1][len(PX):]}"
               + (f"; nearest reference {near[0][len(PX) + 4:]} ({near[1][0]:.2f} bpw, KLD {near[1][1]:.5f})" if near else ""))
    grp = 0 if 2.0 <= float(T) <= 5.0 else 1 if float(T) <= 7.0 and float(T) > 5.0 else 2 if float(T) < 2.0 else 3
    for f in fin: order.append((grp, float(T), f[1]))
    prefs.append(pref[1])
ref = os.environ.get("VP_REFCHECK", "sc4.00-e4")
with open(OUTQ, "w") as o:
    if ref: o.write(f"pubref-{ref} {MODELS}/q38-27b-{ref}.rad -\n")
    for _, T, t in sorted(order):
        o.write(f"pub-{t[len(PX):]} {MODELS}/q38-27b-sc4.00-e4.rad {mixes[t]}\n")
if REP: open(REP, "w").write("\n".join(rep) + "\n")
with open(OUTQ + ".pref", "w") as o:   # the preferred mix per target (long-context check)
    for t in prefs: o.write(f"long-{t[len(PX):]} {MODELS}/q38-27b-sc4.00-e4.rad {mixes[t]}\n")
print(f"vpick: {len(order)} finalists over {len(cands)} targets -> {OUTQ}")
