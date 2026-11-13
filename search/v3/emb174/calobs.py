#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""calobs.py PLAN.json DRVROOT REGISTRY DOCSDIR OUT.obs.jsonl [--prefix emb174] -- the embedding calibration evals
(calsubmit.py) as fit rows for the search (w135, #174), in the driver's obs format ({"tag", "assign", "label", "kld",
"se", "size"}), one row per slice part as driver.add_obs: the row = the incumbent's assignment with the calibration
embedding, its KLD on the part's documents, SE = the paired delta-method SE vs the incumbent on the same documents.
Each label is the driver's whole-slice label with a prefix ("emb174-s171b|c|full"): its own offset with the incumbents'
own rows as anchors (se 0, included), so the rows stand on their own whether or not the live run keeps the slice's
old (f16) labels after the A8 switch. Also prints the measured log ratios per part and their pmean (the accept
statistic) with SE. A run may name "assign_of" (the incumbent whose assignment the rows carry) when "of" is another score
of it (a8anchor174.sh: the A8 reference w132 scored, docs written by a8docs.py)."""
import argparse, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import seqrule as SR         # noqa: E402


def docs(p): return json.load(open(p))["m"]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="calobs.py")
    ap.add_argument("plan"); ap.add_argument("root"); ap.add_argument("registry"); ap.add_argument("docs"); ap.add_argument("out")
    ap.add_argument("--prefix", default="emb174"); ap.add_argument("--json")
    a = ap.parse_args(argv)
    P = json.load(open(a.plan)); reg = json.load(open(a.registry))["slices"]
    sizes = {}
    for l in open(f"{a.root}/state.jsonl"):
        r = json.loads(l)
        if r.get("ev") == "submit": sizes[r["tag"]] = r["size"]
    rows, anchors, rep = [], {}, {}
    for run in P["runs"]:
        t, of = run["tag"], run["of"]
        sl = next((s for s in reg if os.path.exists(f"{a.docs}/{t}-{s}.json")), None)
        if sl is None: print(f"{t}: no result yet", file=sys.stderr); continue
        A, B = docs(f"{a.docs}/{of}-{sl}.json"), docs(f"{a.docs}/{t}-{sl}.json")
        parts = [d["part"] for d in reg[sl]["docs"]]
        if len(A) != len(B) or any(x is None for x in A + B): raise SystemExit(f"{t}: not a whole-slice pair with {of}")
        if any(x[1] != y[1] for x, y in zip(A, B)): raise SystemExit(f"{t}: scored positions differ from {of}'s")
        ao = run.get("assign_of", of)                                   # A8 anchors: "of" names the reference score, assign_of the incumbent
        asg = json.load(open(f"{a.root}/assign/{ao}.json")); asg_c = dict(asg, **{"token_embd.weight": run["emb"]})
        size = sizes.get(ao); Pall = SR.Paired(len(A), {"kind": "full", "stat": "pmean"}); e = {}
        for p in sorted(set(parts)):
            ix = [i for i in range(len(A)) if parts[i] == p]
            Pp = SR.Paired(len(ix), {"kind": "full"})
            for i in ix: Pp.add(A[i][0], B[i][0]); Pall.add(A[i][0], B[i][0], p)
            label = f"{a.prefix}-{sl}|{p}|full"; npos = sum(A[i][1] for i in ix)
            ka, kb = sum(A[i][0] for i in ix) / npos, sum(B[i][0] for i in ix) / npos
            if (of, label) not in anchors:
                anchors[(of, label)] = {"tag": f"{of}@{label}", "assign": asg, "label": label, "kld": ka, "se": 0.0, "size": size}
            rows.append({"tag": f"{t}@{label}", "assign": asg_c, "label": label, "kld": kb, "se": Pp.se(), "size": size})
            e[p] = {"kld_inc": ka, "kld": kb, "logr": math.log(kb / ka), "se": Pp.se(), "docs": len(ix), "dkld": kb - ka}
        rep[t] = {"of": of, "size": size, "emb": run["emb"], "slice": sl, "parts": e, "pmean": Pall.d(), "se": Pall.se()}
        print(f"{t:22s} {size:>4} {run['emb']:5s} " + " ".join(f"{p}: {v['logr']:+.4f} ± {v['se']:.4f} (dKL {v['dkld']:+.6f})" for p, v in e.items())
              + f" | pmean {Pall.d():+.4f} ± {Pall.se():.4f}")
    with open(a.out + ".part", "w") as f:
        for r in list(anchors.values()) + rows: f.write(json.dumps(r) + "\n")
    os.replace(a.out + ".part", a.out)
    if a.json: json.dump(rep, open(a.json, "w"), indent=1)
    print(f"{a.out}: {len(anchors)} anchors + {len(rows)} rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
