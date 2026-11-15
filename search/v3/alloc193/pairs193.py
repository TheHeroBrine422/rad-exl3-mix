# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""pairs193.py SNAP OUT.jsonl -- RULES-193 §3 data: every paired outcome (candidate vs its reference, one slice) on the
allowed slices.

  stage1   : `decision` events without `stage` (generated + manual candidates, incl. early-stopped) and `tentative`
             events (the stage-1 result of the candidates that went to confirmation); slice = the candidate's submit
             slice; ref = the tentative's `vs`, else the submit's `ref`
  stage2   : `decision` events with stage 2 whose confirm slice is s186b / s186d (never s190b); ref = the `confirm` event's
  reselect : the s186e re-selection rankings (each model vs the size's incumbent at the start)
Row: {tag, kind, src, size, slice, logr, se, k, stopped, ref, cand (assign tag), refa (assign tag), n_result, pred_live}"""
import json, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common193 as C  # noqa: E402


def build(snap):
    st = C.load_state(snap)
    sub = {e["tag"]: e for e in st if e.get("ev") == "submit"}
    nres = {}; n = 0
    for e in st:
        if e.get("ev") == "result": n += 1; nres.setdefault(e["tag"], n)
    stopped = {e["tag"]: bool(e["r"].get("stopped")) for e in st if e.get("ev") == "result"}
    cf = {}
    for e in st:
        if e.get("ev") == "confirm": cf[e["tag"]] = e
    rows = []
    for e in st:
        ev = e.get("ev")
        if ev in ("decision", "tentative") and e.get("logr") is not None and e.get("stage") != 2:
            s = sub.get(e["tag"])
            if s is None or not C.ok_slice(s["slice"]): continue
            ref = e.get("vs") or s.get("ref")
            if ref is None: continue
            rows.append({"tag": e["tag"], "kind": s["kind"], "src": "stage1", "size": float(e["size"]), "slice": s["slice"],
                         "logr": e["logr"], "se": e.get("se"), "k": e.get("k"), "stopped": stopped.get(e["tag"], False),
                         "cand": e["tag"], "refa": ref, "n_result": nres.get(e["tag"]), "pred_live": s.get("pred_logr"),
                         "decision": e.get("decision") or "tentative"})
        elif ev == "decision" and e.get("stage") == 2 and e.get("logr") is not None:
            c = cf.get(e["tag"])
            if c is None or not C.ok_slice(c["slice"]): continue
            rows.append({"tag": e["tag"] + "@" + c["slice"], "kind": sub[e["tag"]]["kind"], "src": "stage2", "size": float(e["size"]),
                         "slice": c["slice"], "logr": e["logr"], "se": e.get("se"), "k": e.get("k"), "stopped": False,
                         "cand": e["tag"], "refa": c["ref"], "n_result": nres.get(c["eval"]), "pred_live": None, "decision": e["decision"]})
        elif ev == "reselect_size" and C.ok_slice(e.get("slice")):
            for r in e.get("ranking", []):
                if r.get("logr") is None or r["of"] == e["prev"]: continue
                rows.append({"tag": r["eval"], "kind": "reselect", "src": "reselect", "size": float(e["size"]), "slice": e["slice"],
                             "logr": r["logr"], "se": r.get("se"), "k": r.get("k"), "stopped": False, "cand": r["of"],
                             "refa": e["prev"], "n_result": nres.get(r["eval"]), "pred_live": None, "decision": "reselect"})
    return rows


if __name__ == "__main__":
    rows = build(sys.argv[1])
    with open(sys.argv[2], "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    import collections
    print(len(rows), collections.Counter(r["src"] for r in rows), collections.Counter(r["slice"] for r in rows))
    print("dropped", C.dropped)
