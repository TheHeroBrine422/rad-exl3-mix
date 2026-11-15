# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""table193.py OUT.md OUT.json -- #193 outcome table from the live run's state (read-only): per inbox file of mine,
the tag, the stage-1 decision (log r, z, docs, stopped), and the stage-2 verdict (log r, z) or, for probes, the probe's
whole-slice paired log r vs the size's incumbent at the probe's time on the same slice (KLD ratio; both whole).
Stage-2 numbers are read for the report only (RULES-193 §0)."""
import json, math, os, sys

DRV = "/workspace/rad-local/workers/lead-srch/run/drv"


def main(out_md, out_js):
    st = [json.loads(l) for l in open(f"{DRV}/state.jsonl")]
    mine = [e for e in st if e.get("ev") == "inbox" and e["file"].startswith("w172-")]
    res = {}; inc_at = {}; cur_inc = {}
    for i, e in enumerate(st):
        if e.get("ev") == "incumbent": cur_inc[float(e["size"])] = e["tag"]
        if e.get("ev") == "submit" and e["tag"] in {m["tag"] for m in mine}: inc_at[e["tag"]] = dict(cur_inc)
        if e.get("ev") == "result": res[e["tag"]] = e["r"]
    sub = {e["tag"]: e for e in st if e.get("ev") == "submit"}
    d1 = {}; d2 = {}; tent = {}
    for e in st:
        if e.get("ev") == "decision" and e.get("stage") != 2: d1[e["tag"]] = e
        if e.get("ev") == "decision" and e.get("stage") == 2: d2[e["tag"]] = e
        if e.get("ev") == "tentative": tent[e["tag"]] = e
    # whole-slice s190a kld per assignment hash (for probes)
    kl = {}
    for t, r in res.items():
        s = sub.get(t)
        if s and r.get("slice") == "s190a" and not r.get("stopped") and r.get("kld"): kl.setdefault(s["hash"], r["kld"])
    rows = []
    for m in mine:
        t = m["tag"]; f = m["file"]; alloc = f[5:].split("-")[0]
        r = {"file": f, "alloc": alloc, "mode": m["mode"], "size": m["size"], "tag": t}
        if m["mode"] == "decide":
            e = tent.get(t) or d1.get(t)
            if e:
                r.update({"s1": "tentative" if t in tent else e["decision"], "s1_logr": e["logr"], "s1_z": e["z"], "s1_k": e.get("k"),
                          "s1_vs": e.get("vs") or sub[t].get("ref"), "stopped": bool(res.get(t, {}).get("stopped"))})
            e2 = d2.get(t)
            if e2: r.update({"s2": e2["decision"], "s2_logr": e2["logr"], "s2_z": e2["z"]})
        else:
            rr = res.get(t)
            if rr and rr.get("kld"):
                it = inc_at.get(t, {}).get(float(m["size"])); ih = sub[it]["hash"] if it in sub else None
                r.update({"kld": rr["kld"], "inc": it, "logr_vs_inc": math.log(rr["kld"] / kl[ih]) if ih in kl else None})
        rows.append(r)
    json.dump(rows, open(out_js, "w"), indent=1)
    # per-allocator tally (lead note 5:00 AM CST): tried / stage-1 decided / tentative / confirmed accept, join / stage-2 reject
    T = {}
    for r in rows:
        if r["mode"] != "decide": continue
        t = T.setdefault(r["alloc"], {"files": 0, "s1": 0, "tent": 0, "accept": 0, "join": 0, "s2rej": 0, "s2pend": 0, "sizes": []})
        t["files"] += 1
        if "s1" in r: t["s1"] += 1
        if r.get("s1") == "tentative":
            t["tent"] += 1
            if "s2" not in r: t["s2pend"] += 1
            elif r["s2"] == "accept": t["accept"] += 1; t["sizes"].append(f"{r['size']:g}A")
            elif r["s2"] == "join": t["join"] += 1; t["sizes"].append(f"{r['size']:g}j")
            else: t["s2rej"] += 1
    L = ["| allocator | files | stage-1 decided | tentative | confirmed accept | confirmed join | stage-2 reject | stage-2 pending | confirmed at (A accept, j join) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for a, t in sorted(T.items(), key=lambda kv: -(kv[1]["accept"] + kv[1]["join"])):
        L.append(f"| {a} | {t['files']} | {t['s1']} | {t['tent']} | {t['accept']} | {t['join']} | {t['s2rej']} | {t['s2pend']} | {' '.join(t['sizes'])} |")
    L += ["", "| file | alloc | size | tag | stage 1 (log r, z, docs) | stage 2 (log r, z) |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r["mode"] == "decide":
            s1 = f"{r.get('s1', 'pending')} {r['s1_logr']:+.4f} z {r['s1_z']:+.2f} k {r['s1_k']}{' stopped' if r.get('stopped') else ''}" if "s1_logr" in r else "pending"
            s2 = f"{r['s2']} {r['s2_logr']:+.4f} z {r['s2_z']:+.2f}" if "s2" in r else ("pending" if r.get("s1") == "tentative" else "-")
        else:
            s1 = f"probe KLD {r['kld']:.5f}, log r vs incumbent {r['logr_vs_inc']:+.4f}" if r.get("logr_vs_inc") is not None else "probe pending"
            s2 = "-"
        L.append(f"| {r['file']} | {r['alloc']} | {r['size']:g} | {r['tag'].split('-', 2)[2]} | {s1} | {s2} |")
    open(out_md, "w").write("\n".join(L) + "\n"); print("\n".join(L[:len(T) + 2]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
