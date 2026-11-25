#!/usr/bin/env python3
"""report.py -- markdown tables for the final report from manifest.jsonl / sources.json / dedup.json:
cell budget vs final, underfilled sources, dedup totals, chat tags, agent-session tool definitions, language and code balance."""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT


def md(rows, head):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def main():
    rows = [json.loads(l) for l in open(f"{ROOT}/manifest.jsonl")]
    sj = json.load(open(f"{ROOT}/sources.json")); dd = json.load(open(f"{ROOT}/dedup.json"))
    xg = [r for r in rows if r["origin"] == "xgap-v1"]
    out = []
    cells = collections.OrderedDict()
    for s in sj["xgap-v1"]:
        c = cells.setdefault(s["cell"], [0, 0, 0, 0]); b = s["budgets"]
        c[0] += sum(v for k, v in b.items() if k != "xl"); c[1] += b.get("xl", 0); c[2] += s["docs_final"]; c[3] += s["tokens_final"]
    # code long-files budget counts toward long-docs, not code-langs
    out.append("### cells: budget vs final (xgap-v1)\n\n" + md([(k, f"{v[0]:,}", f"{v[1]:,}", f"{v[2]:,}", f"{v[3]:,}") for k, v in cells.items()],
                                                             ["cell (by source registry)", "budget", "xl budget", "docs final", "tokens final"]))
    bycell = collections.defaultdict(lambda: [0, 0])
    for r in xg: bycell[r["cell"]][0] += 1; bycell[r["cell"]][1] += r["tokens"]
    out.append("### cells as tagged (a code file of 8k+ tokens is a long-docs document)\n\n" + md([(k, f"{v[0]:,}", f"{v[1]:,}") for k, v in sorted(bycell.items())], ["cell", "docs", "tokens"]))
    under = []
    for s in sj["xgap-v1"]:
        b = sum(v for k, v in s["budgets"].items() if k != "xl")
        if s["tokens_final"] < 0.85 * b: under.append((s["id"], f"{s['tokens_final']:,}", f"{b:,}", f"{100 * s['tokens_final'] / b:.0f} %", s["stopped"]))
    out.append("### underfilled sources (< 85 % of budget, final tokens after dedup)\n\n" + md(under, ["source", "final", "budget", "%", "stopped"]))
    out.append("### long-docs by band and source category (tokens)\n\n" + md(
        [(c, *[f"{sum(r['tokens'] for r in xg if r['cell'] == 'long-docs' and r['category'] == c and r['length_band'] == b):,}" for b in ["8-16k", "16-32k", "32-64k", "64-128k"]])
         for c in sorted({r["category"] for r in xg if r["cell"] == "long-docs"})], ["category", "8-16k", "16-32k", "32-64k", "64-128k"]))
    # code languages
    cl = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r in xg:
        if r["category"] == "code":
            a = cl[r["lang"]]
            if r["cell"] == "code-langs": a[0] += 1; a[1] += r["tokens"]
            else: a[2] += 1; a[3] += r["tokens"]
    out.append("### code languages (code-langs = files 1k-8k tokens; long = 8k+ files, inside long-docs)\n\n" + md(
        [(k, v[0], f"{v[1]:,}", v[2], f"{v[3]:,}") for k, v in sorted(cl.items())], ["lang", "docs 1-8k", "tokens", "docs 8k+", "tokens"]))
    lg = collections.defaultdict(lambda: [0, 0])
    for r in xg:
        if r["cell"] in ("multilingual-text", "multilingual-chat", "long-chat"): lg[(r["cell"], r.get("lang"))][0] += 1; lg[(r["cell"], r.get("lang"))][1] += r["tokens"]
    top = sorted(lg.items(), key=lambda kv: -kv[1][1])
    out.append("### human languages: tokens by (cell, lang), top 45 of %d\n\n" % len(top) + md([(f"{c} / {l}", n, f"{t:,}") for (c, l), (n, t) in top[:45]], ["cell / lang", "docs", "tokens"]))
    ct = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r["kind"] == "chat": ct[(r["origin"], r.get("thinking"), r.get("has_system"))][0] += 1; ct[(r["origin"], r.get("thinking"), r.get("has_system"))][1] += r["tokens"]
    out.append("### chat docs: thinking x has_system by origin\n\n" + md([(*k, n, f"{t:,}") for k, (n, t) in sorted(ct.items(), key=str)], ["origin", "thinking", "has_system", "docs", "tokens"]))
    ag = [r for r in xg if r["cell"] == "agent-sessions"]
    t = collections.defaultdict(lambda: [0, 0, 0])
    for r in ag:
        k = r["src_id"][7:]; t[k][0] += 1; t[k][1] += r["tokens"]; t[k][2] += bool(r.get("tools_inferred"))
    out.append("### agent sessions by source: docs, tokens, docs with any inferred tool definition\n\n" + md([(k, v[0], f"{v[1]:,}", v[2]) for k, v in sorted(t.items())], ["source", "docs", "tokens", "docs w/ inferred tools"]))
    real = collections.Counter(); inf = collections.Counter()
    for r in ag:
        for n in r.get("tools_real") or []: real[(r.get("format"), n)] += 1
        for n in r.get("tools_inferred_names") or []: inf[(r.get("format"), n)] += 1
    out.append("### harness tools with REAL definitions (sessions using them)\n\n" + md([(f, n, c) for (f, n), c in sorted(real.items(), key=lambda kv: -kv[1])[:40]], ["format", "tool", "sessions"]))
    out.append("### most common INFERRED tools (sessions)\n\n" + md([(f, n, c) for (f, n), c in inf.most_common(25)], ["format", "tool", "sessions"]))
    lens = collections.Counter(r.get("length") or "-" for r in ag)
    out.append("### agent sessions: xl / pieces / whole\n\n" + md([("xl (whole, > 128k)", lens["xl"]), ("pieces", sum(1 for r in ag if "piece" in r)), ("whole <= 128k", sum(1 for r in ag if r.get("length") != "xl" and "piece" not in r))], ["kind", "docs"]))
    tot = collections.Counter()
    for k, v in dd["dups_at_build_vs_calib_public_xcorp"].items():
        for kk, vv in v.items(): tot[kk] += vv
    out.append("### dedup\n\n" + md([("rejected at build vs calib / public / xcorp", json.dumps(dict(tot))), ("dropped in assembly (xgap vs xgap)", json.dumps(dd["dropped_in_assembly"])),
                                      ("candidate docs entering assembly", dd["xgap_sources_docs"])], ["step", "count"]))
    hold = collections.Counter(("holdout" if r["holdout"] else "train") for r in xg)
    out.append("### holdout\n\n" + md([(k, v) for k, v in hold.items()], ["split", "docs"]))
    print("\n".join(out))


if __name__ == "__main__":
    main()
