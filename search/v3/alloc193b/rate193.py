# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""rate193.py [FROM_TAG_NUMBER] -- #193 report: confirmation rate per source since the two-stage rule, the search's own
move kinds vs my allocators (manual rows split by file prefix), on the live state (read-only). Counted per candidate
with a stage-1 decision (stale rows excluded): tentative, confirmed accept, confirmed join; rate = (accept + join) /
decided with a Wilson 80 % interval. FROM_TAG_NUMBER (default: my first inbox tag) restricts to the same period."""
import json, math, sys

DRV = "/workspace/rad-local/workers/lead-srch/run/drv"


def wilson(k, n, z=1.2816):
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main(frm=None):
    st = [json.loads(l) for l in open(f"{DRV}/state.jsonl")]
    mine = {e["tag"]: e["file"] for e in st if e.get("ev") == "inbox"}
    num = lambda t: int(t.rsplit("-", 1)[1])
    if frm is None: frm = min(num(t) for t, f in mine.items() if f.startswith("w181-"))
    T = {}; kind = {e["tag"]: e["kind"] for e in st if e.get("ev") == "submit"}; d1 = set(); tent = set(); d2 = {}
    for e in st:
        if e.get("ev") == "tentative": tent.add(e["tag"])
        if e.get("ev") == "decision" and e.get("stage") == 2: d2[e["tag"]] = e["decision"]
        elif e.get("ev") == "decision" and e.get("decision") != "stale": d1.add(e["tag"])
    for t in d1 | tent:
        if num(t) < frm or kind.get(t) in (None, "incumbent", "confirm"): continue
        src = kind[t]
        if t in mine: src = (mine[t][:4] + ":" + mine[t][5:].split("-")[0]) if mine[t][:5] in ("w172-", "w181-") else "inbox:other"
        r = T.setdefault(src, [0, 0, 0, 0, 0]); r[0] += 1
        if t in tent:
            r[1] += 1; v = d2.get(t)
            if v == "accept": r[2] += 1
            elif v == "join": r[3] += 1
            elif v is None: r[4] += 1
    def line(name, r):
        n = r[0] - r[4]; k = r[2] + r[3]; lo, hi = wilson(k, n)
        return f"| {name} | {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {100 * k / max(n, 1):.0f} % [{100 * lo:.0f}, {100 * hi:.0f}] |"
    print(f"from tag number {frm}\n| source | decided | tentative | accept | join | stage 2 pending | confirmed / decided (80 % Wilson) |\n|---|---|---|---|---|---|---|")
    own = [0] * 5; my = [0] * 5
    for s, r in T.items():
        tgt = my if s.startswith("w181:") else (own if not s.startswith(("inbox", "w172:")) else None)
        if tgt is not None:
            for i in range(5): tgt[i] += r[i]
    print(line("**search's own moves (all kinds)**", own)); print(line("**w181 allocators (all)**", my))
    for s, r in sorted(T.items(), key=lambda kv: (kv[0].startswith("w1"), -kv[1][0])): print(line(s, r))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
