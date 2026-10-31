#!/usr/bin/env python3
"""finalsel.py N [OURS] -- the V2 final's candidates (evalq2 queue lines on golden-ver2, stdout; a report on stderr).

Per target bpw T (from the mix names: every round / stream fit, mean, Thompson and ours-only mixes, measured on a fit
slice a / f2 / f3), rank the measured mixes by slice-normalised KLD = KLD / offset(slice), offsets from the latest
fit report ("slice offsets f2 x f3 y"; a = 1), and take the best N distinct mixes (identical mix files count once).
OURS=1 (default): only all-ours mixes (every weight from ours-K* or bf16; the embedding may come from any container);
OURS=0: any mix, plus the best all-ours one if it is not already among the N.
Rationale (2026-10-30): the mean model's own optimum loses to Thompson draws 70-90% of the time above 2.5 bpw, so the
final confirms the best *measured* mixes on the untouched ver2 slice instead of a last fit's mean mixes.
Run from /root."""
import glob, hashlib, json, os, re, sys
# paths (README: Paths)
RXM_WORK = os.environ.get("RXM_WORK", "/root")
RXM_MODELS = os.environ.get("RXM_MODELS", f"{RXM_WORK}/models/final")
RXM_OURS = os.environ.get("RXM_OURS", "/data/ours")

N = int(sys.argv[1]); OURS = (sys.argv[2] if len(sys.argv) > 2 else "1") == "1"
MX = f"{RXM_WORK}/knap/mixes"; BASE = f"{RXM_MODELS}/q38-27b-sc4.00-e4.rad"; P = 26895998464
G2 = "golden-ver2 corpus/ver2.jsonl"

off = {"a": 1.0, "f2": 1.0, "f3": 1.0}; src = "none (offsets 1.0)"
fits = sorted(glob.glob(f"{RXM_WORK}/auto/fit-*.txt"), key=os.path.getmtime)
for f in reversed(fits):
    m = re.search(r"slice offsets f2 ([\d.]+) f3 ([\d.]+)", open(f).read())
    if m: off["f2"], off["f3"] = float(m.group(1)), float(m.group(2)); src = f; break

res = {}
for l in open(f"{RXM_WORK}/evq/results.jsonl"):
    r = json.loads(l)
    if "kld" in r: res[r["tag"]] = r

def mixinfo(name):
    p = f"{MX}/{name}.mix"
    if not os.path.exists(p): return None
    lines = [l.split() for l in open(p) if l.strip()]
    w = [(t, q) for t, q in lines if t != "token_embd.weight"]
    ours = len(w) >= 337 and all(f"{RXM_OURS}/" in q or q.endswith("q38-27b-bf16.rad") for _, q in w)
    return p, hashlib.sha1("".join(sorted(open(p))).encode()).hexdigest(), ours

GRID = {float(x) for x in os.environ.get("FINAL_TARGETS", "1.5 1.75 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 9.0 10.0").split()}   # release sizes (2026-10-31)
cands = {}   # target -> [(norm kld, tag, mixname, path, ours, bpw)]
for t, r in res.items():
    # any measured mix at a target (2026-10-31: was a list of candidate kinds, which missed new kinds), but only the
    # current target grid (FINAL_TARGETS; the pre-stream 2.99 / 3.99 / 5.49 and the dropped 1.2-1.5 / 8.5+ are out)
    m = re.fullmatch(r"(a|f2|f3)-(knap-.+-sc4\.00-e4-([\d.]+))", t)
    if not m or float(m.group(3)) not in GRID: continue
    info = mixinfo(m.group(2))
    if not info: continue
    T = m.group(3); b = r["bytes"] * 8 / P
    if b > float(T) + 0.005: continue   # 2026-10-31: was >= T, which dropped the DP's exactly-filled high-bpw mixes (9.0003)
    cands.setdefault(T, []).append((r["kld"] / off[m.group(1)], t, m.group(2), info[0], info[2], info[1], b))

print(f"finalsel: slice offsets {off} (from {os.path.basename(src)}), N={N}, ours-only={OURS}", file=sys.stderr)
for T in sorted(cands, key=float):
    pool = sorted(cands[T])
    # 2026-10-31: top 3 at FINAL_N3 (default 2.5-4.5 bpw), N elsewhere
    lo3, hi3 = (float(x) for x in os.environ.get("FINAL_N3", "2.5,4.5").split(","))
    NT = 3 if lo3 <= float(T) <= hi3 else N
    pick, seen = [], set()
    for c in pool:
        if c[5] in seen or (OURS and not c[4]): continue
        if len(pick) < NT: pick.append(c); seen.add(c[5])
    # plus the candidate measured best on held-out ver1 (w- tags) if not already in: guards against a lucky fit-slice
    # winner pushing out a better mix (2026-10-31); ver2 then decides
    wv = [(res[f"w-{c[2]}"]["kld"], c) for c in pool if f"w-{c[2]}" in res and (not OURS or c[4])]
    if wv:
        c = min(wv, key=lambda x: x[0])[1]
        if c[5] not in seen: pick.append(c); seen.add(c[5])
    if not OURS and not any(c[4] for c in pick):
        best_ours = next((c for c in pool if c[4] and c[5] not in seen), None)
        if best_ours: pick.append(best_ours)
    for c in pick:
        print(f"v2-{c[2]} {BASE} {c[3]} {G2}")
    print(f"  {T:>5}: " + "  ".join(f"{c[1]} ({c[6]:.3f} bpw, norm {c[0]:.5f}{', ours' if c[4] else ''})" for c in pick)
          + f"   [{len(pool)} measured]", file=sys.stderr)
