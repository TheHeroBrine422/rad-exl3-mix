#!/usr/bin/env python3
"""windows.py GOLDEN POOLS [L] [NSEQ] [SEED] -- A4 (ranked.md): which documents the YAQA collectors' qad windows cover,
against the corpus's scored (gen) tokens. CPU only, reads GOLDEN/{kld.json, gen.u8} and POOLS (pools-v3.json).

collect.py takes, from every qad document of >= L tokens, the windows [a, a + L) for a = 0, L, 2L, ... (shorter
documents never), shuffles them with default_rng(SEED) and uses the first NSEQ. Its loss covers window positions
0 .. L-2 (all of them; YAQA_GENMASK=1: only the gen-scored ones). Per source category this prints:
  corpus  : share of the calib golden's gen rows (the scoring set has the calib corpus's distribution)
  qad     : share of the qad pool's gen rows (what the pool could offer)
  win     : share of the gen positions inside all qad windows
  draw    : the same in the NSEQ windows the collector actually uses
  pos     : share of all loss positions in those NSEQ windows (unmasked loss: what it is trained on today)
  gen/pos : gen fraction of the loss positions in the draw (the masked loss keeps this share)
W_SHORT=MIN (what-if): qad docs of MIN .. L-1 tokens join as one sequence each.
Plus ratios draw / corpus and pos / corpus, zero-gen windows, and the tokens lost to the >= L rule."""
import collections, json, os, sys
import numpy as np

gd, pf = sys.argv[1:3]
L = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
NSEQ = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0
SHORT = int(os.environ.get("W_SHORT", 0))
m = json.load(open(f"{gd}/kld.json")); docs = m["docs"]
gen = np.fromfile(f"{gd}/gen.u8", np.uint8).astype(bool)
pool = json.load(open(pf))["pool"]
assert len(pool) == len(docs) and len(gen) == m["rows"], (len(pool), len(docs), len(gen), m["rows"])
cat = lambda d: d["source"].split("/")[0] if d["source"].startswith("agent-trace") else d["source"]

def doc_gen(i):                     # gen flag per document position p (row of the prediction made AT p), p = 0 .. tokens-2
    d = docs[i]; g = np.zeros(d["tokens"] - 1, bool); n = d["tokens"] - 1 - d["score_from"]
    g[d["score_from"]:] = gen[d["row0"]:d["row0"] + n]
    return g

C = collections.defaultdict(lambda: collections.Counter())
wins = []                           # (doc, start) in collect.py's order (before the shuffle)
for i, d in enumerate(docs):
    c = cat(d); g = doc_gen(i); C[c]["corpus_gen"] += int(g.sum()); C[c]["corpus_tok"] += d["tokens"]
    if pool[i] != "qad": continue
    C[c]["qad_gen"] += int(g.sum()); C[c]["qad_tok"] += d["tokens"]; C[c]["qad_docs"] += 1
    if d["tokens"] < L: C[c]["short_gen"] += int(g.sum()); C[c]["short_docs"] += 1
    for a in range(0, d["tokens"] - L + 1, L):
        wins.append((i, a, L)); C[c]["win_gen"] += int(g[a:a + L - 1].sum()); C[c]["win_n"] += 1
    if SHORT and SHORT <= d["tokens"] < L:     # W_SHORT=MIN: what-if, each such doc one sequence of its own length
        wins.append((i, 0, d["tokens"])); C[c]["win_gen"] += int(g.sum()); C[c]["win_n"] += 1
# collect.py's order exactly: its list of global window starts, shuffled by default_rng(SEED)
off = np.cumsum([0] + [d["tokens"] for d in docs]); at = {int(off[i] + a): (i, a, n) for i, a, n in wins}
gl = [int(off[i] + a) for i, a, n in wins]; np.random.default_rng(SEED).shuffle(gl); w_sh = [at[s] for s in gl]
zero = 0; gfrac = []
for i, a, n in w_sh[:NSEQ]:
    c = cat(docs[i]); g = doc_gen(i)[a:a + n - 1]
    C[c]["draw_gen"] += int(g.sum()); C[c]["draw_pos"] += n - 1; C[c]["draw_n"] += 1
    zero += int(g.sum() == 0); gfrac.append(g.mean())
tot = {k: sum(v[k] for v in C.values()) for k in ("corpus_gen", "qad_gen", "win_gen", "draw_gen", "draw_pos", "qad_tok", "corpus_tok")}
print(f"L {L}, NSEQ {NSEQ}, SEED {SEED}{f', short docs >= {SHORT} as own sequences' if SHORT else ''}: {len(wins)} qad windows; corpus gen rows {tot['corpus_gen']:,} of {m['rows']:,}")
print(f"qad gen rows {tot['qad_gen']:,}; inside windows {tot['win_gen']:,} ({tot['win_gen'] / tot['qad_gen']:.1%}); "
      f"in docs < {L} tokens (never used): {sum(v['short_gen'] for v in C.values()):,} "
      f"({sum(v['short_gen'] for v in C.values()) / tot['qad_gen']:.1%}) in {sum(v['short_docs'] for v in C.values())} of "
      f"{sum(v['qad_docs'] for v in C.values())} qad docs")
print(f"draw: {tot['draw_gen']:,} gen of {tot['draw_pos']:,} loss positions ({tot['draw_gen'] / tot['draw_pos']:.1%}); "
      f"zero-gen windows {zero}; per-window gen fraction p10/50/90 "
      f"{np.percentile(gfrac, 10):.2f}/{np.percentile(gfrac, 50):.2f}/{np.percentile(gfrac, 90):.2f}")
sh = lambda c, k: C[c][k] / max(tot[k], 1)
print(f"\n{'category':22s} {'corpus':>7s} {'qad':>7s} {'win':>7s} {'draw':>7s} {'pos':>7s} {'gen/pos':>7s} "
      f"{'draw/corp':>9s} {'pos/corp':>9s}  windows(draw/all) short-docs")
out = {}
for c in sorted(C, key=lambda c: -C[c]["corpus_gen"]):
    r = {k: sh(c, k) for k in ("corpus_gen", "qad_gen", "win_gen", "draw_gen", "draw_pos")}
    r["gen_per_pos"] = C[c]["draw_gen"] / C[c]["draw_pos"] if C[c]["draw_pos"] else None
    r["draw_vs_corpus"] = r["draw_gen"] / r["corpus_gen"] if r["corpus_gen"] else None
    r["pos_vs_corpus"] = r["draw_pos"] / r["corpus_gen"] if r["corpus_gen"] else None
    r.update(win_n=C[c]["win_n"], draw_n=C[c]["draw_n"], short_docs=C[c]["short_docs"], qad_docs=C[c]["qad_docs"]); out[c] = r
    f = lambda v: f"{v:7.1%}" if v is not None else "      -"
    g = lambda v: f"{v:9.2f}" if v is not None else "        -"
    print(f"{c:22s} {f(r['corpus_gen'])} {f(r['qad_gen'])} {f(r['win_gen'])} {f(r['draw_gen'])} {f(r['draw_pos'])} "
          f"{f(r['gen_per_pos'])} {g(r['draw_vs_corpus'])} {g(r['pos_vs_corpus'])}  {r['draw_n']:5d}/{r['win_n']:5d} "
          f"{r['short_docs']:4d}/{r['qad_docs']}")
flag = [c for c, r in out.items() if r["corpus_gen"] >= 0.01 and not (0.5 <= (r["draw_vs_corpus"] or 0) <= 2.0)]
tv = lambda k: 0.5 * sum(abs(r[k] - r["corpus_gen"]) for r in out.values())
under = sum(r["corpus_gen"] for r in out.values() if (r["draw_vs_corpus"] or 0) < 0.5)
print(f"\ntotal variation vs corpus gen mix: qad pool {tv('qad_gen'):.3f}, drawn gen {tv('draw_gen'):.3f}, "
      f"drawn loss positions (unmasked loss) {tv('draw_pos'):.3f}; corpus gen mass in categories drawn at < 0.5x: {under:.1%}")
print(f"\ncategories with >= 1% of corpus gen rows whose drawn gen share is off by > 2x: {flag or 'none'}")
json.dump({"tv_qad": tv("qad_gen"), "tv_draw_gen": tv("draw_gen"), "tv_draw_pos": tv("draw_pos"), "under_half": under, "L": L, "nseq": NSEQ, "seed": SEED, "zero_gen_windows": zero, "totals": tot, "categories": out,
           "flag_2x": flag}, open(sys.argv[6] if len(sys.argv) > 6 else "/dev/null", "w"), indent=1)
