#!/usr/bin/env python3
"""bsum155.py RES -- (w155, #180-02) per configuration (main / copy / none) and context: bench.py's warm-dropped runs over
rounds 1-6 (tokens a step: mean; server ms a step and client tok/s: median, the r4-6 median too: the final kernel) and
the fixed-prompt acceptance (metrics-acc-*: gen-prompts 16 x 256 greedy tokens)."""
import json, re, statistics as st, sys, glob

d = sys.argv[1]
print("| config | ctx | runs | tok/step (mean, 6 rounds) | ms/step median (r1-6 / r4-6) | client tok/s mean | client tok/s median (r1-6 / r4-6) |")
print("|---|---|---|---|---|---|---|")
for c in ("main", "copy", "none"):
    rows = {}
    for r in range(1, 7):
        for l in open(f"{d}/bench-{c}-r{r}.jsonl"):
            j = json.loads(l)
            if j.get("warmup"):
                continue
            rows.setdefault(j["ctx"], []).append((r, j["srv_tokens_per_decode_step"], j["srv_ms_per_decode_step"], j["client_tok_s"]))
    for ctx, v in rows.items():
        late = [x for x in v if x[0] >= 4]
        print(f"| {c} | {ctx} | {len(v)} | {st.mean(x[1] for x in v):.2f} | {st.median(x[2] for x in v):.2f} / {st.median(x[2] for x in late):.2f} | "
              f"{st.mean(x[3] for x in v):.1f} | {st.median(x[3] for x in v):.1f} / {st.median(x[3] for x in late):.1f} |")
print()
for c in ("main", "copy", "none"):
    m = {}
    for l in open(f"{d}/metrics-acc-{c}.txt"):
        g = re.match(r"^radiance:(draft_tokens_total|draft_accepted_total)\{.*\} (\S+)", l)
        if g:
            m[g.group(1)] = float(g.group(2))
    steps = m["draft_tokens_total"] / 7
    print(f"acc-{c}: {steps:.0f} steps, {m['draft_accepted_total']:.0f} accepted, {(steps + m['draft_accepted_total']) / steps:.3f} tok/step, "
          f"acceptance {m['draft_accepted_total'] / m['draft_tokens_total']:.3f}")
