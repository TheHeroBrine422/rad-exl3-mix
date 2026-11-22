#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""readout220.py [W220_DIR] -- the w220 GPU gates' readout (box2: /data/q27/w220), markdown to stdout (and W220_DIR/readout220.md):
  d    default equivalence: qad220 (d220) vs qad200 (d200), eager, 2 steps: step records + parameters bit for bit?
  b    frozen prefix (gb-pre vs gb-full, eager; gbc-* compiled; gb-pre-r = a repeat of gb-pre: the run-to-run floor): mean gradients of the first 2 plan steps of the
       trainable scales (layers 32-63) and the two batches' KL / gradient norms
  c    speed table: seconds per optimizer step (median of steps 3-5) of sp-eager / sp-comp / sp-comp-32 / sp-comp-48 (and sp-eager-32)
  a    compiled-default check: 25C (compiled 25A recipe) vs w214's eager 25A on j217, paired per document (cmp214's arithmetic), 0.25 M / 0.5 M rows; pass = ratio within +-2 %
Verdicts (summary table first): a PASS = |ratio - 1| <= 2 % at 0.25 M and 0.5 M; b PASS = the frozen-prefix run's mean gradients differ from the full-backward run's by <= 1e-4 of each
vector's max |g| (rounding level; bitwise identical is reported as such), KL / row identical to 1e-6 relative and batch gradient norms to 1e-4; d PASS = bit-identical step records and parameters."""
import io, json, os, sys, contextlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "qad214"))
import cmp220


def cap(f, *a):
    b = io.StringIO()
    with contextlib.redirect_stdout(b): rc = f(*a)
    return rc, b.getvalue().strip()


def main():
    W = sys.argv[1] if len(sys.argv) > 1 else "/data/q27/w220"; R = f"{W}/runs"; out = []
    def say(s=""): out.append(s); print(s)
    def have(*names): return all(os.path.exists(f"{R}/{n}/bench.done") for n in names)
    say("# w220 GPU gates")
    V = {}
    out.append("<!--SUMMARY-->")
    if have("d200", "d220"):
        rc, t = cap(cmp220.steps, f"{R}/d200", f"{R}/d220"); rc2, t2 = cap(cmp220.state, f"{R}/d200", f"{R}/d220")
        V["d"] = "PASS" if rc == 0 and rc2 == 0 else "FAIL"
        say(f"\n## d. defaults = qad200 (2 steps, eager): **{'IDENTICAL' if rc == 0 and rc2 == 0 else 'DIFFERENT'}**\n```\n{t}\n{t2}\n```")
    for lab, pre, full, rep in (("eager", "gb-pre", "gb-full", "gb-pre-r"), ("compiled", "gbc-pre", "gbc-full", "gbc-pre-r")):
        if have(pre, full):
            _, t = cap(cmp220.grad, f"{R}/{pre}", f"{R}/{full}")
            st, bt = cmp220.grad_stats(f"{R}/{pre}", f"{R}/{full}")
            okb = st["max_rel"] <= 1e-4 and all(abs(ka / kb - 1) <= 1e-6 and abs(ga / gb - 1) <= 1e-4 for ka, kb, ga, gb in bt)
            V[f"b ({lab})"] = ("PASS" if okb else "FAIL") + (" (bitwise identical)" if st["max_abs"] == 0 and all(ka == kb for ka, kb, _, _ in bt) else f" (max rel diff {st['max_rel']:.2e})")
            say(f"\n## b. frozen prefix ({lab}; layers 32-63 trainable; frozen-prefix run vs the full-backward run, same trainable set)\n```\n{t}\n```")
            if rep and have(rep):
                _, t = cap(cmp220.grad, f"{R}/{pre}", f"{R}/{rep}"); say(f"run-to-run floor ({pre} vs its repeat {rep}):\n```\n{t}\n```")
    sp = [x for x in ("sp-eager", "sp-comp", "sp-comp-32", "sp-comp-48", "sp-eager-32") if have(x)]
    if sp:
        _, t = cap(cmp220.speed, *[f"{R}/{x}" for x in sp]); say(f"\n## c. speed (2.50 student, V 24 x Q 1024 plan = ~850 gen rows a step; other cards busy at the same time as in production)\n```\n{t}\n```")
        med = {}
        for x in sp:
            w = [json.loads(l)["wall"] for l in open(f"{R}/{x}/train.jsonl") if json.loads(l)["what"] == "main" and json.loads(l)["step"] >= 3]
            med[x] = float(np.median(w))
        if "sp-eager" in med:
            say("\n| config | s / step | speed-up vs eager |\n|---|---|---|")
            for x in sp: say(f"| {x} | {med[x]:.1f} | {med['sp-eager'] / med[x]:.2f}x |")
    import cmp214
    eager, comp, g = f"/data/q27/w217/scores", f"{W}/scores", "/data/q27/w217/slices/golden-j217"
    for ck, lab in (("ckpt-2e+05", "0.25 M"), ("ckpt-5e+05", "0.5 M")):
        fe, fc = f"{eager}/25A-{ck}.json.rows", f"{comp}/25C-{ck}.json.rows"
        if os.path.exists(fe) and os.path.exists(fc):
            xe, n = cmp214.doc_sums(g, fe); xc, nc = cmp214.doc_sums(g, fc)
            d = cmp214.paired(xe, xc, n); ctl = f"{eager}/ctl-2.50.json.rows"
            V[f"a {lab}"] = ("PASS" if abs(d["ratio"] - 1) <= 0.02 else "FAIL") + f" (ratio {d['ratio']:.4f} [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}])"
            r = f"\n## a. compiled 25C vs eager 25A at {lab} (j217, paired per document; ratio compiled / eager): **{d['ratio']:.4f}** [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}] " \
                f"(docs better {d['docs_better']} / worse {d['docs_worse']}); KL {xc.sum() / n.sum():.6f} vs {xe.sum() / n.sum():.6f}; **{'PASS' if abs(d['ratio'] - 1) <= 0.02 else 'FAIL'}** (|ratio - 1| <= 2 %)"
            if os.path.exists(ctl):
                x0, _ = cmp214.doc_sums(g, ctl); r += f"; vs the unhealed release: compiled {xc.sum() / x0.sum():.4f}, eager {xe.sum() / x0.sum():.4f}"
            say(r)
    summ = "\n| gate | verdict |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in V.items()) + ("" if V else "| (no gate has finished yet) | |") + "\n"
    out = [summ if x == "<!--SUMMARY-->" else x for x in out]
    print("\n--- summary ---" + summ)
    open(f"{W}/readout220.md", "w").write("\n".join(out) + "\n")


if __name__ == "__main__": main()
