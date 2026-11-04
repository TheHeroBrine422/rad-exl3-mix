#!/usr/bin/env python3
"""hdmon.py HDIAG_DIR [OUT.jsonl] -- convergence of the 27B diagonal-H_out collection from its snapshots
(hdiag-snapN{,-dy,-xw}: cumulative means over N sequences, a snapshot every 128).

Per snapshot N and estimator (dy = Σdy², the primary; xw; B = Sketch B), over all linears:
  step     relative L2 change of each linear's diagonal from the previous snapshot (median / p90 over linears)
  split    split-half agreement from independent 128-sequence blocks (block_k = N_k m_k - N_{k-1} m_{k-1}): the sum
           of even-numbered blocks vs odd-numbered ones, per linear: relative L2 difference of the two half means
           and their cosine (median / p10 / worst). Noise in the full mean ≈ rel_diff / 2 (two halves, each with
           half the sequences); it should fall as 1/sqrt(N).
  B also reports collect.py's own even/odd-sequence halves.json (median cosine).
Rows are appended to OUT.jsonl (default HDIAG_DIR/hdmon.jsonl) for snapshots not yet in it; a table is printed.
"""
import glob, json, math, os, re, sys
import torch
from safetensors import safe_open

D = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else f"{D}/hdmon.jsonl"
EST = {"dy": "-dy", "xw": "-xw", "B": ""}
snaps = sorted({int(m.group(1)) for p in glob.glob(f"{D}/hdiag-snap*") if (m := re.search(r"hdiag-snap(\d+)$", p))})
def load(n, suf):
    d = f"{D}/hdiag-snap{n}{suf}"; out = {}
    for f in glob.glob(f"{d}/*.safetensors"):
        with safe_open(f, "pt") as h: out[os.path.basename(f)] = h.get_tensor("hout_diag").double()
    return out
def q(xs, p): xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))]
done = set()
if os.path.exists(OUT):
    for l in open(OUT): r_ = json.loads(l); done.add((r_["n"], r_["est"]))
rows = []
for est, suf in EST.items():
    prev, prevn = None, 0; sums = {}; blocks = {"even": {}, "odd": {}}; k = 0
    for n in snaps:
        cur = load(n, suf)
        if not cur: continue
        tot = {key: v * n for key, v in cur.items()}
        blk = {key: tot[key] - sums.get(key, 0) for key in tot}                 # this block's sum (n - prevn seqs)
        half = "even" if k % 2 == 0 else "odd"; k += 1
        for key, b in blk.items(): blocks[half][key] = blocks[half].get(key, 0) + b
        r = {"n": n, "est": est}
        if prev is not None:
            st = [float((cur[key] - prev[key]).norm() / prev[key].norm()) for key in cur if key in prev]
            r.update(step_med=q(st, 0.5), step_p90=q(st, 0.9))
        if blocks["odd"]:
            rd, cs = [], []
            for key in blocks["even"]:
                if key not in blocks["odd"]: continue
                a, b = blocks["even"][key], blocks["odd"][key]
                a = a / a.sum(); b = b / b.sum()                                   # shape only: half sizes may differ
                rd.append(float((a - b).norm() / ((a + b) / 2).norm())); cs.append(float(a @ b / (a.norm() * b.norm())))
            r.update(split_rel_med=q(rd, 0.5), split_rel_p90=q(rd, 0.9), split_cos_med=q(cs, 0.5),
                     split_cos_p10=q(cs, 0.1), split_cos_min=min(cs))
        if est == "B" and os.path.exists(f"{D}/hdiag-snap{n}/halves.json"):
            h = json.load(open(f"{D}/hdiag-snap{n}/halves.json"))
            r.update(halves_med=h.get("median"), halves_p5=h.get("p5"))
        rows.append(r); sums = tot; prev, prevn = cur, n
with open(OUT, "a") as f:
    for r in rows:
        if (r["n"], r["est"]) not in done: f.write(json.dumps(r) + "\n")
print(f"{'est':3} {'N':>5} {'step med/p90':>15} {'split rel med/p90':>18} {'split cos med/p10/min':>24} {'B halves':>9}")
for r in sorted(rows, key=lambda r: (r["est"], r["n"])):
    s = f"{r['step_med']:.4f}/{r['step_p90']:.4f}" if "step_med" in r else "-"
    sp = f"{r['split_rel_med']:.4f}/{r['split_rel_p90']:.4f}" if "split_rel_med" in r else "-"
    sc = f"{r['split_cos_med']:.4f}/{r['split_cos_p10']:.4f}/{r['split_cos_min']:.3f}" if "split_cos_med" in r else "-"
    print(f"{r['est']:3} {r['n']:5d} {s:>15} {sp:>18} {sc:>24} {str(r.get('halves_med', '')):>9}")
