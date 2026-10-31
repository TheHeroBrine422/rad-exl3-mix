#!/usr/bin/env python3
"""uncswaps.py ROUND N -- targeted calibration swaps for the options the round's mean mixes rely on but that have no
direct measurement yet (evalq2 queue lines on calib-A, stdout; mix files -> $RXM_WORK/auto/mixes/).

An option here is (donor, family group, 16-layer band). It counts as measured when a cal-sc4.00-e4-DONOR-GROUP-LBw16
swap exists (scored or queued). The unmeasured ones are ranked by how many tensors of the round's mean mixes use them
(summed over targets) x 4^max(0, 4 - bits): low-K options are where the cost curve extrapolates (the 2026-10-30 K1/K2
misfit), so they come first; common high-K options follow by usage. The top N are emitted. Each swap puts DONOR on that group over that band, inside sc4.00-e4,
so the fit gets the damage of that one choice in isolation (cheaper and sharper than learning it from whole mixes).
bf16, the base container and the head are skipped. Tags are kfit2's cal- form."""
import collections, glob, json, os, re, sys
# paths (README: Paths)
RXM_WORK = os.environ.get("RXM_WORK", "/root")
RXM_MODELS = os.environ.get("RXM_MODELS", f"{RXM_WORK}/models/final")
RXM_OURS = os.environ.get("RXM_OURS", "/data/ours")

R, N = sys.argv[1], int(sys.argv[2])
F = RXM_MODELS; BASE_C = "sc4.00-e4"; BASE = f"{F}/q38-27b-{BASE_C}.rad"; G = "golden-calA corpus/calA.jsonl"
GROUPS = {"attn_in": ["attn_qg", "attn_k", "attn_v"], "attn_out": ["attn_output"], "ssm_in": ["ssm_qkv", "ssm_z"],
          "ssm_out": ["ssm_out"], "ffn_gu": ["ffn_gate_up"], "ffn_down": ["ffn_down"]}
T2G = {t: g for g, ts in GROUPS.items() for t in ts}
SEL = {"attn_in": lambda l: l % 4 == 3, "attn_out": lambda l: l % 4 == 3, "ssm_in": lambda l: l % 4 != 3,
       "ssm_out": lambda l: l % 4 != 3, "ffn_gu": lambda l: True, "ffn_down": lambda l: True}


def cname(path):
    m = re.search(r"q38-27b-(.+)\.rad$", path) or re.search(r"/(ours-K\d+)\.rad$", path)
    return m.group(1) if m else None


def drad(c): return f"{RXM_OURS}/{c}.rad" if c.startswith("ours-") else f"{F}/q38-27b-{c}.rad"


have = set()
for l in open(f"{RXM_WORK}/evq/results.jsonl"):
    t = json.loads(l)["tag"]
    if t.startswith("cal-"): have.add(t)
for f in (f"{RXM_WORK}/auto/q", f"{RXM_WORK}/auto/q.prio"):
    if os.path.exists(f):
        for l in open(f):
            if l.startswith("cal-"): have.add(l.split()[0])

use = collections.Counter()
for p in glob.glob(f"{RXM_WORK}/knap/mixes/knap-{R if R[0].isalpha() else 'r' + R}-{BASE_C}-*.mix"):
    for l in open(p):
        w, path = l.split()
        m = re.match(r"blk\.(\d+)\.(\w+)\.weight$", w)
        c = cname(path)
        if not m or not c or c in (BASE_C, "bf16"): continue
        g = T2G.get(m.group(2))
        if g: use[(c, g, int(m.group(1)) // 16 * 16)] += 1

def bits(c):
    m = re.match(r"ours-K(\d)$", c) or re.match(r"(?:sc|plain)(\d+\.\d+)", c)
    return float(m.group(1)) if m else 4.0
rank = sorted(use.items(), key=lambda kv: -kv[1] * 4.0 ** max(0.0, 4.0 - bits(kv[0][0])))
n = 0
for (c, g, b0), k in rank:
    tag = f"cal-{BASE_C}-{c}-{g}-L{b0}w16"
    if tag in have: continue
    with open(f"{RXM_WORK}/auto/mixes/{tag}.mix", "w") as f:
        for l in range(b0, b0 + 16):
            if SEL[g](l):
                for t in GROUPS[g]: f.write(f"blk.{l}.{t}.weight {drad(c)}\n")
    print(f"{tag} {BASE} {RXM_WORK}/auto/mixes/{tag}.mix {G}")
    n += 1
    if n >= N: break
print(f"uncswaps {R}: {n} swaps (of {sum(1 for o in use if f'cal-{BASE_C}-{o[0]}-{o[1]}-L{o[2]}w16' not in have)} unmeasured options in use)", file=sys.stderr)
