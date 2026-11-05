#!/usr/bin/env python3
"""r90report.py [SUB1.jsonl ...] -- #90 numbers and verdicts (README "#90", rules fixed before data).

Rule 1 (curve usable for pricing): rel(K) falls at every 1/8 step K0.125 -> 0.25 -> ... -> 1 for every tensor (a step
       may be flat within 1%: rel(K + 1/8) <= 1.01 rel(K)), and the K1 anchor agrees with the existing fracsweep K1
       (v3/e3/data/fracsweep.jsonl, same tensors) within 10%. Also shown, not part of the rule: the K1 -> K2 slope vs
       the hproxy ladder's (kfit2 model's ours-K1 / ours-K2 errors, the K1 price knap4 uses), the 1 -> 1.25 -> 1.5 -> 2
       anchors, and how often the scale search sits on its lowest factor (0.5: error overstated there).
Rule 2 (worth its kernel work at a size): knap4 with sub-1 options (rel(K) / rel(K1) x the tensor's K1 price, family
       ratios from here) predicts >= 3% lower KLD than the same menu without them. Read from r90knap.py's output
       (data/e3/sub1/r90knap.jsonl) when it exists.
Default input: data/e3/sub1/sub1-cpu1.jsonl + sub1-cpu2.jsonl (cpu1: GDN layers, cpu2: attention layers)."""
import glob, json, math, os, sys
from collections import defaultdict
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
paths = sys.argv[1:] or sorted(glob.glob(f"{ROOT}/data/e3/sub1/sub1-cpu*.jsonl"))
CB = "ldlq_cubic0.02_i8"
R = {}
for p in paths:
    for l in open(p):
        try: o = json.loads(l)
        except ValueError: continue
        if o["method"] == CB: R[(o["layer"], o["w"], float(o["bits"]))] = o    # last line wins (reruns)
TENS = sorted({(l, w) for l, w, _ in R})
SUB = [k / 8 for k in range(1, 9)]                 # 0.125 .. 1
ANCH = [1.0, 1.25, 1.5, 2.0]
FAMS = ["ssm_qkv", "ssm_z", "ssm_out", "attn_qg", "attn_k", "attn_v", "attn_output", "ffn_gate_up", "ffn_down"]


def rel(l, w, K):
    o = R.get((l, w, float(K)))
    return o["rel"] if o else None


def gm(v): return math.exp(sum(map(math.log, v)) / len(v))


print(f"# #90 report from {', '.join(os.path.relpath(p, ROOT) for p in paths)} ({len(R)} lines, {len(TENS)} tensors)\n")
missing = [(l, w, K) for l, w in TENS for K in SUB + ANCH if rel(l, w, K) is None]
complete = len(TENS) == 49 and not missing
if missing: print(f"missing points: {len(missing)} (first {missing[:5]})\n")

print("## Family curves: rel(K) / rel(K1), geometric mean over the family's tensors [min, max]\n")
print("| family | n | " + " | ".join(f"K{K:g}" for K in SUB[:-1] + ANCH[1:]) + " |")
print("|---|---|" + "---|" * (len(SUB) - 1 + len(ANCH) - 1))
for f in FAMS:
    ts = [(l, w) for l, w in TENS if w == f]
    if not ts: continue
    cells = []
    for K in SUB[:-1] + ANCH[1:]:
        v = [rel(l, w, K) / rel(l, w, 1) for l, w in ts if rel(l, w, K) and rel(l, w, 1)]
        cells.append(f"{gm(v):.3f} [{min(v):.2f}, {max(v):.2f}]" if v else "-")
    print(f"| {f} | {len(ts)} | " + " | ".join(cells) + " |")
allv = {K: [rel(l, w, K) / rel(l, w, 1) for l, w in TENS if rel(l, w, K) and rel(l, w, 1)] for K in SUB[:-1] + ANCH[1:]}
print("| all | %d | " % len(TENS) + " | ".join(f"{gm(v):.3f}" for v in allv.values()) + " |")
print("\nerror halvings per bit (log2 rel(a) / rel(b) / (b - a)), all tensors, mean:")
PTS = SUB + ANCH[1:]
for a, b in zip(PTS, PTS[1:]):
    v = [math.log2(rel(l, w, a) / rel(l, w, b)) / (b - a) for l, w in TENS if rel(l, w, a) and rel(l, w, b)]
    if v: print(f"  K{a:g} -> K{b:g}: {np.mean(v):.2f} (min {min(v):.2f}, max {max(v):.2f})")

print("\n## Rule 1a: monotone at every 1/8 step (flat within 1% allowed)\n")
viol, flat = [], []
for l, w in TENS:
    for a, b in zip(SUB, SUB[1:]):
        ra, rb = rel(l, w, a), rel(l, w, b)
        if ra is None or rb is None: continue
        if rb > 1.01 * ra: viol.append((l, w, a, b, rb / ra))
        elif rb >= ra: flat.append((l, w, a, b, rb / ra))
steps = [rel(l, w, b) / rel(l, w, a) for l, w in TENS for a, b in zip(SUB, SUB[1:]) if rel(l, w, a) and rel(l, w, b)]
print(f"{len(steps)} steps: step ratio rel(K+1/8)/rel(K) median {np.median(steps):.3f}, max {max(steps):.3f}; "
      f"flat-within-1% {len(flat)}, violations {len(viol)}")
for v in viol: print(f"  VIOLATION L{v[0]} {v[1]} K{v[2]:g}->K{v[3]:g}: x{v[4]:.4f}")
for v in flat: print(f"  flat L{v[0]} {v[1]} K{v[2]:g}->K{v[3]:g}: x{v[4]:.4f}")
anc = [(l, w, a, b) for l, w in TENS for a, b in zip(ANCH, ANCH[1:]) if rel(l, w, a) and rel(l, w, b) and rel(l, w, b) >= rel(l, w, a)]
print(f"anchors K1 -> 1.25 -> 1.5 -> 2 non-decreasing steps: {len(anc)}")
edge = defaultdict(int); cnt = defaultdict(int)
for (l, w, K), o in R.items():
    cnt[K] += 1; edge[K] += o["scale_f"] <= 0.5 + 1e-9
print("scale search on its lowest factor (0.5): " + ", ".join(f"K{K:g} {edge[K]}/{cnt[K]}" for K in sorted(cnt)))
sfm = {K: np.median([o["scale_f"] for (l, w, k), o in R.items() if k == K]) for K in sorted(cnt)}
print("median scale factor: " + ", ".join(f"K{K:g} {v:g}" for K, v in sfm.items()))

print("\n## Rule 1b: K1 anchor vs fracsweep K1 (mul1, same rows / seed; within 10%)\n")
F = {}
for l in open(f"{ROOT}/v3/e3/data/fracsweep.jsonl"):
    o = json.loads(l); F[(o["layer"], o["w"], float(o["bits"]))] = o["rel"]
anc_r = {}
for l, w in TENS:
    if (l, w, 1.0) in F and rel(l, w, 1):
        anc_r[(l, w)] = rel(l, w, 1) / F[(l, w, 1.0)]
        extra = "  ".join(f"K{K:g} {rel(l, w, K) / F[(l, w, K)]:.3f}" for K in (1.25, 1.5, 2.0) if (l, w, K) in F and rel(l, w, K))
        print(f"  L{l} {w:12s} K1 sub1 {rel(l, w, 1):.4f} / fracsweep {F[(l, w, 1.0)]:.4f} = {anc_r[(l, w)]:.3f}   ({extra})")
for K in (0.5, 0.75):
    v = [rel(l, w, K) / F[(l, w, K)] for l, w in TENS if (l, w, K) in F and rel(l, w, K)]
    if v: print(f"  (also K{K:g}: sub1 / fracsweep median {np.median(v):.3f}, n={len(v)}; fracsweep had scale factors >= 1)")
md = json.load(open(f"{ROOT}/data/v3/kfit2-model-final.json")); MT = {t["w"]: t["opts"] for t in md["tensors"]}
sl = []
for l, w in TENS:
    o = MT.get(f"blk.{l}.{w}.weight")
    if o and rel(l, w, 1) and rel(l, w, 2): sl.append(((rel(l, w, 1) / rel(l, w, 2)) / (o["ours-K1"][1] / o["ours-K2"][1]), l, w))
v = [x[0] for x in sl]
print(f"  K1/K2 error ratio, sub1 vs the hproxy ladder (knap4's K1 / K2 prices), {len(sl)} tensors: "
      f"median {np.median(v):.3f}, min {min(v):.3f}, max {max(v):.3f}  (reported only); outside 0.9..1.1: "
      + (", ".join(f"L{l} {w} {x:.3f}" for x, l, w in sl if abs(x - 1) > 0.1) or "none"))
ok_mono = not viol
ok_anchor = bool(anc_r) and all(abs(v - 1) <= 0.10 for v in anc_r.values())
print(f"\nRule 1 (monotone: {len(viol)} violations; K1 anchor ratios {min(anc_r.values()):.3f}..{max(anc_r.values()):.3f} "
      f"on {len(anc_r)} tensors): " + ("PASS (curve usable for pricing)" if ok_mono and ok_anchor else "FAIL")
      + ("" if complete else "  [INCOMPLETE]"))

print("\n## Rule 2: knap4 with vs without sub-1 options (r90knap.py)\n")
kp = f"{ROOT}/data/e3/sub1/r90knap.jsonl"
if not os.path.exists(kp): print("no r90knap.jsonl yet: run v3/e3/sub1/r90knap.py"); sys.exit()
K = defaultdict(dict)
for l in open(kp):
    o = json.loads(l)
    if o["bk"] == 1 << 18: K[o["menu"]][(o["arm"], o["target"])] = o     # r90knap's main solves (256 KiB buckets)
for menu in ("v3e", "v3"):
    d = K.get(menu, {})
    print(f"menu {menu}" + (" (rule: the V3 ladder's 1/8 steps from K1.125)" if menu == "v3e" else
                            " (knap4's default V3 ladder, levels from K2; reported only)") + ":")
    for tg in sorted({t for _, t in d}):
        a, b = d.get(("base", tg)), d.get(("sub1", tg))
        if not a or not b: print(f"  {tg:.2f}: incomplete"); continue
        ch = b["kld"] / a["kld"] - 1
        tag = ("PASS (>= 3% lower)" if ch <= -0.03 else "FAIL (< 3%)") if menu == "v3e" and tg >= 1.5 else "(reported)"
        print(f"  {tg:.2f} bpw: KLD {a['kld']:.5f} -> {b['kld']:.5f} ({ch:+.2%}), trunk params at K < 1: "
              f"{b['sub1_share']:.1%} ({b['sub1_n']} tensors) -> {tag}")
print("details, bucket convergence and the tensors that go sub-1: data/e3/sub1/r90knap.md")
