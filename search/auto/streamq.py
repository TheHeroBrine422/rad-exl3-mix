#!/usr/bin/env python3
"""streamq.py FITTAG SLICE NFIT -- streaming mode (2026-10-30): rewrite /root/auto/q after fit FITTAG (call with the
queue's claim lock held). Output: the new queue in place, a summary line on stdout.

  kept:    every line already scored, failed or claimed (history; evalq2 skips them), every line that is not a mix
           candidate (no "-knap-(rN|sN)" tag), and the tags in /root/auto/state/protect (one per line)
  dropped: unstarted mix candidates of earlier fits / rounds (superseded by this fit)
  added:   this fit's candidates on SLICE:
             - mean-model and Thompson mixes (knap-FITTAG-ts*) in one list, ranked by VALUE (below) x a share:
               the Thompson share w_b of the target's bpw band (<2.5, 2.5-5, >=5) is the smoothed win rate of
               Thompson over mean mixes (same round / fit, target and slice) in the last 30 such pairs of that band,
               clamped to 0.15..0.85; the two value-ordered lists are merged so ~w_b of the slots are Thompson. (2026-10-30: Thompson won 70-90% above 2.5 bpw.)
             - every 4th fit: held-out ver1 (w-) copies of the mean mixes at 2, 3, 4, 5, 6, 7, 8, 9 bpw, at the end
           a candidate whose mix content is identical to an already measured mix (any slice) is skipped.
  LOCAL SEARCH: knap-FITTAG-ls* mutations take ~SQ_LS_SHARE (0.25) of the slots.
  EXPLORATION (2026-10-31): tilted solves knap-FITTAG-tl* / -rd* take ~SQ_EX_SHARE (0.15) of the slots, inserted after the band ordering.
  ORDER: round-robin over bpw bands (<3, 3-5, >=5); inside a band, the value-ordered mean / Thompson merge.
  VALUE of a target (2026-10-30) = (0.5 * staleness + change + miss) / GPU cost (3.5 for TP2-sized mixes, else 1):
    staleness: rank of how long ago its last mean mix was measured, 0..1 (never measured: 1)
    change:    share of tensors this fit's mix changes vs the last measured mean mix at the target, /10%, capped at 1
               (never measured: 1)
    miss:      |log(measured / predicted)| of that last measured mix, capped at 1 (predictions from the
               frontier-*.txt reports; unknown: 0)"""
import collections, glob, hashlib, json, math, os, re, sys
# paths (README: Paths)
RXM_WORK = os.environ.get("RXM_WORK", "/root")
RXM_MODELS = os.environ.get("RXM_MODELS", f"{RXM_WORK}/models/final")

FT, S, NFIT = sys.argv[1], sys.argv[2], int(sys.argv[3])
A = f"{RXM_WORK}/auto"; Q = os.environ.get("SQ_Q", f"{A}/q"); MX = f"{RXM_WORK}/knap/mixes"
BASE = f"{RXM_MODELS}/q38-27b-sc4.00-e4.rad"
GOLD = {"a": "golden-calA corpus/calA.jsonl", "f2": "golden-fit2 corpus/fit2.jsonl", "f3": "golden-fit3 corpus/fit3.jsonl"}
CAND = re.compile(r"^(?:a|f2|f3|w)-knap-(?:r\d+|s\d+)")

done, order = set(), {}
for i, l in enumerate(open(f"{RXM_WORK}/evq/results.jsonl")):
    t = json.loads(l)["tag"]; done.add(t); order[t] = i
claimed = set(os.listdir(f"{RXM_WORK}/evq/claim"))
protect = set(open(f"{A}/state/protect").read().split()) if os.path.exists(f"{A}/state/protect") else set()


def mixhash(path):
    try: return hashlib.sha1("".join(sorted(open(path))).encode()).hexdigest()
    except FileNotFoundError: return None


measured = set()
for t in done:
    m = re.match(r"^(?:a|f2|f3|w)-(knap-.+)$", t)
    if m:
        h = mixhash(f"{MX}/{m.group(1)}.mix")
        if h: measured.add(h)

# last measured mean mix at each target (any round / fit, any fit slice): results index, tag
last, last_tag = {}, {}
for t, i in order.items():
    # any measured mix at the target counts (mean, Thompson, ours-only): 2026-10-31 fix -- counting only mean mixes
    # kept targets whose evals were Thompson draws permanently "stale", so 4.75 / 5.75 were drawn over and over while
    # 3.0-4.0 and 6.0+ got nothing
    m = re.match(r"^(?:a|f2|f3)-knap-(?:r\d+|s\d+)(?:oo)?(?:-(?:ts|ls|tl|rd)\d)?-sc4\.00-e4-([\d.]+)$", t)
    if m and i > last.get(m.group(1), -1): last[m.group(1)] = i; last_tag[m.group(1)] = t
kld = {}
for l in open(f"{RXM_WORK}/evq/results.jsonl"):
    r = json.loads(l)
    if "kld" in r: kld[r["tag"]] = r["kld"]
pred = {}
for f in glob.glob(f"{A}/frontier-*.txt"):
    for l in open(f):
        m = re.match(r"^(knap-\S+)\s+bpw [\d.]+\s+pred KLD ([\d.]+)", l)
        if m: pred[m.group(1)] = float(m.group(2))


def tgt(p): return re.search(r"-([\d.]+)\.mix$", p).group(1)


def readmix(path):
    try: return dict(l.split() for l in open(path) if l.strip())
    except FileNotFoundError: return None


means = glob.glob(f"{MX}/knap-{FT}-sc4.00-e4-*.mix")
tg = sorted({tgt(p) for p in means})
stale_rank = {t: i / max(1, len(tg) - 1) for i, t in enumerate(sorted(tg, key=lambda t: -last.get(t, -1)))}
val = {}
for p in means:
    t = tgt(p); lt = last_tag.get(t)
    if lt is None: val[t] = 0.5 + 1 + 0; continue
    a, b = readmix(p), readmix(f"{MX}/{lt.split('-', 1)[1]}.mix")
    chg = 1.0 if b is None else min(1.0, sum(a.get(w) != b.get(w) for w in set(a) | set(b)) / max(1, len(a)) / 0.10)
    nm = lt.split("-", 1)[1]
    miss = min(1.0, abs(math.log(kld[lt] / pred[nm]))) if lt in kld and pred.get(nm) else 0.0
    val[t] = 0.5 * stale_rank[t] + chg + miss
# GPU cost: a mix over TP2_BYTES (26e9, ~7.73 bpw) runs at TP2 on both cards for ~215 s vs ~100 s on one card, so
# ~3.5x the GPU time (2026-10-30: the Large tier is low priority). Value per unit of GPU time.
for t in val:
    if float(t) * 26895998464 / 8 > 26e9: val[t] /= float(os.environ.get("SQ_TP2_COST", 3.5))
# KIND REWARDS (2026-10-31): every candidate of fit N is measured on fit N's slice, so measured / (fit N's predicted
# KLD of its MEAN mix at that target, offset by fit N's own slice offset) cancels slice-offset and model-level errors
# within the fit: "beat" = the candidate did better than the model's best guess. Smoothed beat rate over each kind's
# last 40 evals (bands pooled). It replaced the paired Thompson-vs-mean pairs, which had dried up (8 in the stream).
fpred, foff = {}, {}
for f in glob.glob(f"{A}/frontier-s*.txt"):
    n_ = re.search(r"frontier-s(\d+)\.txt", f)
    if not n_: continue
    n_ = int(n_.group(1))
    for l in open(f):
        m = re.match(rf"^knap-s{n_}-sc4\.00-e4-([\d.]+)\s+bpw [\d.]+\s+pred KLD ([\d.]+)", l)
        if m: fpred[(n_, m.group(1))] = float(m.group(2))
for f in glob.glob(f"{A}/fit-s*.txt"):
    n_ = re.search(r"fit-s(\d+)\.txt", f)
    if not n_: continue
    m = re.search(r"slice offsets ((?:\w+ [\d.]+ ?)+)", open(f).read(6000))
    o = {"a": 1.0}
    if m: v = m.group(1).split(); o.update({v[i]: float(v[i + 1]) for i in range(0, len(v), 2)})
    foff[int(n_.group(1))] = o
KR = re.compile(r"(a|f2|f3)-knap-s(\d+)(?:-(ts|ls|tl|rd)\d+)?-sc4\.00-e4-([\d.]+)")
beats = collections.defaultdict(list)
for t_, i_ in sorted(order.items(), key=lambda x: x[1]):
    m = KR.fullmatch(t_)
    if not m or t_ not in kld: continue
    n_, k_ = int(m.group(2)), m.group(3) or "mean"
    if (n_, m.group(4)) in fpred and n_ in foff:
        beats[k_].append(kld[t_] / foff[n_].get(m.group(1), 1.0) < fpred[(n_, m.group(4))])
RATE = {k: (sum(beats[k][-40:]) + 1) / (len(beats[k][-40:]) + 2) for k in ("mean", "ts", "ls", "tl", "rd")}
NEV = {k: len(beats[k]) for k in RATE}
# Thompson share per bpw band: smoothed win rate vs the paired mean mix, last 30 pairs of the band (results order)
def band(t): t = float(t); return 0 if t < 2.5 else 1 if t < 5 else 2
pairs = collections.defaultdict(list)
for t, i in sorted(order.items(), key=lambda x: x[1]):
    m = re.fullmatch(r"(a|f2|f3)-knap-(r\d+|s\d+)-ts\d-sc4\.00-e4-([\d.]+)", t)
    if not m or t not in kld: continue
    mt = f"{m.group(1)}-knap-{m.group(2)}-sc4.00-e4-{m.group(3)}"
    if mt in kld: pairs[band(m.group(3))].append(kld[t] < kld[mt])
W = {b: min(0.85, max(0.15, (sum(v[-30:]) + 1) / (len(v[-30:]) + 2))) for b in range(3) for v in [pairs.get(b, [])]}
if os.environ.get("SQ_AUTO", "1") == "1":   # kind-reward shares (above); all bands alike
    W = {b: min(0.70, max(0.15, RATE["ts"] / (RATE["ts"] + RATE["mean"]))) for b in range(3)}
thomp = glob.glob(f"{MX}/knap-{FT}-ts*-sc4.00-e4-*.mix")
# proportional merge of the two value-ordered lists: take the next Thompson mix whenever Thompson's count so far is
# below w x (slots so far + 1), w = the share of the band of the next Thompson mix in line
skipped = 0
def fresh(lst):
    global skipped
    out = []
    for p in lst:
        if mixhash(p) in measured: skipped += 1
        else: out.append(p)
    return out
mq = fresh(sorted(means, key=lambda p: -val[tgt(p)]))
tq = fresh(sorted(thomp, key=lambda p: -val.get(tgt(p), 0)))
seq, nt = [], 0
while mq or tq:
    if tq and (not mq or nt < W[band(tgt(tq[0]))] * (len(seq) + 1)): seq.append(tq.pop(0)); nt += 1
    else: seq.append(mq.pop(0))
# exploration (tilted / random-direction solves at the focus targets): ~EX_SHARE of the mean / Thompson slots (merged before local search,
# which keeps its own ~LS_SHARE of all slots: 2026-10-31, local search set new bests 9x as often per eval as Thompson), alternating kinds
EXS = float(os.environ.get("SQ_EX_SHARE", 0.15))
if os.environ.get("SQ_AUTO", "1") == "1" and min(NEV["tl"], NEV["rd"]) >= 20:
    EXS = min(0.30, max(0.10, 0.15 * (RATE["tl"] + RATE["rd"]) / 2 / 0.5))
tlq = fresh(sorted(glob.glob(f"{MX}/knap-{FT}-tl*-sc4.00-e4-*.mix"), key=lambda p: -val.get(tgt(p), 0)))
rdq = fresh(sorted(glob.glob(f"{MX}/knap-{FT}-rd*-sc4.00-e4-*.mix"), key=lambda p: -val.get(tgt(p), 0)))
xq = [p for pair in zip(rdq, tlq) for p in pair] + rdq[len(tlq):] + tlq[len(rdq):]
if NFIT % 2: xq = [p for pair in zip(tlq, rdq) for p in pair] + tlq[len(rdq):] + rdq[len(tlq):]
# local-search mutations (knap-FT-ls*, from the best measured mix at each target): ~LS_SHARE of the slots
LSH = float(os.environ.get("SQ_LS_SHARE", 0.25))
if os.environ.get("SQ_AUTO", "1") == "1": LSH = min(0.40, max(0.15, 0.25 * RATE["ls"] / 0.5))
lq = fresh(sorted(glob.glob(f"{MX}/knap-{FT}-ls*-sc4.00-e4-*.mix"), key=lambda p: -val.get(tgt(p), 0)))
s0, seq, nl = seq, [], 0
while s0 or lq:
    if lq and (not s0 or nl < LSH * (len(seq) + 1)): seq.append(lq.pop(0)); nl += 1
    else: seq.append(s0.pop(0))

# coverage (2026-10-30): round-robin over bpw bands (<3, 3-5, >=5), the value / Thompson order kept inside each band,
# so every fit's first ~6 slots reach all three bands (value alone kept the slots on 5-7 bpw, never below 3)
# 2026-10-31 (the 0.05-0.005 KLD range matters most, < 0.001 barely): weighted bands --
#   <2.75: 20%   2.75-4.5 (core): 45%   4.5-6: 25%   >=6: 10%   (deficit round-robin; empty bands give up their share)
BW = [(0.0, 2.75, 0.20), (2.75, 4.5, 0.45), (4.5, 6.0, 0.25), (6.0, 99, 0.10)]
def cband(p): t = float(tgt(p)); return next(i for i, (lo, hi, _) in enumerate(BW) if lo <= t < hi)
bands = [[p for p in seq if cband(p) == b] for b in range(len(BW))]
seq, taken = [], [0] * len(BW)
while any(bands):
    live = [b for b in range(len(BW)) if bands[b]]
    tot = sum(BW[b][2] for b in live)
    b = max(live, key=lambda b: BW[b][2] / tot * (len(seq) + 1) - taken[b])
    seq.append(bands[b].pop(0)); taken[b] += 1
# exploration last (2026-10-31: inserted before the band weighting it took 3 of the first 6 slots, the tilts being
# focus-target only): one every ~1/EX slots of the final order, the first at slot ~1/EX
s0, seq, nx = seq, [], 0
while s0 or xq:
    if xq and (not s0 or nx + 1 <= EXS * (len(seq) + 1)):
        p = xq.pop(0); seq.append(p); nx += 1
        # its paired control: this fit's mean mix at the same target, right after it (same slice) -- the only clean
        # measure of whether exploration helps (cross-slice and per-slice "new best" counts are biased, 2026-10-31)
        mp = f"{MX}/knap-{FT}-sc4.00-e4-{tgt(p)}.mix"
        if mp in s0: s0.remove(mp); seq.append(mp)
    else: seq.append(s0.pop(0))
new = [f"{S}-{os.path.basename(p)[:-4]} {BASE} {p} {GOLD[S]}" for p in seq]
if False and NFIT % 4 == 0:   # 2026-10-31: never reached the workers (queue rewritten first); auto.sh queues ver1 every 2 h on q.prio
    for p in means:
        if float(tgt(p)) in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0):
            n = os.path.basename(p)[:-4]
            new.append(f"w-{n} {BASE} {p} golden-ver1 corpus/ver1.jsonl")

kept, dropped = [], 0
for l in open(Q):
    t = l.split()[0] if l.strip() else ""
    if not t or not CAND.match(t) or t in done or t in claimed or t in protect: kept.append(l.rstrip("\n"))
    else: dropped += 1
with open(Q + ".new", "w") as f:
    f.write("\n".join(kept + new) + "\n")
os.replace(Q + ".new", Q)
print(f"stream fit {FT} ({S}): Thompson share " + "/".join(f"{W[b]:.2f}" for b in range(3)) + f", ls {LSH:.2f}, ex {EXS:.2f} (beat rates "
      + " ".join(f"{k} {RATE[k]:.2f}/{NEV[k]}" for k in RATE) + f"); {len(new)} candidates queued ({skipped} identical to measured mixes skipped), "
      f"{dropped} superseded dropped")
