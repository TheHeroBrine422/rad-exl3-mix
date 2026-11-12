# #166 boundary simulation: rules (w126, written 2026-11-12 ~5:52 PM CST, before any simulation run; git commit time is the attestation)

Question (TODO "Continuous early stop", user 11-12): one scoring run per candidate with documents in a fixed shuffled
order (x / c interleaved), the paired z vs the incumbent checked after every document from a warmup on, the run killed
when z > b(t). Which boundary? The user decides; this file fixes how the pick is made before any number exists.

## Data
- w123's stored 27B rows: colo `/data/work/w13/w123/rows.npz` (+ `.index.json`), the two-part pairs on 27d / 27e /
  27f (protocol.py `two_part`); `data/searchperf/timing.json`; V2's candidates `data/v2/results.jsonl` (empirical prior).
- Standard slice as w123: 130 x pieces + 50 calib docs, x per-document noise RX measured as protocol.py does.

## Rules compared (all reject early only; accept only after the whole slice: pooled log r < 0 and both parts < 0)
- **F**: fixed full slice.
- **S3**: w123's recommendation, staged 1/8 → 1/2 → 1 (reject at 1/8 if z ≥ 0, at 1/2 if z ≥ 0.5), costed as w123
  did (one extra radiance run per stage reached, goldcat = 1 run a stage).
- **S1**: the same checkpoints in one run (`--kld-progress`): checked after the document that completes 1/8 and 1/2;
  no extra runs.
- **B(p, w)**: continuous: t = documents scored / documents in the slice; from t ≥ w on, after every document, reject
  when z > b(t) = 2 · ((1 − t) / 0.9)^p, p ∈ {0.5, 1, 2}, w ∈ {0.05, 0.10, 0.20} (9 rules; for t < 0.1 the formula
  gives b > 2 and is used as is).
- z = pooled log ratio over the scored documents of both parts / its delta-method SE over those documents
  (u_d = B_d / mean B − A_d / mean A, the parts pooled), as protocol.py's real-document check.
- Order: each part's documents in a seeded random order, merged so that after k documents each part has its
  proportional share (the k-th document is x iff floor(k · Nx / N) steps up): both parts are present from the start.

## Two evaluations
1. **Normal model** (protocol.py's): each document's contribution drawn independently, part SEs as protocol.py
   (X = RX · s / √130 per unit, C = s / √50), KL share w of the x part from the 27d–f pairs. Priors: the four normal
   close-call priors (μ, σ) ∈ {(0.5%, 1%), (0, 1%), (0.5%, 0.5%), (1%, 2%)} × per-document noise s ∈ {0.01, 0.02,
   0.04}, and V2's measured candidates (empirical, s = 0.01 + 2|δ|). 4,000 draws per δ grid point.
2. **Real documents** (protocol.py section 4): each 27d–f pair with 1% ≤ |log r| ≤ 15%, per-document ratios scaled
   to δ ∈ {−1%, −0.5%, +0.5%, +1%}, slices drawn with replacement at the pair's own sizes, a fresh interleaved order
   per draw, 400 draws: P(accept) per rule.

## Cost per candidate (card seconds)
mix_s + run_s (one goldcat run) + τ · tokens of the scored documents (each part's mean tokens per document) + 2 s kill
latency when a run is stopped; + the incumbent's full-slice eval once per 25 decisions (rotation). S3: w123's model
(extra runs per stage). Constants from timing.json.

## Metrics (per rule and prior)
- gain/h = true KL banked per card-hour (E[−δ · accept] / cost), reported relative to F;
- wrong accepts per decision (δ > 0 accepted);
- **good-move loss** = 1 − (gain per decision) / (F's gain per decision): the share of F's banked gain a rule gives up;
- P(reject | δ < 0); mean fraction of the slice scored.

## Pick (fixed now)
1. Eligible: wrong accepts per decision ≤ F's on every prior (holds for any reject-only rule: a code check), and on
   real documents P(accept | δ = +0.5%) and P(accept | δ = +1%) each ≤ S3's + 0.005.
2. Primary: the highest median over the 13 priors (12 normal + V2) of gain/h relative to F.
3. Guard: the pick's worst prior (min ratio) must be ≥ S3's worst prior − 0.05; else the next rule by step 2.
4. Ties: rules within 2% of the best median count as tied; among them the lowest median good-move loss wins.
5. Reported beside the pick: S3, S1, F, and the full 3 × 3 grid; real-document P(accept) at δ = −1% / −0.5% (the
   good-move loss on real documents).
The pick is a recommendation for the user, who decides the boundary for the search.
