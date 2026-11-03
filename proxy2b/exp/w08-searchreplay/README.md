# w08-searchreplay: move caps (F7) and posterior-mean selection (F8), replayed on V2's search

**Questions** (ranked.md F7, F8; V2 stream #70):
- **F7:** does the stream's model get less accurate, and do its candidates win less often, the more tensors a
  candidate changes from the best measured mix? If so, cap the tensors changed per move (2–8) and re-anchor often.
  Evidence for it: 2609.09867 found composition error 25–42× the per-tensor pricing error.
- **F8:** should a target's pick be the candidate with the lowest posterior-mean KLD (measurement shrunk toward
  kfit2's prediction), instead of the lowest measured KLD?

**Data.**
- On the VM (enough for F8):
  - `data/v2/results.jsonl`: 1,450 evals;
  - `data/v2/fit-final.txt`: kfit2's final fit. Every fit-slice measurement has its cross-validated prediction
    ("CV pred") and the slice offsets.
  - 123 of the stream's mixes were also evaluated on a held-out slice (ver1 `w-`, ver2 `v2-`).
- **Not on the VM:** the stream's 800-odd `.mix` files, and the per-fit `frontier-s*.txt` predictions made before
  each measurement. Both live only on iceman (`/root/knap/mixes`, `/root/auto`). Nothing on colo, the 4070 or in
  the repo has them, and I did not touch iceman.
- So F8 is replayed here. F7 is a ready script for iceman (`f7replay.py`, read-only, CPU, about a minute).

## F8 result (`f8replay.out`; replayed here; this pass rule was ranked.md's, not pre-registered by the lead)

1. **Calibration.** Within each (held-out slice, target) group, the held-out log KLD is predicted best by an
   equal-weight blend: post = w·log(measured) + (1 − w)·log(CV pred).
   - Leave-one-group-out best w: **0.40–0.45** (25 groups, 113 mixes).
   - Within-group RMS error: measured alone 0.048, blend 0.038, model alone 0.043.
   - Noise split from the held-out deviations: measurement σ ≈ 0.039, model τ ≈ 0.034 in log KLD, implying
     w = 0.43. A fit-slice eval is about as noisy a guess of held-out KLD as kfit2's prediction is.
2. **Selection on the held-out groups.** Rule: posterior (w 0.5) vs measured-best.
   - The picks differ in 4 of 25 groups: 2 better (ver2 4.5: −9.3%; ver1 4.5: −2.7%) and 2 marginally worse
     (+0.2%, +0.2%).
   - Geomean over all 25 groups: **−0.48%**; ver2 groups alone (11): −0.87%.
   - **ranked.md's bar** ("posterior picks ≥ 1% better on ver2"): **not met.** With 2 differing ver2 picks the
     test cannot really decide.
   - The held-out evals went mostly to measured-best finalists, which biases this test toward measured-best.
3. **Simulation over whole targets** (each target's real candidate set, truth = model + N(0, τ),
   measurement = truth + N(0, σ)):
   - expected regret vs the best candidate: measured-best +1.80%, posterior +1.42–1.44%, model-best +2.95%;
   - so posterior-mean selection gains **~0.35–0.4% KLD** per pick.
   - Per target it ranges 0.1–0.8 points, larger where targets have fewer candidates or noisier evals.

**Reading.** F8 is real but small: ~0.4%, against ranked.md's −0.5…−1.5%. It costs nothing, though. Use the
posterior with w ≈ 0.45 in three places:
- (a) the "best measured mix" each local-search round starts from (`kfit2.py` KFIT_LS);
- (b) which candidates go to the ver3 read (V3's design reads ver3 only for a band's new best);
- (c) the tie rule between finalists.

Thresholdout for ver3 reads (F8's second half) is a process rule. Replay cannot measure it.

## F7 (to run on iceman: `run.sh`)

`f7replay.py` takes every measured stream candidate. For each one it computes:
- **anchor:** the best mix measured at its target before it;
- **n:** the tensors it changes from that anchor;
- **err:** its log prediction error against the prediction its own fit logged before the eval;
- **win:** whether it beat the anchor by more than 1%.

It reports, by n-bin (1–2, 3–4, 5–8, 9–16, 17–32, 33+) and by kind (mean / Thompson / local search / tilt /
random): bias, RMS error, win rate and gain per eval (E[max(0, 1 − meas/anchor)]). It then replays caps
c ∈ {2, 4, 8, 16}: the evals above the cap, moved to the gain rate below it, per band.

**Draft pass rule (for the lead to commit before the run).** Adopt cap c* = the smallest c in {2, 4, 8, 16} such
that, in the < 2.75 or the 2.75–4.5 band (the ones that matter), all three hold:
1. evals with n > c win at most half as often as those with n ≤ c;
2. their RMS prediction error is ≥ 1.5× that of n ≤ c;
3. the estimated final-best gain at c is ≥ 0.5%.

The cap then applies to local search and Thompson proposals; a Thompson draw is clipped to its c* best-predicted
changes. Mean solves stay uncapped, since they are the model's re-anchoring. If no c qualifies: no cap, and F7 is
closed.

**Kind confound.** Local-search mutations change 1–4 tensors plus downgrades; mean and Thompson solves change
many. The per-kind table is the check. A cap holds only if the n-gradient shows up *within* Thompson (and within
mean) too. Otherwise the gradient is just "local search beats the rest", which V2 already knew: on 2026-10-31,
local search set new bests 9× as often per eval as Thompson.

**The cap estimate is optimistic.** It assumes the moved evals would win at the small-move rate. Small moves get
used up, and later wins start from a better best. Treat it as an upper bound.

## Files

| file | what |
|---|---|
| `f8replay.py`, `f8replay.out` | F8 replay (VM, 0.2 s) |
| `f7replay.py` | F7 replay (iceman, read-only) |
| `selftest_f7.py`, `selftest_f7.out` | synthetic stream with a planted composition error: f7replay finds the RMS rising and the win rate falling with n, and positive cap gains |
| `run.sh` | F8 locally (if not done), then F7 over ssh on iceman |

**GPU time:** none. **Iceman:** a ~1 min niced CPU read-only job. It takes no card or RAM lock, but run it when
iceman is otherwise idle.

## What changes what
- **F8 (ranked #14 with F7):** E drops from −0.5…−1.5% to ~−0.4%. Adopt it anyway, as a free default (the three
  places above).
- **F7:**
  - If the cap rule passes: change kfit2's Thompson / local-search generators (cap n). That is a small patch of
    `search/kfit2.py`, not written until the replay says so.
  - If it fails: F7 is closed, and the V3 stream keeps V2's generators.
- **V3 stream design (v3.md §5):** the measured-vs-model noise split (σ ≈ 0.039, τ ≈ 0.034 per fit-slice eval vs
  held-out) is the number to set the shrinkage by. The streaming "new best" rule should use the posterior, not
  the raw measurement.
