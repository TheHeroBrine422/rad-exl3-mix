# w09-racing: early rejection of stream candidates (G3) and lower-variance comparisons (G6)

Ideas **G3** (group-sequential early rejection / racing) and **G6** (control variates, stratified / Neyman slices)
from `docs/ideas/ranked.md`. These affect search throughput only, not KL directly. All CPU; **no GPU**.

## Question
1. **G3.** The V2 stream scored each candidate on a whole search slice, then kept it only if it beat the best mix
   measured at that target (`search/auto/streamq.py`). If a candidate's partial scores already show it will lose,
   how much eval time does stopping save, and how often does it throw away a candidate that would have won?
2. **G6.** How much smaller can the standard error of a paired slice comparison get at equal tokens? Three routes:
   - category stratification;
   - Neyman allocation of documents to categories;
   - a control variate.

## Mechanism
- **Stopping mid-run is possible without engine changes.** The recorder pre-sizes `score.f32` and `pwrite`s every
  row as it is scored (`eval/radiance-patches/kld.cpp` l.108/539). A watcher can read the partial file and stop the
  run, so the ~80 s fixed cost is paid once.
- **The incumbent's per-document scores are already known in full.** It was scored on the whole slice. That makes
  this a ratio (CUPED-style) estimation problem with a known auxiliary total.
- **Decision rule.** After a token fraction f, with documents fed in a fixed random order interleaved by category:
  - predict candidate B's full-slice sum as `B_seen + R_seen · A_unseen`;
  - predictive variance = residual variance (proportional to A_d) over the unseen documents + the ratio's uncertainty;
  - P(win) = P(B_full < A_full); reject when P(win) < α.
  - Stages at 10 / 20 / 35 / 50 / 70% of the tokens.
- **Variants scored in the replay:**
  - `cat`: a separate ratio per category;
  - `elast`: a document's past "elasticity" as a covariate;
  - `prior`: kfit2's predicted ratio from the frontier reports, weighted by its measured error τ.
- **G6 estimators.** Delta-method SE of log(ΣB/ΣA), which equals pcmp.py's document bootstrap (self-test: within 3%):
  - plain;
  - additive strata;
  - separate ratio per category (`strat-ratio`);
  - Neyman allocation at equal tokens (allocation fitted on the first half of the pairs, scored on the second);
  - elasticity control variate.

## Data
- **Where the data is.** The V2 per-row scores (`evq/scores/<tag>.f32`, about 770 stream candidates on
  calA / fit2 / fit3) exist **only on iceman** (`/root/evq`). Iceman was unreachable on 2026-11-03 (port 2223 refused).
  Nothing is mirrored on colo, the 4070 or thor.
- **What ran today:**
  - `aggsim.py` on the V2 slice-level KLDs in `data/v2/results.jsonl` (no per-row data needed);
  - `race.py` on the 0.8B proxy's 113 per-row score files (`/data/yb/*.f32` on g08-search1, copied to colo
    `/data/work/w09/`) as the real-data self-test.

## Results so far (no V2 per-row data yet)
**Aggregate estimate of G3 (`aggsim-v2.txt`).**
- Inputs: 664 stream candidates had an incumbent, and 41% beat it. The median |log ratio| was 0.045, and 27% of
  candidates were within 2%. A document-iid SE of 1.6–2.0% per slice (capstudy) was assumed.
- Mean eval time relative to today, at ≤ ~1% wrong rejections (α 0.01):

| fixed cost T0 | eval time vs today | saved |
|---|---|---|
| 80 s (today) | 0.88–0.89 | 11–12% |
| 15 s (resident-base runner, v3.md §5) | 0.76–0.79 | 21–24% |

- **The ceiling is the fixed cost.** The V2 median wall was 130 s, of which only ~50 s scales with tokens. Even
  instant rejection of every loser could save at most ~22% at T0 = 80 s.
- A slice twice as large with racing costs 1.11–1.13× today's time (T0 80 s). Racing does not pay for a larger slice.

**0.8B real-data replay (`proxy08-race.txt`; 1,499 ordered pairs of quant arms).** This is a different regime:
pairs are random arm pairs, not candidate vs best.
- **`ratio` variant:** P(win) is well calibrated (stage-1 bins: 0.004 / 0.06 / 0.16 / 0.38 / 0.62 / 0.95 observed).
  At α 0.01 it rejects 43.5% with 0.40% wrong rejections, saving 16.5% (T0 80) or 27.9% (T0 15).
- **`cat` variant:** 1–1.5 points more time saved, but overconfident early. It fails the wrong-rejection check in the
  synthetic self-test (2.2% at α 0.01) and is excluded from the pass rule.
- **`elast` variant:** no help (control-variate SE 0.98× plain) and worse calibration.
- **G6 SE ratios to plain** (median over pairs):

| estimator | SE ratio |
|---|---|
| additive strata | 0.935 |
| separate ratio per category | 0.892 |
| Neyman allocation at equal tokens | 0.882 |
| elasticity control variate | 0.98 |

- Neyman allocation would take ~2.9× more multilingual docs and ~0.25–0.4× the agentic / tools / long-agentic docs.

**What the kfit2 prediction can and cannot do (G6, "predicted KL as the covariate").** kfit2 gives one number per mix,
not a per-document value. So it cannot reduce the document-sampling noise of a slice comparison. It can only act as a
prior on the candidate's ratio, which is the `prior` variant here. V2's model error was ~12% per mix (frontier
`+-12%`), against a ~1.8% slice SE, so the prior is expected to add little. The replay measures τ and the gain.

**Important caveat for G6.** On a category-matched slice, the stratified-ratio point estimate equals the plain one.
Its smaller SE gives better-calibrated z-scores but changes no stream decision. Only the **Neyman slice design**
changes the actual noise per GPU-second. It also needs weighted means: per-document weights in `kld.json`, used by
`aggscore.py`, `pcmp.py` and `evalq.py`.

## Arms / what `run.sh` does
- Per V2 slice (calA, fit2, fit3), `race.py` in stream mode runs:
  - variants `ratio`, `cat`, `elast` and `prior`;
  - α ∈ {0.001, 0.003, 0.01, 0.02, 0.05};
  - 20 document orders;
  - T0 ∈ {80, 15} s, RATE 2,500 tok/s;
  - the G6 table.
- `aggsim.py` is re-run on iceman's full results.

## Draft pass rules (for the lead to commit before the V2 replay is run)
**G3.** Variant `ratio`, `--vw A`, stages 0.1/0.2/0.35/0.5/0.7.
1. **Choose α on calA:** the largest grid α whose wrong rejections are ≤ 1.0% of calA's accepts.
2. **Judge at that α on fit2 + fit3** (each its own replay, results pooled by candidate count):
   - wrong rejections < 2% of accepts;
   - forfeited gain (Σ of the wrongly rejected candidates' −log ratio) < 1% of the accepted candidates' total gain;
   - calibration: observed accept rate in the stage-1 P < 0.01 bin ≤ 0.02.
3. **Then by time saved:**
   - ≥ 30% at T0 = 80 s → adopt racing in `evalq2.sh` now (ranked.md's bar).
   - < 30% at T0 = 80 s but ≥ 20% at T0 = 15 s → adopt only together with the resident-base runner (v3.md §5,
     ranked.md G4/G5), and count racing in that runner's value.
   - Otherwise → drop G3.
4. **`prior` variant:** adopted only if, at the same α rule, it saves ≥ 5 points more time on fit2 + fit3 with
   wrong rejections still < 2%.

**Expected outcome**, from the aggregate estimate and the 0.8B replay: the T0 = 80 s bar fails (11–17%); the
T0 = 15 s branch is borderline-to-pass (21–28%).

**G6.**
1. **Neyman search slices:** adopted for the V3 search slices if the median se ratio (Neyman, equal tokens,
   allocation fitted on the first half of the pairs) is ≤ 0.85 pooled over calA / fit2 / fit3 and ≤ 0.90 on each.
   0.85 means ≥ 28% fewer tokens at equal SE. ranked.md's bar of 0.80 is reported beside it.
2. **Stratified-ratio SE in pcmp-style z-scores:** adopted if its median ratio is ≤ 0.90 on every slice (reporting
   only, no decision change).
3. **Control variates:** elasticity is dropped unless its median ratio is ≤ 0.90.

**Expected outcome**, from the 0.8B: Neyman ~0.88 and strat-ratio ~0.89 → G6 likely fails 0.85, passes the
reporting rule. Both are inside 0.80–0.95 on the 0.8B; the 27B slices (8k cap, long traces) may differ.

## Cost
- **GPU time: 0.**
- Copy: ~770 score files plus metadata from iceman to colo (~2–4 GB, one niced tar).
- CPU: ~1–3 min per slice on colo.

## What changes what
- **G3 passes (T0 80):** add a racing watcher to `search/auto/evalq2.sh`, as a patch written after the replay.
  - Corpus fed in the category-interleaved order.
  - Every 10 s a watcher reads the partial `score.f32` against the incumbent's per-document sums and stops the server
    on reject.
  - Rejected candidates are logged with `"raced": f`, so kfit2 treats them as partial measurements or skips them.
  - It raises candidates per GPU-hour by 1/(1 − saved), which matters at ≤ 2.5 bpw where V2's search had not
    flattened (ranked.md G3, INDEX #70).
- **G3 passes only at T0 15:** racing becomes part of the resident-base runner's case (v3.md §5 step 1 measures T0
  first).
- **G6 Neyman passes:** rebuild the V3 search slices with Neyman allocation, using `eval/pools/mkver.py`-style
  drawing with per-category document counts from `neyman_alloc_rel_docs`. Then patch weighted means into
  `aggscore.py`, `pcmp.py` and `evalq.py`. Not written yet: it only makes sense if the rule passes.

## Files
- `race.py`: the replay (G3 + G6), on any per-row score files plus golden. Stream mode with `--results`.
- `aggsim.py`: the aggregate-only estimate.
- `selftest.py`: the self-test. `selftest.txt` is its output: **PASS**. It covers:
  - synthetic calibration and wrong rejections;
  - strat-ratio and Neyman gains on built-in structure;
  - the stream pairing and the frontier prior parser;
  - delta-method SE vs document bootstrap on 8 real 0.8B pairs (ratios 0.97–1.01);
  - aggsim's bridge variance.
- `aggsim-v2.txt`, `proxy08-race.txt`, `proxy08.jsonl`: results so far.
- `run.sh`: the V2 replay (VM → iceman tar → colo). Needs iceman up.
- No shared-code patches.

## Runtime cost (user rule, 2026-11-03)
n/a (eval / search statistics).
