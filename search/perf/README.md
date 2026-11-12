# search/perf -- search performance pass (#161, w123, 11-12)

What one V3 search decision costs on box2 / ds8 (mix → pack → score), what can be made faster with byte-identical
containers and score rows, and the eval-slice size / decision rule that gives the most correct accept / reject
decisions per card-hour. Data: `data/searchperf/` (README there). LOG: #161.

## Files

| file | what |
|---|---|
| `tscore.sh` | one timed radiance KLD run (phase times from the log: start, init, load, prep, score, exit) |
| `timing.py` | the cost model from tscore / radmix timings (`data/searchperf/timing.json`) |
| `goldcat.py` | combined goldens (a slice's parts in one run), per-part rows / reports split back, `stages` (sequential-stage goldens: seeded disjoint document subsets), `sub` |
| `hostd-evalperf.patch` | rxrun: per-host ram lock rule for pack / mix (`eval.ram_mode`, RamGate: the 11-12 freeze fix), goldcat scoring, several eval cards; box2 / ds8 host confs; `check_rxrun.py --part evalstall` |
| `test_hostd_patch.py` | offline check of the patch's eval path (stub pack / mix / score, real golden layout): 17 checks |
| `rad-convert-no-error.patch` | radiance `rad-convert --no-error`: skip the error report's decode (container bytes unchanged) |
| `decbench.c` | rad-convert's error-report decode of one exl3i weight, 1 call vs T threads (bit-identical) |
| `rowpack.py`, `slicese.py` | stored 27B score rows → npz; paired SE vs documents / per-doc cap / heavy documents |
| `adhoc/x1-x7.py` | the SE study's readouts on colo's `se2/` (outputs in `data/searchperf/se/`) |
| `box2/` | the box2 drivers as run (w123 scratch `/data/q27/w123`): timing runs, V2 3.50 group moves (`moves.*`), settings (`gputest*.sh`), the k3 pack test (`packtest4.sh`), the rad-convert build (`rcbuild.sh`) |
| `protocol.py` | noise per document, SE vs slice fraction, the decision model (fixed vs sequential rules, cost per card-hour, V2's measured move distribution), rotation, the real-document check |

## Part 1 -- one search eval (box2 R9700, 210 W cap; V2 3.00 bpw mix; golden-search27 = 103 calib docs, 172k tokens)

| step | measured | note |
|---|---|---|
| score phase | 130-137 s (1,285 tokens/s) | ~87% of an eval; kernel speed is #164's job |
| radiance fixed cost per run | 7.7 s (start 0.3, init 1.5, load 3.8-5.8, prep 0.55, exit 0.3) | one per slice part today (x and c: 2 runs) |
| radmix (ext4, full 10.1 GB copy) | 10-19 s, median 11.2 s | its 10 GB "max RSS" is the page cache of its mmap'd sources |
| pack (rad-convert exl3i, per level, cached) | stock k3: 2,083 s on one core (nice 19 beside the ladder); `--no-error` 140 s, byte-identical; anonymous RSS 2.4-4.1 GiB, the rest of its 30-40 GB RSS is the mmap'd bf16 source (page cache) | ~all of it is the error report's decode (decbench: 15.8 s per K3 gate_up on one core) |
| lock waits | ev-p1-u3 (11-12): 11.7 min before its pack, then the host froze | pack and mix held ram.lock exclusive (below) |

Scoring settings (h2, 84k positions): `--max-model-len 12288`, `--max-num-batched-tokens 4096 / 8192` give
byte-identical rows and the same time (±1%); `--max-num-seqs 2` is 2.4% faster but its rows differ (not allowed);
`--max-num-seqs 4` fails at start; `--max-model-len 8192` refuses the 8192-token documents. Two radiance processes on
one card: 92 s each vs 49 s alone, rows identical: +6% throughput, only where two models fit (≤ ~4 bpw).

**The 11-12 freeze (lead notes):** an eval's pack holds ram.lock exclusive; it gets it only when every card session is
out at once (a level boundary: production sessions are 32 jobs a card), and then each card worker takes its card lock
and waits in `flock(ram.lock, LOCK_SH)` (`pool.Locks`), as do the CPU lane's assemble / post (`run.py`), for the whole
pack: box2 and ds8 sat idle 15:23-15:50 CST. Reproduced on the CPU stand-in (`check_rxrun.py --part evalstall`: a 20 s
stub pack leaves the level's card idle 20.0 s, the first job starts the instant the pack ends) and fixed per host:
`eval.ram_mode "shared"` (box2, ds8) = ram.lock shared after a RAM admission (MemAvailable >= `pack_ram_gib` 16 /
`mix_ram_gib` 4), conv.lock still one pack / mix a host; with it the longest idle stretch is 2.2 s (the session
yield) and the level's bytes are the same. check_rxrun 21 / 21 PASS on the patched tree. A real stock pack ran beside
the ladder this way on box2 (11-12 ~16:00-16:30 CST) without stalling it.

**Speed-ups (all byte-identical):**
1. **goldcat** -- a slice's x and c parts in one radiance run (one load): −7.7 s per eval. Rows split back are byte for
   byte each part's own run; top1 / positions per part from the rows equal the part's own report (checked on
   golden-search27 halves).
2. **mix outside the exclusive lock, overlapped with scoring** -- radmix under conv.lock + ram.lock *shared* (the
   ram_mode rule above; its RSS is page cache) and `slots.eval` = eval cards + 1, so the next candidate's mix runs
   while the current one scores: −11.2 s per eval and no more waits behind ladder sessions.
3. **several eval cards per host** (`eval.cards`): today each host scores on one fixed card.
4. **pack: `rad-convert --no-error`** (radiance patch): the error report decodes every quantised weight on one core
   (= the pack time); the container does not depend on it. k3 level: **2,083 s → 140 s (×14.9), container
   byte-identical** (14,288,037,888 bytes, `cmp`).
   Packs should be made as soon as a level is DONE (under ram.lock shared on box2 / ds8), not on the first eval that
   needs them.
5. Not done (proposals): a radiance `--kld-progress` file (per-document KL as it goes) so one run serves every stage of a
   sequential decision (−7.7 s per extra stage); a resident engine that swaps weights (design/v3.md "resident base"):
   load is 4-6 s of 190 s, not worth it now.

**Per eval** (standard two-part slice, 130 x pieces + 50 calib docs ≈ 227k tokens): before 11.2 + 2 × 7.7 + 176.7 =
**203 s → 17.7 evals / card-hour** (plus lock waits and on-demand packs: ~35 min per new level, the host frozen); after (1 + 2) 7.7 + 176.7 = **184 s → 19.5
evals / card-hour** (+10%; +6% more with two processes per card at ≤ 4 bpw). The bigger lever is Part 2.

## Part 2 -- slice size and decision rule

Measured on 1,450 stored 27B pair × part reports (`rows.npz` from gpubox w92 / w99 / w110 / w111 / w113 / w114 and the
w123 box2 runs):
- Per-document noise s = SE·√docs of log r grows with the effect: ≈ |log r| for real level moves (eight 8-layer
  3.00 → 3.50 moves: −3.4 … −8.1%, SE 0.39-0.72% on 103 docs), 2-3 |log r| for mixed pairs; x parts have ~1.9× the
  per-document noise of calib parts. Moves of disjoint tensor groups have uncorrelated per-document noise (mean corr
  0.00).
- The document bootstrap SE is calibrated: the same pair on two different slices, z sd 0.83 (n 40); SE ∝ 1/√f when a
  fraction f of the documents is kept (ratio 0.91-1.06).
- Per-doc caps: whole documents (calib cap 4096, heavy x categories 2048 as mkslice27 builds them) are right; caps
  ≤ 1024 save tokens but shift the estimate by 0.5-1.5 SE (another estimand). Heavy documents: in 58% of x-part pairs
  one document holds > 20% of the KL; the 20% water-filled readout does not lower the SE (moves the estimate 0.06 SE).
- The x / calib split (130 / 50) is within 2% of the noise-optimal split at its token budget: keep it.
- V2's search: 711 candidates vs the best mix measured before them at the same target and slice: median +1.5%,
  p10 / p90 −8% / +15%, 27% within ±2% -- most decisions are easy, which favours a cheap first stage.

**Decision model** (`protocol.py`, normal per-part estimates; priors: four normal close-call priors and V2's measured
candidates; costs from Part 1 incl. scoring the new incumbent on the rest of the slice after an early accept and one
full incumbent eval per rotation). Relative to scoring every candidate on the full slice (gain = true KL banked per
card-hour):

| rule | s / decision | gain / h | correct / h | wrong accepts / decision |
|---|---|---|---|---|
| fixed full slice | 194 | ×1.00 | ×1.00 | ×1.00 |
| fixed 1/4 slice | 61 | ×2.7 median, **×−5.4 worst** (mostly-bad moves) | ×3.0 | ×2.4 (max ×4.2) |
| **seq 1/8 → 1/2 → 1, reject early only** | **92** | **×1.89 (worst ×1.35)** | **×2.05** | **×0.60** |
| same rule, V2's candidates | 103 | ×1.70 | ×1.79 | ×0.58 |

Noisy moves (s = 0.08 for every effect size): the rule still gains ×1.29 (worst ×1.01), wrong accepts ×0.60.
On real documents (76 27d-f pairs, per-document ratios scaled to ±0.5 / ±1%, their own slice sizes, heavy documents
included) the rule accepts bad moves 0.8-1.1% of the time (model 0.7-0.8%; fixed full slice 1.5-2.0%) and good ones
67-68% (model 70-71%; fixed full slice 74-75%).

## Recommended V3 search eval protocol

1. **Slice:** the w120 two-part slice (≈130 xcorp pieces + ≈50 calib search-pool docs, mkslice27 caps). Split once into
   three stage goldens per part, seeded disjoint documents: 1/8, 3/8, 1/2 of each part
   (`goldcat.py stages GOLDEN PREFIX SEED 0.125,0.5,1`), then goldcat each stage's x + c.
2. **Incumbent:** scored on the whole slice (all stages) at each rotation.
3. **Candidate** (paired with the incumbent on the same documents = common random numbers):
   - stage 1 (1/8): **reject** if the pooled log ratio ≥ 0 (z ≥ 0);
   - stage 2 (cumulative 1/2): **reject** if z ≥ 0.5;
   - stage 3 (whole slice): **accept** iff the pooled log ratio < 0 **and** both parts < 0; then it is the new
     incumbent (all its rows exist, nothing to rescore).
   Never accept before the whole slice (an early accept must score the rest anyway and adds wrong accepts).
   z = pooled log r / delta-method SE over the scored documents (per-document u = B_d / mean B − A_d / mean A).
4. **Rotation:** a new slice every **~25 decisions** (the calib part's selection optimism reaches 1 SE after 25-66
   decisions in the close-call regimes, thousands at V2's move sizes): a fresh x part every time (xcorp, ~53k docs,
   never reused) and a fresh random draw of ~50 calib docs from the remaining search budget (93 never-used + 121
   proxy-only = 214 docs, so each calib doc returns in ~1 rotation in 4; record per-document decision counts in the
   slice registry). Cost: one incumbent eval per rotation (~4% at 25).
5. **When candidates are scarce** (each costs a knapsack fit), score them on the whole slice (fixed 1): the sequential
   rule trades missed good moves (gain banked per decision 0.83 vs 0.93 of an oracle) for speed, which pays only
   with a steady supply of candidates.
6. Held-out checks stay as in design/v3.md (ver3 for band-best candidates); best-of-many picks on one search slice are
   optimistic (V2: ~3%), so re-measure finalists on fresh slices.

## Rerun

    python3 -I search/perf/protocol.py rows.npz data/searchperf/timing.json OUT runs_per_eval=1 mix_s=0 v2=data/v2/results.jsonl
    python3 search/perf/test_hostd_patch.py <pipeline tree with hostd-evalperf.patch applied>
