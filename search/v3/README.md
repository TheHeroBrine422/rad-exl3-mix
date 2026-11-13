# search/v3 — the V3 search driver (#166, w126, 2026-11-12)

Per release size an incumbent mix; candidates from the fitted model (its predicted-best changes) plus uncertainty /
exploration moves (V2's kinds); a stream of evals through `rxrun.multi` (box2 / ds8); each candidate scored once with
documents in a fixed interleaved x / c order and **stopped early** by the host when it is clearly worse; accept on the
whole slice; every eval (accepted, rejected, stopped) is a row for periodic **kfit3 refits**; slices rotate. The driver
decides nothing on its own: CONF `"go": true` is the user's go for a real run (without it only `"dry": true` configs
submit anything). Design: TODO "Search driver", "Continuous early stop" (+ the user's boundary), "Final picks",
"Low sizes in parallel", "Heads before w124's un-permute"; w123's protocol (`search/perf/README.md`).

| file | what |
|---|---|
| `driver.py` | the loop: ingest results → decide (paired per document vs the size's current incumbent) → fit rows → refit (background, `fit_cmd` on colo) → rotate → generate (`queue_depth` outstanding) → report (`status.txt`, `frontier.json` per release band). State `ROOT/state.jsonl` (append-only), restart = same command |
| `moves.py` | kinds `solve` (Lagrangian MCKP at the size's byte cap, re-linearised), `partial` (part of the way to it), `pair` (one-step upgrade paid by the cheapest downgrades), `block` (group × 8-layer band), `thompson` (solve under a Laplace draw), `explore` (UCB over posterior draws / random bands); shares by V2's kind rewards |
| `fit.py` | kfit3's model (`search/refit/kfit3.py`, subclassed, not edited) on the search's rows: labels `S|full` / `S|pre<k>` (k on a 5 % grid) with the generating incumbent's KLD on the same documents as anchor, residual / √(SE² + τ²), weak prior on family factors, Laplace draws. `python3 fit.py IN OUT` is the remote fit |
| `seqrule.py` | the accept / reject rule (paired delta-method z over the scored documents; `curve` b(t) = 2((1 − t)/0.9)^p after a warmup, `stages`, `full`); also the hosts' `rxrun/earlystop.py` (byte-identical copy) |
| `slices.py` | slice registry (`ready` → `active` → `retired`, decisions per size, per-document decision counts) and `goldmix` (a slice's parts merged into one golden, seeded order, x / c interleaved) |
| `menu.py`, `mkmenu.py` | the option menu (per tensor: options with bytes, absolute kfit3 error, family; head / embedding options with donor paths); `mkmenu.py` builds it from ladder level dirs (+ w122's head price table) |
| `finalpick.py` | the final pick of a band (w127): top ~5 on ver4 (calib) + a fresh xcorp part, size-adjusted score with the band's frozen b, pooled best, calib part as a veto, per-part / pooled / part-gap paired bootstrap CIs vs the pooled best |
| `abreport.py`, `RULES-sizeadj.md` | the #167 A/B (raw vs size-adjusted accept on the stand-in; rule committed before data, 1a3a449) |
| `earlystop-sizeadj.patch` | pipeline/rxrun/earlystop.py = seqrule.py again (the size offset `off` in the early-stop rule): stage with the next restage |
| `boundsim.py`, `RULES-boundary.md` | the early-stop boundary simulation (rule committed before data, c17045c) — result below |
| `standin.py` | CPU stand-in end-to-end (synthetic truth, fake executor speaking the queue's file protocol) |
| `hostd-earlystop.patch`, `test_hostd_es.py` | rxrun.hostd: `early_stop` evals scored with `--kld-progress`, followed, killed at the boundary (+ README, box2 / ds8 `score_cmd_progress`); offline check |
| `RULES-parts.md`, `earlystop-parts.patch`, `run171/` | #171 (w131): per-part fit labels + x ratio scale (`fit_opts.xscale`), the equal-weight per-part statistic (seqrule `stat` "pmean", default) for accept and the hosts' early stop (the patch makes rxrun/earlystop.py = seqrule.py; supersedes `earlystop-sizeadj.patch`), the first real run's tools and launch (`run171/README.md`) |
| `radiance-kld-progress.patch` | radiance `--kld-progress FILE` (branch `qc-exl3-v3-kldprog` = qc-exl3-v3 81bad51 + ca5491d) |

## Boundary simulation (RULES-boundary.md; `data/searchv3/boundsim/`)
w123's 446 two-part pairs (27d–f), standard slice 130 x + 50 c, M = 4000. Gain/h vs full slice, median (min) over 13
priors; good-move loss = share of F's banked gain given up; real documents P(accept) at δ = −1 / −0.5 / +0.5 / +1 %:

| rule | gain/h | good-move loss | slice scored | real P(acc) −1 % / −0.5 % / +0.5 % / +1 % |
|---|---|---|---|---|
| F full slice | ×1.000 | 0 | 1.00 | 0.739 / 0.753 / 0.0202 / 0.0154 |
| S3 w123 staged (separate runs) | ×1.767 (1.300) | 10.6 % | 0.40 | 0.673 / 0.686 / 0.0108 / 0.0080 |
| S1 staged in one progress run | ×1.811 (1.365) | 10.6 % | 0.40 | same as S3 |
| B p0.5 w5 / w10 / w20 | ×1.713 / 1.665 / 1.580 | 2.0 / 0.8 / 0.3 % | 0.50–0.57 | |
| B p1 w5 / w10 / w20 | ×1.785 / 1.739 / 1.651 | 1.8 / 0.9 / 0.5 % | 0.47–0.53 | |
| **B p2 w5 (pick)** | **×1.867 (1.399)** | **2.5 %** | 0.44 | 0.704 / 0.718 / 0.0104 / 0.0074 |
| B p2 w10 / w20 | ×1.826 / 1.730 | 1.8 / 1.4 % | 0.46 / 0.49 | |

Pick by the rule: **b(t) = 2 · ((1 − t)/0.9)², warmup 5 %** — the best median gain per card-hour (×1.87 vs ×1.77 for
S3), worst prior ×1.40 (S3 ×1.30), real-document wrong accepts at or below S3's, and it gives up a quarter of S3's
good moves (2.5 % vs 10.6 % of the full slice's banked gain). B p2 w10 (×1.83) misses the 2 % tie band by 0.004.
The user decides.

## Run
```
# menu (on the host with the level dirs; numpy only)
python3 -I search/v3/mkmenu.py menu.json --levels /data/q27/ladder/q27v3-* --base BASE.rad --diag LEVELDIR [--heads hcurve122.json --head-donor '/path/hd-{plan}.rad']
# slices: parts by eval/slices/mkslice27.sh, merged per host, registered on the VM
python3 -I search/v3/slices.py goldmix OUT SEED x=GOLDEN_X c=GOLDEN_C ; python3 search/v3/slices.py add REG NAME MIXDIR
# the driver (VM, light; fits through fit_cmd on colo; evals through the coordinator's CONF "evals")
setsid nohup python3 search/v3/driver.py CONF.json > driver.out 2>&1 &      # --status; touch ROOT/STOP to end
```
**Changes before the first run (w127, #167, 11-12):**
- **Size-adjusted score** (TODO 11-13): accept / reject on log KLD + b·bpw: a candidate is accepted iff
  log r + b (bpw_c − bpw_inc) < 0, pooled and on every part; the hosts' early stop uses the same statistic (`off` in the
  early-stop rule: restage `earlystop-sizeadj.patch`). b per size = the current fit's local slope −d log KLD / d bpw (the
  fitted model's knapsack optimum at size ∓ `b_delta` 0.1 bpw from the incumbent), refreshed after every refit, clamped to
  `b_range` [0.05, 6]; before the first fit `b_default` (V2 board: 1.6 at ≤ 3, 1.1 at 4–5.5, 0.7 at ≥ 6, linear between).
  Each candidate keeps the b of its submit (`state.jsonl` submit `b`). Band bests (monitor reads, frontier points, `top`)
  use log KLD + b_T (bpw − T). `size_adjust: false` = the old raw rule. **Freeze for the final picks:**
  `driver.py CONF --freeze-b ROOT/b-frozen.json` (refuses to overwrite) before any ver4 read; `finalpick.py` reads it.
- **Monitor alert-only** (user 11-13): no automatic stop and no N. A size gets a soft ALERT (driver.log, status line,
  `monitor.json` "alerts", `events.jsonl` for `convplot3.py --events`) when its size-adjusted monitor best has not improved
  by more than the paired SE over its last `alert_frac` 0.2 of decisions, once it has ≥ `alert_min` 50; it clears when
  the best improves. End a size by hand: `touch ROOT/STOP-<size>` (e.g. STOP-3.25). Curves: the monitor reads are in
  the coordinator's results.jsonl under the monitor's slice name: `convplot3.py RESULTS OUT --heldout-re '^ver3' --events
  ROOT/events.jsonl`; convplot3 normalises by V2 references on the same slice, so list them in CONF
  `monitor.refs` ([{"tag": "ref-V2-3.00", "mix": EVAL MIX}, ...]: scored once on the monitor golden, never a decision).
- **Warmup** of the boundary is CONF `warmup` (default 0.05, overrides `rule.warmup`); the rule stays the pre-registered
  B p2 w5 unless the user says otherwise.
- **Byte offset prior:** CONF `bytes_offset0` (w126 measured 8,255,936 B) is applied from the start; the first
  donor-free container still re-calibrates it (with head donors in every mix nothing else would).
- **Pools merged** (user 11-13): `slices.py draw REG LEDGER POOLS N SEED` (default mode `merged`) draws from fit + search
  (1,134 docs) least-used first by ledger decision counts (head-input collections w94 / w119-headx count 0) plus this
  registry's own; `slices.py poolcheck POOLS GOLDEN..` checks that a slice's (or a prior_obs golden's) calib docs are all
  fit + search. fit.py / `prior_obs` rows may come from any fit + search slice (one pool; each label keeps its own offset).
  Today: 220 docs with 0 counted uses, 177 with 1.
- **Final picks:** `finalpick.py SPEC.json` (docstring: SPEC = band, frozen b file, parts c / x, candidates with
  their rxrun docs files from whole runs).

CONF keys: root, menu (+ menu_remote for the fit host), registry, base (host alias), sizes, size_tol (0.02 bpw below
the target), evals / results / docs (the coordinator's), rule, queue_depth, refit_every, rotate_after (25), audit_frac
(0.1: scored without early stop, decided by the rule after the fact, logs the early-stop bias), size_adjust (true),
b_default, b_delta, b_range, warmup, bytes_offset0, topn (5), monitor {name, golden, alert_min, alert_frac, refs}, ndraw, fit_cmd
(`{in}` `{out}`), fit_opts (tau 0.02, beta, ...), prior_obs (rows of the calibration swaps, obs JSONL), init
({size: assignment JSON | "nearest"}), init_head / init_emb, size_weights, go / dry.

## Tests (11-12)
- radiance `kld_test` (12 cases, incl. `progress_lines_follow_documents_and_leave_the_rows_unchanged`) and
  `config_test` (68) PASS (host build, colo).
- `test_hostd_es.py` 11 / 11 PASS (stub radiance with progress; stopped runs killed, prefix sums == plain run's;
  unstopped results == plain).
- Real engine (box2, V2 3.00, the dry golden): rows with `--kld-progress` **byte-identical** to without (cmp);
  the interleaved golden's per-document sums == the original golden's (max diff 0.0); progress in document order,
  its per-position KL == the rows file; KLD (all positions) 0.63949 = w117's.
- **CPU stand-in** (`standin.py`, colo; `data/searchv3/standin/`): 16-layer synthetic menu, 21 levels, truth = kfit3's
  form with per-tensor deviations, documents with per-block sensitivities (x noise 2x c), 400 evals. si6 (sizes 1.5 /
  2.5 / 4.0 / 6.0): **7 / 7 PASS** -- true KL −57 / −33 / −35 / −17.5 %; 0 of 54 accepts truly worse; 230 of 266
  rejects stopped early, card time 11.5 h vs 21.7 h all-full; fit vs truth on one-step moves corr 0.990; 4 rotations;
  restart from state; monitor 53 reads, never fitted. si7 (fit groups: low 1.0 / 1.5 on every row, main 3.0 / 4.0 on
  their own): 7 / 7 PASS (1.0 below that menu: skipped as designed). Early-stop bias from audit runs: prefix − whole
  log r +0.0004 / +0.0009 (stopped candidates read slightly worse, as expected; negligible here).
- **Dry run on box2** (11-12 6:21–6:49 PM CST; one card (card3.lock); own `rxrun.multi` instance + the patched
  hostd, eval-only; menu = 6 pass-1 levels k2.625–k3.75 in V2 3.00's head / embedding; sizes 3.0 / 3.25; slices
  dryA / dryB = the retired golden-search27c-c's 26 calib docs split into two pseudo-parts (no new calib docs;
  not decisions anyone uses); rule B p2 w5, rotate after 4, audit 0.2; fits on colo through fit_cmd, ~10–20 s a fit
  round trip): 17 evals (4 incumbent scores incl. the rotation's re-scores, 13 candidates) → 12 decisions, 5 accepts,
  1 stale; **6 runs stopped early by the host** (after 3–19 of 26 documents, 16–26 s after the score started vs
  ~60 s for the whole slice), 1 audit decided by the rule after the fact; packs `--no-error` 2–3 min a level; byte
  offset container − menu = 8,255,936 bytes, calibrated on the first donor-free result: every later container ==
  its menu size (≤ the cap); the two candidates made before it were 0.0025 bpw over (fresh runs calibrate on the
  incumbents first; init incumbents keep a 0.005 bpw margin). Logs: `data/searchv3/dryrun/`.

## Before the first real run (lead / user)
1. **Go** (CONF `"go": true`) and the boundary (the pick above or another).
2. **Restage** box2 / ds8 with `hostd-earlystop.patch` (+ w123's evalperf changes already in HEAD) and radiance
   built from `qc-exl3-v3-kldprog` (`score_cmd_progress` in the host confs points at the staged radiance).
3. **Menu**: `mkmenu.py` over all pass-1 levels on each host (levels stay where they were built: an eval whose levels
   sit on two hosts is BLOCKED -- copy level dirs or keep each size's menu to one host's levels), w122's head price
   table + donors (`--heads`, `--head-donor`), the embedding options; packs of every level made when DONE.
4. **Initial model**: the knapsack refit's rows as `prior_obs` (obs JSONL: tag, assign, label, kld, se), or the
   refit's per-size picks as `init` assignments.
5. **Slices**: c parts need a calib budget decision: by today's ledger only **27** search-pool docs are unused
   (`slices.py draw ... never`); `no27b` gives the same 27; **177** if docs used only by the head-input collections
   w94 / w119-headx-search (2 decisions on short windows) are allowed (`headx`). x parts: fresh xcorp (mkslice27 /
   pickx on the host with night1105's goldens), merged per host with `goldmix`; ledger rows (role select) per slice.
6. **Monitor**: CONF `monitor` = golden-ver3-8k on each eval host (alert-only since w127: no N to fix) + `refs` (V2
   containers for convplot3's scale).
7. **Fit rows**: settled by the pool merge (user 11-13): search slices and prior_obs both come from fit + search.
8. Fit groups for the low sizes (TODO "Low sizes in parallel"): `fit.py check OBS.json [SEED]` compares main-only vs
   joint on held-out 2.5–4.5 rows once real rows exist; set CONF `fit_groups` from it.
