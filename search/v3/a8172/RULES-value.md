# RULES-value — did the #171 search buy anything? (w132, #172 item 2; lead note 11-13 10:15 AM CST, user)

Written and committed before any of these mixes is scored. Pre-registration is attested by this file's git commit.

**Question.** For every size, is the search's current incumbent better than its initial pick (the starting fit's
knapsack solve, `mkinit171.py`)? Both are measured as A8 containers on documents the search never saw.

**Arms (from `workers/w131/run/drv/state.jsonl` after the 10:11 AM CST stop).**
- Initial = each size's first incumbent event (`v3s171-<size>-init-*`).
- Current = each size's last incumbent event.
- Sizes 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5: 12 pairs.
- 7.0 is excluded: its current incumbent has the initial pick's assignment.
- 1.0 and 1.25 had no incumbent (below the menu).

**Containers.** Each tag's hostd `.mix` (`/data/q27/mixes/<tag>.mix`: the trunk levels' packs + the R16 head donor,
on the V2-3.00 base) plus `act <w> a8g128` on all 336 trunk weights, built by radmix. Scored with the rad10 build and
hostd's score command: bf16 KV, max-model-len 16384.

**Data.** golden-s171c and golden-s171d, the two-part slices w131 built for the search and never activated: 0 evals,
0 decisions, 0 fit rows. They are already in the ledger, so there are no new documents. The verify pool is not used.

**Statistic.** Per document, the KL sum over the golden's generated positions (`rxrun.hostd.doc_sums`). The documents
of both slices are pooled by part (c = calib merged pool, x = xcorp).
- d_s = mean over p ∈ {c, x} of log(Σ_p KL_current / Σ_p KL_initial), the search's own pmean statistic.
- **Primary:** the size-adjusted D_s = d_s + b_s (bpw_current − bpw_initial). b_s is the driver's last recorded slope
  per size: 1.5 2.755, 1.75 1.836, 2.0 1.686, 2.5 1.495, 3.0 1.382, 3.5 1.277, 4.0 1.145, 4.5 0.985, 5.0 0.781,
  5.5 0.565, 6.0 0.350, 6.5 0.158.
- **Pooled:** the mean of D_s over the 12 sizes.
- **CIs:** 95% from 2,000 bootstrap resamples of documents, stratified by part. One resample is shared by all sizes,
  so the pooled CI keeps the cross-size correlation.

**Reading (fixed now).**
- Size s **gained** if D_s's 95% CI is entirely < 0, **lost** if entirely > 0, and is **null** otherwise.
- The pooled result reads the same way.
- Also reported, not used for the reading: raw d_s, each slice separately (sign consistency), each part separately,
  and KLD per arm.

**Runs.** ds8, all 8 cards through `/data/q27/locks/card*.lock` (radmix under conv.lock + ram.lock shared). A mix is
deleted after its two scores. Script: `run-value.sh`; analysis: `value.py`.

## Amendment 1 (lead note 11-13 10:45 AM CST): third arm, w133's M1 re-solve picks

Written and committed after the initial / current results were read (11:35 AM CST), before any M1 mix is built or
scored. Amendment 1 adds an arm; it does not change the rules for the first two.

- **Arm M1:** `data/refit/w133/fin/picksT/pickT133-<size>.json` (13 picks, 1.5–7.0; M1 fit-M1-full on
  `menu133e.json`; none of its rows is on s171c / s171d). The mix is `Menu.eval_mix` of the pick on menu133e:
  - the trunk levels map to ds8's packs, named as hostd names them (`packs/<level>-s<scales[:8]>.rad`);
  - the head maps to the R16 donor (`heads/donors/hd-u*.rad`);
  - the embedding maps to w135's donor (`emb/emb-e2.rad`, `emb-e5.rad`), or to the V2 base's own for "base".
  It is built on the same V2-3.00 base plus `act <w> a8g128` on the 336 trunk weights and scored like the other two
  arms.
- **Comparisons**, with the same statistic, bootstrap (the same 2,000 resamples, seed 172), b_s and reading rules:
  - (i) **M1 vs current** at the 12 sizes: the primary for this arm;
  - (ii) M1 vs initial at the 12 sizes;
  - 7.0 (M1 vs initial = current; b = 0.70, the driver's 7.0 b) is reported apart, not pooled.
  - bpw_M1 = menu133e bpw + the driver's 8,255,936 B offset, as the driver computes bpw. Container bytes are reported
    beside it.
- Pooled readings for (i) and (ii): the mean over the 12 sizes, as before.
