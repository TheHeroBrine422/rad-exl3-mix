# Redteam report: where V3's plan and the H_out decision can be wrong (2026-11-07)

Ideas are numbered A1–J5 and grouped by theme. Facts I checked against the repo or colo today are marked **[checked]**. Everything else comes from the attached plan documents.

## What changed since the instructions were written

The "8.5 h 27B diagonal run" is done: 1,923 sequences, #33-24. The current equivalent cost is the H_out ladder. About 24 levels at K≥4 move from ~37 min to ~3.2 h each, so roughly +60 card-h. The per-mix writer passes add ~4–5 card-h per mix. Every check below is aimed at not wasting those hours.

## Headline findings (all from cheap checks)

**1. Two adopted settings can't run together.** `pipeline/rxq/config.py:36` raises `ValueError("tile_cd and H_out together…")`, and `tilecd.py` refuses `ldlq_2hess` **[checked]**. Tile CD on attn_qg (#77, adopted) and H_out at K≥4 (#33-28, adopted) therefore conflict. Also, `v3.toml [inputs.hout]` has only `apply_k_max = 8.0` and no `apply_k_min`, so turning H_out on would apply it to every level **[checked]**.

**2. H_out at K4 has not been checked on verify.** H27.md says: "Not checked on verify (rule decides on search)" **[checked]**. The same slice produced the swj "−22% at K4" that a head-matched verify run later withdrew.

**3. About a quarter of the official score lies beyond 8k context, and no 27B decision has looked there.** On a character proxy over `calib.jsonl` (chat docs scored on assistant spans, text docs scored fully), ~24% of scored characters sit beyond ~8k tokens and ~16% beyond ~32k. Most of it is the 8 long agentic traces (12.6%) and the "long" text docs **[checked, char proxy]**. KL also roughly doubles past 32k (#52), so the share of KL mass is larger still. Every 27B adoption used `golden-search27` (8k cap), and the stream's search slices cap at 2k.

**4. The knapsack prices fractional levels from the one-sided proxy.** `knap5`'s `KN_LEVELS` places levels using ourq's `proxy_err`, which is one-sided H_in **[checked]**. H_out quants trade one-sided error for KL. The K3.875→K4 boundary and every K≥4 level will be mispositioned, and H_out's gain is not in the model at all (reprice.md says so).

**5. The non-trunk parts are a bigger lever than assumed.** The swj head-matched controls show that swapping V2 3.00's head+embedding for V2 4.00's changes total KL by **0.761/0.757** on verify at a K4 trunk, for +0.095 bpw. Trunk bytes at the margin buy about −13% for the same bpw (board slope at 4.0). The design doc's "head ≈ 5–7% of KLD" is likely low at 3.5–4.5 bpw.

**6. bf16 tensors that sit outside the knapsack.** `gdn_ab` is 23.6M params = **0.014 bpw** at bf16 **[checked]**. At K4 it would cost 0.0035 bpw, which frees ~0.0105 bpw ≈ −1.6% KL at 2.5 by the board slope. The visual tower plus MTP is 885M params = **0.53 bpw** if the board counted it **[checked]**.

**7. Ruled out: GDN state precision.** The GDN recurrent state is f32 by default in radiance, and no eval script passes `--state-cache-dtype` **[checked]**. STEPQuant/LeapQuant-style state drift is not a measurement confound for us.

---

## A. Decision statistics and slice hygiene

**A1. Verify-check H_out at K4 before the ladder.** The K4 PASS (0.961/0.967) rests only on `golden-search27`, the same 103 docs behind the withdrawn swj K4 claim. Rescore base4, bn4, base4s1 and bn4s1, which already exist, on `golden-verify27`. All four share V2 3.00's head, so pairs within the set stay matched.
- *Source:* new; applies SWJ27 amendment 2's lesson.
- *Why:* repeated adoption on one slice is a forking-paths risk, and 0.8B H_out kept only 52% of its gain on xcorp (#93). A failed verify would save ~60 card-h of H_out levels.
- *Fit:* gate before the ladder's K≥4 levels.
- *Test:* 4 evals ≈ 12 min on card 0. Count it if geo ≤ 0.985 and both pairs < 1 (keeps ≥ ⅔ of the search gain).
- *Gain / risk:* protects a ~1–2% claim at 3.5–4.5 bpw and ~60 card-h. No risk.
- *Conflicts:* #33-28. It adds the confirmation that #104 had and #33-28 lacks.

**A2. Pool the seed floor before quoting σ.** K4's σ = 0.0062 came from one pair. The bn arm's own seed pair (bn4→bn4s1, 1.0124) was left out. Pooled: σ ≈ 0.0098 (2σ/√2 = 0.014 vs |log geo| 0.036). K4 still passes, with half the margin quoted.
- *Source:* new.
- *Why:* σ from one pair has a 95% interval spanning more than 10×. Future rules (#105, #112-02) will inherit this habit.
- *Fit:* methodology.
- *Test:* CPU, minutes. Recompute every rule's noise term from all seed pairs at that level, both arms.
- *Gain / risk:* avoids false PASS/FAIL calls. None.
- *Conflicts:* H27 amendment 5. This is a refinement, not a re-litigation.

**A3. Fit one cross-K model to the five H_out pairs instead of per-level thresholds.** All five pairs are below 1: 0.972, 0.992, 0.976 at K2.5 and 0.961, 0.967 at K4. A log-linear fit gives ≈0.980 at K2.5, ≈0.975 at K3 and ≈0.969 at K3.5. The "K2.5 decides ≤K3.5, K4 decides ≥K4" split hides that K3–K3.875 is untested and sits between two near-pass results.
- *Source:* new.
- *Why:* the 3.0–3.5 bpw sizes live mostly at K2.75–K4. H_out there is plausibly −2.5…−3%.
- *Fit:* per level.
- *Test:* one bn3.5 vs base3.5 pair (K3.5 base ≈ 37 min; bn ≈ 3.2 h; 2 evals), or use ladder2's surviving K3 base. Read it against the K4 rule.
- *Gain / risk:* −1…−2% KL at 3.0–3.5 bpw for ~21 card-h of extra quant time. Risk: noise. One pair can't pass a 2σ/√n test alone, so it has to be pooled.
- *Conflicts:* #33-28. User decision stands. This only tests the band nobody measured.

**A4. Track multiplicity on golden-search27.** I count at least 12 27B arms judged on the same 103 docs: bn, dy, xbn, s-pairs, K4 pairs, swj pairs, cd pairs, then #105 and #112-02. Doc-level idiosyncrasies are shared across all of them. Keep a decision ledger (arm, slice, ratio, z, decision) and require a verify read for every adoption.
- *Source:* 2610.20826 (a 12k-token slice flipped rankings, absolute KL +24%).
- *Why:* at ~2σ rules, ~1 in 20 null methods passes by chance.
- *Fit:* methodology.
- *Test:* none (bookkeeping).
- *Gain / risk:* avoids a false adoption.
- *Conflicts:* —

**A5. Cut a fresh golden-search27b for the remaining 27B decisions.** Draw 103 new search-pool docs, disjoint from golden-search27 and ver3/ver4, and use them for #105, #112-02 and any K3.5 pair. Their decisions then don't share slice noise with H_out and swj.
- *Source:* new.
- *Why:* decorrelates decisions.
- *Fit:* methodology.
- *Test:* calib_slice.py on CPU, 0 card time to cut. Scoring cost is unchanged.
- *Gain / risk:* none / none.
- *Conflicts:* —

**A6. Score fixed-code changes on several slices by default.** swj and C9 carry no rounding noise within a pair. Their only questions are transfer across docs and across positions, and each extra slice costs ~3–5 min. Score search, verify27, agent-search32 and an xcorp-8k golden for every fixed-code arm.
- *Source:* new.
- *Why:* turns "z −12.5 on 103 docs" into a robustness profile for almost nothing.
- *Fit:* methodology.
- *Test:* +3 evals per arm.
- *Gain / risk:* none / none.
- *Conflicts:* #104, #112.

**A7. Re-score one container twice.** Score base on golden-search27 twice, ideally on both cards. The 27B seed floor (2.5% per pair at K2.5) is far larger than per-tensor proxy seed spread suggests (0.5%, #41). An eval non-determinism term would change how every 27B rule should read.
- *Source:* new.
- *Why:* tells apart "KL is seed-sensitive" from "eval is noisy".
- *Fit:* diagnostic.
- *Test:* 1–2 evals. Count it if rows are bit-identical; otherwise measure ε.
- *Gain / risk:* none / none.
- *Conflicts:* #74 showed parity on V2, not on exl3i.

**A8. Break down every 27B pair by category and by document.** Use the stored per-position rows on gpubox's CPU. Report ratios per calib category (agentic, chat, reasoning, code, multilingual, web) and the share of the gain carried by the top 5 docs.
- *Source:* #93 (category transfer 52–91%).
- *Why:* a gain carried by 3 docs or by one category is fragile against a fresh private draw.
- *Fit:* diagnostic.
- *Test:* CPU only. If any category's ratio is > 1.01 with a CI excluding 1, flag the method for that category.
- *Gain / risk:* none / none.
- *Conflicts:* —

**A9. Use a value-of-information rule instead of hard thresholds.** Decisions like "0.9801 vs ≤ 0.98" are threshold artifacts. Rank pending tests by expected KL·usage gain minus card-hours, and spend extra pairs only where the decision is close and the stakes are large. That is H_out ≤K3.5 now, and #105 if it lands near 0.97.
- *Source:* new.
- *Why:* better use of card time.
- *Fit:* process.
- *Test:* none.
- *Gain / risk:* indirect / none.
- *Conflicts:* H27 amendment 5. Not a re-litigation, a rule for next time.

**A10. Watch whether the own-trace selection share drifts the stream.** Stream accept/reject partly uses private agentic text. #91 showed own-agentic and calib objectives trade off (−5…−15% own vs +1…+6% calib for H_in). Log the rank correlation of candidate KL between the own-trace and calib parts of each read. If it drops below ~0.8, cut the own-trace share.
- *Source:* #91, #101.
- *Why:* the private corpus is drawn like calib, not like our sessions.
- *Fit:* stream.
- *Test:* CPU over stream logs.
- *Gain / risk:* avoids selecting the wrong direction.
- *Conflicts:* —

**A11. Make the non-trunk equality check a hard gate in the swap tooling.** The swj head-mismatch bug produced a −22% artifact. Every pair should assert byte-identical non-trunk tensors and the same plugin build hash before scoring.
- *Source:* LOG 11-06 7:50 PM (listed as a TODO).
- *Why:* that one bug class was larger than any real 27B effect measured so far.
- *Fit:* tooling.
- *Test:* none.
- *Gain / risk:* none / none.
- *Conflicts:* —

**A12. Recompute the ledger after golden-verify27.** Confirm that verify27's 71 docs are disjoint from ver3/ver4 and from the reserved long traces. Make sure each further verify27 read is logged with the decision it informed.
- *Source:* eval-data.md, ledger.
- *Why:* verify27 already informed swj K2.5/K4 and will inform H_out K4.
- *Fit:* bookkeeping.
- *Test:* CPU.
- *Gain / risk:* none / none.
- *Conflicts:* —

## B. Coverage of the official score

**B1. Re-score every adopted 27B pair on agent-search32.** This is the most important check in the report. ~24% of scored rows (char proxy) sit beyond 8k and ~16% beyond 32k, and none of base/bn ×5 pairs, swj ×4 or cd ×2 has been seen there. Once w67 fixes the broken agent-search32 golden (the ppl 57,304 input), score all existing containers on it.
- *Source:* new. The shares come from today's calib.jsonl count.
- *Why:* H_out's Σdy² came from 2048-token windows and swj was fit on ≤4096-token windows. Their effect beyond 8k is unmeasured.
- *Fit:* gate for adoption.
- *Test:* ~12 evals × ~5 min ≈ 1 card-h. Count it if each adopted method's ratio on agent-search32 is ≤ its search ratio + 0.01.
- *Gain / risk:* protects claims on roughly 25–40% of the score's KL mass. No risk.
- *Conflicts:* #52 and #73 (2 docs, V2 mixes only).

**B2. Report KL by position bin for each V3 change.** From the B1 rows, split into 0–2k / 2–8k / 8–16k / 16–32k. If a method's gain fades with position, the 2k-capped search slices will over-reward it.
- *Source:* #52.
- *Why:* the capstudy agreement (0.993–0.997) was measured on V2 mixes only.
- *Fit:* diagnostic.
- *Test:* CPU on B1's rows.
- *Gain / risk:* none / none.
- *Conflicts:* #52.

**B3. Get the exact gen-row share by position from the calib golden's gen.u8.** My 24%/16% numbers assume text docs are fully scored and chat docs only on assistant spans. Run the exact count over `gen.u8` and doc lengths on gpubox's CPU, then weight stream objectives (2k slices vs 32k chunks) to match.
- *Source:* today's char-proxy count.
- *Why:* sets how much search budget should go to long chunks.
- *Fit:* stream design.
- *Test:* CPU, minutes.
- *Gain / risk:* better-aimed selection.
- *Conflicts:* eval-data.md's slice mix.

**B4. Check the fast variant (A8) on long agentic chunks.** E1b/E1c priced A8 on a K6 base on golden-calA (≤8k). The fast variant will be released, and ~1/4 of the score is long context through GDN in_proj and attention at long positions.
- *Source:* #26, #27.
- *Why:* activation outliers at long positions could raise A8 cost beyond the knapsack's price.
- *Fit:* per-variant pricing.
- *Test:* E1b fake-quant hook, A8-g128 vs f16 on agent-search32 with the V2 3.00 / 4.00 containers. 4 evals.
- *Gain / risk:* protects the fast variant's predicted KL (±5%).
- *Conflicts:* BACKLOG "Activation cost on a low-K base" (still open).

**B5. Use xcorp-8k as the fresh-draw stand-in.** The private corpus is new docs from calib's sources. xcorp is new docs from those same sources; verify docs are old ones already inside the published H_in. Score adopted 27B pairs on the existing xcorp-8k golden and quote gains as a range (verify vs xcorp).
- *Source:* #93 (0.8B: H_out kept 52%/71%, swj 63%/70%).
- *Why:* honest projections. A big xcorp-vs-calib gap means calib-fitted statistics partly fit calib's own documents.
- *Fit:* reporting / gate.
- *Test:* ~8 evals ≈ 25 min.
- *Gain / risk:* none / none.
- *Conflicts:* #101 (xcorp *blend* dropped). This is scoring, not fitting.

**B6. Check multilingual and code categories at 27B.** The 0.8B #101 categories failed for code (1.03/1.04) and multilingual. Multilingual is ~7% of scored characters. Read the per-category ratios of H_out K4 and swj from A8's breakdown.
- *Source:* #101, #32.
- *Why:* rare-token / multilingual rows are known to be fragile (#32).
- *Fit:* gate.
- *Test:* CPU (A8). Flag if ratio > 1.0 with CI excluding 1.
- *Gain / risk:* none / none.
- *Conflicts:* —

**B7. Confirm the board's eval configuration matches ours.** Our measurements use bf16 KV, 2048-token prefill chunks and radiance 1.16/1.17. The board's private run may use fp8 KV or other chunking. Ratios are probably robust, but absolute anchoring ("3–13% short of the freeze expectation") is not.
- *Source:* LOG pitfalls (fp8 KV moves KLD).
- *Why:* projections are board-anchored.
- *Fit:* reporting.
- *Test:* ask the challenge author. 0 compute.
- *Gain / risk:* none / none.
- *Conflicts:* #53.

**B8. Recorded negative: GDN state precision is not an issue.** f32 state is the default and no eval sets f16 **[checked]**. Keep it that way in the V3 plugin and assert it in the selftest census. STEPQuant (2609.38169) on Qwen3.8-27B shows INT8 state costs ~9 points of long-generation accuracy.
- *Source:* 2609.38169, 2609.38166, 2610.39931.
- *Why:* a later "faster fp16 state" deployment flag would silently move long-context KL.
- *Fit:* selftest.
- *Test:* add a census line.
- *Gain / risk:* none / none.
- *Conflicts:* —

## C. Pipeline and freeze blockers

**C1. Fix the tile CD × H_out refusal.** Options:
- (a) Gate per level: CD at K<4, H_out at K≥4. Needs per-level `tile_cd.k_max` plus `hout.apply_k_min`.
- (b) Implement CD under the two-sided objective, i.e. D-weighted with row weights from H_out diag.
- (c) Drop CD. Its 27B whole-model effect was 0.9996, inside noise, at +5 min per level.

My recommendation is (a), and don't build (b).
- *Source:* `rxq/config.py:36` **[checked]**.
- *Why:* the ladder can't run with both settings on.
- *Fit:* config / code, ~1–2 h.
- *Test:* rxq dry run on 2 layers at K3.875 and K4.
- *Gain / risk:* removes a freeze blocker. CD's KL at K≥4 is ≤ 0.1%.
- *Conflicts:* #77.

**C2. Add `hout.apply_k_min` and test it.** Otherwise `enable=true` applies H_out from K0.125 upward, which costs 5× time on sub-1 / i4 levels where it was never tested.
- *Source:* v3.toml **[checked]**, LOG flag.
- *Why:* ladder waste.
- *Fit:* config.
- *Test:* rxq `--dry` census over all 71 levels shows hout on exactly K≥4.
- *Gain / risk:* saves up to ~100 card-h if the bug went unnoticed.
- *Conflicts:* —

**C3. Smoke-test H_out with mul1 and with fractional K above 4.** The 27B H_out runs covered i8 at K2.5 (fractional) and K4 (integer). Untested: two-sided LDLQ with `mul1` at K6.5–8, with the qtf2 KA 5–7 HIOPT instances (K5.25–K7), and with the 0x8888/0xEEEE masks at K4.125–4.875.
- *Source:* new.
- *Why:* code paths that have never run will run for ~3 h per level.
- *Fit:* gate.
- *Test:* 3 tensors (attn_qg, gate, down) at K4.25, K5.5, K6.75 (i8) and K7.5 (mul1), one layer. Check: no fallback, proxy within band, finite, bit-stable rerun. ~30 card-min.
- *Gain / risk:* avoids wasted level runs.
- *Conflicts:* —

**C4. Cap H_out's upper K until a high-K pair exists.** "≥K4" was set from one level and covers ~24 levels up to mul1 K8. At K≥6 KL is near the non-weight floor (V2 8.0 = 0.00034), and the 0.8B only showed gains through K5 (#82). Run H_out at K4–5.5 now. Extend to K≥5.75 only after a K6 pair, or if donated compute makes the time free.
- *Source:* new.
- *Why:* ~14 levels × 2.6 h ≈ 36 card-h for an unmeasured gain on a tiny KL.
- *Fit:* config.
- *Test:* one K6 pair later (~7 h) if anyone wants it.
- *Gain / risk:* saves ~36 card-h. Risk: forgoes a ≤2% gain at 6.0–7.0 sizes.
- *Conflicts:* #33-28 scope.

**C5. Re-price the ladder's wall clock with H_out.** 24 levels × 3.2 h ≈ 77 card-h, plus the rest at 37 min, plus 1,600 s of CPU packing per level. Check that two-sided LDLQ on `ffn_down` (17408 inputs) fits small donor cards; the 4.1 GB limit was measured without H_out.
- *Source:* PROFILE.md.
- *Why:* donor routing and the schedule.
- *Fit:* planning.
- *Test:* measure peak VRAM of one bn gate/up/down job on a 16 GB card.
- *Gain / risk:* none / none.
- *Conflicts:* —

**C6. Check for swj grid-edge hits.** The swj c_g grid is 0.6..1.4 and the clamp is [0.5, 2] **[checked]**. 0.8B logs show c_g 1.07 ± 0.10. At 27B K2 / K1.x the per-channel optimum may exceed 1.4 (C9's α also wanted the edge). Count channels at 1.4 in the 27B fit logs (swiglu.jsonl) per K.
- *Source:* SWJ27.md README.
- *Why:* clipping would silently cap swj's gain at low K, where it matters most.
- *Fit:* per level.
- *Test:* CPU, minutes. If > 1% of channels hit 1.4 at any K, widen to 2.0 and refit (fixed codes, ~1 card-h per level).
- *Gain / risk:* 0…−1% at 2.0–2.25. None.
- *Conflicts:* #104.

**C7. swj at K≥5 has no data.** `k_max = 8.0` was set from K4 head-matched (−1.1%). Fixed-code swj at K6 costs one fit plus 2 evals and would show no harm or a gain.
- *Source:* reprice.md (extrapolated above K4).
- *Why:* small but unverified.
- *Fit:* per level.
- *Test:* ~1.5 card-h.
- *Gain / risk:* ~0 / tiny.
- *Conflicts:* #104.

**C8. Reconcile `sigma_reg_in_sub1`.** #38 says 0.05 below K1, ourq uses 0.005, and `apply_sub1_damping=false`. Only research levels are affected, but decide before the ladder because sub-1 levels need a requant otherwise.
- *Source:* v3.toml **[checked]**.
- *Why:* freeze hygiene.
- *Fit:* config.
- *Test:* settle64 CPU at K0.5 / 0.75, ~1 h.
- *Gain / risk:* ~0 at release sizes.
- *Conflicts:* #38.

**C9. Pin the H_out diag inputs with checksums and write-time checks.** The xcorp H_in for L43 o_proj had 1e18–1e29 garbage. Add finite, non-negative and heavy-tail-bounded checks per tensor for `hout_diag`, plus a sha256 manifest pinned in v3.toml.
- *Source:* LOG 11-06 02:05.
- *Why:* one corrupt diag spoils a 3.2 h level silently.
- *Fit:* pipeline.
- *Test:* CPU scan of the 400 files.
- *Gain / risk:* none / none.
- *Conflicts:* —

**C10. Settle the #106 converter question before the ladder, or drop it.** o_proj head-permutation act order needs a requant of o_proj at all levels and a "heaviest heads last" converter. It was flat on the real 0.8B quantizer (addendum).
- *Source:* addendum.
- *Why:* avoids a late requant of 16 tensors × 71 levels.
- *Fit:* format.
- *Test:* none, just decide.
- *Gain / risk:* none / none.
- *Conflicts:* #106.

## D. Knapsack pricing once V3's stack is on

**D1. Fix the boundary distortion in KN_LEVELS.** `level_pos` interpolates fractional levels using log ratios of the ladder's **one-sided** `proxy_err` between integer neighbours. It also requires `m[b] < m[K] < m[a]` and falls back to the midpoint otherwise **[checked]**. With H_out at K4 the K4 anchor's one-sided error rises, so K3.x positions shift toward K4 (priced too cheap) or fall back. Inside K4–K5 the shift is roughly uniform, so less harm there.
- *Source:* knap5a.py.
- *Why:* allocation errors around the most-used band at 3.5–4.5 bpw.
- *Fit:* knapsack.
- *Test:* CPU minutes. Count fallbacks and position shifts with ladder proxies once K3.875/K4 exist. Fix by logging a two-sided proxy tr(E H_in Eᵀ diag(H_out)) for *all* levels, plain included, and positioning on that.
- *Gain / risk:* 0.5–2% at 3.5–4.5 bpw. None.
- *Conflicts:* #45.

**D2. Integer levels are still priced by V2's hproxy.** kfit2 uses V2's hproxy (V2 quants, mul1 codebook, no H_out, no swj) × DAMPF, with V3 levels positioned between them. Measure 6–8 group swaps (e.g. one family × one depth band at K2.5 and K4, V3 level vs V2 level) to recalibrate per-family factors for V3 levels.
- *Source:* D3 in BACKLOG; #13 (family factors 1.15–1.23 differ by quantizer).
- *Why:* V3 levels are a different quantizer family.
- *Fit:* knapsack prior.
- *Test:* ~8 evals ≈ 25 min once levels exist.
- *Gain / risk:* 1–3% from better allocation.
- *Conflicts:* #45, #51.

**D3. Put H_out into the knapsack per family, not as a uniform factor.** H_out's 0.8B gain split was gdn_in 39%, mlp_in 44%, attn_in 13%, writers 3% (#82). A uniform ×0.964 on every K≥4 tensor misprices writers.
- *Source:* #82.
- *Why:* allocation.
- *Fit:* knapsack.
- *Test:* CPU re-solve. Check pick sensitivity as with swj `gu` (which moved ≤0.5%). If < 0.5%, drop the item.
- *Gain / risk:* ≤0.5% likely. None.
- *Conflicts:* reprice.md `gu`.

**D4. Measure composition error on V3 mixes.** Score 8 random knapsack-feasible mixes around 2.5 and 4.0 built from V3 levels. Fit measured = α + β·predicted. Papers find β compressive, 0.26–0.77 (2610.09227), and composition error 26–42× the pricing error (2609.09867).
- *Source:* 2607.12266, 2609.09867, 2610.09227.
- *Why:* kfit2's compounding κ was fitted on V2. V3's coherent fixes (swj) change compounding if the "coherent errors compound" hypothesis holds.
- *Fit:* knapsack calibration.
- *Test:* ~8 evals plus CPU.
- *Gain / risk:* 1–3%. None.
- *Conflicts:* #68, #11.

**D5. Re-measure sensitivities in the context of a near-final mix.** GEMQ found sensitivities measured at the compressed model beat ones measured at FP (Mixtral 1.5 bpe 9.39→9.01). MixQuant (context-aware scores) gained most at low budgets. Re-measure group sensitivities on the 2.5 knapsack mix rather than on uniform bases.
- *Source:* 2605.23078, 2607.23047. MixFrag 2607.28589 found the opposite on ViTs, so this is unsettled.
- *Why:* low-bpw misallocation.
- *Fit:* knapsack.
- *Test:* ~10 evals at 2.5.
- *Gain / risk:* 0–2%. Small risk of overfitting to one mix.
- *Conflicts:* —

**D6. Check that the knapsack reproduces the head/emb swap.** Ask knap5 to predict base4v4 / base4 (V2 4.00's head+emb vs 3.00's on a K4 trunk). The measured ratio is 0.761 (verify), 0.798 (search). If the model predicts e.g. 0.88, head/emb are underpriced at 3.5–4.5 bpw.
- *Source:* SWJ27 amendment 2 controls.
- *Why:* +0.095 bpw of head/emb bought −24%, about 2× the trunk's marginal rate.
- *Fit:* knapsack.
- *Test:* CPU minutes.
- *Gain / risk:* if mispriced, re-allocating 0.05–0.1 bpw to head/emb could be worth −3…−8% at 3.5–4.5 (speculative).
- *Conflicts:* #34, #35.

**D7. Measure the non-trunk share at K4.** Swap base4v4's head to K8 (or K7), then its embedding to e8 / t8, and score each. This separates head from embedding and gives their share of KL at ~4 bpw.
- *Source:* new; #99's floor split is the high-bpw version.
- *Why:* decides whether head/emb work outranks trunk work. The bf16 head is unsafe in the old plugin path (#66), so use K8.
- *Fit:* diagnostic.
- *Test:* 3 evals ≈ 10 min.
- *Gain / risk:* changes priorities. None.
- *Conflicts:* #24 (V2 high bpw only).

**D8. Run the slab head whole-model check before the freeze.** #34's slab head (−36…−40% head-only KL in headsim at K5–6) never had its whole-model check. If head/emb carry ~10–20% of KL at 3.5–4.5 (D7), that check outranks several trunk items still in the queue.
- *Source:* #34, HEADWM.md.
- *Why:* the largest unused lever, per the head-swap evidence.
- *Fit:* format (head GEMM + gather).
- *Test:* the existing headwm.sh plan, ~1 card-h.
- *Gain / risk:* −2…−5% at 4–5 bpw (head share × 0.6). Decode cost of the gather.
- *Conflicts:* #34, #32 (vocab-tiered dropped; this is row-slab by Fisher).

**D9. Offer GDN a/b projections as knapsack options.** `in_proj_a` / `in_proj_b` (23.6M params, 0.014 bpw) are bf16 and outside the knapsack. On Qwen3.8-27B NVFP4 they had the smallest effect of any GDN projection (2.1–2.6% GDN-layer output error vs ~10% for qkv/z).
- *Source:* 2609.04098 (same model). Different here: trellis at K4–K6, priced by measured swap, not a global setting.
- *Why:* freeing 0.0105 bpw at 2.5 ≈ −1.6% KL by the board slope, if the quantization cost is below that.
- *Fit:* per-level option (tiny quant jobs, N = 48 rows).
- *Test:* quantize a/b at K4 and K6 for all 48 layers (minutes), swap into the K2.5 base mix, 2 evals. Adopt as an option if the K4 swap costs < 0.7% KL.
- *Gain / risk:* −0.8…−1.6% at 2.0–2.5. Risk: gates sit in log space (exp(−A·softplus)), so check tails (KL99).
- *Conflicts:* shortlist "GDN a/b int8" (int8 saves only half as much). Design §1 "GDN a/b bf16, closed" was about speed, not bytes.

**D10. A8 activation prices at low K are still open.** The fast variant at 2.0–3.0 uses E1c costs calibrated on a K6 base. Run one E1b batch on a K2.5 base, gated at M ≥ 512 vs ungated.
- *Source:* BACKLOG.
- *Why:* the fast variant's predicted KL could be off ±30%.
- *Fit:* pricing.
- *Test:* ~1 card-h.
- *Gain / risk:* protects fast-variant claims.
- *Conflicts:* #26, #27.

**D11. Hold out the knapsack's own predictions.** Before the stream, freeze knap5's predictions for 3 sizes and score them once on ver3. This gives the pre-search starting gap without selection bias.
- *Source:* #72 (fit-slice picks were ~3% optimistic).
- *Why:* honest baseline.
- *Fit:* process.
- *Test:* 3 evals.
- *Gain / risk:* none / none.
- *Conflicts:* —

## E. Library vs mix compatibility

**E1. Recorded OK: swj pairing is library-safe.** knap5 maps gate_proj and up_proj to one `ffn_gate_up` item **[checked]**, so gate and up always share K and the per-level swj fit applies exactly. Re-check this if the fused-GEMM K-tie question (§1d) is ever answered "untie".
- *Source:* knap5a.py.
- *Fit:* knapsack invariant.
- *Test:* add an assert.
- *Gain / risk:* none / none.
- *Conflicts:* —

**E2. Apply the per-mix asym writer pass to the top-3 finalists, not just the winner.** If #105 passes, the stream runs on plain writers and the per-mix writer pass (0.81–0.90 on 0.8B, #111) applies at the end. The pass's gain varies by mix (0.81 vs 0.90), so it can reorder finalists.
- *Source:* #111.
- *Why:* selection under a final transform.
- *Fit:* final pass.
- *Test:* 3 × ~4–5 card-h per size.
- *Gain / risk:* 0–2% / cost.
- *Conflicts:* #105, #111.

**E3. Schedule the per-mix pass budget now.** ~18 shipped mixes (9 sizes × accurate/fast) × ~4.5 card-h ≈ 80 card-h, competing with stream evals. Decide which sizes get it, perhaps 2.0–3.5 where 0.8B gains are largest.
- *Source:* #111.
- *Why:* planning.
- *Fit:* plan.
- *Test:* none.
- *Gain / risk:* none / none.
- *Conflicts:* —

**E4. C9 per level is library-safe; C9 next to asym writers is not.** C9's e_n uses the writer's own codes and calib H_in, so it is fine per level. But "C9 after asym" fits α on the asym target (W·F), which is per mix. Ship C9 on plain writers in the library, and re-fit α in the per-mix pass if asym ships.
- *Source:* w64 item 3, #112.
- *Why:* avoids applying a library α to a per-mix writer.
- *Fit:* per level / per mix.
- *Test:* none.
- *Gain / risk:* none / none.
- *Conflicts:* #112.

**E5. Spell out the healing order.** If healing ever ships, it changes sibling scales after asym and partly undoes asym (#87-07). Healing also overlaps swj and C9 (all scale refits), so their gains must not be added in projections. Write the order (siblings → swj → asym → C9 → healing of writer scales only) into v3.toml.
- *Source:* #87-07, ThinkBit 2610.34886 (scale recalibration does most of the healing).
- *Why:* avoids double-counting.
- *Fit:* process.
- *Test:* none.
- *Gain / risk:* none / none.
- *Conflicts:* #89.

**E6. Check H_out K4 tensors next to plain K<4 siblings.** H_out was only measured on uniform-K trunks. It is per-tensor with bf16 statistics, so it should be additive, but untested. Swap the K4 bn levels of 16 layers into the knapsack's 3.5 mix and compare against the same mix with plain K4.
- *Source:* new.
- *Why:* mixed contexts are the shipped case.
- *Fit:* per level.
- *Test:* 2 evals once levels exist.
- *Gain / risk:* low risk. Confirms D3's pricing.
- *Conflicts:* #33-28.

**E7. Asym and #106's permutation share column indexing.** If #106 ever ships, the o_proj columns are permuted. Asym moments (sibling-sequential), CD and C9 all index columns, so run an end-to-end bit test (decode → original order) on one layer.
- *Source:* w64.
- *Why:* silent mis-indexing.
- *Fit:* format.
- *Test:* CPU.
- *Gain / risk:* none / none.
- *Conflicts:* #106.

## F. Proxy-to-27B transfer and old-proxy results

**F1. Don't use "coherent fixes grow with size" to rank work.** The hypothesis rests on one point, swj at K2.5 (−6.5% at 27B vs −3% / −9% on the 0.8B at K3 / K2). swj at K4 shrank to −1.1%, and #108 (GDN joint refits) failed. Rank C9-port, head-refit and residual items with a ⅓-of-0.8B prior like everything else until a 27B pair says otherwise.
- *Source:* brief §1.
- *Why:* avoids over-investing.
- *Fit:* planning.
- *Test:* none.
- *Gain / risk:* none / none.
- *Conflicts:* —

**F2. Sweep C9's α at 27B before adopting a value.** #112's gain came from gdn_out at α = 3, the grid edge, and #112-01 explores up to 8. c_n = (1−e_n)^(−α/2) with α = 3 overshoots row-norm restoration threefold. That points to compensating something global (residual attenuation through quantized GDN layers) rather than a per-row LS shrink. At 27B, run fixed-code arms α ∈ {1.5, 3, 4.5} on gdn_out per seed.
- *Source:* w64, w75.
- *Why:* the 0.8B's tied head and 24 layers make α transfer doubtful.
- *Fit:* per level (svh).
- *Test:* ~6 evals ≈ 20 min. Pick α by the search pair, check on verify27 (A6).
- *Gain / risk:* −0.5…−2% at 2.5 if real. Overshoot risk.
- *Conflicts:* #85 (C9 α at edge), #112.

**F3. Diagnose the C9 mechanism on the 0.8B.** Per layer, measure ‖h_q‖/‖h_bf16‖ and cos(h_q, h_bf16) of the residual stream for base vs base+C9 (gdn_out α 3). If the effect is a depth-wise residual norm deficit, a per-depth α (or one scalar per writer layer, still free) beats one α per family and transfers better.
- *Source:* 2609.11716 (counteraction / angular error dominates).
- *Why:* tells whether α is a hack or a mechanism.
- *Fit:* per level.
- *Test:* 0.8B, 4070, <1 h.
- *Gain / risk:* understanding. None.
- *Conflicts:* #112.

**F4. i8 vs mul1 in KL at K4–6.5 on the 0.8B.** i8 was judged "free to K6" by proxy (×1.000–1.004) and a Gaussian smoke test. The hash codebook showed proxy error underpriced KL 3–4×. If i8 at K5–6.5 costs 3–5% KL rather than 0.2–1.8%, the accurate variant at 5.0–7.0 should use mul1 there.
- *Source:* #43, #94, #100c.
- *Why:* codebook-structured errors can be coherent.
- *Fit:* per-variant level choice.
- *Test:* 0.8B K4/5/6 × i8/mul1 × 2 seeds, ~1 h on the 4070. Flag if i8/mul1 KL > 1.02 at any K ≤ 6.
- *Gain / risk:* 0–4% at 5.0–7.0. None.
- *Conflicts:* #29-01 (proxy only), #94.

**F5. Re-run old-order settle.py conclusions on settle64.** #38 damping, #39 gain (flat), and #90-02 / #100 re-validation all ran on first-to-last LDLQ order, which moved attn_qg error +14…+26%. Re-run them on colo/Thor in exllamav3's order.
- *Source:* w64 fix.
- *Why:* settings pinned on a biased proxy.
- *Fit:* settings.
- *Test:* CPU ~10–20 h across both boxes. Flag any family with a ≥1% alternative.
- *Gain / risk:* probably ~0.5%; protects settings.
- *Conflicts:* #38, #39, #90-02.

**F6. Retire the 3-layer hconv proxy for effect sizes.** It said −6% for H_out; the whole model said −2.8%. It is still fine for ranking estimators.
- *Source:* #33-25/27/29.
- *Why:* avoids optimistic gating.
- *Fit:* methodology.
- *Test:* none.
- *Gain / risk:* none / none.
- *Conflicts:* #33-29.

**F7. Map 0.8B results to 27B by KL regime, not by K.** 0.8B K3 (KL ~0.09) behaves like 27B at ~2.0–2.25 bpw. 0.8B K2 (~0.5) has no 27B counterpart in our sizes. Quote 0.8B effects at the 0.8B K whose KL matches the 27B size of interest. For 2.5–4.5 bpw that means 0.8B K3.5–K5.
- *Source:* #33-26 notes, CALIBRATION.md.
- *Why:* explains part of the shrink.
- *Fit:* methodology.
- *Test:* none.
- *Gain / risk:* none / none.
- *Conflicts:* —

**F8. Screen head/embedding ideas on the 9B, not the 0.8B/2B.** 0.8B, 2B and 4B tie embedding and head. Any head-side idea (slab head, head KL-refit, embedding healing) needs the 9B staged path (BACKLOG).
- *Source:* BACKLOG.
- *Why:* proxy validity.
- *Fit:* planning.
- *Test:* none.
- *Gain / risk:* none / none.
- *Conflicts:* —

**F9. Use enough seeds for small 0.8B effects.** With 0.8B seed SD ~1% (K3) / ~2% (K2), any 0.8B effect under ~2% needs ≥3 seeds and a verify read before a 27B slot. #106-01, #112-01 and #107 are in that regime.
- *Source:* #33-16.
- *Why:* noise.
- *Fit:* methodology.
- *Test:* none.
- *Gain / risk:* none / none.
- *Conflicts:* #107.

## G. Seeds and noise

**G1. Localize the 27B seed variance with layer-split replicates.** Whole-model seed spread is 1–2.5% per pair at K2.5. Per-tensor proxy seed spread is 0.5% with A-vs-B correlation 0.22 (#41). That gap means KL-relevant seed variance sits in error direction relative to output curvature, which the proxy doesn't see, or in a few tensors. Build half-mixes from the existing base/s1/s2 K2.5 levels: layers 0–31 from s1 + 32–63 from base, the reverse, then quarters.
- *Source:* BACKLOG review idea 2.
- *Why:* if 1–3 layer groups carry most of the variance, per-group seed choice is free KL.
- *Fit:* per level.
- *Test:* 6 evals ≈ 20 min. Count it if one quarter carries ≥ 50% of the spread.
- *Gain / risk:* enables G2. None.
- *Conflicts:* #41 (proxy-based null).

**G2. Choose seeds by KL for a few tensor groups.** If G1 localizes the variance, pick the best of 3 seeds per group on the search pool and confirm on verify27. It is per level and library-safe because seeds are a property of the level.
- *Source:* new; #41 failed because it selected by proxy.
- *Why:* the 2.5% whole-model spread at K2.5 is larger than many methods' gains.
- *Fit:* per level (needs 2–3 quants of the chosen groups per level).
- *Test:* K2.5 first, ~2 card-h. Count it if the verify gain ≥ 1% and ≥ ⅔ of the search gain.
- *Gain / risk:* −0…−2% at 2.5. Risk: winner's curse, which the verify step handles.
- *Conflicts:* #41.

**G3. Build a noise model for the stream.** Each library level is one seed draw (±1.3% per whole-model quant at K2.5). The stream's measured differences between mixes that differ only in level choice include this level noise, and no amount of search can average it away. Use 27B σ per K (A2) in the stream's accept rule and validation curve.
- *Source:* new.
- *Why:* avoids chasing level noise.
- *Fit:* stream.
- *Test:* none.
- *Gain / risk:* none / none.
- *Conflicts:* —

**G4. Check whether H_out reduces seed sensitivity.** At K4 the bn seed pair is 1.012 vs base 1.006. That is no sign of reduced variance, but n = 1. Pool across K once K3.5 exists.
- *Source:* H27.
- *Why:* curiosity with a decision link: lower variance would favour H_out on ranking grounds too.
- *Fit:* diagnostic.
- *Test:* none extra.
- *Gain / risk:* none / none.
- *Conflicts:* —

## H. Interactions in the stack

**H1. Test C9 on H_out-quantized writers.** C9's e_n uses one-sided H_in. Two-sided LDLQ moves error into low-H_out rows, so those rows get the largest C9 multipliers, exactly where H_out decided error matters least. Fixed-code test: bn4 + C9 vs bn4.
- *Source:* new.
- *Why:* C9 would apply at K≥4 alongside H_out.
- *Fit:* per level.
- *Test:* 2 evals ≈ 10 min. If ratio > 0.995, use e_n weighted by diag(H_out) or skip C9 at K≥4.
- *Gain / risk:* avoids a possible harm. Small.
- *Conflicts:* #112.

**H2. swj on H_out codes at K4 is measured; record it.** bn4v4→bn4swj = 0.987 on verify, head-matched. The stack holds at K4. Use 0.987 at K4 for H_out levels, not the plain value.
- *Source:* SWJ27 amendment 2.
- *Fit:* pricing.
- *Test:* none.
- *Gain / risk:* none / none.
- *Conflicts:* —

**H3. swj and A8 in the fast variant.** swj rescales gate/up per channel, which changes the per-channel magnitude of down_proj's input (post-SwiGLU, the outlier-prone tensor), and A8 uses g128 per-token scales on that input. Re-check A8 cost on down with swj on.
- *Source:* E1b (down A8 cost small at K6).
- *Why:* the fast variant ships swj levels.
- *Fit:* per variant.
- *Test:* E1b hook, 2 evals at K2.5 with/without swj.
- *Gain / risk:* likely ~0. None.
- *Conflicts:* #26.

**H4. Check C9 and swj under A8 together.** Same as H3 for C9's writer multipliers. Writers are not A8 inputs, but their outputs feed the next layer's RMSNorm, which is scale-invariant. Expect ~0, so one eval is enough.
- *Source:* new.
- *Fit:* per variant.
- *Test:* 1 eval.
- *Gain / risk:* ~0.
- *Conflicts:* —

**H5. Recorded: fractional K × H_out × i8 was covered at K2.5.** That pair was rate 2.5, i8. Remaining gaps are in C3.
- *Fit:* —
- *Test:* —
- *Gain / risk:* —
- *Conflicts:* —

**H6. Turboderp tensors in V3 mixes must stay f16-only.** reprice.md says turboderp's tensors return for 10–18% of params at 5.0–7.0. They are unrounded mul1. Read through the int8 plane, post-hoc rounding cost +17% (K5) to +60% (K6) in E4. Assert in knap5 that A8 is forbidden for them, and in the census that the plugin decodes them as mul1, not mul1i8.
- *Source:* #31, #65.
- *Why:* a silent fallback would cost 17–60% weight error on those tensors.
- *Fit:* knapsack / plugin census.
- *Test:* CPU assert plus one census.
- *Gain / risk:* none / none.
- *Conflicts:* §1d.

**H7. H_out at the fast/accurate split.** None expected, since activation errors are independent of the weight Hessian. Listed so it isn't re-asked.

## I. Ladder waste prevention

**I1. Dry-run the ladder end to end.** Run rxq with the final v3.toml on 2 layers at K1.125 (i8), K2.5, K3.875, K4 (H_out), K5.5 (H_out), K7 (i8 + H_out) and K7.5 (mul1 + H_out). Then pack, import as exl3i, run the census and a selftest decode. This catches C1–C3, H6 and code-stamp mismatches in under 2 card-h, before 71 levels.
- *Source:* new.
- *Why:* every failure found later costs a level or more.
- *Fit:* gate.
- *Test:* ~2 card-h.
- *Gain / risk:* none / none.
- *Conflicts:* —

**I2. Run unblocked levels first.** K2.5–K3.875 at i8 need neither H_out nor C1's decision. Start them now while A1/C1/C3 resolve. The ladder order already starts at 2.5–4.5.
- *Source:* —
- *Fit:* schedule.
- *Test:* none.
- *Gain / risk:* none / none.
- *Conflicts:* —

**I3. Keep pre-refit artifacts for every level.** Store raw codes plus the original suh/svh, and log one-sided proxy, two-sided proxy and per-row e_n. swj, C9 and healing can then be redone without a requant, and D1/E4 have the inputs they need.
- *Source:* new.
- *Why:* future fixed-code changes.
- *Fit:* pipeline.
- *Test:* none.
- *Gain / risk:* none. Storage only.
- *Conflicts:* —

**I4. Make the rqah score bands per setting.** Donated jobs are verified by a Hessian-weighted error band built from neighbouring results and anchors. At K≥4 the H_out result has higher one-sided error than the plain K3.875 neighbour predicts. Either honest results get rejected or the band gets loosened, which weakens cheat detection. Bands must be per (family, K, setting) and use the setting's own objective.
- *Source:* rqah 0.3 design.
- *Why:* donated-ladder failure mode.
- *Fit:* tooling.
- *Test:* CPU replay of verification on our own K3.875/K4 jobs.
- *Gain / risk:* none / none.
- *Conflicts:* —

**I5. Run a consistency check after each level.** Compare each tensor's proxy to its K−⅛ and K+⅛ neighbours (monotone and inside the expected band). This catches hardware faults; two HSA aperture violations hit card 0 in one night.
- *Source:* LOG 11-06.
- *Why:* silent damage.
- *Fit:* pipeline.
- *Test:* CPU.
- *Gain / risk:* none / none.
- *Conflicts:* —

**I6. List which pending decisions force a requant.** Only H_out scope, #106 (o_proj only), #113 (format) and sub-1 damping force one. #105 (per-mix pass), #112 (svh), swj extensions and healing don't. Publish this list so the freeze doesn't wait on non-blocking items.
- *Source:* —
- *Fit:* plan.
- *Test:* none.
- *Gain / risk:* time.
- *Conflicts:* —

**I7. Stamp mixed hardware.** Donor GPUs differ, and results aren't bit-identical across GPU types. Keep per-tensor code stamps and the producing device in the level manifest so a shipped container can be reproduced and audited.
- *Source:* rqah.
- *Fit:* tooling.
- *Test:* none.
- *Gain / risk:* none / none.
- *Conflicts:* —

## J. Release and board risks

**J1. Confirm the bundled-tensor rule before shipping.** Visual tower plus MTP is 885M params = **0.53 bpw** at bf16 **[checked]**, and DFlash2 adds more. B10 says board bytes are the tensor payload. IQR-V3's listing suggests the tower isn't counted, but that is inference. Get the rule in writing from the challenge author, or ship two containers.
- *Source:* #53, §8.
- *Why:* a 0.5 bpw shift would be catastrophic.
- *Fit:* release.
- *Test:* 0 compute.
- *Gain / risk:* none / avoids disaster.
- *Conflicts:* —

**J2. Anchor projections to verify, xcorp and agent32 together.** Quote V3 projections as ranges (B1, B5), not single board-anchored numbers.
- *Fit:* reporting.
- *Test:* none.

**J3. Check engine-version parity once.** 27B swap tests used radiance 1.16.0, and the ladder and release use 1.17. Re-score two pair containers on 1.17 before quoting cross-version ratios.
- *Source:* LOG 11-02 rebase note.
- *Fit:* reporting.
- *Test:* 2 evals.

**J4. Run a long-trace sanity check on finalists.** Use `agent-verify-whole`, both traces together, once on finished mixes, as planned.
- *Fit:* release.
- *Test:* TP2 at high bpw.

**J5. Read the public corpus once, at the end.** If the public numbers differ from calib verify by more than ~10% for V3 but not for V2, that is the private-corpus shift signal. Report it; don't act on it.
- *Fit:* reporting.
- *Test:* end only.

---

## INDEX experiments resting on the thinnest evidence

| # | conclusion as used | why thin | cheapest re-check |
|---|---|---|---|
| #33-28 K4 | H_out PASS at K≥4, applied to K4–K8 | 2 pairs, search slice only, one level; σ from 1 pair; never at K≥5, never with mul1 | A1 (verify, 12 min), C3, C4 |
| #33-28 K2.5 | not adopted ≤K3.5 | geo 0.9801 vs 0.98; K3–3.875 never measured | A3 (one K3.5 pair) |
| #104 at K≥4 | swj "every level", k_max 8 | K4 −1.1% (2 head-matched pairs); nothing above K4; grid 0.6–1.4 edge unchecked | C6 (CPU), C7 |
| #77 | CD on attn_qg adopted | whole-model 0.9996 (inside noise); proxy gain only; blocked with H_out | C1 decision |
| #112 | C9 α=3 on gdn_out | 0.8B, grid edge, mechanism unclear, tied model | F2, F3, H1 |
| #106 | o_proj act order "passes" | biased proxy at −7.3%, −3% order-fixed, flat on the real 0.8B quantizer | drop (C10) |
| #43 / #29-01 | i8 "free" to K6 | proxy plus Gaussian smoke only; hash proxy underpriced KL 3–4× | F4 (0.8B KL, 1 h) |
| #94 / #109 | i8 K7 = ×1.27 | proxy on real tensors; KL never measured | F4 extension |
| #38, #39, #90-02 | damping, flat gain, i4 at K≤1 | first-to-last order proxy | F5 (settle64, CPU) |
| #41 | best-of-N seeds dead | proxy-selected; whole-model KL seed spread 1–2.5% is 20× what the proxy predicts | G1 / G2 |
| #34 | slab head −36…−40% head KL | whole-model check never run; head/emb share likely underestimated | D7, D8 |
| #35 | trellis embedding pricing | KLD swaps on golden-ml (fit pool), scaled by Lloyd points | one 27B swap at 2.5 |
| #52, #73 | KL vs position flat across quants | 2 docs, V2 mixes only | B1, B2 |
| #33-27 | block-normalized estimator | chosen on the 3-layer proxy that overstated the effect 2×; bn vs dy at 27B −1.1%, z −1.4 | none needed unless H_out scope grows |
| #44 | K1/sub-2 factors | 3 K1 swaps | research ladder only |
| #13 / kfit2 family factors | used to price V3 levels | fitted on V2 quantizer families | D2 |
| #46-03 | A4 option | 27B costs still owed; kernel built | as planned |
| #107 | fractional drift null | n=3, noisy (K1.25 0.964 ± 0.028) | none; low stakes |
| #101 | xcorp H_in dropped | 2 pairs, pooled 0.993 | fine; noted for completeness |

---

## Top 20 by expected gain per unit cost

| rank | item | cost | what it buys |
|---|---|---|---|
| 1 | **A1** H_out K4 on golden-verify27 | 4 evals, ~12 min | protects ~60 card-h and the K≥4 claim |
| 2 | **C1+C2** gate CD/H_out per level, add apply_k_min | 1–2 h code | removes a hard ladder blocker |
| 3 | **B1** re-score existing pairs on agent-search32 | ~1 card-h (after golden fix) | first look at ~25–40% of the score's KL mass |
| 4 | **J1** bundled-tensor counting rule | 0 compute | avoids a possible 0.5 bpw error |
| 5 | **D6+D7** head/emb share and knapsack check | 3 evals + CPU | decides whether head/emb outrank trunk work (−3…−8% at 3.5–4.5 if underpriced) |
| 6 | **D9** GDN a/b as knapsack options | ~1 card-h | −0.8…−1.6% at 2.0–2.5 |
| 7 | **C3 / I1** H_out × mul1 / fractional smoke and ladder dry run | ~2 card-h | prevents failed 3 h levels |
| 8 | **D1** two-sided proxy for level positions | CPU hours | 0.5–2% better allocation at 3.5–4.5 |
| 9 | **G1** layer-split seed replicates | 6 evals | may unlock G2 (−0…−2% free per level) |
| 10 | **A7** eval re-run determinism | 1–2 evals | sets how every 27B rule reads |
| 11 | **A8 / B6** per-category and per-doc breakdown of stored rows | CPU | robustness of every adoption |
| 12 | **C4** cap H_out at K≤5.5 until a K6 pair | 0 | saves ~36 card-h |
| 13 | **D4** composition calibration on 8 random V3 mixes | ~8 evals | 1–3% via better κ/β |
| 14 | **F2 + H1** C9 α sweep and C9×H_out at 27B (fixed code) | ~8 evals | right α or no harm; −0.5…−2% |
| 15 | **C6** swj grid-edge check | CPU | 0…−1% at 2.0–2.25 |
| 16 | **B5** score adopted pairs on xcorp-8k | ~8 evals | honest range for private-corpus gains |
| 17 | **A3** one H_out K3.5 pair | ~4 card-h | −1…−2% at 3.0–3.5 if adopted |
| 18 | **F4** i8 vs mul1 KL at K4–6.5 (0.8B) | ~1 h 4070 | 0–4% at 5.0–7.0 accurate |
| 19 | **I4** rqah per-setting bands | CPU | avoids donor-ladder rejections |
| 20 | **D8** slab-head whole-model check | ~1 card-h | −2…−5% at 4–5 bpw if the head share is large |

## The 5 cheapest diagnostics that would most change this ranking

1. **H_out K4 on golden-verify27** (A1, ~12 min). If it fails, items 7, 8, 12 and 14 lose priority, and ~60 card-h go back to the stream.
2. **Existing 27B pairs on agent-search32, split by position bin** (B1+B2, ~1 card-h once the golden is fixed). If any adopted method fades beyond 8k, the 2k-capped search design and every "−x%" projection need rework.
3. **Head and embedding share at K4** (D7: K8 head, then e8/t8 embedding into base4v4; 3 evals) plus knap5's prediction for the 0.761 swap (D6, CPU). A large share moves D8 and head work above every remaining trunk item.
4. **Layer-split seed replicates and a determinism re-run** (G1+A7, ~8 evals). If the 2.5% seed floor is localized and real, KL-selected seeds per tensor group become the cheapest gain we have. If it is eval noise, every 27B rule's σ needs revisiting.
5. **Per-category and per-document breakdown of stored pair rows** (A8/B6, CPU only). If H_out's or swj's gain sits in a few docs or one category, both adoptions need a verify-plus-xcorp confirmation (B5) before anything is quoted.
