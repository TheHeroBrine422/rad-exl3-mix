# Critic report, ideas round 2 (2026-11-09): pre-freeze merge and ranking

**Inputs.** Six lens reports: codes, composable, faraway, lowbpw, measure, redteam. Read against the INDEX, the LOG, the 11-08 mining merge and the freeze review.

**What I checked myself.** Code reads on the VM, one checkpoint read on colo, and paper greps. I ran no new experiments.

| check | result |
|---|---|
| Scale polish ignores H_out (refit_scales, fork `quantize.py:1730–1800`, called at L1988 with `H_orig` only) | **Confirmed.** The per-output-column `svh` step is separable, so a diagonal H_out cannot change it. The per-input-row `suh` step solves `((QQᵀ)∘H) r = rowsum(Q∘HW)` and sums over output columns with no H_out weight. The docstring also confirms the fit uses the *damped* H. Four lenses found this independently (measure A1, redteam R13, lowbpw D1/D2, composable C3). |
| C9 has no K schedule (`rxq/post.c9_rows`, v3.toml `[post.c9]`) | **Confirmed.** It computes `c = (1−e)^(−α/2)` with e clipped at 0.5. At gdn_out α = 3 and e = 0.5, c = 2.83. The config comment says the clip was "never reached at K ≥ 2 on the 0.8B", but C9 is enabled at every level, sub-1 included. |
| Sub-1 drift clamp | **Confirmed.** `LDLQ_DRIFT = {1: 1.08, …}` and every K ≤ 1 uses 1.08 (quantize.py:1299; rxq `engine._drift`). |
| suh underflow tolerance (rqah TESTS.md) | **Confirmed**, with a caveat: the underflow was seen on the CPU **stand-in** (greedy codes). It may not occur with real quants. |
| Size of GDN `in_proj_a/b` (checkpoint headers, colo) | **Each is [48, 5120] bf16; together 23.6M params = 0.0140 bpw.** codes 1.1 said 43M / 0.026 bpw, which is wrong. lowbpw C1 and composable G2 are right. conv1d is 0.0012 bpw. Neither appears in rxq or v3.toml. |
| Is swj mix-dependent? (redteam R19) | **No.** `rxq/swjfit.py` fits on "x = the MLP input of the **bf16** model", so swj is mix-independent apart from the gate = up K tie. **R19 is closed.** |
| Two-sided head LDLQ | **Already captured.** The 11-08 merge, Tier A6 (I001798), estimated two-sided head LDLQ in vocab order at ×0.37–0.40. What composable B1 adds is the ordering rule: Fisher-sorted rows give only ×0.83, so slab rows must not be stored sorted. |
| Paper numbers | Spot-checked and matching: 2609.04098 (a/b 2.1% / 2.6%, out 12.7%, qkv 10.4%); 2610.32127 MPQ (LDLQ 15.01 → +CD 12.76 → 11.63); 2609.33923 (Qwen3.8-27B: block-cosine 6.041, cos×CKA 5.991, oracle Q_real 6.087); 2610.07522 (QuIP# W2 8B 13.76 → 12.70); 2610.09212 CurveTQ (Γ_b 0.038 → 0.00013 after RHT); 2608.12026 SoftWater (0.0619 → 0.00957); 2610.41822 TVQ (Llama-3-70B 4.97 → 4.48, Llama-2-70B 3.70 → 3.72); 2610.34886 ThinkBit (KL 0.641 → 0.282, scales alone). **Not verified (digest only):** STEPQuant ρ ≈ 0.80, the coverage-model numbers in 2607.12266, HQMQ App. R, ShamAN-Q's 17.46 ablation. |

**Method notes for every test below:**
- Differences under about 1.5% on a single 100k-position slice are noise, and requant pairs carry seed noise of about 0.009 per pair (#116). Prefer **fixed-code pairs** (refit / C9 / a-b / splice), where seeds are shared and the C9 pair spread was 0.0019.
- 0.8B gains shrink to about ⅓ at 27B. CPU-proxy gains shrank to about ⅕ in KL for #117.
- Write every pass rule before the data and report the combined SE (seed + document, #116).

Machine abbreviations:
- **card 0** = gpubox R9700 (lead's queue)
- **4070** = 0.8B proxy box
- **colo** = r640 CPU

All gains are KL change at **2.5 bpw** unless stated.

---

## 1. Pre-freeze shortlist (decide before the trunk ladder; each fits in ~1 day of card 0 + CPU)

| # | item | what it decides | test (rule written before data) | cost | gain at 2.5 / P |
|---|---|---|---|---|---|
| **1** | **Scale polish in the right metric** (h-weighted `suh` step, refit in undamped H; keep pre-refit scales in every job) — measure A1, redteam R13, lowbpw D1/D2/H6, composable C3 | Stamp setting for all 55 levels. Storing pre-refit su/sv lets it be redone after the freeze. | card 0, rxq on L10/L39 at K0.5/K2.5/K4, four arms: current / `no_refit` / h-weighted / undamped. Score w81 f2calc. **Pass:** f2 −1% on qkv/z/qg with no family worse. Then one fixed-code 27B reader pair at K2.5. **Pass:** g ≤ 0.997, both parts < 1. **If `no_refit` wins:** the polish is harmful under H_out, and this becomes the largest item. | ~0.5 + 2 card-h | −0.1…−0.5% (more at ≤1.5 bpw) / 0.4 |
| **2** | **Mixed-K H_out splice pair** — redteam R1 | Whether "H_out at every level" holds in a knapsack mix. Analogous to #111, where asym won every uniform test and lost 8–24% in mixes. | Splice existing levels on card 0: even layers K4, odd layers K2.5. bn+bn4 vs base+base4, plus the seed-1 twins, on a **fresh** slice (27c). **Holds:** geo ≤ 0.985, both pairs < 1. **Fires (geo ≥ 1.00):** scope H_out by K band before the ladder. | ~2.5 card-h, no new quants | avoids +2–3% / P(fires) ≈ 0.1 |
| **3** | **Output-row tile interleave for GDN in_proj_qkv and attn q_proj** — composable A1/A2/A4/G1. Codes 2.1 and redteam R10 cover the same spread from a different angle. | A format field (tile-index remap at the store or a gather), and the **storage order of those tensors' rows in the ladder**. This is the only pre-ladder format lever with a sizeable modelled gain. | (a) card 0, M2: real rxq f2 natural vs 16-row tile interleave, L10 qkv/z and L11/L39 q_proj at K2.5/K4. **Pass:** qkv f2 ≤ 0.90. (b) CPU: head-level h stability across Σdy² block halves (halves are on gpubox). **Pass:** out-of-sample keeps ≥ 75% of the gain. (c) If both pass: dense-splice KL pair (M1 harness), two seeds. **Pass:** geo ≤ 0.995, both parts < 1. | 20 min + 2 card-h + CPU | −0.2…−0.7% / 0.3 |
| **4** | **C9 and swj at sub-1, plus sub-1 hygiene** — redteam R14/R15/R18/R22/R23, lowbpw A5/A6/D3, codes 8.2, measure K1, composable D4/E17 | In-job C9 α(K); swj k_min; scale-grid floor; drift below K1; zero-tensor floor; suh clamp. Sub-1 levels are built last, so this is less urgent but cheap. | colo CPU stand-in at K0.125/0.5/1: log C9 c statistics. **Fires if c_mean > 1.3 at K < 1.5** → set α(K) = α·clip(K−1, 0, 1). Also: g_scale histogram (**>10% at the 0.5 floor** → extend to 0.25); measured drift vs 1.08; suh/svh zero/subnormal scan of existing levels; rel-after-refit vs 1.0 on 9 tensors at K0.125 (≥ 0.97 → price as zero / drop the cell). 4070 1 h: swj on/off at K0.5/K1. | CPU ~3 h + 4070 1 h + ~10 card-min | 0 at 2.5; −1…−4% avoided at 1.0–1.25 / 0.5 |
| **5** | **Log-once instrumentation in the job stamp** — measure B1/B2/B3/B4/B7, lowbpw H1/H2/H6, composable C8, faraway (per-position KL rows, bf16 top-2) | What every ladder job writes. Cheap only before the stamp is regenerated. | Add: per-output-channel error `diag(EᵀHE)` (one- and two-sided), 128×128 error map, trajectory (pre-refit / post-refit / post-CD / post-swj+C9), pre-refit su/sv, overload counters, chosen g_scale, `proxy_err_2s`; 1–2% duplicate-seed jobs. **Check:** CPU stand-in reproduces Σ_n e_n = logged proxy to 1e-6. | ½ day dev, ~0 run | enabler: C-pricing worth −0.3…−1% / 0.9 |
| **6** | **swj scope k_max = 4.5** — redteam R17 | Saves about 18 of the ~53 card-h of swj passes. knap5 already prices swj as 1.0 from K5. | None needed. If anyone objects: one K6 fixed-code swj pair (1 card-h). **Pass to keep:** ≤ 0.995. | 0 | ≈0 KL; −18 card-h / 0.9 |
| **7** | **q_proj q\|gate and in_proj_qkv q\|k\|v as separate knapsack tensors** — codes 2.4/2.5, lowbpw F1 | Ladder job structure (which tensors exist). | colo settle64 two-sided proxy at equal bytes on 8 tensors (L3/L19/L39/L63 qg; L2/L10/L30/L50 qkv). **Pass:** ≥ 0.10 bit-equivalent mean. Then one 27B fixed-code pair from rows of two existing levels. Composes with item 3 (interleave inside each section). | 1–2 CPU-h + 1 card-h | −0.2…−0.8% / 0.25 |
| **8** | **H_out shape: σ_out sweep and tempering h^p, held out** — redteam R5/R6, composable A7/A8/C4, codes 7.1–7.3, measure F3 | v3.toml `hout.sigma_out` and an exponent. σ_out has never been swept at 27B. | colo w87 two-sided proxy, factor from even blocks, judged on odd blocks: σ_out ∈ {0.003, 0.01, 0.03, 0.1} × p ∈ {0.5, 0.75, 1} at K0.5/2.5/4. **Pass:** ≥ 2% held out with ≥ ⅔ of tensors agreeing (the #114 rule). A 27B pair only if it passes. Prerequisite: copy the Σdy² block halves to colo (they are not there now). | ~4 CPU-h | 0…−0.4% / 0.15 |

**Already running and gating, not repeated above:** w92 full-stack level, w93 integrity scan (merge B2). For w92 I would add redteam R75's pre-registered rule (full stack ≤ bn+swj, and ≤ 0.93 vs base, with combined SE) and measure A4's same-seed design.

**Faraway items that need something reserved before the ladder:**
- Keep per-position KL rows and bf16 top-2 margins from every fit-pool scoring. This is in item 5 and enables FB-A1, FB-A2 and FB-B2.
- Record each qad window's category in collection order (seconds; enables FB-A3).
- The interleave field (item 3).

Optional head-adapter and side-module fields (FB-C4, FB-D8, FB-H2) are post-freeze plugin weights and need nothing reserved in exl3i.

---

## 2. Top 25 (expected gain × P / cost, at 2.5 bpw)

| rank | item | freeze | cost | gain | P | lenses |
|---|---|---|---|---|---|---|
| 1 | **GDN in_proj_a/b out of bf16** (fp8/int8 per row, int4 option; knapsack-priced) | post-freeze (plugin op; not a ladder level) | ~1 card-h + small op | −0.7…−1.5% (0.007–0.0105 bpw freed) | 0.7 | codes, lowbpw, composable |
| 2 | **#118 per-family attention offsets in knap5** | knapsack input | CPU minutes + 1 eval | −0.2…−1% | 0.5 | lowbpw, codes, composable, measure, redteam |
| 3 | **Scale polish metric** (shortlist 1) | pre | 2.5 card-h | −0.1…−0.5% | 0.4 | measure, redteam, lowbpw, composable |
| 4 | **Head/embedding prices from the calib part** (post-stratified fit) | knapsack input | CPU | −0.2…−1% | 0.5 | composable E1, lowbpw G7, measure E4 |
| 5 | **Two-sided head with no Fisher-sorted rows inside slabs** (vocab order or interleave; gen-position Fisher; Fisher floor) | before head donors (after the trunk) | ~2 card-h headsim | −0.5…−1.5% (calib-like) | 0.4 | composable B1/B2/B4/B7, codes 11.1; merge A6 captured |
| 6 | **swj k_max 4.5** | pre | 0 | 0 (−18 card-h) | 0.9 | redteam |
| 7 | **Mixed-K H_out splice** | pre | 2.5 card-h | avoids +2–3% | 0.1 | redteam |
| 8 | **Output-row tile interleave** (qkv, qg; z/k/v after) | pre (format) | 20 min + 2 card-h | −0.2…−0.7% | 0.3 | composable; codes 2.1 alternative |
| 9 | **Log-once instrumentation** | pre | ½ day | enabler | 0.9 | measure, lowbpw, composable, faraway |
| 10 | **Corrected head slab plans before KN_HEADGRID; guard = 1.0** | knapsack input | CPU | ±1% | 0.6 | redteam R42/R43 |
| 11 | **C9 K schedule + sub-1 hygiene** | pre (sub-1 levels) | CPU 3 h | 0 (−1…−4% at 1.0–1.25 avoided) | 0.5 | redteam, lowbpw, codes, measure |
| 12 | **Section split q\|gate, q\|k\|v** | pre | CPU + 1 card-h | −0.2…−0.8% | 0.25 | codes, lowbpw |
| 13 | **Stored-row readouts** (tail share, category, all positions, trace-clustered SE, CUPED) | diagnostic | CPU minutes | re-ranks themes | — | faraway, redteam, measure |
| 14 | **Two-sided / Fisher pricing in kfit2** (½Σh_n e_n; split lumped attn_in / ssm_in groups) | knapsack input | colo ~1 day | −0.3…−1% | 0.3 | measure C2–C4, lowbpw, composable E2 |
| 15 | **Pass-1 calibration design + fold stored 27B arms into kfit2** | knapsack input | CPU now; ~7 card-h after pass 1 | −0.5…−1.5% (better pass-2 solve) | 0.4 | measure C7/C9, lowbpw G4 |
| 16 | **Low-rank sub-1 options** (H-weighted SVD; factors as two exl3i tensors, A2) | post-freeze (new options; no ladder change) | colo census 3 h + format | 0 at 2.5; −2…−6% at 1.0–1.25 | 0.35 | lowbpw A1/A2/A14/A16, codes 8.1 |
| 17 | **σ_out / tempering sweep** | pre | 4 CPU-h | 0…−0.4% | 0.15 | redteam, composable, codes, measure |
| 18 | **Category-matched Σdy² / H_in weights** (reweight existing blocks; agent share) | pre (statistic) | 4070 2 h; 27B near-free only if block categories differ | −0.3…−1% | 0.15 | faraway A3 |
| 19 | **Low-K (K1.5) and high-K (K6) stack pairs**, plus the i8 two-sided ceiling on CPU | pre (apply_k scoping) | CPU 3 h + ~5 card-h | 0 at 2.5; protects 1.0–1.75 and 6–7 | 0.15 | redteam R2/R3/R29, measure A7 |
| 20 | **Intra-layer pair terms in the layer DP** (pairs measured on the 0.8B first) | knapsack input | 4070 2 h → 27B ~2 card-h | 0…−0.8% | 0.15 | lowbpw G5, measure D1/D2/D6, composable E7 |
| 21 | **Margin (flip-risk) weighted H_out** | pre (statistic) | 4070 3 h; 27B 5 card-h | −0.5…−2% | 0.1 | faraway A1 |
| 22 | **Re-LDLQ / joint code+scale re-search after refit** (core K2–3.25 levels) | pre | f2 test 30 min; ×2 quant time on ~10 levels | −0.1…−0.5% | 0.2 | codes 4.2, composable D2, lowbpw E4 |
| 23 | **Saturation fit and real anchors at ≤ 1.75 bpw** | knapsack input | ~6 card-h after sub-1 levels | 0 at 2.5; −1…−3% at 1.0–1.25 | 0.3 | lowbpw G2/G3, measure K2/K3, composable E24 |
| 24 | **Mixer-first equal-bytes test on agent text** (incl. attention k/v at 8 bit) | knapsack input | 4070 1 h → 1 card-h | 0…−2% | 0.15 | faraway B1/H3 |
| 25 | **Neuron-pruned MLP options for ≤ 1.25 bpw** (OBS-compensated) | post-freeze (plugin: per-layer MLP width) | colo 2 h census, then format | 0 at 2.5; −2…−6% at 1.0 | 0.15 | lowbpw B1 |

---

## 3. Cheapest diagnostics that would change this ranking most

1. **Refit four-way f2 on card 0** (20 min; shortlist 1). Decides rank 3. If `no_refit` wins, it is the largest pre-freeze item; if the h-weighted polish moves f2 by < 0.3%, drop it.
2. **Interleave M2 f2 on card 0** (20 min) **+ head-level h split-half check** (CPU minutes, once halves are on colo). Kills theme A of composable if qkv f2 ≥ 0.95 or the out-of-sample gain is < 75%. Moves it into the top 3 if qkv f2 ≤ 0.85.
3. **Readouts from stored 27B rows** (CPU minutes, lead on gpubox):
   - top-0.1/1/10% share of mean KL;
   - per category;
   - all positions vs gen-masked;
   - trace/source-clustered SE;
   - CUPED.

   The #117 log's pooled p99 of 3.9 at mean 0.058 implies the top 1% of positions carry ≥ 67% of the score, *if* that p99 is per position. This decides faraway A1–A5 and redteam R8/R64/R66, and sets document counts for every later pair.
4. **knap5 what-if sweep** (CPU minutes) at 1.0/1.25/2.5:
   - (a) #118 offsets;
   - (b) a/b at bf16/int8/int4;
   - (c) calib-part head/embedding weights;
   - (d) H_out/#117/swj proxies converted to KL equivalents (measure C1);
   - (e) low-rank/K0 options from lowbpw M1.

   Any item that moves predicted KL by < 0.2% drops out.
5. **a/b dense-splice pair** (0.5–1 card-h, fixed codes): a/b dequantized from fp8, int8 and int4 into a K2.5 mix. If int8 gives r ≤ 1.002, rank 1 is confirmed and only the knapsack decides the bits.
6. **C9 statistics at K0.125/0.5/1 on the CPU stand-in** (minutes). Confirms or kills R15.
7. **headsim three-arm** (one-sided / two-sided vocab order / two-sided sorted; ~1.5–2 card-h). Settles rank 5 and the slab-order rule, with exact head-only KL.
8. **Mixed-K H_out splice** (2.5 card-h). Not cheap, but the only check on the every-level decision; run it before the ladder enqueue whatever 1–7 show.

---

## 4. Full merged list by theme

Each entry gives: lenses · freeze tag · fit · hardware/cost · test (pass rule) · gain at 2.5 · P · evidence and conflicts.

### A. Per-job quantizer settings (all 55 levels)

- **A-1 Scale polish metric.** Shortlist 1.
  - Lenses: measure A1/A2, redteam R13, lowbpw D1/D2/H6, composable C3.
  - Pre-freeze (or post-freeze if pre-refit scales are kept). Per-level. Card 0, 2.5 h.
  - Gain −0.1…−0.5%; P 0.4.
  - Evidence: code confirmed; #118's ×1.076 one-sided ratio on H_out levels; the 0.8B B1 screen gave 0.993 / 0.975 (bdiag, #85-02 inconclusive).
  - Differs from #85-02: it fixes the in-job polish, which is not a new pass.
- **A-2 Sub-step objective audit** (measure A2). Pre-freeze. Code read plus a per-step f2 log, 2 h. Gain 0…−0.3%, P 0.3. Merge into A-1's run.
- **A-3 C9 α(K) schedule.** Lenses: redteam R15, lowbpw D8/F4. Pre-freeze for sub-1. Shortlist 4. Gain 0 at 2.5. Evidence: code confirmed (c up to 2.83).
- **A-4 swj scope.** k_max 4.5 (redteam R17, shortlist 6); k_min ≥ 1 if the 0.8B K0.5/K1 test is worse (R18). Pre-freeze.
- **A-5 swj mix-dependence (R19): CLOSED.** swj fits on bf16 MLP inputs. The only remaining constraint is gate = up K (R20); add an assertion in the mix builder.
- **A-6 σ_out / h^p / floor clamp.** Shortlist 8. Lenses: redteam R5/R6, composable A7/A8/C4/F3, codes 7.1–7.3, measure F3. Conflicts: #114-03 swept σ_in only; #33-27 / #33-29.
- **A-7 Two-sided drift table and sub-1 drift** (composable C1, lowbpw D3). colo CPU stand-in, 1 h. **Pass:** measured drift deviates ≥ 0.5% (K ≥ 1) or ≥ 1.2 at K0.5. Gain 0…−0.3%, P 0.2. #107 was one-sided.
- **A-8 g_scale chosen in the right metric** (codes 4.3/4.4, composable C2/D1, lowbpw D4, measure F6). colo settle64 2 h, or card 0 f2 on 9 tensors. **Pass:** ≤ 0.995. Gain 0…−0.3%, P 0.15. D-1/§0 shows the within-tile metric is flat, so any gain must come from feedback interaction. #39 was MSE-only.
- **A-9 Re-LDLQ / joint code–scale re-search after refit** (codes 4.2, composable D2, lowbpw E4; CoS-VQ and JARQ evidence). card 0 f2 at K2.5, 30 min. **Pass:** f2 ≤ 0.99, then a 27B pair. Doubles quant time on core levels. Gain −0.1…−0.5%, P 0.2.
- **A-10 #39 gain recheck on the order-fixed proxy with H_out** (redteam R50, merge B3). colo 2 h. Freeze review: refit after the gain makes the effect ≤ 0.5%. Gain 0…−0.3%, P 0.1.
- **A-11 Block order with H_out** (redteam R81). colo w87 2 h. **Pass to keep:** not worse. ±0.2%. #117 counts as 0 gain under R53's audit and is kept because it is free.
- **A-12 Tile CD scope.**
  - Keep attn_qg; gate by g_eig ≥ 0.03 (codes 4.1/4.13, composable C5/A9, measure F2, redteam R82/R83).
  - §0/D-1: early attn_qg is the only material in-tile structure (0.21 bit); gate/up ≤ 0.006. That closes merge B12 and its attn_k variant (user-closed anyway).
  - A K6 / K1.5 no-harm f2 check (10 min) is worth doing.
  - Gain ≤ 0.1%.
- **A-13 Held-out acceptance for tile-CD flips** (codes 4.8, measure F8, faraway A6). Safety. ±0.1%. Low.
- **A-14 Gen-weighted H_in for K ≤ 2 levels** (measure F4). #86 gave K2 0.976 on every seed but missed its 0.97 bar. 4070 2 seeds + K1.5, 2 h; a 27B collection is ~12 card-h, forward only. 0 at 2.5; −0.5…−1.5% at 1.0–1.75; P 0.15.
- **A-15 Partial activation denoising** (measure F5, lowbpw E5, faraway E3, composable F1, codes 7.4; 2610.07522 verified).
  - 4070, 3 h, λ > ρ. **Pass:** ≤ 0.98 at K2 and K3.
  - Gain 0…−1%, P 0.08.
  - #88 (level-matched, full denoising) failed at K3 1.031 / K2 1.10. The difference is partial ρ < λ and propagated noise. Still a weak prior.
- **A-16 Two-path lookahead / soft (BCJR, VAMP) LDLQ + CD** (codes 4.6/4.7, lowbpw D6, faraway E1; MPQ verified at 2-bit). Days of development; colo proxy. **Pass:** ≤ 0.98× LDLQ+cd3. Gain −0.5…−2% at 2-bit, less at 2.5. P 0.1. **Not feasible before the freeze;** V4 candidate.
- **A-17 Damped feedback at sub-1** (lowbpw D10; QTEA). colo 1 h. **Pass:** ≥ 2% held out. 0 at 2.5.
- **A-18 Null-space shrink** (codes 4.5). Faraway M0.9 found no eigendirection below 0.01×mean on 27B H_in. **Reject.**

### B. Format and layout that change the ladder

- **B-1 Output-row tile interleave** (qkv, qg; z/v/k as A3). Shortlist 3.
  - Lenses: composable A1–A6/A9/G1/X2. Redteam R10 said "not actionable", which holds only without a gather; with a gather it is actionable.
  - High-rate model (composable perm.py, read): sum of pivots per 128-block after diag(h·sv²) rotation; qkv ×0.775 (16-row), q ×0.857. The model is sound for high-rate LDLQ noise shaping.
  - Unknowns: realized fraction at K2.5, noise in head-level h, gather cost (2–4% of GDN prefill if unfused; tile remap in the store epilogue is about free).
- **B-2 Section split** (q\|gate, q\|k\|v). Shortlist 7.
- **B-3 Trunk row-slab K + shared `suh` across levels** (codes 2.1/2.2/2.3/2.6/2.11/6.5/9.5, lowbpw A8).
  - Ceiling 0.07 bpw (codes D-2, high-rate, Σdy²-weighted), against #96's 0.036.
  - Needs per-slab K, GEMM support, and suh sharing (refit cost unmeasured).
  - The user closed per-head K on 11-08. B-1 exploits the same between-head spread at fixed rate with no rate field.
  - **Run only if B-1's M2 fails.** Then the cheapest check is codes §12.2 (refit split by suh/svh, 2 CPU-h) plus §12.3 split-half.
  - Gain −0.5…−3%, P 0.1.
- **B-4 Per-input-block (column) K, mask alignment, D-weighted Viterbi.** codes 2.8/2.9/4.10, lowbpw E1/E6, measure F1, composable check 1. **Closed:** within-tile D flat to ≤ 4e-4 bit after RHT (three independent measurements plus CurveTQ); across-16 ≤ 0.09 bit.
- **B-5 Heavy-tile special K / tile CD on the massive-channel block** (codes 3.5/3.6). ≤ 0.3%. Low. Conditional on reader-oracle C-3.
- **B-6 Per-head K on writer inputs** (codes 2.7). ≤ 0.3%. Reject for V3.
- **B-7 Learned 128-block rotation for high-g_eig tensors** (codes 6.1, HARP). ≤ 0.2% for +0.02–0.08 bpw metadata. Reject.
- **B-8 Metadata diet: su/sv as 8-bit log** (lowbpw C6). Saves 0.002 bpw ≈ 0.3%. CPU 20 min. **Pass:** proxy ≤ +0.1% after refit. P 0.3. Pre-freeze format field. Low priority: scales carry swj/C9/H_out.

### C. Bytes outside the trellis and reader outliers

- **C-1 GDN a/b quantized.** Rank 1.
  - Lenses: codes 1.1 (bytes overstated 2×), lowbpw C1, composable G2.
  - Post-freeze (not ladder). Plugin op; radiance's qwen35_fp8 path exists.
  - Test: 27B dense-splice of the dequantized a/b, fixed codes. **Pass:** r ≤ 1.002 at int8 or fp8; ≤ 1.005 at int4. Then add the options to knap5.
  - Evidence: on this model a/b are the least sensitive GDN projections (2609.04098, verified).
  - D6 closed only the GEMM merge.
- **C-2 conv1d to 8 bit** (codes 1.2). 0.0006 bpw freed ≈ 0.1%. Fold into C-1's splice.
- **C-3 Reader-side channel-3994 restore oracle** (codes 3.1). card 0, 30 min. **Keep the theme if** R ≤ 0.995. Gain 0…−2%, P 0.1.
  - #120-02 (writers) was null.
  - Codes' evidence (L3 q_proj near low-rank) is at sub-1.
  - lowbpw F2: CPU census of the reader error share in the spike column first, 1 h.
- **C-4 Spike-column side-car / rank-r side term for early readers** (codes 3.2/3.3, lowbpw A10/A4, measure E3). Conditional on C-3. Would need an exl3i field (#120 reserved none). 0…−1%. Low.
- **C-5 Sink-aware q_proj H_in** (codes 3.4). Echoes #33-07. Low; 0.8B only if cheap.

### D. Sub-1 and low-bpw sizes (1.0–1.75; ≈0 at 2.5)

- **D-1 Low-rank sub-1 options** (rank 16). lowbpw M1 measured on 17 real 27B tensors: LR4@K0.125 beats trellis on 13/17, and loses on every down_proj and mid attn_v.
  - Caveats: the trellis side is the #90 CPU proxy (free-start, no H_out, unclear refit); factor quantization error is excluded.
  - Needs a G6-style KL-per-proxy check, since structured error may cost more KL.
- **D-2 Sub-1 hygiene bundle.** Shortlist 4.
- **D-3 Chained tail-biting / NTRY at sub-1** (lowbpw A11/A12, composable D3, codes 8.4). colo viterbi.c, 1 h. **Pass:** chained ≤ 0.95× TB, or NTRY16 ≥ 0.5%. NTRY is encoder-only. #102 tested only K2–3. P 0.3 for NTRY.
- **D-4 Tuned / sparse sub-1 codebook** (codes 5.4, lowbpw A13). colo 1–2 h. **Pass:** ≤ 0.95. P 0.15. Format flag.
- **D-5 Tile CD at K ≤ 1.875 for non-attn_k families** (lowbpw D5, composable D5). colo 2–3 h. **Pass:** median ≤ 0.99. +2–3 card-h ladder. P 0.25.
- **D-6 Neuron-pruned MLP options** (lowbpw B1/H3). Rank 25.
- **D-7 GDN head / q-head pruning, layer drop** (lowbpw B2–B4, faraway D6/D7/G3). Faraway M0.5/M0.6 measured unhealed pruning 11–15× too costly. **Reject for V3.**
- **D-8 Saturation fit / sub-1 anchors / guard extrapolated cells / per-K-band factors** (lowbpw G2/G3/G8/G9/G10, measure K2/K3/K4, composable E11/E24). Rank 23.
- **D-9 Head K1 / K1.5 donors and low floors; low-rank head core for ≤ 1.5** (lowbpw C2/C3/C4). After the trunk; headsim. 0 at 2.5; −1…−4% at 1.0. C3's rank-r generalization of mean-row is merge A6 (I001930), captured.
- **D-10 Attention temperature restoration per q/k level** (lowbpw D9). 0 at 2.5. Low.

### E. Head and embedding

- **E-1 Two-sided head, ordering rule, gen Fisher, Fisher floor, block-normalized Fisher** (composable B1–B7/F4/X2, codes 11.1). Rank 5.
  - Two-sided head LDLQ is merge A6 (captured).
  - New: never store slab rows Fisher-sorted (×0.83 vs ×0.33–0.40 in vocab order or interleaved).
- **E-2 Softmax shift t / Fisher-weighted shift** (codes 11.2/11.3, measure L1). Add to A6's headsim batch. 2609.31291 found ~0 on Qwen3.5 with GPTQ. ≤ 0.3%.
- **E-3 Margin-weighted slab ranking** (faraway A5). headsim CPU minutes. **Pass:** ≥ 10% lower head KL, guard not worse. Gain −0.3…−1%, P 0.2. Add to the A6 batch.
- **E-4 Low-rank head residual** (codes 11.4, ARCHead). Gain over a Hessian-aware head is about 0.4% relative PPL in the paper. Low.
- **E-5 Headsim → whole-model calibration factor; slab-plan corpus robustness** (composable E23, measure L2). CPU. Do it with rank 10.
- **E-6 Embedding row-slab K / Fisher metric / real-KL low-t swaps** (codes 10.1–10.3, composable E21, lowbpw C5). #119's next steps; post-freeze; matters at ≤ 1.5 bpw.
- **E-7 Embedding row-energy restoration** (composable I6). One pair at t2, calib part. ≤ 0.3%. Low.

### F. Statistics (H_out / H_in inputs)

- **F-1 Category-matched statistics** (faraway A3, A4 for healing). Rank 18.
  - Caveat the lens itself raises: if #33-27's heavy blocks 6/9 were agentic, block normalization already down-weighted agent text and *helped*. Check block composition first (seconds).
- **F-2 Margin / flip-risk H_out** (faraway A1). Rank 21. The user opposes new Σdy² collections; run on the 0.8B first.
- **F-3 KL-reweighted IRLS statistics** (faraway A2). 0.8B 3 h. **Pass:** ≤ 0.97. P 0.1. Overfit risk (DRQ).
- **F-4 Tail-aware damping, 27B self-continuations, lifetime-weighted H_out rows** (faraway A7/A8/D4). P ≤ 0.08 each. Low.
- **F-5 Analytic SwiGLU metric as a shrinkage target for gate/up h** (composable F5). #84's cosine gate failed as a replacement. Low.
- **F-6 Quantized-operating-point H_out** (redteam R7). V4 note only.

### G. Knapsack inputs (no ladder change; decide before the first V3 solve)

- **G-1 #118 offsets.** Rank 2.
- **G-2 Price parity across settings** (H_out/#117/swj proxies → KL equivalents; turboderp comparison). measure C1, composable E2, redteam R32/R35/R37. CPU 1 h. Gain 0…−0.3% at 2.5; −0.2…−1% at 5–7 bpw. P 0.4.
- **G-3 Calib-part pricing.** Rank 4.
- **G-4 Two-sided / Fisher pricing, split lumped groups, cost exponent per family, Fisher-trace depth prior, GDN half-life prior** (measure C2/C3/C4/C8/C10/C16, lowbpw F3, codes 2.12/9.8, composable E10). Rank 14 as a bundle. The colo decode of V2 containers makes #118 2(c) affordable.
- **G-5 Separate gate/up sensitivities** (codes 9.3, measure C5). Blocked in practice by the swj gate = up K tie; price as one item. **Reject untying for V3.**
- **G-6 Stack factor by K; family-resolved H_out/swj/C9 factors from fixed-code toggles** (composable E4/E5, measure C6/C11, redteam R36/R38). ~3–4 card-h. Gain −0.1…−0.3%. Do after w92.
- **G-7 Pair terms / non-additivity certificate / coherent-component test** (measure D1/D2/D3/D5/D6/D8, lowbpw G5, composable E6/E7, faraway B5). Rank 20.
- **G-8 Operating-point probes** (amplitude α-sweeps, row-shuffle, antithetic ±E, band restore; measure C13/C14/C15/D7, composable E8/E9). 2–8 card-h each. Validation of the model; post-pass-1.
- **G-9 Block-output (probe-propagated) sensitivity feature** (measure C12; 2609.33923 verified on this model). colo 2–3 h per level. **Pass:** Spearman vs group swaps beats one-sided proxy. Gain −0.2…−0.8%, P 0.2. Worth a CPU try, since it is the one paper result on Qwen3.8-27B allocation.
- **G-10 Flip-attributed prior / flip-count feature** (faraway B2/B6). Needs the per-position rows (shortlist 5). P 0.1.
- **G-11 Heal-aware knapsack, differentiable allocation audit** (faraway B3/B4). V4 / 0.8B. Low priority now.
- **G-12 Level-menu pruning by pick probability; defer pass 2; value-of-level** (measure G1/G2/G3, composable H1, redteam R76, codes 9.9). Saves 10–40 card-h. CPU 1 day. Do it after pass 1.
- **G-13 Bucket resolution 256 KiB** (codes 9.4). Captured (merge: KN_BK 256 KiB).

### H. Evaluation rules and integrity

- **H-1 Rule audit (null-calibrated), combined SE, fresh slice 27c, calib-part sign weighting, source/trace clustering, CUPED** (redteam R53–R59, measure A4/H1/H2/H3/H6). CPU minutes. Adopt before any further pair.
  - Most remaining candidates (0.3–1%) are below the single-slice resolution (~2.5–3%).
  - **Decide them on fixed-code pairs or proxy plus cost, not on requant pairs.**
- **H-2 Pair arms differ only in the intended setting** (stamp-diff assertion; redteam R60). Guardrail.
- **H-3 Integrity:**
  - Hessian/H_out scan: merge B2 / w93, running.
  - H_out coverage assertion (R69).
  - Allocator env refusal (R71, merge A1).
  - Fork 0008 pinned in donor images (R72).
  - Config-time refusal of tile CD + dense H_out (R78).
  - Donor scorer two-sided (R73, composable C6). The freeze review says the `outlier_better` flag covers this. Acceptable.
- **H-4 GDN state dtype parity and engine floor; top-512 truncation at high bpw** (measure I2/H4/H5, redteam R66). Code read 30 min; a 27B floor needs TP2 (~1.5 card-h). Matters for 6–8 bpw. I2 is cheap: do it.
- **H-5 Long-context and position-resolved family sensitivity** (measure I1/I3/I4, redteam R63). ~1–4 card-h. Gain −0.1…−0.5% for long/agentic. Post-freeze unless it fires.

### I. Post-freeze (healing / QAD / per-mix; listed only for completeness)

- GDN A_log, dt_bias and conv1d as healing parameters (faraway D2/D3; merge B13 captured).
- Agent-matched healing data (faraway A4).
- STE trellis QAD with periodic Viterbi (faraway D1; TVQ verified: no gain on Llama-2-70B).
- KL head adapter / routed correctors (faraway C4/H2/J3).
- Sequential statistics per mix (faraway D9).
- Distillation from a 4.5-bpw teacher (faraway D10).

All need the 0.8B first; none blocks the freeze.

---

## 5. Rejected (reason)

- **D-weighted Viterbi, fractional-mask alignment, column-split K:** within-tile metric flat after RHT (codes D-1, measure §0, composable check 1, CurveTQ).
- **Residual-stream channel permutation (route A), cross-block Kronecker mixing, KLT-then-Hadamard, learned rotations, 512-point Hadamard:** input side already within 1–3% of the GM bound (composable check 5); SpectrumQuant shows no gain at 70B; user-closed.
- **i8c revival (#78-04), hash codebook (#100), trained table, 2^20 trellis (#113), stored start state (costs 0.055 bpw for ~0.039-bit gain), entropy-coded indices, random-coding residual, E8 / Leech replacement:** measured or arithmetically negative.
- **Zero-byte logit post-processing (temperature, bias, tail floor, feature temperature), fixed head-side linear map:** faraway M0.1/M0.7. Fit-on-A ratios on B were 1.00–1.04.
- **Residual slicing, unhealed neuron / layer pruning, vocabulary pruning (243 pad rows), sibling joint coding, null-space shrink:** faraway M0.5/M0.6/M0.8/M0.9/M0.10.
- **Gate/up untied K:** the swj fit requires equal K; gain ≤ 0.036 bit.
- **Logit-averaging two half-size models, uncounted-tensor storage, per-document precision:** arithmetic or against the rules' intent.
- **Best-of-N seeds** (#41), **per-group scales** (#40), **A4** (#46-03), **per-level asym** (#105/#111), **xcorp H_in** (#101), **REAL-Q** (#92): already measured negative; no lens gave a real difference.
- **Bucket resolution, K1.125–1.875 in the menu, head 1/16 grid, donor scorer one-sidedness, integrity scan:** already captured (merge 11-08 / freeze review), so not new.
- **codes 1.1's "0.026 bpw":** corrected to 0.014 (checkpoint read). The idea survives at about half the claimed gain.

</div>
