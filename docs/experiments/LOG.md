# Experiment log (newest first)

Append-only; template and rules in `README.md`. Every experiment's entries are here, from 2026-10-29 on (entries
before 2026-11-02 ~05:00 CST were rewritten into this template from the old trackers; notes at the end). Earlier
versions of anything: git history.

### 2026-11-11 9:55 PM CST · #145 · C9 and swj at sub-1 on the 27B (fixed codes, two mixes, fresh slice 27d): rule says no change (C9 as today, swj at every level); e_clip 0.3 helps at K0.5 (−4.1%, every statistic) but misses the band pattern by 0.0002; swj off costs +12…36% (w110)
- **Status:** done (rule: no change; proposals for the lead) · **Machine:** gpubox card 0 (levels 11:25 AM–3:10 PM,
  swj 3:10–5:23 PM, packs 5:24–7:24 PM under conv.lock + ram.lock excl., scores 7:46–9:37 PM CST) · **Code:**
  `v3/e3/settle/w110/` (levels110.json for rxrun, arms110.sh, c9g1.py, c9e3.py, verdict110.py, slice110.sh) · **Rule:**
  `v3/e3/settle/w110.md` (c6e9cd1, alone, before data; amendments 1 149396a / 2 70f8b72 before any pack or score: G1
  GPU-vs-CPU rounding tolerance 1e-5, e3 = in-job c with only the clipped rows replaced) · **Data:**
  `v3/e3/settle/data/w110/` (`report.md`, `verdict.txt`, `docstat-M{1,2}.txt`, `docsums.json`, hygiene, census)
- **Setup:** three i8 levels at stamp 159591b2 (K0.5 / K1 / K1.5, seed 0, rxrun N = 3, #142 fusion), one 3-source
  swj pass (0.74 h / level), C9 arms cur / off / e3 (e_clip 0.3) × swj on / off as scale-only finals (same codes, one
  seed); mixes into V2 3.00: M1 uniform K1 (6 arms), M2 even K0.5 / odd K1.5 (6 uniform + 6 band arms); all 16 arm
  mixes PAIR OK vs T with `--planes svh`. Fresh slice **golden-search27d**: 130 never-scored xcorp pieces
  (xc8k-b2-c0, 222k pos) + 63 calib search docs never in the ledger (77k pos). T KLD: K1 1.093, splice 2.282.
- **Result (pooled r vs T, both parts same sign unless noted):** swj off: K1 **1.356**, splice 1.187, K0.5-only 1.123,
  K1.5-only 1.061. C9 off: K1 1.097, splice 1.043 [0.956, 1.131], K0.5-only 1.004 (per-doc 0.988), K1.5-only 1.004.
  e3: K1 1.0013 (z +2.6), **K0.5-only 0.9593** [0.945, 0.972] (x 0.957 / c 0.968, per-doc geo 0.965, 89% of xcorp docs
  better), K1.5-only 0.9998 (e3 moves 6 writers there), both 0.959.
- **Rule verdict:** C9 stays; swj stays at every level ≤ K4.5. #140's 0.8B "C9 off at K0.5" and #141-04's swj K0.5
  doubt do not carry to the 27B. e3 passes at K0.5 but the "below K1" pattern also needed worse(K1.5): 0.9998 → report only.
- **Proposals:** (1) e_clip 0.3 for K < 1 (0.5 from K1): post hoc, confirm on a second fresh slice with the kept mixes
  (`/data/w110/mix/mix-sp-cc-ss.rad`, `mix-sp-ec-ss.rad`) + a K0.75 point before adopting; (2) swj at K1 −26% is ~8× the
  0.8B → "unverified" until a second slice (supports the status quo); (3) the levels are 159591b2, now superseded by
  4f831984 (swj bf16 numerics): codes reusable only if the lead re-stamps, otherwise a fresh ROOT.
- **Runtime:** none (scales). Card: ~3.75 h quant, 2.2 h swj, ~1.3 h scores. Cleaned: packs, finals, 15 of 18 mixes.

### 2026-11-11 9:05 PM CST · #148 · P16-TP 27B pair on fresh golden-search27e FAILS (geo 1.0005, z +0.04); full P16 also shows nothing on this third slice (1.0038; joint over three slices 0.9965, z −0.79); TP2 / TP4 functional check PASS (bit-exact) (w113)
- **Status:** done (no adoption proposal: rule FAIL) · **Machine:** gpubox card 0 (T units 01:32–02:04 UTC, packs /
  gates to 02:11, 12 scores 02:11–02:59 UTC, one per card hold between w110's scores), gpubox CPU (slice, tpcheck) ·
  **Code / rule / data:** as below (`v3/e3/settle/w113/`, `w113.md` ed8c0fd, `data/w113/` `a/` `b/` `tp/`)
- **Slice golden-search27e** (first use): 130 pieces of never-scored `xc8k-b2-c1` (187,589 positions; 27b-x / 27c-x /
  27d-x / mX docs excluded) + 72 calib search docs never in the ledger and not in 27d-c (65,334 scored); ledger +72
  rows `golden-search27e-c` (select, v3).
- **#148-02 (b), 112 tensors (ssm_qkv 48, ssm_z 48, attn_qg 16) at K2.5, dense splice into w111's N mixes**; gates
  PAIR OK (only the 112 changed, codes plane), BYTES113 112 / 112, stamp 87a7e4ab. N → P16-TP pooled **0.9933 / 1.0079,
  geo 1.0005** (xcorp 0.9991, calib 1.0284), SE 0.0151, z +0.04 → **FAIL**. Reported: N → full P16 (128) 0.9954 /
  1.0122, geo **1.0038** (xcorp 1.0016, calib 1.0476), z +0.52; P → T 0.9968. Per doc (N → T): per-doc geo 0.992 /
  0.994, 56% / 55% better; xcorp 59% one swe-smith doc (r 0.984), without it 1.004 / 1.038; calib pooled 1.012 / 1.045
  vs per-doc geo 1.002 / 0.995 (a few heavy docs). Joint full P16, 27c + 27d + 27e: **0.9965, z −0.79**; 128-tensor
  P16 on 27d + 27e 0.9985 (z −0.26). This T pair with w101 / w111's P16 pairs: 0.9932, z −1.31 (arms differ).
- **#148-03 TP functional check (CPU): PASS.** The C map (`p16tp_map.c`) == the table for all 5 families at TP 1 / 2 /
  4; on the 8 real T-quantized tensors every TP2 / TP4 rank, computing from its loader-cut stored rows only (whole
  stored blocks, output Hadamard inside the rank) and storing through the closed-form map, equals TP1 un-permuted by
  the table bit for bit.
- **Decision / next:** no adoption on this evidence (rule). P16-TP keeps P16's whole f2 gain on qkv / z / qg and is TP
  1 / 2 / 4-safe with radiance's row cut unchanged, but P16's own KL gain has not repeated (0.9933 → 0.9903 → 1.0038).
  w113's recommendation: do not adopt P16 (either form) before the freeze; if the user wants it, ship P16-TP on
  qkv / z / qg (kernel requirements in `w113/README.md` "Proposal"); v_proj has no TP-safe form. User decides.
  Cleanup: w111's four mixes (66 GB) and all w113 mixes / patches / codes deleted; golden-search27e kept on gpubox.
- **Runtime:** none (no adoption); P + D if P16-TP were adopted (store remap, untimed in this form).

### 2026-11-11 7:35 PM CST · #148 · P16-TP (interleave inside each TP4 rank's rows): real-quant f2 keeps 96 / 114 / 99.5% of full P16's log-gain on ssm_qkv / ssm_z / attn_qg (PASS); attn_v at TP4 width 2 keeps nothing (FAIL); 27B pair on fresh golden-search27e running (w113)
- **Status:** running · **Machine:** gpubox card 0 (units 00:28–01:27 UTC in 4-unit holds of ~3 min between w110's packs),
  gpubox CPU (f2, slice) · **Code:** `v3/e3/settle/w113/` (perm113.py, p16tp_map.c, a113.sh, f2il113.py, report148a.py,
  slice113.sh, b113.sh, tpcheck113.py) · **Rule:** `v3/e3/settle/w113.md` (ed8c0fd, committed alone before data) ·
  **Data:** `v3/e3/settle/data/w113/` · README `w113/README.md` (radiance sharding with file:line refs)
- **Why:** user 11-11: P16 must work across cards. Full P16 (gperm(N, 16)) is not TP-safe: at TP4, 48 / 80 ssm_qkv
  and 16 / 48 ssm_z stored blocks hold rows of two ranks; attn_v / attn_k blocks span ranks already at TP2; elsewhere
  blocks sit at other ranks' positions (loader gather). Radiance's row cut is per stacked part (qkv q | k | v), every TP4
  rank share is a multiple of 128 rows, TP4 sets nest in TP2 sets; exl3i weights refuse TP > 1 today.
- **P16-TP:** gperm(N/4, 16) inside every TP4 rank's row set; one stored order serves TP 1 / 2 / 4 with the loader's cut
  unchanged (no gather); widths ssm_qkv / ssm_z / attn_qg 8 (attn_qg = exactly full P16's block row sets), attn_v / k 2.
  Kernel map closed form (`p16tp_map.c`; parts' tiles per TP4 rank, nb, m = 4 / t) == the table for every family / TP /
  rank.
- **#148-01 (a), real rxq f2, stamp 87a7e4ab, K2.5 / K4 × seeds 0 / 1** (R = geo f2_X / f2_N; keep = ln R_T / ln R_P):
  ssm_qkv (L10) R_P 0.677, R_T 0.688, **keep 0.959**; ssm_z (L10) 0.948 / 0.941, **1.144**; attn_qg (L11, L39) 0.873 /
  0.873, **0.995**; attn_v (L3 / L11 / L39 / L63) 0.865 / **1.002, keep −0.01 → FAIL**; report-only V (TP2-only, width 4)
  R 0.975, keep 0.18. Model ratios agree tensor by tensor (e.g. L63 v_proj P 0.66, T 1.00). v_proj's gain needs rows of
  different KV heads in one block, which no TP4-safe order allows. Seeds agree to ±0.003.
- **Next:** (b) scope ssm_qkv + ssm_z + attn_qg (112 tensors), T vs w111's N mixes on **golden-search27e** (fresh: 130
  pieces of never-scored xc8k-b2-c1, 187,589 positions + 72 calib search docs never in the ledger, 65,334 scored; ledger
  rows added), plus full P16 (128) reported as a third slice.
- **Runtime:** prefill + decode if adopted (store remap); the experiment changes nothing.

### 2026-11-11 6:20 PM CST · #144 · swj bf16 numerics ADOPTED (user) · settings stamp 4f8319840e59c5d0
- User 11-11: "swj: do it". Applied `v3/e3/settle/w109/swj-numerics.patch` (`rxq.swjfit --numerics`, rxrun passes
  `[post.swj] numerics`) and set `numerics = "bf16"`. Checks on colo: check_diag 20/20, check_cpujob 25/25, check_ladder
  5/5, check_rxrun 11/11. swj per ladder ~20 → ~9 card-h; ladder ~2.4–2.6 card-days on card 0.
- **Runtime:** none (swj scales fold into svh).

### 2026-11-11 12:50 PM CST · #146 · P16 interleave: A8 store remap costs +0.05…0.25% of prefill gemm (bar PASS; a raster tweak takes it to noise); second slice with v_proj: D geo 0.9903 PASS, joint z −1.41; the 112 alone 0.9976 here (w111)
- **Status:** done (adopt-proposal under the rule; the user decides) · **Machine:** dad1 9070 XT (dev, card0.lock),
  gpubox card 0 (bench 17:40 / 17:54 UTC, ~10 s each; v_proj units + packs + 16 scores 17:29–18:47 UTC) · **Code:**
  `v3/e3/settle/w111/` (kl111.sh, hold111.sh, verdict111.py, proj111.py, mksplice111.py, bytes111.py); radiance fork
  branch `qc-exl3-v3-p16` (758528b, 4aee43b, fc1032a P16.md: exl3i_8a8.hip.h / exl3i_a8_gemm.hip / exl3i_a8_dev.h, tests/bench_p16.hip,
  tests/p16_build.sh) · **Rule:** `v3/e3/settle/w111.md` (1fa29c9, committed alone before data) · **Data:**
  `v3/e3/settle/data/w111/` (`k/` bench CSVs + projections, `kl/` verdict, docstat, gates) · README `w111/README.md`
- **Correction to w101's kernel note:** its closed form "stored 16 (8 b + t) + c → natural 16 (b + nb t) + c" is the
  inverse of gperm (equal only at nb = 8); the quantized map is natural = 16 (8 (k mod nb) + ⌊k / nb⌋) + c, k = j >> 4.
  Implemented that way (⌊k / nb⌋ by multiply-shift, exact for nb < 378, checked at launch).
- **#146-01 kernel (A8 GEMM, `Exl3iA8Args.il16`):** remap off = today's kernel instruction for instruction (also A4's);
  C1 (off == base) byte-identical at all 9 trunk shapes × K2.5 / K4 × T 512 / 2048 / 8192; C2 (on == host-gathered off)
  byte-identical at all 24 remapped cells. R9700, 30 ABBA rounds, GEMM on / off geo (paired): ssm_qkv **1.010**, ssm_z
  1.004, attn_qg **1.016**, attn_v 1.004 (worst cell 1.020); A / A run 0.998–1.001. Projected prefill gemm (A8, 48 GDN +
  16 attention layers; the four families are 21% of it): **+0.05 / +0.17 / +0.16…0.25%** at T 512 / 2048 / 8192 →
  **kernel bar PASS** (≤ +0.5%, ≤ 1.03 per shape). Reported variant (`il16 = 2`, the stored blocks that fill natural
  blocks 8c…8c+7 run back to back): 1.003 / 1.001 / 1.005 / 1.002, projection ≤ +0.10% (noise). Decode: same remap on
  `epilogue_block`'s y address (nt / rows / rate / T2), A4 one line; expected ≈ 0, not implemented or timed.
- **#146-02 KL, golden-search27d (w110's fresh slice, first #130 use; 193 docs, 299,548 positions):** D = 112 + 16
  v_proj at K2.5, dense splice into w101's mixes (gates: PAIR OK, BYTES111 16 / 16, stamp 87a7e4ab): N → P16 pooled
  **0.9911 / 0.9896, geo 0.9903** (xcorp 0.9904, calib 0.9900), SE 0.0090, z −1.08 → **PASS**. 112 scope alone (w101's
  mixes) 0.9993 / 0.9958, geo **0.9976** (xcorp 0.9987, calib 0.9843): w101's −0.7% does not repeat in size. v_proj
  increment r(128) / r(112) 0.9917 / 0.9937, geo **0.9927 → attn_v joins**; large for 16 tensors → **unverified**
  (one slice; hygiene holds). Joint with w101: 112 scope 0.9941 (z −0.93); w101 + D 0.9922 (**z −1.41**). Per doc (D):
  xcorp geo 0.987 / 0.990, calib per-doc geo 1.002 / 1.003 (calib gain pooled only).
- **Proposal (rule):** adopt P16 for ssm_qkv / ssm_z / attn_qg / attn_v (k_proj model 0.952, untested): a screen pass,
  not significant; needs a format bit and the remap in every writing kernel. Lead / user decide; a v_proj check on a
  third slice is cheap (the four D mixes are kept on gpubox `/data/w111/mix/`, 66 GB; w101's 65 GB of mixes deleted).
- **Runtime:** prefill (+0.05…0.25% gemm as implemented, ≤ 0.1% with the raster) + decode (≈ 0, untimed) if adopted.

### 2026-11-11 12:40 PM CST · #144 · swj level pass: f32 GEMMs are 70% and can't be sped up bit-identically; more sources per pass is the only identical lever (S=8: −30% swj card-h); proposal `--numerics bf16` 2.2x faster, held-out within noise (w109)
- **Status:** done (proposals for the lead / user) · **Machine:** dad1 RX 9070 XT (gfx1201, 16 GB) under dad1's
  card0.lock, 11-11 16:59–18:25 UTC; gpubox: read-only (w92 / w99 / swj27 timing.jsonl and w92's L0-7 scales), card 0
  not used · **Code:** `v3/e3/settle/w109/` (README) · **Rule:** `v3/e3/settle/w109.md` (b6433ad, committed alone before
  data) · **Data:** `v3/e3/settle/data/w109/` (runs.jsonl, prof*.json, judge-*.txt, projection.txt)
- **Setup:** dev passes on the bf16 27B cut to layers 0–7 (HF shards 1–3), w92's K2.5 i8 levels fs0 / fs1, swj27-qad
  (first 100 docs = 143k tokens for speed, all 725 for the held-out judge); today's code = `python -m rxq.swjfit --fused
  --no-cache` at 640ecfd. The 9070 XT matches the R9700: S=2 49.6 vs 52.9 min a level (scaled vs w92's pass).
- **Profile (S=2, synchronized timers):** f32 GEMMs **70%** (sources 46%, the bf16 reference gate / up 25%), bf16 forward
  13%, fused grid sums 10%, load / decode / boundary I/O / fit 6%. Shared per pass B ≈ 52 s, per source P ≈ 32.4 s (dev);
  R9700 from production passes: B 41.9 + S × 30.4 min a full pass.
- **(a) output-identical (judge: scales torch.equal, same box):** the hook's f32 GEMM = hipBLASLt's sequential FMA over k
  (CPU emulation 8/8; all split / interleave orders fail); layouts, merged N, stacked documents and all BLAS backends give
  the same bits at the same 15–17 TFLOP/s; Triton f32 `tl.dot` ≤ 11 TF and not identical. Grid kernel compute-bound:
  launch reorder identical but ×0.91–1.05; identical tile configs all slower. `--fast` (async boundary writes, prefetch,
  one aa² sum) identical but −3.8% (S=8) / +1.4% (S=12): under the 5% bar, dropped. **Sources per pass:** per level 83 /
  57 / 46 / 39 / 37 s at S = 1 / 2 / 4 / 8 / 12 (dev); S=12 peaks at 11.3 GiB (fits 16 GB, one layer a stage). The
  reference reproduces itself bit for bit (3 runs).
- **Not identical across boxes:** today's code on the 9070 XT vs w92's R9700 production scales (L0–7, 725 docs): gate c
  differs on 2.2% of channels, up c on 99.5% (max |dc| 0.06), held-out gain equal to 5e-5. Same box = bit-identical.
  Likely machine-specific kernel choices in the forward (unverified).
- **(b) proposals (swj stays ~29% of the ladder after (a) → in scope; judge: held-out gain on 8 layers × 2 sources):**
  `--numerics bf16` (reference GEMMs bf16 × bf16 → f32, source weights as hi + lo bf16 in one GEMM over 2K): loss
  −0.06% / −0.05% vs spread 1.19% → **PASS**; gate c moved on 1.4–1.5% of channels (less than the box-to-box 2.2%);
  the reference-GEMM change alone already moves 1.5% (summation-order noise, not the split). Speed ×2.2 (S=2) / ×2.3
  (S=8). **Fewer fit tokens: FAIL** (fit on half the fit half: −5.6% of the held-out gain vs spread 1.2%: the fit is
  data-limited). Non-IEEE divide in the grid kernel: no-op (same bits, same time). Coarser grid: not tried.
- **Patch** `v3/e3/settle/w109/swj-numerics.patch` (`rxq/swjfit.py --numerics f32|bf16`, split per tensor (peak back to
  10.6 GiB); `rxrun/run.py` passes `[post.swj] numerics`): tested on dad1, f32 = today's bit for bit, bf16 = the dev arm
  bit for bit, S=12 bf16 203 s / 11.3 GiB. Merging moves rxq's code stamp (all rxq/*.py hashed) and `numerics` the
  settings stamp: before the ladder or never.
- **Projection (w105's method, quant 49–53 h, 36 swj levels):** card 0 today S=2 **30.8 h** swj → 3.33–3.49 card-days;
  S=8 21.4 h → 2.93–3.10; S=12 20.3 h → 2.89–3.06 (swj 28–29%); (b) S=12 9.2 h → 2.43–2.59 (15–16%). swj on dad1
  instead: card 0 2.04–2.21 days, dad1 19.0 h (today, S=12) / 8.6 h ((b), S=12) in parallel; needs the full bf16 model
  there, each level's gate / up (~3.6 GB at K2.5) up gpubox's 300 Mbps link, a remote swj slot and swj off rxrun's main
  loop (not built).
- **Runtime:** none (pass time only).

### 2026-11-11 11:40 AM CST · #147 · Stored layout vs IQR-V4's kernels before the ladder: no layout change needed; output-Hadamard epilogue measured at 3.0% of the A8 GEMM; fractional-K wdec8 1.35× K4 (w112)
- **Status:** done (research; proposals for the lead) · **Machine:** dad1 RX 9070 XT card 0 (card0.lock, 17:34–17:35 UTC,
  ~45 s) · **Code:** scratch worktree of `qc-exl3-v3` 42960d5 + a bench-only epilogue ablation (`EXL3I_W112_NOHAD`, not
  committed) · **Data / write-up:** `local/research/iqr-v4-kernels.md` (+ `-data/`: raw CSVs, patch, scripts)
- **Question:** does any layout property behind IQR-V4's kernel speed need to be in our stored bytes before the ladder?
- **Rule:** none (descriptive measurement; no decision hinges on a threshold).
- **Result:** IQR stores codes row-major and permutes them at load (size-neutral relayout hook), as exl3i already does
  (k-major tiles), so tile order, packing, lane layout, scale placement and alignment stay free for the perf pass.
  Pre-ladder items: **P16** (#130 / #146, format bit, being timed; add: P16 as gperm(R, 16) is not row-splittable for
  TP ≥ 2, so define it per rank if TP2 could ever be timed) and **suh tying** (recommend: keep per weight; a fused
  producer writing one int8 plane per distinct suh gets the gain, tying adds only ≈ 0.4% of the A8 layer at M 2048).
  Codebook (#100) already decided.
- **Measured (#147, layer mean 48 : 16, median of 4 interleaved rounds):** output-Hadamard epilogue = **3.0% of the
  A8 GEMM** (K4 M 512 / 2048 / 8192: ratio 0.970 / 0.968 / 0.969; K2.5 M 2048 0.971), 2.3–2.8% of the A8 path; down
  1.2–1.4%, attn_k / v 5–8%. A8 path at M 2048 K4: actq 7.5%, wdec8 11.2%, GEMM 81.3% (8938 µs). **wdec8 at K2.5 is
  1.35× K4** per layer (1344 vs 999 µs; 2.85× on gdn_z / attn_o / ssm_out shapes) although it reads fewer bytes: the
  run-time plan in `RateFrac`, a kernel fix (the even-spread mask makes the offsets depend only on lane bit 3 and the
  run); knap4's speed table overprices fractional levels at M ≈ 512 until then.
- **Decision / next (lead):** confirm a size-neutral load-time permutation is outside the AGENTS.md "VRAM size" rule;
  record "suh stays per weight" as a freeze decision; settle P16's TP form with #146.
- **Runtime:** none (research); the items it lists are P (prefill) for the perf pass.

### 2026-11-11 11:10 AM CST · #141 · measured drift table helps at every sub-1 K (held out −2.4 … −0.1%) and takes production g off the 2.05 edge (13/13 tensors better at K0.125 and K0.5); NTRY 4 keeps 59–96% of NTRY 16's gain at ~¼ the cost; swj fails the sub-1 rule at K0.5 on the 0.8B, keeps K1 (w107)
- **Status:** done (proposals for the lead) · **Machine:** Thor + colo (-01 / -03: 170 + 23 single-threaded jobs, 11-11
  14:10–15:59 UTC), gpubox card 0 (-02 14:42–15:38 UTC, -04 evals 16:55–16:58 UTC, card0.lock → ram.lock shared) ·
  **Code:** `v3/e3/settle/w107/` · **Rule:** `v3/e3/settle/w107.md` (69cbd3e, committed alone before data; its text says
  "~8:20 AM CST", the commit is 8:06 AM CST) · **Data:** `v3/e3/settle/data/w107/` (`report.md`; rerun:
  `python3 v3/e3/settle/w107/report107.py v3/e3/settle/data/w107`)
- **Gates:** G1 PASS (cur re-run = w100's rows: same box bit-identical, cross-box rel2_B ≤ 5.6e-8, d_all ≤ 9.2e-10
  relative); G0 PASS (T1 = w102's stored rows, worst 9.7e-8).
- **-01 drift (w100's proxy, 27 × 2, held-out two-sided rel2_B):** D(K) = 1.3475 / **1.3425 / 1.2452** / 1.1838 /
  **1.1425 / 1.1137 / 1.0922** / 1.0892 at K0.125 / 0.25 / 0.375 / 0.5 / 0.625 / 0.75 / 0.875 / 1. Log-linear interp of
  w100's three points misses K0.25 by +4.0% (the drift is flat from K0.125 to K0.25), K0.625–0.875 by −1.5 … −1.9% →
  **the eight-point table is needed.** Table arm vs cur (r, tensor-bootstrap 95% CI): 0.9758 [0.968, 0.983], **0.9654**
  [0.956, 0.975], 0.9756, 0.9830, 0.9914, 0.9960, 0.9993 [0.9983, 1.0002], 0.9981 [0.997, 0.999]; r ≤ 1.000 at all 8 →
  **proposed: T = 1.348 / 1.342 / 1.245 / 1.184 / 1.142 / 1.114 / 1.092 / 1.089** (K ≥ 1.5 unchanged). Patch:
  `v3/e3/settle/w107/drift-measured.patch` (drift = "measured" in v3.toml + rxq/engine.py table; stamp → 159591b2670f304b on HEAD).
- **-02 production (rxq, layers 2–3, 13 tensors, i8, in-sample calib H):** cur g = 2.050 (edge) on 13/13 at K0.125; with
  the table g = 1.73–1.85, none at the edge. Two-sided error tab / cur: K0.125 geo **0.9877**, 13/13 ≤ 1, worst 0.9997
  (q_proj 0.923); K0.5 geo **0.9816**, 13/13, worst 0.9981; one-sided agrees (0.9873 / 0.9835, 13/13). **Rule PASS → the
  drift table is proposed for adoption.**
- **-03 NTRY 4 (w102's setup, 27 × s0):** R4 = **0.9678 / 0.9859 / 0.9920** at K0.125 / 0.5 / 1 (NTRY 16: 0.9456 / 0.9812 /
  0.9917: NTRY 4 keeps 59% / 75% / 96% of its gain); CPU factor 3.90–3.93 per tensor. Card-0 level time with w102's
  shares (×4 GPU passes): K0.125 3.3 → ~12 h, K1 1.0 → ~3.0 h. Information only (ladder default stays NTRY 1).
- **-04 swj at sub-1 (#132-01, w100's rule; 0.8B, w106's K0.5 / K1 quants reused, today's rxq stack, g08-search1):**
  swj / base KL: K0.5 **0.9715 (z −22.7) / 1.0483 (z +35.6), geo 1.0092 → fails** (propose swj k_min above K0.5);
  K1 0.9610 / 0.9724, geo **0.9666 → keeps**. Both sides are the jobs' wq, i.e. before C9's sidecar (as d1103's stack, which
  had no C9). At K0.5 the 0.8B is near-random (KL 8.1–8.7, top-1 6–8%), so it is a weak proxy for the 27B at K0.5.
- **Runtime:** none (-03: quant time only).

### 2026-11-11 11:15 AM CST · #140 · C9 at low K (0.8B KL): no arm passes the rule, but the data show C9 hurts at K0.5 and helps from K1 (lead read; w106 driver)
- **Status:** done, rule verdict **no arm passes → C9 stays as today**; post-hoc reading below (proposal, needs a confirm)
  · **Machine:** gpubox card 0 (driver ran headless after the agent was stopped 14:43 UTC; one fix: PYTHONPATH for
  swiglufit's `common` import, 15:40 UTC) · **Data:** `v3/e3/settle/data/w106/report.md` (+ result.txt), gpubox `/data/w106/`
- KL ratio vs C9 as today (g08-search1, per-doc bootstrap), seeds 0 / 1:

  | K | C9 off | α(K) taper (ak) | e clip 0.3 (e3) |
  |---|---|---|---|
  | 0.5 | **0.966 / 0.925** | = off | 0.982 / 0.944 |
  | 1 | 1.028 / 1.041 | = off | **0.995 / 0.996** |
  | 1.5 | 1.075 / 1.077 | 1.022 / 1.028 | 0.999 / 0.999 |
  | 2 | 1.035 / 1.038 | = cur | 1.000 / 1.000 |

- Rule (≤ 0.99 at K0.5 **and** K1 on both seeds, K2 ≤ 1.003): every arm fails — off / ak win big at K0.5 but lose
  3–4% at K1; e3 wins everywhere below K2 but only 0.5% at K1. The proposed α(K) = α·clip(K−1, 0, 1) is **wrong**:
  it turns C9 off at K1 (−3…4%) and halves it at K1.5 (+2.2…2.8% worse).
- **Post-hoc reading (not a decision):** C9 should be OFF at K0.5 (−3.4 / −7.5%), ON from K1, and e clip 0.3 is a
  small free gain at K1–1.5 (−0.5% / −0.1%). Crossover between K0.5 and K1 unmeasured. Proposal: confirm on a fresh
  0.8B slice (g08-search2) with the arm "off at K < K*, e3 at K ≥ 1" at K0.25 / 0.5 / 0.75 / 0.875 / 1 / 1.5, rule
  written first. Affects only sub-1 / 1.x levels (built last).
- **Runtime:** none (C9 folds into svh).

### 2026-11-11 11:01 AM CST · #130-03 · P16 tile interleave, 27B dense-splice pair: geo 0.9933, both parts < 1 → PASS (−0.7%, z −0.95); v_proj f2 0.87 (report) (w101)
- **Status:** #130 (a) / (b) / (c) all PASS; adoption is the user's call after a timing (kernel note) · **Machine:** gpubox
  card 0, 11-11 14:03–16:52 UTC (waited ~35 min on card0.lock behind another job; 4 quant units 11–12 min each, packs
  ~10 s with `--reuse` + radmix ≤ 7 min, 8 scores 57–84 s) · **Code:** `v3/e3/settle/w101/il101c.sh` (+ `mksplice101.py`,
  `bytes101.py`, `verdict130c.py`) · **Rule:** `v3/e3/settle/w101.md` amendment 1 (0446e9b, before data) · **Data:**
  `v3/e3/settle/data/w101/c/` (verdict, docstat, per-doc sums, f2, census, gates)
- **Arms:** all 112 ssm_qkv / ssm_z / attn_qg at K2.5 i8, stamp 87a7e4ab (refit "hout", H_out blocks, #117 order, tile
  CD), natural vs P16 rows (P Wq un-permuted), dense bf16 into mix-base (seed 0) / mix-s1 (seed 1). Gates: rxq.pairs
  PAIR OK (787 weights, 675 identical, the 112 arm weights differ in the codes plane), BYTES101 PASS 112 / 112, stamp.
- **Result (golden-search27c, 91 docs, 145,597 positions):** N → P16 pooled **0.9930** [0.982, 1.005] / **0.9937**
  [0.983, 1.003], **geo 0.9933**; xcorp 0.9931 / 0.9941, calib **0.9904 / 0.9856**; combined SE 0.0070 (doc 0.0035 +
  #116 seed floor), z −0.95. Rule: geo ≤ 0.995 and both parts < 1 → **PASS**. Not significant alone: the rule's bar
  is a screen. Per document: 62–69% of docs better, per-doc geo 0.979 / 0.979; leave-one-out ≤ 0.996; without the two
  xcorp docs > 20% of their part 0.989 / 0.998. No confounder flag (size as predicted, same sign on every part and seed).
- **Realized f2 on all 112 tensors (report):** qkv 0.806, z 0.936, q_proj 0.881 vs model 0.787 / 0.922 / 0.857.
- **v_proj (amendment 2, report, (a)'s code):** f2 P16 / N L3 0.906, L11 0.976, L39 0.971, L63 0.657 (geo 0.87),
  = model per tensor. The first launch failed in every unit (my copy of il101.sh pointed at a fork without 0008; rxq
  refused, nothing written); fixed and rerun.
- **Slice:** golden-search27c, its last use (retire, R55).
- **Lead / user decide:** (1) whether to time the store-side remap (`w101/kernel-note130.md`: closed-form column map in
  exl3i_8a8 / nt / rate kernels + a per-tensor format bit; plan for prefill microbench + e2e prefill + decode), and
  whether to extend P16 to v_proj (and k_proj, model 0.95) in the same format bit; (2) a second-slice check before
  adoption (the four spliced mixes are kept on gpubox `/data/w101/c/mix/`, 4 × 17 GB; delete when not needed).
  B16 (h-fitted tiles) would cut the qkv model cost ~8% more than P16 (0.72 vs 0.78) but needs a per-tensor table, so P16 first.
- **Runtime:** prefill + decode if adopted (store remap, ~0 expected, untimed); none for the experiment.

### 2026-11-11 10:25 AM CST · #143 · local ladder runner: persistent workers +58%, N = 3 +11% more; outputs identical; ladder ≈ 3.2–3.7 card-days
- **Status:** done (measurement); runner built and tested, **not started** (freeze first) · **Machine:** gpubox card 0
  (5 bench arms + one driver run, ~23 min of card time in the gaps of w103 / w101 / w106), colo CPU (stand-in checks)
  · **Code:** `pipeline/rxrun/` (worker, pool, run, cmp, seedsd; uncommitted, for the lead to merge),
  `pipeline/tests/check_rxrun.py`, bench `v3/e3/settle/w105/bench105.{sh,py}`, configs `ladder-card0.json`
  · **Rule:** `v3/e3/settle/w105.md` (committed 0c6c7a0 as #140 before the first arm; renumbered #143: #140-142 were taken meanwhile) · **Data:** `v3/e3/settle/data/w105/`
- **Arms** (`q27v3-87a7e4ab-k2.5-i8`, layers 8-12 = 20 group jobs incl. L11 attention with tile CD and 5 down_proj,
  cold page cache, staged tree = 11-11 morning working tree, stamp 87a7e4ab, fork b0052f7c):

  | arm | wall | jobs/h | Σ tensor walls | GPU busy | VRAM peak | vs p1 |
  |---|---|---|---|---|---|---|
  | p1 proc N=1 (w92's way) | 342 s | 211 | 240 s | 60.5% | 11.2 GiB | ref |
  | w1 persistent N=1 | 216 s | 333 | 214 s | 93.6% | 11.3 GiB | 0 / 67 files differ |
  | w2 persistent N=2 | 199 s | 362 | 391 s | 97.8% | 17.8 GiB | 0 / 67 |
  | w3 persistent N=3 | 194 s | 370 | 551 s | 98.0% | 23.0 GiB | 0 / 67 |
  | p2 proc N=2 | 225 s | 320 | 381 s | 88.2% | 17.7 GiB | 0 / 67 |
- **Reading:** the overhead was process start + first-call warm-ups (p1: job walls − tensor walls = 102 s / 20 jobs;
  `scaled_ldl` / `finalize` first-use costs vanish in a warm worker, tensors 11% faster). Once warm the card is ~94%
  busy, so concurrency adds little: N=2 +8.7%, N=3 +11.1% over N=1; rule 2 (≥ 10%) picks **N = 3** (one run each:
  a thin margin; N = 2 is the fallback if card 0 must share VRAM). Model calibration: Σ est (rxq.ladder.job_req) 330 s
  vs p1 342 s (the model is the proc path's).
- **Identity:** whole-file sha256 can never match between runs: safetensors writes `__metadata__` from a Rust HashMap
  (random key order on every write, even in one process). rxrun.cmp compares tensor bytes / dtypes / shapes / offsets
  + the metadata dict. Every arm = p1 on all t- / c9- / diag- files; the full driver (`rxrun.run`, its own specs) on
  layers 8-9 = p1's files (26 / 26). w92's files: settings ed97adb7 ≠ 87a7e4ab, no spec matches (rule 4, n/a).
- **Projection** (50 levels, 12,800 jobs, 82.8 modeled R9700-h): N=3 rate 1.70 est-s/s → quant 49.4 h (+1.5% dups);
  overhead-split model (drop 5 s a job, tensors × 0.89, / 1.11) 52.7 h; + swj 50 × 0.75 h = 37.5 h →
  **3.6–3.8 card-days**, vs 5.2 the w92 way. swj is now ~42% of the card time: the next target. The K2.5 rate is
  applied to every K (sub-1 levels are trellis-bound, so the overhead share there is smaller: the split model covers it).
  **With today's settings** (working tree 11-11 ~10:30 AM CST, stamp 524bc868: swj k_max 4.5 per #129-02 → swj on 36
  of 50 levels = 27 h): **≈ 3.2–3.3 card-days**; if #142's sub-1 fusion (layer ×2.7) lands, the 7 sub-1 levels (25% of
  the modeled quant time) drop ~8.5 h more → ≈ 3.0.
- **Duplicate-seed jobs (user 11-11 item 5):** 1.5% of a level's tensors (min 1; ~6 a level, ~300 pairs), one tensor
  a job, seed_offset + 1, families in rotation across levels, layer random (seeded by stamp + level), stored in
  `<level>/dup/`, never assembled; `python -m rxrun.seedsd ROOT` → seed SD of ln proxy per family × K band with a
  bootstrap interval. First pair (L9 gate_proj, K2.5): codes differ, proxy differs by 6.5e-5 relative (SD 0.0046%).
- **Checks:** `check_rxrun.py` on colo (CPU stand-in): pool 6/6 (proc vs proc, persistent N=1 / 2, two devices on one
  queue, worker SIGKILL + resume, hold), run 5/5 (two levels assembled, dups, status + ETA, rerun skips in 1.4 s,
  seedsd).
- **Note:** the shared tree's settings stamp moved to 59a104cb during this session (uncommitted `quant.scale_clamp`,
  R14 #132): the runner takes whatever settings file it is pointed at; a stamp change needs a new ROOT.
- **Runtime:** ~23 min card 0, ~3 min colo CPU.

### 2026-11-11 9:35 AM CST · #142 · qtf2 sub-1 0-bit-step fusion: bit-identical (incl. a real K0.125 layer via rxq job on card 0), kernel ×3.8 (K0.125) to ×1.7 (K0.875), layer ×2.7; step 2 skipped (w108)
- **Status:** done (judge (a) PASS; patch not merged: lead) · **Machine:** dad1 (RX 9070 XT, gfx1201, `/data/w108`), gpubox card 0 · **Code:**
  `v3/e3/exl3frac2/sub1fuse/` (patch `qtf2-sub1fuse.patch`, not merged) · **Rule:** `v3/e3/exl3frac2/SUB1FUSE.md`,
  **committed alone (a99c06d) before any data** · **Data:** `v3/e3/exl3frac2/data/sub1fuse/`
- **Question:** LOG #37's "not done" item: fuse the KA = 0 kernel's 0-bit steps (a sub-1 state only moves at a
  branching position) so the per-position global cost traffic and barriers go; identical output → pure speed change.
- **Setup:** HEAD's `v3/e3/exl3frac2` (ref) vs the patched `qtf2.cuh`, both built from the fork at b0052f7c; every KA = 0
  instance (K0.125-0.875), i8 and mul1; qtf2check's tile set + a second Gaussian set (2,456 tiles).
- **What changed:** each thread keeps 128 states' costs in registers between branching positions (bit 15 local, so the
  (0, 1) min is in registers); codebook reads from a 1,024-entry byte-sum table in LDS (the mul1 decode depends only on
  the product's byte sum, so the table, filled with `decode_mul1_product_2` itself, is exact for every codebook build);
  lanes on state bits 10-14 (LDS bank conflicts of the table reads ~2× lower than lanes 8 states apart, measured 2×);
  the 15-bit cost / history scratch in a swizzled layout (vector + coalesced for writer and reader; the traceback reads
  it through `qtf2_adr0`); per-MASK step kinds only (registers). Same half arithmetic in the same order per state.
- **Result:** (a) **PASS**: 0 differing indices / values, both codebooks, all seven sub-1 rates, LUT 1 / 0, batches 32 /
  128; KA ≥ 1 instances identical; 4 runs deterministic; the `QTF2_NOFUSE` build = HEAD. Speed (i8, batch 128 =
  production, tiles/s): K0.125 12.0k → 45.3k (×3.77), K0.25 ×3.26, K0.375 ×2.89, K0.5 14.1k → 36.5k (×2.60), K0.625
  ×2.11, K0.75 ×1.80, K0.875 16.5k → 28.3k (×1.71); mul1 alike; at batch 32 ×6.1 … ×2.6. Step 2 (macro-step trellis):
  global traffic + barriers are 5.0% of K0.125's kernel time (timing build `QTF2_F0NOGLOB`) < the rule's 30% → skipped;
  what is left is the per-position codebook read of every state, which macro-steps would still need.
- **Real layer (gpubox card 0, 9:38-9:43 AM CST):** `rxq job`, settings 524bc868, K0.125 i8, layer 10 (4 group jobs, 6
  tensors), HEAD's tree vs the patched one: **all 6 tensor files bit-identical**; tensor seconds 167.6 → 62.0 (×2.70),
  job wall 186 → 82 s (×2.27); implied R9700 kernel ratio ×3.3 at K0.125 (dad1 ×3.8).
- **Projection (R9700-calibrated):** the seven KA = 0 levels' job wall 21.8 → 12.2 card-h (K0.125 3.53 → 1.48 h, K0.5
  3.11 → 1.69 h, K0.875 2.73 → 2.01 h); trellis at K ≠ 0.125 scaled from PROFILE.md §0 by the old kernel's tile rate.
- **Decision / next:** ships as a pure speed change if the lead merges it (judge (a)): settings stamp unchanged; the
  code stamp changes automatically (it hashes `v3/e3/exl3frac2/*.*`; note: so does the rule file SUB1FUSE.md committed
  there, a99c06d). CUDA build of the new path not compiled (no nvcc here, 4070 off limits).
- **Runtime:** none (quantizer speed only; same codes).
- **Files:** `v3/e3/exl3frac2/data/sub1fuse/README.md` (tables, development path), `dad1-*.jsonl`, `gpubox/`.

### 2026-11-11 8:15 AM CST · #141 · sub-1 follow-ups of #132 / #136: drift table at every sub-1 K, g recheck on production, NTRY 4, swj at sub-1 (w107): rule committed, queues running
- **Status:** running · **Machine:** Thor + colo (-01 / -03, q87 runners, single-threaded, nice 19; colo ≤ 8 GB of ours,
  no ffn_down), gpubox card 0 (-02 now, -04 after w106's card steps; card0.lock → ram.lock shared) · **Code:**
  `v3/e3/settle/w107/` · **Rule:** `v3/e3/settle/w107.md`, **committed alone (69cbd3e) before any data** · **Data:**
  `v3/e3/settle/data/w107/` (when done)
- **Question:** (-01) measured LDLQ drift D(K) at K0.25 / 0.375 / 0.625 / 0.75 / 0.875 (w100's proxy, same 27 × 2) and
  what a measured table buys held-out at all eight sub-1 K; does log-linear interp of w100's three points (1.348 / 1.184 /
  1.089) predict the new ones within 1%? (-02) on the real quantizer (layers 2–3, K0.125 / 0.5), does the table move g
  off the 2.05 edge and lower the error on ≥ 80% of tensors with none worse by > 0.5%? (-03) NTRY 4 ratio and time
  (information for a per-tensor knapsack option). (-04) #132-01 on today's 0.8B stack, w100's rule unchanged.
- **Setup:** settings stamp 87a7e4ab (committed v3.toml; the lead's uncommitted `scale_clamp` / swj k_max edits are not
  in these runs). G1 (Thor): the cur re-run reproduces w100's rows (same box bit-identical; cross-box rel2_B 3e-8, d_all
  8.5e-10 relative).
- **Runtime:** none (-03: quant time only).
- **9:30 AM CST, -01 phase a done (all 54 cur rows at the five new K):** measured table written before any tab row of
  the new K (`data/w107/gates/table.json`): D = 1.3425 / 1.2452 / 1.1425 / 1.1137 / 1.0922 at K0.25 / 0.375 / 0.625 / 0.75 /
  0.875 → T = 1.342 / 1.245 / 1.142 / 1.114 / 1.092. Colo's leftover jobs moved to Thor (colo runner stopped; its rows kept).

### 2026-11-11 8:14 AM CST · #130 / #131 · P16 tile interleave: (a) f2 0.68 on qkv and (b) held out kept 0.98, both PASS → 27B pair (c) queued on card 0; MLP / k_proj gain nothing in the model; per-section K FAILS (−0.02 bit) (w101)
- **Status:** #130-01 PASS, #130-02 PASS, #130-03 running (queued on card0.lock), amendment 2 (v_proj f2) queued · #131
  **FAIL** · **Machine:** gpubox card 0 ((a) 11-09 03:56–05:14 UTC; (c) / v_proj from 11-11 14:03 UTC), colo ((b), #131
  queue 11-09 04:04–09:24 UTC, model extras 11-11) · **Code:** `v3/e3/settle/w101/` (README) · **Rule:**
  `v3/e3/settle/w101.md` (acf54f2; amendment 1 = (c) recipe 0446e9b; amendment 2 = v_proj f2 80016b2; each before its
  data) · **Data:** `v3/e3/settle/data/w101/`. Read on 11-11 after the 11-09 stop: every 11-09 job had finished, so
  nothing was rerun.
- **#130-01 (a), real rxq two-sided f2, P16 / natural** (geo over K2.5 / K4 × seeds 0 / 1; stamp ed97adb7):
  L10 in_proj_qkv **0.678** (model 0.658), L11 q_proj 0.825, L39 q_proj 0.926, L10 in_proj_z 0.949. Seeds agree to
  ±0.002. The unweighted f1 rises (qkv ×1.17), as expected for an H_out-driven gain. Pass (≤ 0.90); scope of (c)
  = qkv + q_proj (geo 0.874 ≤ 0.95) + z (0.949 ≤ 0.95).
- **#130-02 (b), stability over Σdy² block halves** (112 tensors, G-b 6e-8): P16 cost ratio in → out of sample qkv
  0.779 → 0.784 (**kept 0.977**), z 0.918 → 0.922 (0.959), q_proj 0.851 → 0.857 (0.960); head-level Spearman ρ
  0.97 / 0.95 / 0.97. Pass (≥ 0.75). B16 (h-fitted balanced tiles, report only) 0.72 on qkv, also stable.
- **Lead note (where else):** model P16 / N, in-sample bn1920: v_proj 0.923, k_proj 0.952, mlp gate 1.000, up 1.000,
  down's input side (π on its columns, 4 layers) **1.007, worse** → the MLP gets nothing (no kernel change would be
  needed there, but there is no gain). v_proj ≤ 0.95 → real f2 per amendment 2 (L3 / L11 / L39 / L63, report only).
- **#130-03 (c):** 112 tensors (48 qkv, 48 z, 16 q_proj) at K2.5, N vs P16, seeds 0 / 1, stamp 87a7e4ab (refit
  "hout"), dense bf16 splice into mix-base / mix-s1, gates (rxq.pairs, bytes, stamp), golden-search27c (its last use,
  R55; docstat per-document view printed). Pass: geo ≤ 0.995, both parts < 1. Kernel / timing note written
  (`w101/kernel-note130.md`): store-side remap is closed-form (16-column tiles, `uint4` stores stay whole), ~4 int ops
  per store in exl3i_8a8 / nt / rate kernels, plus a format bit; a separate gather costs ~0.1 ms per qkv call at T 2048.
- **#131, per-section K at equal bytes** (w87 two-sided block proxy, 8 tensors × B2.5 / B4, chosen on one block half,
  scored on the other): mean held-out gain **−0.018 bit** (attn_qg −0.003 / +0.003, ssm_qkv −0.037 / −0.034) vs ≥ 0.10
  → **FAIL**, no 27B pair. In-sample +0.05–0.13 bit does not survive. Section boundaries lie on 128-row blocks (asserted).
- **Runtime:** #131 none. #130: if adopted, prefill + decode (store-side un-permute in every kernel writing ssm_qkv /
  ssm_z / attn_qg; ~0 expected for the store remap, not yet timed; timing plan in the kernel note).

### 2026-11-11 8:25 AM CST · #129-02 · swj at K6 on 2× the documents: cap swj at k_max = 4.5 (by the rule) · R14 scale clamp adopted
- **#129-02:** `mix-k6{c9,swj}` (same K6 codes, svh only) scored on golden-search27b x + c (w91 builds; ~4 min card 0),
  pooled with #129's 27c rows (`v3/e3/settle/data/w99/s129b.txt`, `eval/docstat/docstat.py`). 192 docs: pooled r
  **1.026** [0.925, 1.038], per-doc geo 0.989, median 0.989, 57% of docs better; xcorp 1.026 (one doc 77% of the
  KL; without it 0.994 [0.881, 1.046]); calib pooled 0.982 [0.953, 1.003] but per-doc geo **1.021** (47% better). On 27b
  alone the heavy doc is swe-rebench-openhands (73%, r 1.089), on 27c smoltalk2/longalign: each slice's xcorp part is one
  document in pooled terms. Rule (w99.md #129-02, before data): pooled ≥ 1.00 → **cap**. No reading shows a gain worth
  keeping; v3.toml `post.swj.k_max` 8.0 → **4.5** (~0.75 card-h saved per level above K4.5, ~15–18 card-h per ladder).
- **R14 clamp (#132, user 11-11):** `quant.scale_clamp = true`: after the refit, |su| / |sv| ≥ max(f16 min normal,
  1e-3 · median) per vector, else the pre-refit value, else the floor (sign kept); Qf, the logged proxy and every refit
  arm follow the clamp (`pipeline/rxq/refit.py` clamp_scales). On the real w92 K0.125 codes (3 tensors with values under
  the floor, 49 values) the one- / two-sided proxy changes by **≤ 4e-6 relative** (`v3/e3/settle/r14/r14.jsonl`,
  `r14chk.py`). First check_diag run caught a stale logged proxy and unclamped arms when the clamp fires (random test
  tensors), both fixed; checks re-running on colo.
- **Settings stamp → `524bc8684c7fda53`** (swj k_max 4.5 + scale_clamp). Checks on colo: check_diag 20/20, check_cpujob 25/25, check_ladder 5/5 (rerun with the final v3.toml). User 11-11: agrees with the cap ("barely has an impact if at all").
- **Runtime:** none (swj scales fold into svh; the clamp changes stored values only).

### 2026-11-11 8:05 AM CST · #140 · C9 at low K on the 0.8B (off / today / α(K) / e clip 0.3), KL: rule committed, infra rebuild on gpubox (w106)
- **Status:** running · **Machine:** gpubox card 0 (`/root/card0.lock` → `ram.lock` shared, behind other card-0 jobs;
  the 4070 is out) · **Code:** `v3/e3/settle/w106/` · **Rule:** `v3/e3/settle/w106.md`, **committed alone (a139199)
  before any data** · **Data:** `v3/e3/settle/data/w106/` (when done)
- **Question:** #138-01 / R15: C9's c = (1−e)^(−α/2) has no K schedule (c up to 2.83 at gdn_out α 3). Does α(K) =
  α·clip(K−1, 0, 1), or e clipped at 0.3, beat C9 as today by ≥ 1% at K0.5 and K1 (both seeds) without costing > 0.3% at K2?
- **Setup:** today's settings via `rxq quant` on the 0.8B (stamp 87a7e4ab, block order off: the 27B's order), H_out c18 dy,
  swj (g08-fit1); K0.5 / 1 / 1.5 / 2 × seeds 0 / 1; arms on fixed codes (writer row multipliers only), scored on g08-search1.
  The 4070's 0.8B files are unreachable: h08 and the two goldens are rebuilt on card 0 (same documents), c18 copied from colo.
- **Runtime:** none (C9 folds into svh).

### 2026-11-11 8:05 AM CST · #136-01, -03, -06 · NTRY 16 passes at every sub-1 level (−5.4 / −1.9 / −0.8%) and chains pass at K0.125 (−10.7%); tile CD helps at K6 and K1.5; card-0 rxq is reproducible (tensors bit-identical; only the safetensors header key order varies) (w102)
- **Status:** done (all of #136 now read) · **Machine:** colo + Thor (-01, 81 CPU jobs, 11-09 ~04:15–06:30 UTC), gpubox
  card 0 (-03 / -06, 11-09 04:03–05:11 UTC, under card0.lock + ram.lock shared) · **Code:** `v3/e3/settle/w102/`
  (`job102.py d3` + `ring102.c`, `report102.py`, `cd102.sh` + `f2cd102.py`, `det102.sh`) · **Rule:** `v3/e3/settle/w102.md`
  (136d89b; amendment 1 for -06 e3f5653, before data) · **Data:** `v3/e3/settle/data/w102/` (`d3/`, `report102.txt`,
  `cd/`, `det/`). Read on 11-11 after the 11-09 stop (all jobs had finished; nothing rerun).
- **#136-01 (D-3, tail-biting search width / chained rings at sub-1, w87's two-sided held-out proxy, 27 tensors × s0, i8):**
  G0 PASS (arm F = w95's stored rows to 3.3e-7). Per K, geo over 27 tensors with the tensor-bootstrap 95% CI:

  | K | T16 / T1 (NTRY 16 vs 1) | O4 / O1 (chain of 4 vs 1 ring) | T1 / F (tail-biting cost vs free start) | O4 / T16 |
  |---|---|---|---|---|
  | 0.125 | **0.9456** [0.936, 0.954] PASS | **0.8930** [0.880, 0.905] PASS | 1.283 | 0.992 |
  | 0.5 | **0.9812** [0.978, 0.985] PASS | 0.9831 fail | 1.105 | 1.073 |
  | 1 | **0.9917** [0.988, 0.994] PASS | 0.9909 fail | 1.074 | 1.109 |

  Bars: NTRY ≤ 0.995, chain ≤ 0.95. All 9 families gain at every K (K0.125 0.90–0.96, K1 0.981–0.996). B1, B2 and in-sample A
  agree to ±0.005.
  - **Quant-time cost of NTRY 16:** the CPU search takes ×15.7–16.1 per tensor (T16 846 / 1,458 / 1,566 s vs T1 54 / 92 / 97 s, median).
    On the GPU the 2 trellis passes per tile become 32. In w95's real card-0 levels the trellis stages are **89% of
    wall at K0.125** (400 tensors, 11,720 s summed) and **67% at K1** (3,731 s). Scaling the whole trellis share by
    16 gives an upper bound of ~47 h for K0.125 and ~11 h for K1 per level, vs 3.3 h / 1.0 h today.
  - **Reading:** this is encoder only (format, decoders and bpw unchanged), but at that price NTRY 16 is not cheap. The
    lead decides: (a) NTRY 16 only at K0.125, where it buys 5.4%; (b) a smaller NTRY (4 was not measured); (c) chained
    rings at K0.125, which match NTRY 16 there (O4 / T16 0.99) at ~1× quant time but need the #102 decoder change in
    every kernel.
  - **Side finding for every sub-1 proxy screen:** the free-start search used by #90 / #126 / w87 / w100 understates
    production's tail-biting error by **28% / 10.5% / 7.4%** at K0.125 / 0.5 / 1 (T1 / F). Ratios between arms are
    probably fine, but absolute sub-1 proxy numbers are optimistic.
- **#136-03 (A-12, tile CD no-harm at the ladder ends, real rxq, w81's f2, q_proj of L3 / 23 / 43 / 63):** f2_cd / f2_off
  **K6 0.9973 (0.9962–0.9982), K1.5 0.9958 (0.9915–0.9979)**: CD helps at both ends → **PASS, CD stays at every
  level**. The f1 ratios are 1.001 / 1.003 (unweighted error, which CD does not optimise). In-job tilecd ratios are
  0.995–0.998 (K6) and 0.991–0.997 (K1.5); cd1 accepts 3.4–5.6% of tiles; CD takes 12 s (K6) / 19 s (K1.5) per tensor.
- **#136-06 (GPU repeat-run determinism, w92's fs0 L03-g0 / L02-g0 / L02-g2 rerun on card 0 with w92's staged code):**
  **all 7 tensor files are bit-identical in every tensor** (q / k / v with tile CD + blocks H_out, GDN qkv / z, MLP
  gate / up; `torch.equal` on every tensor, with the same offsets, dtypes and shapes), and the metadata values are equal. tilecd.jsonl is
  identical; log.jsonl differs only in timings and maxrss.
  - **But a byte `cmp` of the files fails:** safetensors writes `__metadata__` in a per-process order (rxq builds the
    dict in a fixed order, `pipeline/rxq/job.py:86`; the serializer reorders it). So **a reproducibility or
    verification check (e.g. rqah) must compare parsed tensors and metadata, or hash the data section, never whole
    file bytes or file sha.**
  - The 11-02 race does not reproduce on the V3 path.
- **Runtime:** none for -03 / -06. -01: none if NTRY only (quant time ×~16 in the tile search at the chosen levels);
  chains = decoder change in every kernel.

### 2026-11-11 8:00 AM CST · #132, #133 · Sub-1 hygiene: C9, drift, g_scale edge and scan rules fire, zero floor stands; H_out shape: σ_out 0.01, p 1 stands (1 / 36 cells, edge-flagged) (w100)
- **Status:** done (#132: proposals for the lead; #132-01 swj at sub-1 written, not run) / null (#133) · **Machine:** colo +
  Thor (54 tensor × samples × {sub1, g125, hshape}, all finished 11-09 before the stop; nothing rerun), gpubox read-only scan ·
  **Code:** `v3/e3/settle/w100/` (`report100.py` + amendment-1 / production-log / scan-verdict sections, added after data,
  report-only) · **Rule:** `v3/e3/settle/w100.md` (372b555 / 7028b0c / e7561d3) · **Data:** `v3/e3/settle/data/w100/`
  (`report.md`, `report.json`, `out/{colo,thor}`, `gates/`, `real/`, `scan/`, `w92logs/`)
- **Gates:** G1 / G2 / G3 PASS (11-09 entry; G3 max 1.7e-10).
- **#132-1 C9 at sub-1: FIRES.** gdn_out (α 3) c_mean **2.78 / 1.95 / 1.31** at K0.125 / 0.5 / 1 (95% of rows at the e clip
  at K0.125); attn_o (α −1) 0.725 / 0.851 / 0.938 (fires at K0.125); real w92 i4 codes agree (2.81 / 1.47 at K0.125 / 0.875).
  **Proposal: `post.c9` α(K) = α · clip(K − 1, 0, 1)** (C9 off at K ≤ 1). K1 gdn_out (1.310) is marginal.
- **#132-2 g_scale edge: FIRES on the production logs, not on the proxy.** w92 K0.125: 13 / 13 at g 2.05; proxy 0 / 54
  at any edge (picks 1.63–1.66; ext arm changes nothing). But amendment 1's fixed-g arms (K0.125, i8, 54 tensor × samples)
  say a higher g is **worse**: held-out rel2_B vs the proxy's pick ×1.033 (g 2.05), ×1.076 (2.6), ×1.109 (3.3), 0 / 54
  better. The proposal as written is the top grid to 3.9, but this evidence says **don't widen it**. Production's K0.125
  search sits at the edge for its own reasons (drift 1.08 vs measured ~1.35, see 3). Suggest a production-side check of
  K0.125 at g ≈ 1.65 instead.
- **#132-3 drift: FIRES at K0.125 and K1.** Measured geo d_all **1.348** (K0.125; inputs 1.44, rest 1.24), 1.184 (K0.5:
  within 1.0–1.2, no fire), **1.089** (K1, +0.85% vs 1.08). Per-tensor measured-drift arm, held-out rel2_B vs cur: 0.970 /
  0.979 / 0.997 (all ≤ 1). **Proposal: drift table 1.35 at K0.125, 1.09 at K1** (to be checked against `drift = "interp"`
  between table points). The arm used each tensor's own drift (an oracle), so the table value will buy somewhat less.
- **#132-4 zero floor: stands.** Max rel at K0.125: real codes 0.788 (L3 up_proj), proxy 0.690 (L59 attn_v); nothing near 0.97.
- **#132-5 suh / svh scan: FIRES (one value).** 71 dirs, 28,160 scale tensors: 0 zeros, 0 non-finite, 93 f16 subnormals
  (92 at K0.125: w92 L3 q_proj / o_proj suh, down to 3.6e-7; 1 at K1: a4cost i4 L17 out_proj suh 5.5e-5, median 1.5e-2).
  **Proposal: R14 clamp** (refit scales ≥ max(f16 min normal, 1e-3 · median)). The K1 hit is a retired i4 level.
- **#132-01 swj at sub-1:** `v3/e3/settle/w100/swj4070.sh` written for the 4070, not run. The 4070 is now the user's,
  so it would need porting to gpubox card 0 (lead's call).
- **#133 H_out shape: null.** No "all" cell passes at any K. Best-picked held-out gains: K0.5 +0.40% (σ_out 0.1, p 1:
  edge), K2.5 +0.05%, K4 +0.06% (σ_out 0.03, p 1). p 0.5 is worse everywhere (+0.6% at K0.5, +2.3% at K2.5 / 4); p 0.75
  is worse at K ≥ 2.5. One family cell passes: **K0.5 ssm_z, σ_out 0.1, p 0.75: g +2.16%, 3 / 3 agree, swap +1.88%**. It
  sits on the grid edge, so it is flagged (a candidate override that needs a wider grid), not proposed. **σ_out 0.01, p 1
  stands.** The half collection costs 1–2% vs the full bn1920 (full / cur on (B, O): 0.990 / 0.983 / 0.983), so the
  quantizer's own data could favour more smoothing than production wants.
- **Caveat:** the proxy predates 87a7e4ab (refit "hout", drift "interp"); it uses w93's per-output-channel refit.
- **Runtime:** none.

### 2026-11-11 7:55 AM CST · #128 / #129 · per-document re-read (diagnostic, post hoc): one xcorp document holds 85% of #129's xcorp KL
- **Status:** diagnostic (no decision changed by it; #129-02 still decides) · **Machine:** gpubox CPU · **Code:**
  `eval/docstat/docstat.py` (new: pooled / per-doc geo / median / KL shares / > 20% exclusion / leave-one-out / drop-k
  curve, document bootstrap) · **Data:** `data/docstat/w99-docstat.txt`
- **#129 (k6c9 → k6swj):** smoltalk2/longalign (the non-breaking-space PDF text) holds **84.9% of the xcorp KL** (r 1.035).
  Without it: xcorp r **0.963** [0.824, 1.006]; calib pooled 0.975 [0.942, 1.001] (no calib doc > 20%). Per-doc: xcorp
  geo 0.973 / median 0.982 (62% of docs better), calib geo **1.047** / median 0.988 (54% better); all 91: geo 0.993,
  median 0.984. Drop-k curve is non-monotone (k3 0.876, k10 1.060): noise dominates. Reading: once the outlier is out,
  swj at K6 leans ~3% better pooled but nothing is significant and calib's per-doc geo disagrees → #129-02's extra
  documents are still needed.
- **#128 (hp → hb, H_out off → on):** per-doc geo **0.980 / 0.951**, median **0.947 / 0.946**, 65–67% of docs better,
  calib pooled 0.939 / 0.953 (z −2.6 / −2.3). The inconclusive pooled 1.0054 (seed 0) is longalign (54% share, r 1.000) +
  swe-rebench (23%, r 1.027). Supports the user's call (H_out everywhere).
- **Slice note:** golden-search27c's xcorp part is effectively one document in pooled terms; retire it (already due,
  R55) and cap per-document KL share in the next two-part builds (w92's proposal).
- **Runtime:** none.

### 2026-11-11 7:40 AM CST · #128 / #129 · user decisions: H_out stays at every level; #129 gets more documents
- **#128:** user: keep H_out at all levels (it helped in both uniform-K tests; the splice was inconclusive, not
  negative). No config change (apply_k_min 0, apply_k_max 8).
- **#129:** user asked whether more data makes it significant. Rule #129-02 (`v3/e3/settle/w99.md`, committed before
  data): same two K6 mixes scored on golden-search27b too (~2× documents), pooled + per-document statistics; queued for
  card 0 (both cards in the user's radiance today).
- **Runtime:** none.

### 2026-11-10 2:24 AM CST · #— · Overfit audit of 11-08 → 11-10 (user cadence; run while DS d100 grinds, VM-local, no cards touched)
- **Status:** done · **Machine:** VM (ledger + LOG mining only) · **Runtime:** none
- Decisions since the 11-08 audit and the slices that made them: head Fisher/LDLQ (#124/125, w94: decided on
  `w94-headx-search` select-role, fit on `w94-headx-qad` — correct pools), #136/#137/#138/#139 + #130-133 held-out
  two-sided proxy calls (0.8B/2B proxies on their own held-out draws, rules committed only after both parts agreed),
  #122 K2.5 stack donor gate (xcorp + calib-search part), #126 i8 switch (held-out two-sided proxy, PASS), the
  11-09 freeze edit (user decision, config-only). Nothing was fitted or picked on the verify pool since the audit:
  the only verify-pool `select` rows in the ledger are 2026-10 legacy (golden-ver1/ver2, pre-pools); golden-verify27
  is check-role only, 11-06, and its entry already declares "5 routine uses: no more on it".
- Rules-after-data: none found (each entry pre-registers or states its rule before the arms' numbers).
- Adopted settings vs verify: H_out, tile CD, gain-1.0, permutation all have verify rows or explicit
  report-only framing (#117 permutation decided on the fresh two-part search slice with parts' sign agreement,
  honestly flagged ~−0.3 % not significant, adoption left to the user). **i8-everywhere has no 27B verify row** —
  a proper check needs two full ladder-adjacent 27B quants; deferred by design: the first ladder mixes will carry
  i8 codebooks and get their routine verify scores as they are built (logged here as the standing action).
- Public: never used — no new golden since the audit is cut from public (all six new goldens checked by corpus
  hash: calib-pool c parts + xcorp x parts); public rows in this period: none.
- Slice rotation: `golden-search27` retired (824 decisions, single-part, pre-rule — its re-use on 11-08 was
  report-only); active search slices are two-part `golden-search27b/c` builds with fresh xcorp redraws
  (xc8k-b1 → xc8k-b2-c2 between w90/91 and w92 ✓); next 27B search round draws a new x batch + fresh calib
  search docs per the design doc.
- Ledger gap found and fixed process-wise: w90/w91/w92/w94 goldens were added (ledger commit 11-09), but no
  whole-calib collections were logged this period (none ran). Action: worker reports must call ledger_add.py in
  the same turn as the result (was already the rule; reminder added to the watchdog checklist).

### 2026-11-09 12:17 PM CST · freeze edit · int8 everywhere: the A16 / bf16 options and the mul1 codebook are retired (user + deadcode decision)
- **Status:** done (decision, applied in the freeze edit; settings stamp → **87a7e4ab32f9d5b9**) · **Machine:** — (config +
  solver-menu edits; checks 11-09 12:40 PM CST on the VM niced: **check_ladder 5/5, check_knap 3/3 (w63 byte-repro
  intact), check_cpujob 25/25 with K6.5-mul1 swapped for K7.5-i8 — the real fork quantizes i8 at K7.5 fine —,
  check_diag 20/20, rqah rxq_e2e running**) · **Code:** `pipeline/config/v3.toml`, `pipeline/rxq/{knap,config,ladder}.py`,
  `v3/levels/reprice/knap5.py`, tests `check_ladder` / `check_cpujob`
- **Decision (user, after the conversation with Code 000 / deadcode):** V3 serves **int8 activations only**: the accurate
  (f16 / A16) variant, the per-tensor f16-prefill and bf16-tensor options in the knapsack menu, and the mul1 codebook are
  retired — one kernel path, simpler long-term maintenance, and the accuracy edge is invisible at the sizes that matter.
  Evidence read for it: A8's KLD cost ×1.0014 at K2.5 full stack (#122 w92); fast sizes were already priced all-A8 (w58);
  the A16-only region was mul1's K6.5–8 byte advantage alone.
- **Applied:** v3.toml `[quant.codebook]` mul1_levels = [], i8 by_level to K8 (**user 12:10 PM CST: "i8 K7, K7.5, K8"** —
  the #94 penalties accepted: ×1.27 K7 (~0.18 bit), ×2.33 K7.5, ~×4.5 K8; knap5 R8 table extended with the w45 medians
  7.25 1.69 / 7.5 2.33 / 7.75 3.23 / 8 4.5); `[ladder]` i8_extra = [7.5, 8.0] → **ladder 55 → 50 levels** (pass 1: 20);
  `[knap] modes = "a8"` → rxq.knap passes KN_MODES=a8, knap5's menu drops the f16 / bf16 rows except the **head's f16
  rows** (headdec / slab serve the head, it was never in the trunk tile; the bf16 head option is out too — #99's floor
  split can run the legacy arm). --accurate in rxq.knap stays only to reproduce w63 byte for byte (check_knap).
- **Consequences:** one mix per size (no accurate/fast pairs, no per-mix mode mixing — the planned A8/A16 switch-cost
  test is closed); an a8-only smoke solve fits 2.5 / 4.0 / 7.0 and picks i8 K7.5 / K8 at 7.0 (head ours-K8, no bf16, all
  trunk flops a8); an 8.0 size (#99) now means i8 K8 + slab head; kernel work stays i8a8-only; decode-side wording keeps
  the f16 dequant GEMV (memory-bound, not an A16 option).

### 2026-11-09 5:27 AM CST · #126 · card-0 smoke (step 3): both i8 levels run end-to-end in radiance, SMOKE OK (w95)
- **Status:** done (**pass**: proxy + smoke) · **Machine:** gpubox card 0 (after w92's `W92 DONE` 11:23 PM 11-08) · **Rule:**
  `v3/e3/settle/w95.md` (5b7caff, before data) · **Code:** `v3/e3/settle/w95/` (`w95.sh`, `smoke95.py`) on the patched
  tree, stamp **80b5e6d77ef57fc6** · **Data:** `v3/e3/settle/data/w95/` (`smoke95.txt`, `result.txt`,
  `smoke95-scores-and-recipes.txt`; full score rows on gpubox `/data/w95/scores/`)
- Config change (delivered as `local/workers/w95/i8-sub1.patch`, applied in the staged worktree only): `[quant.codebook]`
  by_level = one i8 row K0.125–7, `i4_twins = []`, `[ladder]` low series from the codebook; rqah `tests/rxq_e2e.py`
  0.5-i8 (`rqah-i8-sub1.patch`). CPU stand-ins on the patched tree: check_cpujob 25/25, check_ladder 5/5, rxq_e2e 9/9.
- Run (256 jobs/level, assembled 80b5e6d703ae4dd9, 64 shards, 64 C9 writers; SWJAPPLY post; rad-convert packs rc=0;
  KLD on golden-search27c, both parts, score only, no gate):
  - **K1 i8:** 4,758 s; mix 1.2677 bpw; x KLD **2.23959** (top-1 0.5741, p99 23.163), c KLD **1.17926** (top-1 0.6305,
    p99 8.152); pooled **1.86254**.
  - **K0.125 i8:** 12,724 s; mix 0.4763 bpw; x KLD **11.68833** (top-1 0.0295, p99 22.501), c KLD **11.55326** (top-1
    0.0415, p99 22.751); pooled **11.64030**.
  All rows rc=0, KL finite. Both mixes census PASS: trunk exactly `336 exl3i K mul1i8` at rate 1 / 0.125 (rest of the
  mix unchanged; no i4 anywhere). **SMOKE OK.**
- Note: the first launch's card-0 wait fired 10:07 PM in a w92 restart gap (the wait only counted the driver gone ≥ 10
  min); stopped 10:10, `w95.sh` fixed (also require `W92 DONE`/`VERDICT` in w92's result tail, or 60 min gone) and
  relaunched 10:10:36 PM — that wait fired correctly at 11:32 PM; no quantized work was lost (jobs skip finished work).
- **Runtime:** none (mul1i8 decodes through the same exl3i f16-activation path as i4; one codebook removes the i4 code
  paths — `local/workers/w95/dead-i4.md`).
- **Not committed (lead's no-commit rule, rule files only):** this entry, the INDEX row, `v3/e3/settle/w95.md` results,
  `v3/e3/settle/w95/`, `v3/e3/settle/data/w95/` — for the lead's batch commit. Worktrees to remove on merge:
  `/workspace/wt/w95-rxm`, `/workspace/wt/w95-rqah`.

### 2026-11-09 12:25 AM CST · #138-03 · H_out-weighted refit held out: the gain holds → rule says PROPOSE ADOPTION (user decides)
- **Status:** done (positive by the #138-03 rule) · **Machine:** gpubox card 0 (lead-run, ~15 min) · **Rule:**
  `v3/e3/settle/w98.md` #138-03 (5e3aa34, before data) · **Code:** `v3/e3/settle/w98/w98h.sh` · **Data:**
  `v3/e3/settle/data/w98/report138-03.txt` (gpubox `/data/w98h/`)
- Every job quantized and refit with the EVEN-block Σdy² factor (1,024 seqs, block-normalized, `mkhalf.py`); f2 scored
  with the ODD-block factor (896 seqs). c1 / c2 fail as expected (different factor than w92 / than the in-job f2); c3 OK.
  report98's own line applies #138-01's old bar (T-geo ≤ 0.990); the #138-03 rule is read here.
- **hout arm, held out, all 39 cells:** geo **0.99907**, 87.2% of cells < 1, max 1.0000 → **PROPOSE** (rule: geo < 1,
  ≥ 80% < 1, no cell > 1.002). Per K: K0.5 0.99764, K2.5 0.99970, K4 0.99987. Targets (qkv, z, qg): T-geo **0.99722
  held out vs 0.99678 in sample**: the gain is real, not fitting the statistic. hout+undamped 0.99692 (not the
  pre-registered arm).
- **Size:** tiny: ~0.03% proxy at K2.5 overall (0.1% on the targets), 0.24% at K0.5; KL gains are usually smaller
  than proxy gains. Zero runtime cost; a quantizer change (refit mode = hout) and a settings-stamp change.
- **Decision (user, 12:30 AM CST): ADOPT** `refit = hout`, folded into the freeze re-stamp with i8 sub-1.
- **Runtime:** none.

### 2026-11-09 12:20 AM CST · #138-01 · Scale polish in the right metric: no arm passes, `refit = true` stays; C9 at K0.5 raises the two-sided proxy ×2.7 (w98's card-0 screen, read by the lead)
- **Status:** #138-01 negative · #138-02 (27B pair) not run (conditional) · #139 instrumentation: files measured, not
  wired · **Machine:** gpubox card 0 · **Rule:** `v3/e3/settle/w98.md` (7f4b421, before data) · **Data:**
  `v3/e3/settle/data/w98/report138.txt` (gpubox `/data/w98/`). w98 (Opus) was stopped at 04:15 UTC for weekly usage
  after this step; the lead read its report file.
- 13 tensors (L10 GDN + MLP, L39 attention + MLP) × K0.5 / 2.5 / 4, fixed codes, four refit arms, w81's two-sided f2.
  Checks c1–c3 OK (no-op identical to w92 fs0; in-job f2 = CPU fp64 to 5e-5).
- **Result:** no refit (`none`) is worse everywhere (T-geo 1.017; K0.5 ×1.05): the polish helps. H_out-weighted
  `hout` T-geo **0.9968** (K0.5 0.992, K2.5 0.9988, K4 0.9996), `undamped` 0.9988, both 0.9964; rule (a) T-geo ≤ 0.990
  fails for all → **refit = true stays**. Best single cells: K0.5 in_proj_z 0.984, qkv 0.993. Proxy gains shrink in KL
  (~⅕ for #117), so ≤ 0.1% KL at K2.5.
- **Side finding (item 4 evidence):** the trajectory shows **C9 multiplies the two-sided f2 at K0.5 by ×2.67 (L10
  out_proj 0.168 → 0.449) and ×1.07 (L39 o_proj)**, vs ×1.04 / ×1.005 at K2.5 and ×1.005 / ×1.001 at K4. C9 targets
  KL (gain shrink), not the proxy, so a proxy rise is expected, but ×2.7 at sub-1 supports the critic's R15 (C9 needs
  an α(K) schedule below K≈1.5). Sub-1 sizes only.
- **#139 instrumentation:** per-level diag files would be ~143 MB (2.2 MB a layer); not wired into rxq yet.
- **Runtime:** none.

### 2026-11-08 11:40 PM CST · #122 · Full adopted stack as one K2.5 level through the donors' job path: gate PASS (0.972 pooled, = bn+swj); pooled number is two xcorp documents; calib part 0.908 (w92)
- **Status:** done · **Machine:** gpubox card 0 (~3.9 card-h: jobs 2 × 1.1 h, swjfit 1.75 h, scores + e2e ~0.5 h; packs
  4 × 22 min CPU) · **Code:** `v3/e3/settle/w92/` (README) · **Rule:** `v3/e3/settle/w92.md`, committed 26ec0c9 before
  any data; result section there · **Data:** `v3/e3/settle/data/w92/` (per-document sums `docsums92.json`, verdict,
  job logs, timings, e2e); rows and levels on gpubox `/data/w92/`
- **Path exercised on the GPU, no failure:** 512 group jobs (K2.5 i8, seed offsets 0 / 1) from `rxq.ladder.level_jobs`
  via `python -m rxq job` (H_out bn1920 blocks + #117 block order + tile CD on attn_qg + C9 in the writers' jobs) →
  `rxq assemble` (64 shards, 64 C9 writers) → `rxq.swjfit` on the assembled levels (one pass, 105 min) → `rxq.post
  apply LEVEL SWJ,LEVEL/c9` → `rxq.mix recipe` → rad-convert / radmix / census → hygiene PAIR OK on all 4 pairs.
  Per level 4,111 / 4,069 s of job wall incl. ~4–5 s process overhead a job (rxq.ladder estimates 1.54 h with
  overhead; actual 1.13 h); ffn_down peaks 8.5 GiB at K2.5; tile CD two-sided ratio geo 0.9988 (16 q_proj).
- **Gate (fresh two-part slice golden-search27c: 65 xcorp pieces of the never-scored xc8k-b2-c2 + 26 calib search
  docs; ledger role select):** ctl → full stack pooled **0.9714** [0.941, 1.035] (seed 0) / **0.9728** [0.908, 0.990]
  (seed 1); xcorp 0.975 / 0.976; calib **0.899** [0.865, 0.936] / **0.917** [0.884, 0.961]. Bar ctl → bn+swj (#104's
  mixes) 0.9658 / 0.9806. geo 0.9721 ≤ 0.9732 and nothing > 1 → **PASS**. Direct bn+swj → full stack 1.006 / 0.992
  (|z| < 0.6); calib part 1.020 / 1.017 (z +1.8 / +1.5, within the seed floor). The full stack equals bn+swj here;
  C9 / #117 / tile CD are not resolved by this slice. No integration alarm.
- **Slice caveat (post hoc):** xcorp holds 95% of the pooled KL sum and one xcorp piece (smoltalk2/longalign, PDF text
  with a non-breaking space between every word, base KL 3.2 / position) 51% of it. Without it the pooled ratios are
  0.998 / 0.956. Per document (equal weight), **xcorp gains as much as calib**: xcorp per-doc geo 0.909 / 0.900
  (median 0.880 / 0.898), calib 0.877 / 0.998 (median 0.877 / 0.930); all 91 docs 0.900 / 0.927. So the lead's 11:25
  reading ("gains concentrated on calib-like text, small on xcorp") does not hold per document: the xcorp part's 0.976
  is mostly the longalign piece (0.944) and the swe-rebench piece (1.012). **User's bar (< 0.95 vs old base): pooled
  0.971 / 0.973 ≥ 0.95, calib part 0.899 / 0.917 < 0.95**; the adopted bn+swj arm is also ≥ 0.95 pooled on this
  slice (0.966 / 0.981). Proposal for later slices: cap a document's weight, or decide on the per-part / per-document
  statistic, so that one pathological piece cannot carry a decision.
- **Sub-1 job check** (K0.875 / K0.125 i4, layers 2–3): PASS 26 / 26 (σ per group as v3.toml, H_out blocks, block
  order, tile CD, C9). The K1 i4 arm was dropped (lead / user: i4 retired).
- **e2e on the fs0 mix (lead's add-on):** A8 (a8g128, all trunk) 1.0014 [0.9997, 1.0034] on the calib part, rows
  finite; generation coherent (correct `merge_intervals` + doctests, a sensible itinerary, a sensible tool call; with
  thinking on, 300 tokens of on-topic reasoning); prefill 1,432 tok/s vs V2 3.00's 1,736 (−17.5%, uniform fractional
  K2.5 i8 vs V2's mix); static weights 8.20 vs 9.38 GiB.
- **Found (GPU-only):** (1) **rxq's code stamp is platform-dependent**: `engine.code_stamp` hashes ROCm hipify outputs
  (~240 in the fork's ext dir, 14 written into rad-exl3-mix's kernel dirs by the first build), so a CUDA and a ROCm
  donor get different code stamps for the same code and `rxq assemble` would refuse their mixed level. Proposed fix
  `v3/e3/settle/w92/codestamp-hipify.patch` (skip `*.hip`, `*_hip.*`, `hip_*`; no tracked source has "hip" in
  its name), not applied. (2) gpubox's `/data/exllamav3` (the default `exllamav3` import) lacks fork 0008: the blocks
  path needs a 0008 fork first on PYTHONPATH (rxq refuses otherwise; the donor image must carry 0008 / 0009). (3) The
  swap27 radiance home has no mul1i4 importer (moot now). (4) w92's own driver bug: `post()` declared
  `local t=$1 F=$A/lv/$t/final` in one statement, so $t expanded to the swj loop's stale global and fs0's post wrote
  into fs1/final. The driver stopped on its own check (fs0/final had no shards); the lead fixed it (two statements)
  and relaunched at 10:07 PM CST, bad output in `/data/w92/bad-1109/`. No result was affected (fs1/final was rebuilt).
- **Anchors:** settings ed97adb7ed1f9919, code 92132b43df398cb8 (50902fc + fork 6242f7ae, gpubox). Re-made with
  5ce222f + fork b0052f7c (code 27e7b3eb41ee8e05), 3 jobs are bit-identical (PAIR OK 3 / 3). rqah's series key is the
  settings stamp only, so the job dirs import as trusted rows unchanged.
- **Runtime:** none (no setting, format or kernel change).

### 2026-11-08 11:25 PM CST · #122 · Full adopted stack at K2.5 through rxq (w92): PASS; calib part 0.908, xcorp part 0.976 vs old base
- **Status:** done (PASS by the rule) · **Machine:** gpubox card 0 · **Code:** `v3/e3/settle/w92/` · **Slice:**
  golden-search27c (fresh two-part: xcorp 65 docs + calib search-pool docs)
- **Arms:** fs0 / fs1 = K2.5 i8 levels quantized by `rxq job` from v3.toml (H_out blocks, #117 order, tile CD,
  C9, swj applied), packed into V2 3.00 (2.6244 bpw) vs fresh-packed old ladder K2.5 controls; reference arms
  bn+swj. All four hygiene pairs PAIR OK (only the 336 trunk weights differ).
- **Result:** ctl → full stack pooled **0.9714 / 0.9728 (geo 0.9721)**; **xcorp part geo 0.9755, calib part geo
  0.9080**; bar (bn+swj on the same slice) geo 0.9732 → **PASS**; full stack vs bn+swj direct z +0.37 / −0.50 (equal
  within noise: C9 + #117 + tile CD add ≲ 1%, as expected).
- **Lead reading, CORRECTED after w92's report (11-09 12:10 AM CST):** the part split is not a text-type effect. xcorp holds 95% of the pooled KL sum, and two xcorp pieces hold 73% of the xcorp sum (one PDF-text piece with a non-breaking space between every word, 51%; one agentic piece, 22%), so the pooled and xcorp ratios mostly measure those two documents. Per-document (equal weight, post hoc): xcorp geo 0.909 / 0.900, calib 0.877 / 0.998 (medians 0.880 / 0.898 vs 0.877 / 0.930): xcorp gains about as much as calib. The 11:25 PM claim "gains concentrated on calib, small on xcorp" is withdrawn. Full stack vs bn+swj: 1.006 / 0.992 pooled (C9, #117, tile CD not visible here; calib part +1.8%, inside the seed floor). Proposal (w92): cap per-document weight in two-part slices or decide on a per-document statistic. e2e: A8 ×1.0014 on calib, generation coherent, prefill 1,432 vs V2 3.00's 1,736 tok/s (not like for like: uniform fractional K2.5 vs V2's mix), weights 8.20 vs 9.38 GiB.
- **Bug found and fixed during the run:** w92.sh `post()` used one `local t=$1 F=…$t…` (stale $t): fs0's post wrote
  into fs1/final; caught by the driver's layer check; fixed and rerun (bad output in `/data/w92/bad-1109/`).
- **Runtime:** none (quantizer settings).

### 2026-11-08 10:10 PM CST · #136-02, -04, -05 · 8-bit log su / sv passes only at K0.5; GDN state is f32 everywhere (no flag); Σdy² heavy blocks 6 / 9 are not agentic (w102)
- **Status:** -02 done (pass at K0.5 only, fail at K2.5 / K4) · -04 done (no flag) · -05 done (caveat cleared) · **Machine:**
  colo + Thor (-02, 27 CPU jobs, 03:47–03:58 UTC), gpubox CPU (-05, seconds), code read (-04) · **Code:**
  `v3/e3/settle/w102/` (`job102.py b8`, `report102.py`, `f1comp102.py`) · **Rule:** `v3/e3/settle/w102.md` (136d89b, time
  fix dc45f47, both before data) · **Data:** `v3/e3/settle/data/w102/` (`b8/`, `f1comp.json`)
- **#136-02 (B-8, su / sv as 8-bit log, w87's two-sided held-out proxy, 27 tensors × s0, i8):** G0 PASS (the unrounded arm
  equals w87 / w95's stored rows to 1.2e-7). L8 = sign + 7-bit log2 grid over each vector's range, rounded after the
  refit, vs fp16: **K0.5 ×1.0001 [1.0000, 1.0002] PASS; K2.5 ×1.0034 [1.0028, 1.0042] fail; K4 ×1.0228 [1.0186, 1.0272]
  fail** (bar ≤ 1.001). fp16 itself costs nothing (≤ 1e-4). The output side dominates: sv alone ×1.0027 / ×1.0180 at K2.5
  / K4, su alone ×1.0007 / ×1.0057. The log2 steps are 0.012 (su) and 0.019 (sv), i.e. ±0.4% / ±0.7% per entry. Report
  only: L9 (sign + 8 bit) ×1.0008 / ×1.0060 at K2.5 / K4; L8r (re-solving the output refit after rounding su) ≈ L8;
  E4M3 ×1.20 / ×2.45 (unusable).
  **Saving:** 8 (N + Kd) / (N Kd) = **0.0022 bpw over the trunk** (0.002 per ffn tensor, 0.009 per attn_k / v).
  **Reading:** worth it at most at the sub-1 levels, where 0.002 bpw is ~0.4% of K0.5's bits and the proxy cost is nil.
  It would need a per-level exl3i scale-format field, plus a LUT dequant of the 8-bit scales (128 entries) in the
  kernels that apply suh / svh, to save VRAM and not only file bytes. **Not proposed for K ≥ 2.5.** At sub-1 it is
  the lead's call (format field + kernel work for ~0.002 bpw at ≤ 1.3 bpw sizes). Caveat: the proxy does not refit su.
- **#136-04 (H-4, GDN state dtype parity): no flag.** The recurrent state is f32 in every path that produces or scores
  our numbers:
  - **bf16 recorder (the published calib golden and our xcorp goldens, radiance `--kld-record`):** radiance-hero
    `arch/qwen35_bf16/qwen35_bf16.cpp:235–240` declares `sd.dtype = RAD_F32` with no flag. The same holds in radiance
    1.9 (`references/radiance`, line 226) and in latest.
  - **Our evals (qwen35_exl3 plugin):** `plugin/qwen35_exl3/qwen35_exl3.cpp:158–159` declares `RAD_F32`.
    `core/engine_bringup.cpp:1250–1259` refuses `--state-cache-dtype f16` for any plugin that does not declare an f16
    state, so it cannot slip in.
  - **Serving:** f32 by default; f16 only under `--state-cache-dtype f16` and only for qwen35_fp8
    (`qwen35_fp8.cpp:678–688`).
  - **Kernels:** both arches use the same ops (`gdn_chunk_scan` / `gdn_kkt_solve` / `gdn_recurrent_update`; libr4d
    `r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip` keeps the state in fp32 WMMA accumulators across chunks and stages a
    bf16 copy in LDS as a matmul operand, the same rounding as FLA). That rounding is shared by the reference and every
    quant, so it is not an engine floor between them.
  - **Unknowns:** the published golden's `kld.json` records no engine version or step size. Our own 0.8B check found
    chunk 2048 identical to whole documents (#33-26), and #99 measured the whole non-weight floor at 8.0.
  - Nothing biases the 6–8 bpw numbers through the state dtype, so no TP2 floor run is needed for this.
- **#136-05 (F-1 step 1, category composition of bn1920's blocks):** collect.py's window order reconstructed (1,350 qad
  windows of 2,048, rng 0, sequence s → window s mod 1,350; 15 blocks of 128).
  - The heavy blocks reproduce #33-27: layer-0 / layer-10 block energy vs the median block is **6.3× / 5.4× (block
    6)** and **18.7× / 6.9× (block 9)**. All layers: 1.56 / 2.31. Layers 39 / 50 / 63: 0.8–1.25.
  - **Agentic share** (agentic + tools + long-agentic, by windows): block 6 0.289 (**1.00× the collection mean**),
    block 9 0.336 (**1.16×**). Every block is 0.73–1.22×, and each block mixes ~88 documents. Their highest-loss
    sequences are books / web (doab, gutenberg, pg19, fineweb-edu), not agent text.
  - **Rule outcome:** not agent-heavy → the FB-A3 caveat is cleared. By the rule, the category-matched statistic
    deserves its 0.8B test (on hold: the 4070 is off limits).
  - **But:** the 27B collection is not agent-starved. Agentic windows are **28.9%** of the 1,920 sequences (long
    traces 17.6%), close to calib's 34% token share and the lens' ~25% scored-row target. The lens' "6%" does not hold
    for this collection (no gen mask; the long qad traces give many windows). At 27B a category-matched reweighting
    would barely move the weights, so expect ≈ 0 there. The 0.8B test only says something about the 0.8B collection's
    mix.

### 2026-11-09 9:50 PM CST · #137 · A-15 partial activation denoising (H + ρN) vs equal-trace damping, 0.8B: rule committed, running (w103)
- **11-11 8:00 AM CST: moved to gpubox card 0** (the 4070 is out), **amendment 1 committed alone (48d559f) before any data**: the 4070 run had been stopped 11-09 inside its first unit, nothing scored or read. Inputs rebuilt on gpubox (`gpubox/setup.sh`, `/data/w103`): model (w106's HF download, reflinked), g08-fit1 from the ledger's 87 docs (score check PASS, max |GPU − score.py| 1.5e-6), g08-search2 / -search3 by more.sh's seeded draw (`gpubox/docs.py`; 145,541 / 152,407 rows), 0.8B calib input Hessians (hcollect.py) and C18's H_out dy (collect.py, yaqaFact7's env). Reproduction gate (information only) dropped: its references are on the 4070. Driver `gpubox/w103.sh` (run.sh with `W103_ENV=gpubox/env.sh`).
- **Status:** running · **Machine:** gpubox card 0 since 11-11 (was: 4070, queue `/data/queue.d/95-w103.sh`) ·
  **Code:** `proxy2b/exp/w103-pad/` (`mkhp.py`, `run.sh`, `report.py`; snapshot `/data/yb/w103/snap/`; moments.py from
  w07-denoise) · **Rule:** `proxy2b/exp/w103-pad/RULES.md`, **committed alone (dc45f47) before data**
- **Question:** quantize every trunk linear against H′ = H + ρN (N = E[ηηᵀ] of the input error in a quantized model,
  metric only, no F / C terms: the partial version #88 did not run); is any gain more than the damping its trace adds?
- **Setup:** 0.8B i8, today's ladder stack (q2b + H_out dy σo 0.01 + swj; no tile CD (not in the 0.8B ourq), no writers
  asym (per-mix)). N from the seed-3 stack per K, `moments.py --mode quant` on fit-pool windows. Arms r0, n10 / n25 / n50
  (ρ 0.1 / 0.25 / 0.5) and d10 / d25 / d50 (H + c·diag H, same added trace per group), K3 / K2, seeds 0 / 1 (28 quants).
  Decides on **g08-search2** (first use), same-sign confirmation on **g08-search3** (first use).
- **Pass rule:** per K, ρ* = best on search2; (a) geo r0 → nρ* ≤ 0.98, (b) dρ* → nρ* ≤ 0.99, (c) both < 1 on search3;
  PASS needs all three at K3 and K2; (a) without (b)/(c) = damping → drop. Gate G0: ρ = 0 reproduces calib H exactly.
- **Runtime:** none (quantizer metric only).

### 2026-11-08 9:40 PM CST · #128, #129 · H_out in a mixed-K splice mix; swj on / off at K6: rule committed, donors checked, queued behind w92 / w95 / w98 (w99)
- **Status:** #128 **INCONCLUSIVE** (11-10 5:45 AM CST, geo 0.9865) · #129 **PROPOSE k_max = 4.5** (11-10 8:50 AM CST,
  r 1.0244, z +0.85) · **DONE 14:50 UTC 11-09** · **Machine:** gpubox card 0 (behind w92 / w95 / w98: the driver waited until their
  drivers are gone 10 min, then `/root/card0.lock` → `/root/ram.lock` per step; radmix / packs conv.lock + ram.lock
  exclusive) · **Code:** `v3/e3/settle/w99/` (`w99.sh`, `trunk99.py`, `verdict99.py`; w92's `mkjobs92.py` / `layer92.sh`
  for the K6 jobs), staged on gpubox `/data/w99/code` = w92's tree (`git archive 50902fc` + w92/, fork 6242f7ae) + w99/ ·
  **Rule:** `v3/e3/settle/w99.md`, **committed alone (addea3e) before any mix, quant or score** · **Data:**
  `v3/e3/settle/data/w99/` (when done)
- **#128 (R1, shortlist item 2):** four mixes into V2 3.00 with `radmix` from the stored swap27 mixes, no new quants:
  even layers' trunk from the K4 level, odd from the K2.5 level, H_out on (bn4 + bn / bn4s1 + bns1) vs off (base4 +
  ladder2 base / base4s1 + s1). Within each seed and K the levels differ only in `--hout` (dense path, block order off:
  `rxq levelargs`). Even layers are all GDN, so the 16 attention layers sit at K2.5 (160 K4 + 176 K2.5 trunk weights).
  Holds: geo ≤ 0.985 and both pairs < 1; fires: geo ≥ 1.00 (then H_out is scoped by K band before the ladder).
- **#128 result (11-10 5:45 AM CST): INCONCLUSIVE.** hp → hb pooled r 1.0054 (seed 0, z +0.57) / 0.9680 (seed 1,
  z −2.72); **geo 0.9865** (xcorp 0.9881, calib 0.9458). SE doc 0.0099, combined (0.0086, #116 same-seed SD) 0.0116:
  z(geo vs 1) −1.16, z(geo vs 0.985) +0.13 — above the HOLDS bar, below FIRES. Seeds split: seed 0 xcorp r 1.0080
  (pooled p99 7.695 → 8.024), calib r 0.9388 (z −2.97); seed 1 xcorp r 0.9685 (p99 7.953 → 7.577), calib r 0.9528
  (z −2.30). top-1 pooled 0.89764 → 0.89904 / 0.89894 → 0.90036 (H_out on is a hair better on top-1 in both seeds).
  Against the uniform-K adoption record (K2.5 geo 0.980, K4 geo 0.964, #33-28), the H_out advantage shrinks to ~1.4%
  on the K4 / K2.5 splice and is unresolvable at this precision. Scores `/data/w99/scores/h{p,b}{0,1}-{x,c}.json`
  (54–79 s each), verdict `/data/w99/verdict-h.txt`.
- **Donor provenance (done, 03:34 UTC):** the stored swap27 mixes' trunk weights are byte-identical to later fresh packs
  of the same levels (`trunk99.py`, 336 / 336 each): bn4 vs cdhout27's bn4ctl, bn4s1 vs bn4s1ctl, base vs w92's basectl;
  s1 vs w92's s1ctl when that pack lands. So the splice from stored mixes equals fresh packing.
- **#129 (R17, shortlist item 6):** a new K6 i8 level (`q27v3-ed97adb7-k6-i8`, seed 0) through w92's job path (est 1.67
  card-h), `rxq.swjfit` on it, then two finals from the same codes: swj + C9 vs C9 only, each packed into V2 8.00 (bf16
  embedding, K7–8 head: the trunk dominates the KL). Keep swj above K4.5 if r ≤ 0.995, else propose `k_max = 4.5`. K5 not
  run (not cheap: ~1.3 card-h quant + ~0.75 swj + two packs).
- **#129 result (11-10 8:50 AM CST): PROPOSE k_max = 4.5.** k6c9 → k6swj (same K6 codes, refit scales only) pooled
  r 1.0244 [0.8961, 1.0335], z +0.85 — swj on is 2.4% worse and not significant; xcorp r 1.0246 (z +0.88), calib
  r 0.9747 (z −1.75); p99 0.3914 → 0.4231; top-1 0.97738 → 0.97698 (91 docs, 145,597 pos). Gate (r ≤ 0.995 → keep
  swj above K4.5) not met → **proposal only**: cap swj at `k_max = 4.5`; no config change made. Hygiene PAIR OK
  (64 K6-layer weights differ, svh plane only, 0 problems); both mixes 6.5878 bpw in V2 8.00. Cost: k6 quant 3872 s
  (est 1.67 h), swjfit 70 min, packs 1347 s + 1332 s (CPU), scores 53–78 s each. Files: `/data/w99/scores/k6{swj,c9}-{x,c}.json`,
  `/data/w99/verdict-k6.txt`, `/data/w99/mix/mix-k6{swj,c9}.rad`.
- **Slice:** golden-search27c (w92's), both parts; its second and third decisions after #122 (R55: retire after this round).
- **Runtime:** none (#128 changes no setting; #129 swj scales fold into svh).
### 2026-11-08 10:10 PM CST · #132, #133 · Sub-1 hygiene (C9 / g_scale edge / drift / zero floor / scale underflow) and H_out shape (σ_out × h^p) on the held-out two-sided proxy: rules committed, queues running (w100)
- **Status:** running · **Machine:** colo (`/data/work/w13/w100`, ≤ 8 GB, no ffn_down), Thor (`/data/work/w100`,
  ffn_down + the rest; restartable per K file), gpubox read-only (suh / svh scan, niced) · **Code:** `v3/e3/settle/w100/`
  · **Rule:** `v3/e3/settle/w100.md` (372b555, before data; amendment 1 7028b0c, report-only fixed-g arms at K0.125; amendment 2: renumbered from #128 / #129, which w99 took first)
- **Task:** pre-freeze shortlist items 4 (sub-1 hygiene) and 8 (H_out shape), ranked.md 11-09; i8 at sub-1 (#126).
- **Gates:** G1 (ldlq2s = ldlq2 bit for bit; scale search reproduced; L_out identical; w93's K2.5 m 1 row reproduced
  to 0 on colo, 3.5e-8 on Thor) PASS; G2 (15 block-normalized Σdy² blocks rebuilt from gpubox's snapshots average to
  bn1920 within 9e-8, 400 keys) PASS; G3 (real w92 writers' C9 recomputed = the jobs' records within 1.2e-10) PASS.
- **First readouts (complete, not proxy):** suh / svh scan of 71 gpubox level / job dirs: 0 zeros, 0 non-finite;
  f16 subnormals 46 at K0.125 (w92 L3 q_proj suh 44, o_proj 2) and 1 at K ≥ 1 (a4cost i4 K1, L17 out_proj suh
  5.5e-5). Real K0.125 codes (13 tensors): max rel 0.788 (L3 up_proj), none near 0.97. Production's scale search
  sits at the grid's top edge (g 2.05) on every logged K0.125 job (w92, w81), the proxy at ~1.65 (flat MSE curve).
- **Runtime:** none (quantizer settings / pricing).

### 2026-11-08 9:55 PM CST · #138, #139 · Scale polish in the right metric (four refit arms on fixed codes) and log-once job instrumentation: rule committed, code tested on CPU, card-0 run queued (w98)
- **Status:** running · **Machine:** colo (CPU tests, `/data/work/w13/w98`), gpubox card 0 (`/data/w98`, queued on
  card0.lock behind w92's packs / w95) · **Code:** rxq patch `local/workers/w98/rxq-refit-diag.patch` (new
  `pipeline/rxq/refit.py`, wiring in engine / quant / job / config, `tests/check_diag.py`), rqah patch
  `local/workers/w98/rqah-diag.patch`; driver `v3/e3/settle/w98/` (`w98.sh`, `mkjobs98.py`, `f2calc98.py`, `report98.py`) ·
  **Rule:** `v3/e3/settle/w98.md`, **committed alone (7f4b421) before any data** (numbered #130 / #131 there; renumbered to #138 / #139 in an amendment, w101–w103 took #130–#137 concurrently)
- **#138 (shortlist item 1):** the fork's polish (`refit_scales`) fits su / sv in the damped H with every output column
  alike, after a two-sided LDLQ. `quant.refit` (stamped, until now unwired) becomes `true | false | "hout" | "undamped" |
  "hout+undamped"` in rxq (no fork patch). Arms evaluated inside one job per (layer, K) from the same pre-refit state
  (fixed codes by construction): L10 / L39 at K0.5 / K2.5 / K4 i8, full stack, scored with w81's f2 (fp64). Pass: T-geo
  (qkv / z / qg × 3 K) ≤ 0.990 and no tensor worse (geo over K ≤ 1.001, no cell > 1.005); then a K2.5 fixed-code reader
  pair on golden-search27c (g ≤ 0.997, both parts < 1).
- **#139 (item 5):** every rxq job writes `diag-<key>.safetensors` (per-output-channel e1 / e2, pre-refit e1 / e2, x1 / d1
  for any later per-channel rescale, r1, the 128 × 128 error map, pre-refit su / sv) and a log `diag` (trajectory pre /
  post refit / post-C9 + tile CD's rel, the proxy check, overload counters, g_scale). CPU stand-in (`check_diag.py`): Σ e1
  / Σ r1 = logged proxy to 1.4e-7 / 2.1e-7 (rule 1e-6); t-files and C9 files bit-identical with diagnostics off and with
  the arms on; `refit = false / hout / undamped` keep the trellis and reproduce the in-job arm's f2 to 3e-8; check_cpujob
  25/25, check_ladder 5/5. Settings stamp unchanged (ed97adb7) with `refit = true`; the code stamp changes (df95e5e2 on
  gpubox for the patched tree).
- **Runtime:** none (fp16 scales / side files).

### 2026-11-08 10:05 PM CST · #130 / #131 · Tile interleave (f2 on card 0, h stability on colo) and per-section K (colo proxy) started (w101)
- **Status:** running · **Machine:** gpubox card 0 (#130-01, 8 rxq units under card0 + ram shared, interleaved with
  w98 / w99), colo (#130-02 stability, #131 proxy queue: 8 preps + 152 (tensor, K) jobs, 8 at once, nice 19) ·
  **Code:** `v3/e3/settle/w101/` · **Rule:** `v3/e3/settle/w101.md` (acf54f2, before data)
- **Gates:** #131 G1 (prep == settle.P2 bit for bit on P2's own block) and G2 (batched two-sided block LDLQ ==
  settle87.ldlq2: identical codes, e2 ratio 1.0) PASS on L10 ssm_qkv and L3 attn_qg. #130-02 G-b (15-block bn mean ==
  hout-bn) 6e-8 on the keys so far. #130-01 prep: P16 checkpoint / H_out un-permute to the source bit for bit.
- **Next:** f2 ratios (pass R(L10 qkv) ≤ 0.90), kept fraction (≥ 0.75), #131 held-out gain (≥ 0.10 bit).

### 2026-11-08 9:05 PM CST · #127 · Head Fisher smoothing / more data on held-out rows: no arm useful (keep today's Fisher); w94's "qad rows in-sample" gap is really Japanese / Thai documents in the search dump (w96)
- **Status:** done (negative) · **Machine:** gpubox CPU (Fisher pass over the calib golden, 13 s, ram.lock shared),
  Thor (xc8k-b1 Fisher; CPU scorer 02:19–02:52 UTC), colo (8 knap5 solves, 3 min) · **Code:** `v3/yaqa/w96/`
  (`score96.py` = headsim94's KL per head block and donor on the CPU; `plans96.py` with `headslab.py` unchanged) ·
  **Rule:** `v3/yaqa/w96/RULES.md` (72e2191, before data) · **Data:** `v3/yaqa/w96/data/`
- **Gates:** G0 my qad-gen Fisher = `headsens.npz` (max rel 4e-15; it is pools-v3's qad, 2,799 documents, not
  pools.json's) PASS; G1 CPU scorer vs w94's GPU sim on all 101 plans: mean ratio within 0.54% (u7 / s7, the smallest
  KLs), per-row corr ≥ 0.99993, top-1 agreement 1.0 PASS.
- **Census:** 3,836,849 qad gen positions. Rows with Fisher 0: 673; rows with < 10 / < 100 positions of p > 1e-4: 49,606 /
  129,017. Those row sets hold 0.0002% / 0.22% / 2.6% of the held-out exact Fisher (1.4% on the search-pool golden).
  Adding xc8k-b1 cuts the < 10 rows to 23,275.
- **Tune (qad split, half A → qad-dump rows in B):** prior U (unigram bins) κ 1–30 within ±0.3%, κ ≥ 100 +7%; prior E
  (head-norm deciles) 2–4× worse (the decile mean is set by a few frequent rows); more-q / more-x ≈ −1%. Chosen: U κ 1,
  more-q w 1, more-x w 0.5, both = more-x + U κ 1.
- **Held out (search-gen, 38,960 rows / 94 documents, paired document bootstrap), arm vs base:**

  | arm | s4-3 | s5-3 | s6-3 | useful |
  |---|---|---|---|---|
  | smooth | +0.0% | −0.6% [−1.0, −0.3] (+0.004 b) | −0.7% [−1.1, −0.2] | no |
  | more-q | +0.1% [−0.0, +0.4] | −2.7% [−4.1, −1.3] | −1.0% [−1.8, −0.4] | no |
  | more-x | +0.0% [−0.3, +0.4] | −3.8% [−5.1, −2.3] (+0.004 b: unequal bytes) | −1.0% [−1.7, −0.4] | no (0 / 3) |
  | both | = more-x | | | report only |

  Rare-token KL (argmax rank ≥ 20k): −0.5 / −5.0 / −2.8% for more-x, smaller for the others (report only). No result
  is borderline near the 3% bar (the only CIs touching 0 have point estimates ≥ 0).
- **knap5 re-solve (best arm more-x, w94's procedure):** −0.00 / −0.02 / −0.01 / −0.02% predicted KLD at 2.0 / 2.5 /
  3.0 / 4.0 vs the base grid; the base grid on the CPU sums reproduces w94's g-search-gen to 6 digits.
- **Post hoc (report only): the qad / search gap is not Fisher overfit.** On the qad-dump rows, plans from half A (never
  saw those documents) = today's full-Fisher plans (+1.1% [−0.1, +2.5] / 0.0 / 0.0%); slab / uniform is 0.14 / 0.18 /
  0.41 on qad and 0.40 / 0.36 / 0.50 on search with either Fisher. The 5 worst held-out documents for s4-3 are 4 jpn
  (fineweb2) + 1 Thai Wikipedia: 43% of its KL, and slab is worse than uniform on them (s/u 1.0–2.3). Without the 6
  jpn / th documents s4-3 / u4 is 0.24. The dump holds 5 of the search pool's 6 jpn documents (the search pool is
  0.45% jpn / 1.05% Thai tokens vs calib 0.24% / 0.20%). Token weighting leaves it at 0.43 [0.18, 0.80]: the head-plan
  price on search-gen rests on a handful of rare-script documents.
- **More data cost (proposal only):** the qad pool has no unused positions (the calib golden covers all of it); more
  would be xcorp / traces on TP2 at ~2.65M rows/h (xc8k-b1: 5.0M rows in 113 min): 2× (+3.8M) ≈ 1.5 h and 16 GB,
  4× (+11.5M) ≈ 4.3 h and 47 GB. Not recommended: 2.3× data (xc8k-b1 at w 1) moved the tune set by 1% and knap by 0.02%.
- **Proposal only (GPU, not run):** two-sided head LDLQ with a smoothed H_out = re-quant K3 / K4 with `hout_diag` =
  smoothed Fisher (prior U, κ 10–30, as w94's `gpu94.sh` writes it) vs w94's T0 donors, then headsim94 on the same dumps:
  ~2 × 100 s quant + ~5 min headsim ≈ 10 card-min. Low priority: σo 0.1 already floors the zero / rare rows, and rows with
  n < 100 hold 2.6% of the held-out Fisher, so expect < 1%.
- **Runtime:** none.

### 2026-11-08 8:20 PM CST · ideas round 2 · six pre-freeze lenses + critic done; pre-freeze shortlist of 8 (lead review pending user)
- **Run:** `local/papers/runs/2026-11-08r2/` (lenses codes, lowbpw, measure, composable, redteam at Opus xhigh;
  faraway xhigh, 83 min, ran its own colo checks; critic Opus high, 7 min). Ranked list `ideas/ranked.md`.
- **Pre-freeze shortlist (critic):** (1) scale polish ignores H_out and fits in the damped H (code-confirmed; card 0
  four-way f2, ~2.5 card-h); (2) mixed-K H_out splice pair guarding "H_out at every level" (~2.5 card-h, no quants);
  (3) output-row tile interleave for in_proj_qkv / q_proj (format; 20-min f2 screen first); (4) C9 α(K) and sub-1
  hygiene (C9's c reaches 2.83 at sub-1; CPU); (5) log-once job instrumentation before the stamp; (6) swj k_max 4.5
  (−18 card-h; user decision); (7) q|gate and q|k|v as separate knapsack tensors (CPU first); (8) σ_out / h^p sweep
  (CPU). Top post-freeze: GDN in_proj_a/b out of bf16 (0.014 bpw; a/b are 23.6M params, codes' 0.026 was wrong).
  Closed by the critic: swj is mix-independent (fits on bf16 MLP inputs; R19).
- **Runtime:** none (planning).

### 2026-11-08 7:13 PM CST · #125-02 · Head donor recipe: two-sided softmax-Fisher LDLQ without mean-row, −9% / −10% head KL at K3 / K4 vs today's donors (PASS) (w94)
- **Status:** done (T0 adopted by amendment 2; for the donor remake: lead / user) · **Machine:** gpubox card 0,
  00:58–01:10 UTC (6:58–7:10 PM CST; 2 quants + 2 headsims) · **Code:** `v3/yaqa/w94/gpu94b.sh`, `report94c.py` (same
  ourq / fork / patch as #125-01) · **Rule:** `v3/yaqa/w94/RULES.md` amendment 2 (c1ee37a, before the arm) · **Data:**
  `v3/yaqa/w94/data/report94c.txt`, `report94c.json`
- **Exact full-head KL, ratio [95% document CI]** (C0' = today's donors, bit-identical KL; T = with mean-row):

  | K | pair | search-gen | qad-even | qad-odd | search-all | guard |
  |---|---|---|---|---|---|---|
  | 3 | T0 / C0' | **0.908** [0.884, 0.932] | 0.874 [0.849, 0.901] | 0.896 | 0.906 | 0.96 |
  | 3 | T0 / T | 0.961 [0.938, 0.985] | 0.941 | 0.986 | 0.964 | 1.00 |
  | 4 | T0 / C0' | **0.899** [0.873, 0.925] | 0.860 [0.836, 0.889] | 0.885 | 0.900 | 0.98 |
  | 4 | T0 / T | 0.943 [0.925, 0.962] | 0.951 | 0.975 | 0.937 | 1.00 |

- **Verdict (amendment 2): T0 PASS** → donor recipe = today's flags + `--hout HD --hout-blocks group --sigma-reg-out
  0.1`, **no** `--head-mean-row`; HD/lm_head.safetensors = `hout_diag` = `fisher_qad_gen[perm]` (f32; `gpu94.sh` writes
  it). Cost: quant 100 s vs 94.5 s (K3), 95 s vs 91 s (K4): +5–6% a donor; same bytes, format and kernels. Peak VRAM
  not measured (the two-sided path puts the whole f32 head on the card; it ran on the 32 GB R9700).
- **Needs:** fork patch 0008 **plus** `v3/yaqa/w94/exl3-two-sided-cpu-weight.patch` (the next exl3-patches number,
  lead); the head is the only tensor above 5e8 weights, so the trunk is unaffected (one-sided and smaller two-sided
  tensors: the line is a no-op).
- **Not tested:** K5–K7 / fractional donors (the gain is the same at K3 and K4); σo other than 0.1 on the real
  quantizer (0.1 = the CPU proxy's selection on qad-odd); the slab plans built from T0 donors (re-run headslab plan +
  headsim94 on the same dumps after the donors: ~5 card-min).
- **Runtime:** none.

### 2026-11-08 7:05 PM CST · #125-01 · Real head quants: two-sided softmax-Fisher LDLQ −12% head KL vs mean-row (PASS); mean-row itself is +7–12% worse than none on gen rows (w94)
- **Status:** done (T PASS; mean-row: rule says recommend dropping) · **Machine:** gpubox card 0, 00:23–00:54 UTC
  (6:23–6:54 PM CST; 6 head quants 90–100 s each + 2 headsims, between w92's jobs, ram.lock exclusive per quant) ·
  **Code:** `v3/yaqa/w94/gpu94.sh`, `report94b.py`; repo `quantize/ourq.py` (HEAD) + fork 6242f7ae (0008) **plus
  `v3/yaqa/w94/exl3-two-sided-cpu-weight.patch`** (one line: the two-sided path returns weight_q on the card while a
  > 5e8-weight tensor keeps weight_r on the CPU, so the proxy metric crashed for the head; the fix only touches that
  subtraction) · **Rule:** `v3/yaqa/w94/RULES.md` amendment 1 (1cc3f12, before data) · **Data:**
  `v3/yaqa/w94/data/report94b.txt`, `report94b.json`, `q/*.proxy.jsonl`
- **Check:** C0' (today's donor flags, re-made with this code) gives head KL identical to `/data/headslab3` K3 / K4 on
  every row (max |dKL| 0), so the arms differ only by their flags.
- **Exact full-head KL, ratio [95% document CI]:**

  | K | pair | search-gen | qad-even | qad-odd | guard (search-gen) |
  |---|---|---|---|---|---|
  | 3 | T / B | 0.880 [0.855, 0.905] | 0.863 [0.838, 0.888] | 0.844 | 0.97 |
  | 3 | B / C0' (mean-row) | 1.074 [1.054, 1.094] | 1.077 [1.051, 1.104] | 1.077 | 0.99 |
  | 3 | T / C0' | 0.945 [0.923, 0.967] | 0.929 [0.904, 0.957] | 0.909 | 0.96 |
  | 4 | T / B | 0.885 [0.854, 0.917] | 0.806 [0.781, 0.833] | 0.807 | 1.02 |
  | 4 | B / C0' (mean-row) | 1.076 [1.053, 1.100] | 1.123 [1.091, 1.155] | 1.125 | 0.96 |
  | 4 | T / C0' | 0.953 [0.929, 0.977] | 0.905 [0.882, 0.932] | 0.908 | 0.97 |

- **Verdicts (amendment 1):** T (two-sided, σo 0.1) **PASS** (≤ 0.97 at both K on search-gen and qad-even, guard
  ≤ 1.3). Mean-row: B / C0' > 1 at both K on search-gen and qad-even → **recommend dropping mean-row** (11-02's −30%
  was measured on older donors and prefill rows; not re-investigated). The CPU proxy got the direction of both right
  (stratum A: T −20…−25%, mean-row +15%) while its additive estimate failed G2.
- **Next:** T without mean-row (T0) is untested: amendment 2 (c1ee37a, before the arm) runs it on card 0 now.
- **Runtime:** none (quantizer: same format, kernels and bytes).

### 2026-11-08 6:30 PM CST · #125 · Head settings on the CPU proxy: rule verdict "proxy unverified" (G2 fails at K3), nothing adopted; GPU confirmation of two-sided + mean-row launched (#125-01) (w94)
- **Status:** #125 done (no adoption: proxy invalid by G2) · #125-01 running · **Machine:** colo (`/data/work/w13/w94`,
  5:11–6:05 PM CST: 4 preps, 8 scale searches, 896 block quants ~27–58 s each, 64 score jobs; ~9 CPU-h), gpubox card 0
  for #125-01 · **Code:** `v3/yaqa/w94/hq94.py`, `q94.sh`, `s94.sh`, `report94.py`; post hoc `s94b.sh`, `strata94.py`,
  `posthoc94.py`; `gpu94.sh` · **Rule:** `v3/yaqa/w94/RULES.md` (650436c), amendment 1 (1cc3f12, before any GPU arm)
  · **Data:** `v3/yaqa/w94/data/report.md`, `strata-posthoc.txt`
- **G2 (proxy validity):** quantizer half PASS (C0 / real donor rows 0.94 [0.85, 1.04] at K3, 1.00 at K4; K3 g_scale
  0.7364 vs 0.7369); estimator half **FAIL at K3**: the additive stratified estimate of the real donor's rows is 1.32×
  headsim's full-head u3 (K4 1.05). By the rule the arms are "proxy unverified": nothing is adopted from the proxy.
- **Arm ratios vs B (additive estimate, unverified):** T(σo 0.1, chosen on qad-odd from 0.01 / 0.1 / 1) 0.906 / 0.887
  (K3 / K4 search-gen), 0.863 / 0.869 (qad-even), guard 0.91 / 0.99; R4 0.878 / 0.843, 0.850 / 0.835; R8 0.886 / 0.943,
  0.804 / 0.822 (guard 1.39 at K4). Every arm would pass the 0.97 bar if the estimate held.
- **Post hoc, report-only:** per stratum (each exact for the rows it changes) two-sided lowers the top 16 blocks' KL by
  20–25% at both K, ~0–10% in blocks 16–255, tail noisy; rank-4 −28% in the top blocks; **mean-row raises the top 16
  blocks' KL by ~15%** (C0 / B 0.85–0.87) and lowers the tail's. Top-4 / -8 principal components hold only 1.5 / 1.9%
  of the mean-removed head's energy. A replicate-weighted joint estimate is unusable (log of a negative mass on rows
  where a sampled tail block holds a likely token).
- **τ (headsim, exact, real donors):** τ* = 1.01 at K3 and K4 (sharpen); KL(τ*) / KL(1) 1.001 / 1.089 on search-gen:
  **not adopted**. A parabola through 0.99 / 1 / 1.01 puts the optimum at τ ≈ 1.005 with −0.7% (K3) / −0.1% (K4):
  below any bar; at K ≥ 4 a 1% τ step costs more than the head's whole KL.
- **#125-01 (amendment 1):** real head quants on card 0 (repo ourq + 0008 fork): C0' (today's donor flags), B
  (+ `--head-mean-row`), T (B + `--hout HD --hout-blocks group --sigma-reg-out 0.1`, Fisher in perm order) at K3 / K4,
  exact headsim94 on both dumps. Rule: T adopted iff T / B ≤ 0.97 at both K on search-gen and qad-even, guard ≤ 1.3;
  mean-row recommended iff B / C0' < 1 at both K on search-gen.
- **Runtime:** none (two-sided, mean-row, τ). Rank-r: P+D (format addition; not on the GPU, post-freeze at the earliest).

### 2026-11-08 5:55 PM CST · #124 · Head inputs at scored rows: gen masking moves head KL +3–7% and changes no plan; pricing on qad rows (the Fisher's own pool) overstates slab heads ~2× (w94)
- **Status:** done (readout; no pass / fail by the rule) · **Machine:** gpubox card 0 (23:11–23:27 UTC = 5:11–5:27 PM
  CST, ~15 card-min: dumps 4.8 min, G1 2 min, headsim 2 × ~3 min), colo (knap5 re-solves, 20 solves) · **Code:**
  `v3/yaqa/w94/` (`headx94.py`, `gate94.py`, `headsim94.py`, `run94.sh`, `report94.py`, `simjson94.py`, `knaprep94.py`)
  · **Rule:** `v3/yaqa/w94/RULES.md` (650436c, before data) · **Data:** `v3/yaqa/w94/data/` (report, headsim-format
  JSON per row set for KN_HEADGRID, plans, gates, knap table); per-row npz in `/workspace/rad-local/workers/w94/data/`
  and gpubox `/data/w94/`
- **G1 PASS:** search gen rows KL vs golden top-512 2.0e-4, top-1 0.994 (38,960 rows); qad 2.0e-4 / 0.994 (33,172).
  The one-card layer-streamed bf16 forward reproduces the golden. (Unscored rows disagree more on search, KL 0.18 a row:
  not used; the golden's unscored rows are not a reference.)
- **Gen mask (search-gen vs search-all, same 103 docs):** uniform u3–u7 head KL ×1.03–1.04; slab plans ×1.06–1.07, so
  the slab / uniform ratio rises ~2–3% relative (s5-3 / u5 0.351 → 0.362, s6-3 / u6 0.493 → 0.499). **The best floor
  at every budget 3.125–7 and the lower hull are identical on gen and all rows: gen masking changes no plan choice.**
- **Pool (qad-gen vs search-gen):** uniform heads 0.88–0.90× (corpus). **Slab plans look ~2× better on qad rows**:
  s5-3 / u5 0.176 [0.156, 0.200] vs 0.362 [0.272, 0.465] held out; s4-3 / u4 0.142 vs 0.398. The plans rank rows by the
  qad Fisher, so qad rows are in-sample; qad picks a different floor at 16 budgets (4.25–6.0, 6.75) that is significantly
  worse held out (up to ×1.53 on search-gen). The held-out −64% at K5 matches #34-02's whole-model reading (s5-3 recovers
  ~65% of the u5 → bf16 gap on the calib part); qad rows would say −82%. **The rule's line "pricing set: qad-gen" is
  wrong for slab plans: price head plans on rows held out from the Fisher pool (search-gen, or fit-pool gen rows).**
- **w88's grid (knap5 KN_HEADGRID, accurate f16, donors K3–K7 only, conservative CI end), predicted KLD vs today's
  menu:** 2.0 −1.9% (search-gen) / −2.0% (search-all) / −2.4% (qad); 2.5 −4.4 / −4.6 / −5.4%; 3.0 −3.2 / −3.5 / −4.8%;
  4.0 −1.9 / −2.1%. Head picks: 3.125 / 3.125 (2.5: 3.25 on all rows) / 3.375 / 4.375 bits. With w88's 11-01 curves
  restricted to K3–K7 the grid gives nothing (= menu): the gain comes from the corrected-rowerr plans, not from the
  rows. Budgets below 3 bits (w88's 2.0 / 2.5 picks used K2 / K2.5 floors) need donors that do not exist.
- **τ:** see #125 (not adopted: < 1%).
- **Decision / next (lead):** price the knap5 head grid on held-out gen rows: `KN_HEADGRID=v3/yaqa/w94/data/sim94-search-gen.json:plans94.json`
  with `KN_HG_DONORS=3,4,5,6,7` until the new donors exist; redo it after the 17 donors (headsim94 on the same dumps,
  ~5 card-min). Do not price on qad rows.
- **Runtime:** none (pricing).

### 2026-11-08 6:15 PM CST · #126 · i8 vs i4 at K0.125–1 on the held-out two-sided proxy: PASS, switch to i8 (w95)
- **Status:** proxy done, **PASS** · card-0 smoke waiting for w92 / w94 · **Machine:** colo 24 + Thor 30 jobs, ~5 CPU-h ·
  **Code:** `v3/e3/settle/w95/` · **Rule:** `v3/e3/settle/w95.md` (5b7caff, before data) · **Data:**
  `v3/e3/settle/data/w95/` (`report.md`, 162 pairs)
- **Gate G0 PASS:** the recomputed i4 rows equal w87's #114-03 rows to 9.5e-8.
- **Result:** R = geo rel_B(i8) / rel_B(i4), held out, 27 tensors × 2 samples, 95% tensor-bootstrap CI:
  - **K0.125: 1.0020** (0.998–1.007)
  - **K0.5: 0.9993** (0.997–1.001)
  - **K1: 0.9999** (0.999–1.002)

  B1 / B2 / in-sample and one-sided agree within 0.1%. Largest family: attn_qg at K0.125, ×1.014 (one tensor, 3:attn_qg,
  ×1.053). Every other family is within ±0.4% at every K. No σ moves.
- **Verdict:** switch to i8 at K0.125–1 (rule: unless > 1% worse at any K). This matches #90-02's one-sided "i4 within
  0.3%" and extends it to the sub-1 rates with H_out. Patches for the config / ladder / tests / design doc:
  `local/workers/w95/i8-sub1.patch` (rad-exl3-mix) and `rqah-i8-sub1.patch` (rad-quant-at-home). Settings stamp
  ed97adb7 → 80b5e6d7.
- **Runtime:** none.

### 2026-11-08 5:35 PM CST · #126 · One codebook: i8 at K0.125–1 too (i4 retired): audit, rule committed, proxy + card-0 smoke launched (w95)
- **Status:** running · **Machine:** colo + Thor (CPU proxy, w87's dirs, `w95/out/`), gpubox card 0 (`/data/w95`,
  waits until w92 / w94 are done) · **Code:** `v3/e3/settle/w95/` (`settle95.py`, `plan95.py`, `report95.py`,
  `w95.sh`, `smoke95.py`) · **Rule:** `v3/e3/settle/w95.md`, **committed alone (5b7caff) before any data**
- **Question (user 11-08 5:20 PM CST):** one codebook (i8) at every level to K7, no i4 anywhere (A4 out). Does i8 work
  at K < 1 everywhere, and is it no worse than i4 there?
- **Audit (step 1):** codebook-generic already: exl3frac2's KA = 0 instances (every variant builds them), engine /
  cpukern / deq (i8 at sub-1), radiance's exl3i importer and kernel (salt in `exl3i_plan.h` is codebook-free; the
  device test covers every rate m 2–128 × mul1 / mul1i8 / mul1i4), knap5 (R8(1) = 1.0; sub-1 options are
  codebook-free). Needs change: `rxq/ladder.py` (the low series hard-codes "i4"), v3.toml (`by_level`, `i4_twins`,
  `[ladder] i4_low` → `low`), tests `check_ladder` / `check_cpujob`, rqah `rxq_e2e` (`0.5-i4` → `0.5-i8`).
  Patches in `local/workers/w95/`. With them: check_ladder 5/5, check_cpujob 25/25 (K0.125 / 0.5 at i8, dense and
  blocks), rqah rxq_e2e 9/9. **New settings stamp 80b5e6d77ef57fc6** (was ed97adb7). Above K1 the level jobs differ
  only in the stamped settings-file pin: w92's K2.5 rows stay valid by content (re-import as trusted rows of the new
  level; the shards' metadata keep ed97adb7).
- **Proxy (step 2):** i4 vs i8 at K0.125 / 0.5 / 1, today's σ, 27 tensors × 2 samples, paired per job; G0 = i4 rows
  equal w87's. Rule: switch unless i8's held-out two-sided error is > 1% worse at any K.
- **Smoke (step 3):** K1 and K0.125 i8 whole levels (256 jobs each, est 1.30 + 3.27 R9700-h) from the patched tree →
  assemble → C9 apply → pack into V2 3.00 → golden-search27c both parts (score only, no swj).
- **Runtime:** none (same decode path, another table; i4 code paths become dead).

### 2026-11-08 5:15 PM CST · #124, #125 · Gen-only head inputs (A2b) and the head-setting CPU batch (A6): rule committed, runs launched (w94)
- **Status:** running · **Machine:** gpubox card 0 (`/data/w94`, waits on card0.lock behind w92, then ram.lock shared
  per step), colo (`/data/work/w13/w94`, nice 19, 16 single-threaded jobs) · **Code:** `v3/yaqa/w94/` (`headx94.py`,
  `gate94.py`, `headsim94.py`, `run94.sh`, `hq94.py`, `q94.sh`, `mkdocs94.py`, `mkblocks94.py`) · **Rule:**
  `v3/yaqa/w94/RULES.md`, **committed alone (650436c; time fix 2ec8d45) before any dump or arm**
- **#124 (A2b):** the old `336.x` prefill dump is gone (11-05 crash), so the new dumps carry every row of a ≤ 512-row
  window per document with its gen flag: search = the 103 calib docs of the retired golden-search27 (no fresh search
  doc spent; 46,712 rows / 38,960 gen), qad = 85 seeded qad docs (36,828 / 33,172 gen). bf16 forward on ONE card,
  layer-streamed (HF modules); CPU smoke test reproduces the golden (KL vs top-512 2e-4, top-1 0.99–1.00). G1 gate per
  dump, then headsim94 (every row kept, document bootstrap) over plan.json + a 1/8-bit curve on today's K3–K7 donors,
  with a τ grid on u3–u7 / s5-3 / s6-3.
- **#125 (A6):** CPU proxy of the head quant on 64 stratified 128-row blocks (sensitivity order, exllamav3 regularize,
  CPU trellis mul1x_i8, σ 0.025): C0 (today's recipe), B (mean-row), T(σo 0.01 / 0.1 / 1) two-sided with the
  softmax-Fisher diagonal, R4 / R8 rank deflation; K3 / K4; exact head-only KL on the new rows from the changed rows
  only (stratified estimate). Proxy fidelity before the rule: K3 g_scale 0.7364 vs the real donor's 0.7369; block 0
  rel 4.61e-4 vs the real donor's 4.81e-4.
- **Found:** today's head donors (`/data/headslab3` K3–K7) were made **without mean-row** (proxy.jsonl
  `head_mean_row: false`, σ 0.025, i8), although mean-row was adopted for every donor on 11-02 (−30% head KL).
- **Runtime:** none (statistics; quantizer settings). Rank-r would be P+D (see the rule).

### 2026-11-08 4:58 PM CST · #39-01, #— · Gain stays 1.0 on today's stack (held out); i4 twins dropped (ladder 55 levels, settings stamp unchanged); stamps and checks (w93)
- **Status:** #39-01 done: **keep 1.0** (null) · **Machine:** colo (16 jobs), Thor (38 incl. ffn_down), ~6 CPU-h ·
  **Code:** `v3/e3/settle/w93/settle93.py` (w87's held-out two-sided proxy + gain + #117 order), `report93.py` ·
  **Rule:** `v3/e3/settle/w93.md` item 2, committed e92d317 before data · **Data:** `v3/e3/settle/data/w93/gain/`
  (`report.md`, 612 rows)
- **Question:** #39 ("flat, keep 1.0", w43) ran on settle.py's reversed LDLQ order without H_out. Does a gain m on
  exllamav3's g_scale beat 1.0 on today's objective (two-sided bn1920 H_out, #117 block order on the readers, i8)?
- **Result:** gate PASS on both boxes (identity order bit-identical; the #117 order changes every tile). Geo r over 27
  tensors × 2 samples, held-out B2 (K2.5 / K4): m 0.9 1.028 / 1.043, 0.95 1.008 / 1.014, **1.0 1**, 1.05 1.006 / 1.004,
  1.1 1.024 / 1.025. m* = 1.0 on B1 and on B2; in-sample A and the one-sided error give the same curve, before the
  refit too. No family qualifies (best: attn_qg K4 m 1.05, +0.63%, one tensor at 1.016; rule ≥ 1% at both K and every
  tensor ≤ 1). **Verdict: keep 1.0** (v3.toml unchanged). Reported: the #117 order on this proxy, at m 1, gives readers
  ×0.978 (K2.5) / ×0.978 (K4) in two-sided held-out error, 36 / 36 better (attn_qg 0.957, attn_k 0.966, gate_up 0.993).
- **Config (review item 4; lead note 4:25 PM CST, user: drop the i4 twins):** the K2 / K3 twins served nothing besides
  #46-03 (rxq.ladder is their only consumer; without A4 the knapsack never picks an i4 twin; gpucheck27's K2 i4 run is
  a kernel test). Applied: `[ladder] i4_twin_range = []` (unstamped) + one line in `rxq/ladder.py` (an empty range) +
  `tests/check_ladder.py` (55 levels); `i4_twins` (stamped) left as an inert value so the **settings stamp stays
  ed97adb7ed1f9919** and w92's K2.5 / K1 levels stay valid anchors (level names unchanged). **71 → 55 levels; quant
  115.1 → 95.4 card-h, swj passes 53.2 → 41.2 card-h, packs 31.6 → 24.4 CPU-h, 18,176 → 14,080 jobs.** v3.toml stale
  comments cleaned (a4 / #46-03, H_out cost, tile CD's K4 pair, swj scope, #105). design/v3.md, pipeline DESIGN /
  README, TODO updated. Patch: `local/workers/w93/ladder-no-i4-twins.patch`.
- **Stamps (review item 7):** settings **ed97adb7ed1f9919**. Code stamp (GPU kernels, fork 6242f7ae = eb5a2139 +
  0008): 908d6198788264f0 at HEAD 26ec0c9 (same at 50902fc; was fd34de5c208b8057 at w86's f15f619, before the rowerr
  patch) → **7b9503f380e845c4 with the twin drop** (rxq/ladder.py is hashed). `w93/stamp93.py` recomputes it from git
  objects; it equals `engine.code_stamp` on the tree and check_cpujob's stand-in stamps.
- **Checks (before / after the twin drop):** `check_ladder` 5/5 / 5/5 (71 / 55 levels); `check_cpujob
  FORK@eb5a2139 FORK@6242f7ae` 25/25 / 25/25 (dense side from a git worktree at eb5a2139); rqah `tests/rxq_e2e.py`
  (branch ladder 1741b5c) 9/9 / 9/9.
- **Runtime:** none.

### 2026-11-08 4:45 PM CST · #123, #39-01 · Published inputs scanned: 1 rule failure, a structural dead channel (L7 MLP input 3994), not corruption; sha256 all agree; #39 gain re-run launched (w93)
- **Status:** #123 done (rule verdict FAIL on 1 of 657 files, diagnosed benign); #39-01 running · **Machine:** colo
  (`/data/work/w13/w93`, nice 19, ≤ 8 GB), Thor (`/data/work/w93`, ffn_down + 2/3 of the gain jobs) · **Code:**
  `v3/e3/settle/w93/` (`scan93.py`, `scan93.sh`, `settle93.py`, `plan93.py`, `report93scan.py`) · **Rule:**
  `v3/e3/settle/w93.md`, **committed (e92d317) before either launch** · **Data:** `v3/e3/settle/data/w93/` (`verdict.md`)
- **#123 question (freeze review blocker 3):** are the ladder's published inputs intact: 257 calib H_in, 400 bn1920 H_out?
- **Result:** selftest PASS (clean Gram passes; 11-06's corrupt span, a NaN, a zero diagonal and an indefinite matrix with
  every 2×2 minor fine all fail). **H_in, exhaustive** (every packed entry, no sampling): 256 / 257 pass every check;
  Cauchy–Schwarz max ratio 0.9868 over all 12.3G entries (none above 1 + 1e-5), 0 non-finite, damped fp64 Cholesky
  (λ = 1e-3 · mean diag) 256 / 256. **The one failure:** `layers.7.post_attention.input`, H[3994, 3994] = 0 and row /
  column 3994 all exact zeros. Cause, checked: L7 `post_attention_layernorm.weight[3994] = −1.0` exactly (the only −1);
  the (1 + w) RMSNorm makes MLP input channel 3994 identically zero, so this is a dead channel of the bf16 model, not
  damaged data. The quantizer adds σ · mean diag to the whole diagonal, so it factors (L7 gate / up have been quantized at
  every level so far). **H_out:** 400 / 400 present, F32 of out_features, seqs 1920 / blocks 15 / blocknorm 1, finite,
  > 0. **sha256:** colo H_in = HF LFS oids (dataset 2dcf0061, unchanged since download) = Thor's copy = w86's dry-run
  pins (8), 0 mismatches; colo H_out = Thor = the VM backup of gpubox `/data/swap27/hout-bn` (the K4 arms' source) =
  w86 pins (13), 0 mismatches.
- **Found, not failing:** (1) H_out dynamic range: within-tensor max/min up to 3e23 (L14 GDN in_proj_qkv min 4e-23;
  L9 / 10 / 16 qkv 1e14–1e19; q_proj L3 / 7 / 11 / 15 1e6–1e9); harmless (prepare_H_out normalizes by the mean and adds
  σ_out 0.01). (2) Four input Gram matrices are numerically singular: Cholesky pivots equal to λ in L17 GDN out_proj
  and L3 / 7 / 11 / 15 o_proj inputs (the same attention layers whose q rows get ~0 H_out: likely dead or gated-off
  heads). Damping covers them; not investigated further.
- **Decision / next:** nothing for the ladder. By the letter of the rule #123 is FAIL (the rule did not foresee a
  structural dead channel); the lead decides whether the 0-diagonal is accepted as is (recommended).
- **#39-01 (running):** w87's held-out two-sided proxy + gain m ∈ {0.9 … 1.1} × s_exl + #117 block order on the readers,
  i8 K2.5 / K4, 27 tensors × 2 samples; gate PASS on colo and Thor (identity order bit-identical; the #117 order
  changes 2,560 / 2,560 tiles, 11:attn_k s0 rel_B ×0.955). 54 jobs: colo 16, Thor 38 (ffn_down there).
- **Runtime:** none (input check; quantizer setting).

### 2026-11-08 4:35 PM CST · #122 · Full-stack pass-1 level on card 0: rule committed, slice cut, jobs running (w92)
- **Status:** running · **Machine:** gpubox card 0 (`/root/card0.lock` → `/root/ram.lock` shared per layer of jobs;
  packs conv.lock + ram.lock exclusive) · **Code:** `v3/e3/settle/w92/` (`w92.sh` driver, `mkjobs92.py`, `layer92.sh`,
  `verdict92.py`, `sub1chk92.py`), staged on gpubox `/data/w92/code` = `git archive 50902fc` + w92/, fork
  `/data/w92/exl3` = 6242f7ae (eb5a2139 + 0008) · **Rule:** `v3/e3/settle/w92.md`, **committed alone (26ec0c9) before
  the slice, any quant or score** · **Data:** `v3/e3/settle/data/w92/` (when done)
- **What:** freeze review 11-08 §1 item 2. Levels `q27v3-ed97adb7-k2.5-i8` (seed offsets 0 and 1) and
  `q27v3-ed97adb7-k1-i4` from `v3.toml` (settings stamp ed97adb7ed1f9919): `rxq.ladder.level_jobs` (256 group jobs a
  level) → the spec rqah's `rxq_quant.stage()` writes → `python -m rxq job` one at a time → `rxq assemble` →
  `rxq.swjfit` (SWJ27's qad meta) → `rxq.post apply LEVEL SWJ,LEVEL/c9` → `rxq.mix recipe` → rad-convert → V2 3.00.
  Gate: control → full stack ≤ control → bn+swj (geo over the pairs) and nothing > 1.0 per pair / part; controls packed
  fresh (CTL_FRESH=1), every pair hygiene-checked (only trunk weights differ).
- **Fresh slice golden-search27c:** calib part 26 search-pool docs / 67,958 tokens (51,773 scored), seed 9222,
  golden-search27 and golden-search27b-c excluded; xcorp part 65 pieces / 93,824 rows of night1105's never-scored
  `xc8k-b2-c2` (stratified, mX docs excluded; no doc shared with golden-search27b-x).
- **Found while setting up:** (1) **K1 is not a sub-1 level in v3.toml**: `sigma_reg_in_sub1` applies at K < 1 and σ 0.2
  at K ≤ 0.125, so the K1 i4 level does not exercise the sub-1 damping. Added a job-level check at K0.875 / K0.125 on
  layers 2–3 (settings logged, H_out blocks, block order, tile CD, C9). (2) The swap27 radiance home (install-116) has
  no mul1i4 importer: K1 i4 is packed / scored with install-117 + `/data/a4cost/home` (#46-03's). (3) gpubox's
  `/data/exllamav3` (on every earlier PYTHONPATH) lacks fork patch 0008; the blocks path needs a 0008 fork first on
  the path (w81's gate used `/data/w81/exl3`; w92 stages 6242f7ae).
- **Runtime:** none (no setting, format or kernel changes).

### 2026-11-08 1:50 PM CST · #120-03 · 0.8B drift / variance split: 70% (K2) / 32% (K3) of the plain-level KL is seed-coherent; a per-channel bias takes 1.1% at K3 (rule: POSITIVE, narrowly); design note says no exl3i field (w91)
- **Status:** done (POSITIVE by the rule) · **Machine:** 4070 (`/data/yb/gpu.lock`, three holds between w79's units,
  ~10 min GPU; queue `/data/queue.d/85-w91drift.sh`, done) · **Code:** `v3/e3/settle/w91/drift91.py`, `q91.sh`,
  `readout91.py` · **Rule:** `v3/e3/settle/w91.md` item 2 (12:38 PM CST, before data) · **Data:**
  `v3/e3/settle/data/w91/drift/`
- **Setup:** the existing 6-seed plain 0.8B levels (`/data/yb/q-k{2,3}base{,_s1..s5}`, ourq i8), g08-search1 (166,155
  gen rows), the challenge's 513-cell KL. G3 PASS: every seed's rows equal the stored ones bit for bit.
- **Split (finite-seed corrected, document bootstrap):** K2 KL 0.537, seed-averaged logits 0.402 (ratio 0.748),
  **coherent share c 0.697 [0.684, 0.711]**; K3 KL 0.0993 → 0.0427 (0.430), **c 0.316 [0.308, 0.325]**. The same
  from the variances (0.694 / 0.321). Uniform across categories; on the worst 1% of rows 0.90 / 0.49.
- **Bias oracle (β = (W − Ŵ) μ on all 150 linears, μ from the quantized model on 32k fit-pool rows):** K3 0.9880 /
  0.9892 / 0.9904 (each z ≈ −10), geo 0.9892; K2 1.0001 / 0.9754 / 0.9924. **Rule: POSITIVE** (c ≥ 0.10, and the K3
  geo ≤ 0.99 with all seeds < 1; a hair under the bar).
- **Reading:** most of the plain-level error is coherent across seeds, but only ~3% of the coherent part is
  bias-shaped. The rest is most likely gain shrink, which swj / C9 target in the adopted stack (not tested here).
  Plain levels make both numbers upper estimates for V3.
- **Item 3 (design note `v3/e3/settle/w91/exl3i-fields.md`):**
  - **reserve no exl3i field.** A bias comes from fixed codes after quantization, so it never forces a ladder requant.
    The clean form is an optional per-linear `blk.N.<w>.bias` weight in the qwen35_exl3 plugin: absent = today's
    container and graph byte for byte. Cost ~1 day, 0.005 bpw, an add per linear (prefill ~10–30 ms per 2048-token
    chunk unfused; must be timed).
  - Sparse exact rows: not reserved (#120-02 DROP).
  - Next check before any build: the same oracle on the **adopted** 0.8B stack (writers / readers, μ from bf16 vs
    quantized). Then one 27B pair with a dev-build bias. Added to the BACKLOG 27B confirmation queue.
- **Runtime:** none (diagnostic); a shipped bias: P+D (see the note).

### 2026-11-08 2:55 PM CST · #34-02, #99 · Slab head with corrected rowerr: BUILD (both pairs pass); the head's whole share at 4.0 is small on xcorp, ~5% on calib (w90, read by the lead)
- **Status:** #34-02 decided (BUILD the slab head into the plugin) · #99 done (report-only) · **Machine:** gpubox card 0
  (headwm90, done 1:57 PM CST) · **Code / rule:** `v3/yaqa/w90/headwm90.sh`, `v3/yaqa/HEADWM.md` amendment 1 ·
  **Data:** gpubox `/data/w90/hw.out` (copy `v3/yaqa/w90/data/hw.out`)
- golden-search27b (101 docs, 143,351 positions; trunk = V2 4.00), pooled r [95% CI] / xcorp part / calib part:

  | pair | pooled | xcorp | calib | p99 | top-1 |
  |---|---|---|---|---|---|
  | v4 → h4-bf16 (whole head error at 4.0) | 0.9938 [0.9688, 0.9981] | 0.9966 | **0.9466** | 1.2237 → 1.2234 | 0.9501 → 0.9531 |
  | h4-u5 → h4-s5-3 (equal bytes, K≈5.0) | **0.9960** [0.9821, 0.9984] | 0.9976 | **0.9682** | 1.2202 → 1.2296 | 0.9506 → 0.9523 |
  | h4-u6 → h4-s6-3 (equal bytes, K=6.0) | 0.9992 [0.9962, 0.9997] | 0.9996 | 0.9936 | 1.2267 → 1.2283 | 0.9520 → 0.9523 |
  | h4-bf16 → h4-s6-3 | 1.0006 | 1.0004 | 1.0049 | | |
  | #99: v8 → h8-bf16 | 0.9972 [0.9859, 0.9989] | 0.9985 | 0.9541 | 0.0348 → 0.0348 | 0.9895 → 0.9901 |

- **Verdict (rule, amendment 1): BUILD** — both pairs pass (pooled upper bound < 1, both parts < 1); validity PASS.
- **Reading (lead):** the head-only predictions (−86% / −61% of head KL) mostly hold relative to the head's share:
  s5-3 recovers ~65% of the u5 → bf16 gap, s6-3 sits within 0.06% of a bf16 head. **But the head's whole share is
  small on xcorp (bf16 head −0.3%) and ~5% on the calib part** (−5.3% at 4.0, −4.6% at 8.0), so the effect depends on
  the corpus: at equal head bytes s5-3 is −0.2% on xcorp and −3.2% on calib-style documents. The private corpus is
  drawn like calib, so the calib part is the relevant size (n = ~35 docs, one slice: not yet checked on verify).
  p99 rises slightly in s5-3 (+0.8%) while top-1 improves. The lead's 11-08 "~0.05 bpw" estimate assumed the calib
  share everywhere; on a calib-like corpus ~0.02–0.03 bpw at 4.0 is the honest figure, plus any head bytes the knapsack
  frees (w88's grid, to be repriced with corrected plans).
- **Follow-ups:** slab-head kernel support (row-slab K per 128-row block) in the plugin; reprice w88's knap5 head grid
  with corrected plans; one verify-pool check of s5-3 vs u5 with the finished mixes. rxq has the same rowerr bug
  (`local/workers/w90/rxq-rowerr.patch`): **applied by the lead 2:55 PM CST** (no ladder jobs queued yet); check_ladder 5/5, check_cpujob 25/25 PASS. gpubox `/data/ourq/ourq.py` patched too (original kept as `ourq.py.pre-rowerr`). w90 / w91 ended rc=1
  at 1:52 PM CST (usage limit) before writing reports; this entry is the lead's reading of their outputs.
- **Not investigated (user, 11-08):** why the head's share is ~15× larger on the calib part than on xcorp (dilution by a larger trunk KLD, rarer vocab on calib, or the qad-Fisher plan targeting calib-style usage). Reason: the head is already near its ceiling (s6-3 within 0.06% of a bf16 head), so we did not expect the answer to lead to any significant further optimization. Consequence kept: **pooled two-part search slices (~65% xcorp) understate head / embedding effects on the board**; size head-side changes on the calib part.
- **Runtime:** decode / prefill: slab head = per-block K in one GEMM (same bytes); cost to measure with the kernel.

### 2026-11-08 1:30 PM CST · #120-01, #120-02, #121 · Massive channel 3994: it dominates the energy, not the error; the bf16-restore oracle is null (DROP); damping normalization: mean stands (w91)
- **Status:** #120-02 negative (DROP) · #121 negative (mean stands) · #120-03 see the 1:50 PM entry · **Machine:** gpubox card 0
  (~14 min: census 71 s + 10 scores) + CPU; colo (B6, 14 jobs, ~2 CPU-h) · **Code:** `v3/e3/settle/w91/` · **Rules:**
  `v3/e3/settle/w91.md` (items 1–3 written 12:38 PM CST, B6 12:52 PM CST, before data) · **Data:** `v3/e3/settle/data/w91/`
  (rows on gpubox `/data/w91/b27/`)
- **Step 0 (#120-01, lead / user: verify the claim first):** census of all 128 writers (bf16, calib H_in) + the 129
  residual-input diagonals (`data/w91/hdiag91.txt`). Channel 3994 IS the massive channel: 0.80–0.93 of the layer-input
  trace at L1–L3 / L7 / L11, 0.4–0.8 to L50, ≤ 0.16 after L51; mean / median of that diagonal 28 at L3 / L7. Writer
  output energy of row 3994: 0.99 / 0.87 / 0.58 (L0–L2 out_proj), 0.96 (L3 o_proj), 0.05–0.62 (L0–L3 down). No other
  channel > 5% of any reader; other writer rows > 5%: 3456 (L21 out_proj), 1089 (L63 o_proj). **But** at K2.5 row 3994
  holds only 1.4–6.3% of the one-sided weighted error of L0–L3's out_proj / o_proj, 0.2–2.4% of the down_projs,
  ≤ 0.7% in L4–L63 (the merge's "82%" was a two-sided figure from one session); its relative error equals its tensor's
  (~4%). Patch set by the rule: S = {3994}; the depth arm was not triggered.
- **#120-02, 27B fixed-code oracle (item 1):** L0–L3's 8 writers stored dense bf16 = decode of the K2.5 codes (C), and
  the same with row 3994 = the bf16 checkpoint row (A), spliced into the stored base / s1 mixes (`rad-convert --reuse
  --changed`, radmix). Gates: slice identical to w84's golden-search27b (rebuilt; `/data/w84` was gone), PAIR OK (only
  the 8 writers differ), byte check (A vs C differ only in row 3994), decode gate (control mix → C r 1.0016, PASS).
  **p0 0.9982 [0.9919, 1.0014] (z −1.06), p1 1.0018 [0.9978, 1.0056] (z +1.12); R = geo 1.0000 (x 1.0001, c 0.9996)
  → DROP** (rule: R ≤ 0.995 to keep the theme). The ceiling of every writer-side 3994 fix at L0–L3 (per-channel bias,
  sparse exact rows, row extraction) is ~0.
- **#121, merge B6 (lead note):** damping δ = σ × N(H_A) with N = mean (today) / median / (tr − λmax)/(n−1) of the
  diagonal, σ as today, on w87's held-out two-sided proxy (5 early layer-input readers L2 / L3 / L7 / L10 / L11 + L50 /
  L63, 2 samples, i8 K2.5 / K3 deciding). The alternatives cut δ 18–28× at L3 / L7. n* = median (B1 0.9990); **held-out
  B2 gain +0.17%, 3 / 7 tensors agree → mean normalization stands** (rule: ≥ 2% and ≥ 5 / 7). Best single tensor: L3
  attn_qg −1.6%. Reported only: at i4 K1 the median is +0.8% worse, at K0.5 +2.4% (early readers +3.2%). No σ re-sweep
  (it was conditional on a win).
- **Runtime:** none (oracle; quantizer setting).

### 2026-11-08 1:25 PM CST · #34-02, #99, #89-07, #89-08 · rowerr fix applied (ourq) + CPU test; HEADWM rule amended to golden-search27b; headwm rerunning; A4 4070 arms dropped / deferred (w90)
- **Status:** #34-02 / #99 running (card 0) · #89-07 dropped · #89-08 planned (deferred) · **Machine:** gpubox card 0
  (with w91's oracle, by the locks), VM CPU (test) · **Code:** `quantize/ourq.py` (+ `v3/e3/settle/reval/ourq-rowerr.patch`),
  `quantize/check_rowerr.py`, `v3/yaqa/w90/headwm90.sh`, `verdict90.py` · **Rule:** `v3/yaqa/HEADWM.md` amendment 1
- **ourq fix (lead note):** w46's `ourq-rowerr.patch` applied to `quantize/ourq.py` (2 lines: CPU copy of w before
  `quantize_exl3` when `--save-wq`, rowerr from that copy). **`check_rowerr.py` (CPU, the fork's real quantize.py
  through rxq's stand-in kernels): 5 / 5 PASS:** quantize_exl3 changes its weight argument (premise); pre-patch
  arithmetic sum err / ref 1.001 vs proxy 0.03157; patched 0.03157 vs 0.03157 (ratio 1.0000); wq / proxy / stored
  tensors identical with and without the copy; ourq.py holds the patch in order. The gpubox copy
  `/data/ourq/ourq.py` (older, no +io lines) is not patched: future `--save-wq` head donors need it.
  **rxq has the same bug** (`pipeline/rxq/quant.py`, ported "unchanged for parity", PORTING.md): patch
  `local/workers/w90/rxq-rowerr.patch` (same 2 lines, applies), not applied: it changes rxq's code stamp (lead).
- **HEADWM amendment 1 (written 1:20 PM CST, before any data on the new slice):** plans from the corrected rowerr;
  every arm scored on golden-search27b's two parts (golden-search27 retired); per pair, pooled r < 1 with 95% upper
  bound < 1 **and** r_x, r_c < 1; validity: h4-bf16 finite, pooled r(v4 → h4-bf16) < 1; #99 report-only, no TP2 arm.
- **Run:** headslab3's broken rowerr, `plan.json` and the s5-3 mix are in `/data/headslab3/broken-rowerr-1108/`; the
  corrected rowerr is installed and the plans are rebuilt (identical to w90's: s5-3 `60928@K7 64512@K5 122880@K4`,
  s6-3 `102400@K7 43520@K6 102400@K5`). Arms: v4, h4-bf16, h4-u5, h4-s5-3, h4-u6, h4-s6-3, then v8 / h8-bf16.
  golden-search27b copies `/data/w90/golden-search27b-{x,c}` (reflinks of w91's rebuild; kld.json = w84's).
- **4070 (user 1:10 PM CST):** the top-512 arm (#89-07) is dropped (our goldens are top-512 already) and the ¼ / ⅛
  data arms (#89-08) are deferred until the trunk quant starts. Neither had started. The 4070 queue now runs only
  #89-06c's student-path reading (`/data/queue.d/80-w79s.sh`).
- **Runtime:** none.

### 2026-11-08 12:55 PM CST · #89-06c, #89-07, #89-08 · 0.8B healing at K4 / K5 running; step 0: the fold path IS the K5 confounder; A4 prerequisites queued (w90)
- **Status:** running · **Machine:** 4070 (user freed it 12:30 PM CST; `/data/yb/gpu.lock`, VRAM cap 7.2 GB) ·
  **Code:** `proxy2b/exp/w79-heal45/` (w79, rule 11-08 1:25 AM CST), `proxy2b/exp/w90-a4heal/` (w90) · **Queues:**
  `/data/queue.d/70-w79heal.sh` (launched 12:39 PM CST), `80-w90a4.sh` (waits for `W79 HEAL DONE`)
- **#89-06c step 0 (12:39–12:42 PM CST), w79's gate:** student path (heal.py's wrappers, no fold, no f16 → bf16
  re-rounding) vs fold path, g08-search1, seed 0. Stacks reproduce exactly (0.0054978 / 0.0185906, ratio 1.0000).
  K5 lr 3e-5 healed / stack: **student 0.9930 (z −4.9) vs fold 1.1441**; K5 lr 1e-4 student 0.9775 (z −9.2) vs fold
  1.0119; K4 lr 3e-4 student 0.9507 (z −14.1) vs fold 0.9494. Jitter (re-rounding alone, SD 0.002): K4 +0.75%
  (z +7.5), K5 +1.85% (z +8.2). By the rule (student < fold − 0.01 at K5) **the fold path's re-rounding is the
  confounder**: every K4 / K5 arm is read on the student path; K4's jitter (0.75%) is its noise band; K5 lr 1e-4 is
  re-read with the K4 rule. w79's `run.sh` / `readout.py` only implement the fold-path reading, so w90 added that
  branch (`w90-a4heal/w79s.sh`, `readout79s.py`; w79's rule unchanged), run when w79's queue ends.
- **For the 27B (not a rule):** at K5 a fold through f16 → bf16 costs more than the healing gains. A 27B healed
  level must fold into the stored f16 suh / svh in the stored precision, with no bf16 re-rounding of the weights.
- **A4 items 2–3 (lead note, rules written 12:40 PM CST before data, `w90-a4heal/README.md`):** #89-07 = heal0 with a
  top-512 + lumped-tail teacher loss (training and held-out); #89-08 = ¼ / ⅛ of heal0's windows at the same 512
  steps; K3 / K2 × seeds 0 / 1 vs d1104's heal0, plus one heal0 rerun (noise floor). ~3.8 h after #89-06c.
- **Runtime:** none.

### 2026-11-08 12:50 PM CST · #34-02 · `ourq --save-wq` head rowerr is broken (err ≈ ref): every slab-head plan was chosen on garbage costs; corrected plans built, headwm held for the lead (w90)
- **Status:** blocked (lead: rebuild the plans before #34-02 is read) · **Machine:** gpubox card 0 (~1 min) + CPU ·
  **Code:** `local/workers/w90/rowerr90.py`, `rowerr90.sh` · **Data:** gpubox `/data/w90/rowerr/` (`plan.json`,
  `plan.log`, `rowerr90.log`; `/data/headslab3` untouched)
- **Check (lead note A2):** the head quants' `wq/lm_head.rowerr.npz` give sum(err) / sum(ref) = 1.0006 / 1.0007 /
  1.0008 / 1.0010 / 1.0011 at K3–K7, against logged proxies of 2.9e-3 … 2.1e-5. Cause: the fork's (and upstream
  a3bfbfd0's) `quantize_exl3` regularizes its weight argument in place (`weight /= sv`, Hadamards, `/= su`,
  `*= g_scale`; `.to(device)` on a tensor already there is no copy), and ourq's save-wq block (since 15f03a7, 10-31)
  compares that transformed w with wq. wq itself (returned in the original space) is fine. w46 found the same on the
  0.8B (`v3/e3/settle/reval.md`, side finding) and wrote `v3/e3/settle/reval/ourq-rowerr.patch` (CPU copy of w when
  `--save-wq`; still applies to `quantize/ourq.py`), never applied.
- **Recomputed** from the saved wq + bf16 lm_head (permuted) + raw calib H: sum(err) / sum(ref) = 2.887e-3 (K3),
  7.3948e-4 (K4: logged proxy 7.3945e-4), 1.9562e-4 (K5: 1.9559e-4), 5.5106e-5 (K6: 5.5078e-5), 2.1457e-5 (K7:
  2.1430e-5): within 0.13% of the proxy. Corrected `headslab.py plan` (predicted head-KL change vs uniform, training
  Fisher): **s5-3 `60928@K7 64512@K5 122880@K4` −85.6%** (headwm's: `50688@K7 48128@K6 149504@K4` −25.8%);
  s4-3 −91.7% (was −36.2%); s6-3 `102400@K7 43520@K6 102400@K5` −60.5% (was nearly the same layout, −6.2%).
- **Consequences:** the built `mix-h4-s5-3.rad` and headslab3's `plan.json` are wrong-plan artifacts. v4, h4-u5 (and
  u6, bf16 arms) do not read rowerr and stand. The 11-01 slab-head plans (`headslab2`: lcurve, fcurve-A/B; behind
  w88's knap5 head grid) were very likely chosen on the same broken costs (same code since 10-31; files deleted, not
  verifiable). Their headsim KL is measured and real, but they are not the plans the planner meant, so the slab-head
  menu probably understates the slab head.
- **Cost of the fix:** ~1 min of card 0 per donor set (recompute from the saved wq, no head requant) + seconds of
  CPU for the plan. To resume headwm: move headslab3's `K*/wq/lm_head.rowerr.npz`, `plan.json` and `mix-h4-s5-3.rad`
  aside, install `/data/w90/rowerr/K*/wq/lm_head.rowerr.npz`, rerun `headwm.sh` (rebuilds plans and s5-3 / s6-3;
  ~2–3 h of card 0 for the remaining scores + V2 8.00 download). Apply `ourq-rowerr.patch` for future donors.
  Also: the HEADWM rule reads golden-search27, which was retired for decisions this morning (the lead may amend it to
  golden-search27b before resuming).
- **Runtime:** none (quantizer-side bookkeeping).

### 2026-11-08 12:35 PM CST · #117 · rxq's block order reproduces w84's route B bit for bit on card 0: PASS 9/9 (w90)
- **Status:** adopted (unchanged; the freeze gate passes) · **Machine:** gpubox card 0, 12:31–12:34 PM CST ·
  **Code:** `pipeline/tools/blkorder27.sh` (HEAD ee475a3) · **Driver:** `local/workers/w90/blk90.sh`
- **Reference regenerated** (w84's `/data/w84/p0/q-perm` was deleted): w84's committed route (`w84q.py` +
  `blkorder84.py` + `order82.json` from cbc4629, staged as w84 did), rxq levelargs of `/data/v3ladder2/K2.5` + tile CD
  off, the 9 readers of w84's noop gate → `/data/w90/p0/q-perm` (82 s). It differs from the control level (27 / 36
  tensors), so the gate is not vacuous.
- **Gate:** `blkorder27.sh` (rxq `quant.block_order` on, HEAD code) vs that reference: trellis / suh / svh / mul1
  identical on **9 / 9** (`BLKORDER27 PASS 9/9`; rxq log `block_order True` on each).
- **Caveat:** both runs used today's `/data/exl3frac/src`; the reference is a regeneration, not w84's original bytes.
- **Runtime:** none.

### 2026-11-08 11:40 AM CST · #114-03 · Damping with H_out on (two-sided LDLQ), held out: settings stand (w87)
- **Status:** null (keep) · **Machine:** colo (`/data/work/w13/w87`, nice 19, ≤ 8 GB) + Thor (`/data/work/w87`),
  9:29–11:28 AM CST, no GPU · **Code:** `v3/e3/settle/w87/` (settle87 / gate87 / plan87 / q87 / report87) ·
  **Rule:** `v3/e3/settle/w87.md` (written 9:25 AM CST, before data) · **Data:** `v3/e3/settle/data/w87/`
  (`report.md`, `out/`, `gates/`)
- **Question:** #114 / #114-02 tuned σ without H_out. Does the best σ move now that every level is two-sided
  (bn1920 diagonal H_out, σ_out 0.01)?
- **Setup:** w77's held-out proxy: same 27 tensors × 2 samples, quantizer on H_A, scored on H_B1 / H_B2, w77's scales
  and row cache. Added: exllamav3's `prepare_H_out` + `ldlq_2hess` (exact on a 128-row block, because the rotated
  diagonal H_out is block-diagonal by 128), and the H_out-weighted error Σ h_i d_i H d_iᵀ.
  - σ ∈ {0.001, 0.005, 0.025, 0.05, 0.1, 0.2}.
  - Levels i4 K0.125 / 0.5 / 1, i8 K2 / 2.5 / 4.
  - 1,944 rows.
- **Gates PASS:**
  - G1: the loop equals the fork's own `ldlq_2hess` + `block_ldl` to 4e-16, random and real blocks.
  - G2: with H_out = I, bit-identical to w77's one-sided path.
  - G3: 54 / 54 H_out shapes OK.
- **Result:** **0 of 66 cells pass** (rule: B2 gain ≥ 2% with ⅔ of tensors agreeing).
  - Max gain +1.24% (K0.5 attn_k, 0.1 vs 0.05). Every group cell is within ±0.5%. B1 / B2 picks agree 49 / 66.
  - The curves are w77's, shifted by ≤ 0.5%:
    - K0.125: minimum at the current 0.2, both groups.
    - K0.5: inputs minimum 0.1 (−0.27% vs 0.05); rest 0.1 (−0.5%), as without H_out.
    - K1: flat.
    - K2–K4: inputs flat 0.001–0.005, rest flat 0.001–0.025, both rising from ~0.05.
- **Decision / next:** the settings stand with H_out: nothing for v3.toml. Covers w81's block-diagonal path (#115,
  same objective).
  - Not measured: the #117 block order (on since 11:15 AM). The proxy uses exllamav3's natural reverse order.
- **Runtime:** none (quantizer setting).

### 2026-11-08 11:15 AM CST · pipeline / rqah · ladder rollout prep: rxq_quant jobs from v3.toml, CPU dry run, #117 on, blocks H_out on (w86)
- **Status:** done (no GPU) · **Machine:** VM (tests), colo CPU (`/data/work/w13/w86`, nice 19, ≤ 8 GB RSS) · **Code:**
  rad-exl3-mix `pipeline/` (uncommitted, for review), `quantize/` (w81's patch applied, `exl3-patches/0008`), rqah branch
  `ladder` (4 commits, not merged); notes `local/workers/w86/README.md`.
- **Settings (lead notes 9:55 / 10:00 AM):** #117 block order **on** (`quant.block_order`, route B ported to
  `rxq/blkorder.py`; == w84's blkorder84.py bit for bit on the CPU stand-in, tile CD off and on); H_out on w81's
  **block-diagonal path** by default (`inputs.hout.ldlq = blocks`; w81's rxm-hout-blocks.patch applied; fork patch 0008
  in the series: upstream a3bfbfd0 + 0001–0008 = fork eb5a2139 + 0008, tree 8b6ab82b). New stamp **`ed97adb7ed1f9919`**.
  `rxq levelargs` maps ourq levels to ldlq dense and block order off.
- **Jobs:** `python -m rxq.ladder` — 71 levels (pass 1 24, pass 2 26, pass 2b 14 conditional i4 twins, pass 3 7 sub-1),
  one rqah `rxq_quant` job per Hessian group (18,176), inputs pinned by sha256 (settings file, group weights, group
  Hessian, H_out per key), requirements from w81's 11-08 R9700 measurements (gate/up H_out 3.17 GiB blocks / 6.40 dense,
  ffn_down 7.72, sub-1 +3 GiB trellis buffer; w80's i4 sub-1 job 4.6–5.6 GiB, RSS 2.5 GiB). **Estimate (R9700
  card-h, blocks):** quant 90 (115 with 5 s a job of process overhead; dense would be 296), pass 1 29 / 37; a K4 level
  41 min (w81's 37 + tile CD + C9); swj level passes 53 card-h on our 32 GB cards; packs 32 CPU-h.
- **CPU stand-in** (`rxq/cpukern.py`, test only): the fork's real quantize.py on a CPU with a greedy trellis encoder;
  codes decode exactly (rxq.deq and rqah's exl3dec, 18/18 rates); stamped `cpu-standin`, refused by assembly and queue.
- **Dry run (colo, real 27B layers 2 GDN + 3 attention, K0.5 i4 / K2.5 i8 / K6.5 mul1, current settings):** datapack →
  18 jobs → rqah server + worker → all accepted, scored (18 pass after the fix below) → exported (rxq assemble) →
  **13/13 tensor + C9 files per level bit-identical to direct `rxq quant`**, exported levels == direct assemblies;
  readers block-ordered, blocks H_out, tile CD, C9 on out_proj / o_proj; job RSS ≤ 3.6 GiB. ffn_down left out: its CPU
  run peaks at 8.4 GB (two 17408² copies in the refit; rule 8 GB); the tiny-model tests cover its path. **Pack check
  blocked on CPU:** colo's host radiance build has no exl3i kernel plugin (device only), so rad-convert's declare fails
  after the importer accepts i8 / mul1 levels; its exl3quant predates mul1i4 (0.3.0). Packing = the first real level on
  gpubox.
- **Bugs found and fixed:** rxq assemble's stamp.json had no `args` → rad-convert's exl3i importer read codebook mul1
  and refused i8 / i4 levels (and would have accepted a wrong `codebook=mul1` recipe); rqah's format rule failed an
  honest K0.5 q_proj (the refit underflowed 1 of 5120 suh to f16 zero → a donor would be quarantined): ≤ 1% zeros now
  valid; rqah hard-coded quant type lists (6 places) → `jobs.QUANT`; export with no accepted job crashed; `check_i4.py`
  still expected the pre-11-08 settings. CPU-only: block_ldl's H.cpu() aliasing, quant_args devices.
- **Tests (CPU):** check_ladder 5/5 (every job of all 71 levels through rqah's checks), check_cpukern 18/18,
  check_cpujob 25/25, check_blkorder 17/17, check_blkorder84 3/3, check_i4 4/4, grouphead 5/5, tilecd 7/7, pairs 22/22;
  w81 check_blockdiag ALL PASS, check_rxq_hout ALL PASS with block order off (with it on, its levelargs case sees the
  new `--set quant.block_order.enable=false`); rqah rxq_e2e 9/9, validity 9/9, sched 40, score 40, statuspage 26,
  probation 16, checker_nan 12, e2e_noop 135, upgrade_e2e 23, update 81, bandwidth 22.
- **For the lead:** GPU checks prepared: `pipeline/tools/blkorder27.sh` (p0perm's 9 readers re-made by rxq on card 0,
  bit for bit). The worker images still need rad-exl3-mix `pipeline/` (docker/build.sh RXM_PATHS + a pin with it).
  The fork working copy lacks the 0008 commit (`git am` it). Runtime: none.

### 2026-11-08 9:55 AM CST · #117 · residual-block LDLQ order ADOPTED (user)
- Route B only (w84): the 272 residual readers' LDLQ visits 128-blocks in the consensus heavy-last order; stored in
  natural order; no model conversion, no format / kernel / runtime change (~30 lines in rxq). Evidence: geo 0.9968 on
  golden-search27b (z −0.62, both parts < 1), proxy 272/272 better. w86 wires it ON in v3.toml (settings stamp).

### 2026-11-08 9:40 AM CST · #115 · Fast H_out: block-diagonal LDLQ passes its gate; an H_out level costs ~1.07× a one-sided one (dense: 5.6×) (w81)
- **Status:** done · **Machine:** gpubox card 0 (23 rxq units between the queue's steps, 09:54–15:31 UTC; ~45 card-min
  in total), CPU f2 on gpubox · **Code:** `local/workers/w81/` (fork patch 0008, `gate.sh`, `f2calc.py`, `gate_report.py`)
- **Setup:** 27B L10 (GDN + MLP) and L39 (attention + MLP), K4 i8, bn H_out (`/data/swap27/hout-bn`, σ_out 0.01), ladder
  settings, tile CD and in-job C9 off. Arms: D dense, G blocks / group schedule, X blocks / diag schedule, N no H_out,
  D2 dense again. f2 = the stored (decoded) weight's two-sided objective in the original basis, from `--save-wq`'s wq
  (`f2calc.py`). Note: rxq's / ourq's `rowerr.npz` is not usable here, because quantize_exl3 regularizes its weight
  argument in place (PORTING.md).
- **Time (quant seconds per layer, R9700):**

  | | dense | blocks / group | blocks / diag | no H_out |
  |---|---|---|---|---|
  | L10 | 177.7 | **34.4** | 44.4 | 32.1 |
  | L39 | 178.9 | **34.4** | 41.0 | 31.4 |

  - gate_proj with H_out: 46.7 s dense vs 6.2 s blocks.
  - Per level (48 GDN + 16 attention layers, quant only): dense ~190 min, **blocks ~37 min**, one-sided ~34 min.
    H_out at every level then costs ~+2.5 min a level instead of ~+156: ~180 card-h saved over 71 levels.
  - Peak VRAM, gate / up: 3.2 GiB blocks vs 6.4 GiB dense at K4, 6.9 vs 10.2 GiB at i4 K0.125. down_proj is unchanged
    at 7.7 GiB (its refit sets the peak).
- **Same objective:**
  - Dense is bit-deterministic: D2 = D on 13/13 tensors.
  - f2, blocks vs dense: L10 max |rel| 9.1e-5, geo −1.6e-5; L39 max 1.8e-4 (k_proj), geo +3.5e-5; i4 K0.125 max 3.5e-4
    (ssm_z), geo −6.4e-5; tile CD on q_proj +1.2e-5 (2.2× faster).
  - 96–100% of trellis words are identical. The group and diag schedules mostly give the very same codes, so the
    float-order source is the batched block Cholesky (factor diff ~1e-6).
  - **18/20 tensors are within the task's 1e-4.** The two misses are near-tie code divergence, with no consistent sign.
    The pre-registered reading passes: every tensor is within ±0.5%, |geo| ≤ 0.1%, and the CPU teacher-forced check
    holds. So **PASS**: the same objective. No KLD pair is needed, because nothing changes the objective.
  - The only behavioral difference: a failing block Cholesky raises instead of retrying with more damping. With
    σ_out 0.01 on a mean-1 PSD block that cannot happen.
- **i4 path (lead's note):** layer 10 with i4 runs at K0.125 (σ 0.2), 0.5, 1, 2 and 3 with blocks H_out: no errors,
  no fallback.
  - H_out lowers its own objective vs no H_out (geo f2): −2.6% at K0.125, −6.4% at 0.5, −10% at 1, −13% at 2, −14%
    at 3. For comparison, i8 K4 is −14% (L10) and −6% (L39).
  - Time ×1.02–1.08 of no H_out.
- **Decision / next (lead):**
  - flip `inputs.hout.ldlq = "blocks"` (w86's flag; schedule group);
  - apply fork patch 0008 (fork commit + `quantize/exl3-patches/0008`; `git am` on eb5a2139 reproduces the tree);
  - apply `w81/patches/rxm-hout-blocks.patch`: `rxq levelargs` maps an ourq level made before 11-08 to
    `inputs.hout.ldlq=dense`, so no-op gates against old H_out levels stay bit-identical (the current repo maps them to
    blocks once the flag is on: `check_rxq_hout.py` FAIL); the `proxy_err_2s` log field; ourq `--hout-blocks`.
- **Runtime:** none (quantizer only).
- **Files:** `local/workers/w81/gate-report.txt`, `data/gate/` (logs, f2, gate.json), `check_blockdiag.py`,
  `check_rxq_hout.py`.

### 2026-11-08 9:35 AM CST · #34 · 1/16-bit slab-head grid wired into knap5 (w88): head leaves uniform K3 at 2.0; −2.4…−3.8% predicted at 2.0–3.0
- **Status:** running (#34; pricing switch ready, not adopted) · **Machine:** colo (CPU, 28 solves, 15:21–15:30 UTC)
  · **Code:** `local/workers/w88/knap5-headgrid.patch` (knap5 `KN_HEADGRID` + new `v3/levels/reprice/headgrid.py`,
  `hgreport.py`; not applied)
- **Setup:** `headgrid.py` takes any headsim outputs (`SIM.json[:PLAN.json]`; default the 11-01 `lcurve` +
  `fcurve-A/B` + `headslab2`: 171 plans, budgets 2.0–8.0, donors K2–K8 with half steps). Plan KL = the conservative end
  of the 95% interval (as KN_HEADSLAB; `KN_HG_CI=0` = mean). Lower hull over every plan and floor (so the floor is
  chosen per budget), one option per 1/16 bit (97 points; between measured 1/8 budgets a linear block mix of the two
  neighbours). knap5 prices option B at r(B) × head KL, with r(K) = model head error / headsim uniform KL at K2–K8
  (0.78–1.04 M, log-linear in K). So uniform heads keep today's price exactly. Bytes linear in K + 1 MiB permutation.
  The grid replaces KN_HEADSLAB's K5 / K6 slabs. Switches: `KN_HG_GUARD=x` (drop plans with rare-token KL > x × the
  uniform head's at equal bits; 1.5 changes nothing, 1.0 cuts the hull to 14 vertices), `KN_HG_DONORS` (only plans
  buildable from listed donors; K3–K7 alone gives back today's menu). Base = w63's `main` env (V3R, swj pts 2:0.91 …
  5:1.0, accurate f16 λ 0; fast λ 3e-7 alongside).
- **Check:** with the switch off, the patched knap5 is byte-identical to knap5 (4 sizes).
- **Result (kfit2-model predictions, accurate):**

  | size | today's menu: head, KLD | grid (CI): head, KLD | grid (mean) | grid, guard 1.0 |
  |---|---|---|---|---|
  | 2.0 | uniform K3, 0.12066 | slab 2.5 (K2 floor), 0.11775 (−2.4%) | −3.2% | slab 3.125 (K3 floor), −1.6% |
  | 2.5 | uniform K4, 0.05340 | slab 3.125 (K3 floor), 0.05136 (−3.8%) | −4.1% | same as grid, −3.8% |
  | 3.0 | uniform K4, 0.02533 | slab 3.6875 (K3.5 floor), 0.02464 (−2.7%) | −3.1% (3.75) | slab 3.25, −1.2% |
  | 4.0 | slab K5, 0.006670 | slab 4.5 (K4 floor), 0.006575 (−1.4%) | −1.9% (4.375) | slab 5.125, −0.3% |

  The head gives up 0.3–0.9 bit (47–133 MiB) to the trunk (25–93 trunk picks go up, none down). Embedding is
  unchanged. Fast λ 3e-7: same heads and the same gains (−2.4 / −3.8 / −2.7 / −1.4%).
- **Caveats:**
  - **Old donors.** The curves are the 11-01 pre-crash donors, before mean-row (−30% head KL on every head, assumed
    to scale out through r). All picked plans need donors in the planned 17-donor set (K2, K3.5, K4.5–6); none
    can be built from today's K3–K7.
  - **Rare tokens.** The 2.0 pick (K2 floor) has rare-token KL 0.051 vs uniform K3's 0.016 (×3.2). At equal bits it
    is −34% mean, and the guard-1.0 arm keeps a K3 floor there. #34-02 checks only K5 / K6 slabs at 4.00, so a
    low-floor head at 2.0–2.5 needs its own whole-model pair before adoption (report KL99 / top-1).
  - **In-pool selection.** headsim's inputs and the per-budget floor choice are both calib-A (fit pool). Pricing on
    fit is allowed; the CI arm is the conservative read.
- **Decision / next (lead):** apply the patch (`git apply`); add `KN_HEADGRID=1` to the V3 search arms once #34-02
  passes. After the 17 donors: headsim at 1/16 budgets (`headslab.py` curve mode, `HS_FLOOR_ALL=1`), pass the new
  files to `KN_HEADGRID=SIM:PLAN,...` with `KN_HG_DONORS` = the built set. Containers: `hdgrid-K<B>.rad` (the
  output line's `head.plan` names the plan / floor).
- **Runtime:** none in the format (row-slab head = #34's D cost); pricing only.
- **Files:** `data/levels/reprice/headgrid/` (`out/*.jsonl`, `report.out`, `jobs.txt`, logs).

### 2026-11-08 9:45 AM CST · #117 · w84 closed: per-part numbers, p99 / top-1, what to merge
- **Status:** done (rule ADOPT; adoption = user) · **Machine:** gpubox card 0 12:05–15:27 UTC, colo CPU · **Code:**
  `v3/e3/settle/w84/` (README) · **Results:** `v3/e3/settle/w84.md` "Results", data `v3/e3/settle/data/w84/`
- **Per pair, per part** (r, z):
  - base → p0: xcorp 0.9951 (z −0.58), calib 1.0083 (+0.57).
  - s1 → p1: xcorp 0.9999 (−0.02), calib 0.9772 (−1.58).
  - So one part in four is above 1 and one sits at 1.000.
  - Pooled 95% CIs: [0.970, 1.015] and [0.969, 1.007].
- **p99 KL (pooled):** 3.909 → 3.830 (p0), 3.797 → 3.831 (p1); mixed.
- **Top-1 (pooled):** +0.0010 and +0.0020.
- **Proxy:** the readers' real-quantizer proxy is 0.9853 at both seeds, matching w82's settle64 prediction. KLD moves
  by about a fifth of the proxy gain, inside the noise.
- **Merge:**
  - `blkorder84.py` (wraps `quantize_exl3` / `scaled_ldl` / `ldlq` for the residual readers; identity order =
    bit-identical no-op, checked on both controls) is the whole change. w86's v3.toml switch should call it, or port
    its 30 lines into rxq.
  - The converter `perm117.py` is not needed. If ever used, a converted quant must also permute the natural sign
    draws and run `regularize` in natural channel order: G2 found that otherwise 98–100% of the tiles change (a
    reseed).
- **Ledger:** golden-search27b-c added (36 docs, role select, new `ledger-roles.json` entry). golden-search27 /
  golden-verify27 have no ledger-roles entry (not added by w84).
- **Disk (gpubox `/data/w84`, ~27 GB):** arm mixes `mix-p{0,1}perm.rad` (8.3 GB each) and `p{0,1}/q-perm` (the 272
  reader tensors, ~5 GB each) are kept for any K4 / verify follow-up. The spliced levels are deleted (a splice
  rebuilds them in seconds). Delete the rest once the lead decides.
- **Runtime:** none.

### 2026-11-08 9:35 AM CST · #117 · residual-stream permutation at 27B: rule says ADOPT, effect ~−0.3% (not significant)
- Two seed pairs, K2.5 plain levels, read on the fresh two-part slice golden-search27b (65 xcorp + 36 calib-search
  docs): pooled r **0.9970 / 0.9967, geo 0.9968, pooled z −0.62**; part geo xcorp 0.9975, calib 0.9926 (same sign).
  Rule (both pooled r < 1, parts agree in sign) → **ADOPT**. Proxy: 272/272 readers better, geo 0.985.
- On the retired golden-search27 (report only): p0 0.983 (z −2.06), p1 1.0039, geo 0.9933 — the old slice's −1.7% on
  p0 was mostly that slice; the fresh slice says ~−0.3%, consistent with zero to ~−1%.
- Zero runtime and build-time cost; changes every level (pre-freeze). Adoption is the user's call (w86 has the v3.toml
  switch, off by default). Runtime none.

### 2026-11-08 8:10 AM CST · #33-28 · K2.5 H_out on golden-verify27: holds — H_out at every level stands
- One-time routine verify check (pre-registered 6:20 AM: stands if verify geo < 1): bn / bns1 / bns2 vs base / s1 / s2,
  **0.9570 / 0.9803 / 0.9883, geo 0.9751, pooled z −3.38** (search was geo 0.9801). Hygiene passed on all three.
  Closes the audit's concern that the every-level adoption rested on a post-hoc standard with no verify rows.
  (Not the fast-H_out gate: that is w81's, still running.) golden-verify27 now has 5 routine uses: no more on it.
- #117: p0 pair on the retired golden-search27, for comparison only: base → p0perm 0.983 (z −2.06); readers' proxy
  geo 0.985 (272/272 < 1). Decision on golden-search27b after p1.

### 2026-11-08 8:00 AM CST · cdhout27 · tile CD + H_out at K4: PASS — tile CD stays at every level
- bn4s1ctl → bn4s1cd 0.007723 → 0.007723, **r 1.0000** (z +0.01); with bn4 0.9995: **geo 0.9998, PASS** (rule g ≤ 1.000,
  both r ≤ 1.01). Fresh controls, noop gates and hygiene passed on both. Tile CD stays on (attn_qg, every level, with
  H_out). Read on golden-search27 (rule written before the audit retired it): a no-harm check, no adoption risk.

### 2026-11-08 7:10 AM CST · cdhout27 · tile CD on top of H_out at K4: first pair no harm; kept (user)
- bn4ctl → bn4cd (fresh control, hygiene PASS): KLD 0.007465 → 0.007461, **0.9995**. Proxy: CD-domain 0.9955, after the
  scale refit **0.9989** on the 16 q_proj (~0.1%; w50's 0.5–1.2% was without H_out). Cost **185 s a level** (3 sweeps,
  16 tensors) ≈ 3.6 card-h over 71 levels, no runtime cost. **User: keep tile CD if bn4s1 passes too** (cost ~nothing). Rule (CDHOUT27.md): g ≤ 1.000 and both r ≤ 1.01, so
  bn4s1 must land ≤ ~1.0005; else tile CD goes off wherever H_out applies (= every level now). bn4s1 ~8:30 AM.

### 2026-11-08 7:05 AM CST · #117 · 27B pairs (w84): converter + gates PASS; the order is a quantizer option, no conversion needed; pairs running
- **Status:** running · **Machine:** gpubox card 0 (card0.lock + ram.lock, `/data/w84`, `run84.sh`), colo CPU
  (`/data/work/w13/w84`, 8 threads, nice 19) · **Code:** `v3/e3/settle/w84/` · **Rule:** `v3/e3/settle/w84.md` (written
  6:25 AM CST before any data; amendment 1 at 6:55 AM CST from the lead's notes: decide on the fresh two-part slice
  golden-search27b, same sign on xcorp and calib parts; golden-search27 report-only)
- **Design (stated before data):** the permutation changes only the order in which LDLQ visits the residual readers'
  128-blocks. Everything else in exllamav3's quantizer is per tile or per channel. So the pairs quantize the natural model
  with that block order (route B, `blkorder84.py`, wrapping `scaled_ldl` / `ldlq` for the 272 readers under rxq with
  the control's settings) and need no converted checkpoint. Signs, writers, embedding, head and V2 base stay
  bit-identical to the control.
- **Converter** `perm117.py` (ckpt / hess / hout / plan): 642 checkpoint tensors permuted, 557 unchanged, every 5120-dim
  tensor classified (vision merger fc2, MTP incl. both halves of mtp.fc). CLI checks (`test117.py`, colo): Hessian
  packed-triangle and shard round trips bit-exact, metadata kept; H_out writers' diagonals round trip (gpubox CPU).
- **G3 bf16 gate PASS** (`gate84.py`, colo, 2 fit docs × 384 tokens, layer-streamed; converted model vs original):
  f32 max |Δlogit| 9.5e-4 (KL 6e-11, top-1 1.0); f64 2.4e-12 (relative 6e-14; the difference shrinks ×4e8 → pure
  float reassociation); MTP head the same (f64 4.2e-12); vision merger fc2 bitwise.
- **G2 route A ≡ route B PASS (sign trap proved, card 0, 19 min):** converted tensor (perm117's own W / packed-H
  functions) with the natural sign draws permuted along and exllamav3's `regularize` run in natural channel order, vs
  the order-patched natural tensor: on L0 qkv / z / out_proj and L3 q / k / v / o_proj the LDLQ inputs (rotated
  weight, L) are bit-identical and the stored trellis codes are identical tile for tile (7/7); suh / svh differ by
  ≤ 1–99 f16 ulps on a few entries (the scale refit's reordered sums). **Two traps found:** a plain converted run
  (signs re-drawn) changes 100% of the tiles (a reseed). A converted run with the signs fixed but `regularize` left
  as-is still changes 98–99%: exllamav3's g_scale tile sample is a wrapped diagonal over tile rows, not
  permutation-equivariant. A converter-based pipeline would have to patch both; route B needs neither.
- **Noop gate p0 PASS:** route B with the identity order reproduces `/data/v3ladder2/K2.5` bit for bit on 9 readers
  (today's rxq vs the 11-02 ourq level).
- **Lead's verify check (item 3)** queued beside it: bn / bns1 / bns2 / s2 on golden-verify27.
- **Runtime:** none (route B: only the quantizer's LDLQ order changes; same format, kernels, and no checkpoint
  conversion).

### 2026-11-08 6:13 AM CST · #119 · Embedding / head representation for the 1.0–2.0 sizes: the embedding is at its floor; LDLQ in a sensitivity metric FAILS on the CPU proxy (w83)
- **Status:** done (CPU; nothing adopted) · **Machine:** colo (`/data/work/w13/w83`, ≤ 8 GB, nice 19) + Thor
  (`/data/work/w83`), 05:05–06:10 AM CST · **Code:** `v3/e3/emb2/` (`IDEAS.md` ranked candidates, `RULES.md` rules +
  2 amendments before the data they govern, `README.md`) · **Data:** `data/emb2/`
- **Bounds (all 248,320 rows):** the shipped trellis embedding is at ×1.03–1.09 the iid-Gaussian D(R) = 2^-2R, so any
  better code (VQ / lattice / LUT / 2^20 trellis) is worth ≤ 0.03–0.06 bit. The table has no structure left: eigenvalue
  AM/GM of the unit-RMS rows 1.138 (KLT + water-filling ≤ 0.093 bit, < an int8 rotation's 0.165 bit), per-128-block
  energy CV 0.126 (iid 0.125), mean row 1% of the energy; the head predicts it poorly (least squares leaves 85%); no
  near-duplicate rows (best other row median cos² 0.05; best-8 least squares leaves 86%). Head in the calib H metric:
  water-filling beats the LDLQ model by only ×0.84 (0.12 bit) before a rotation. **Closed:** KLT / transform coding,
  head-predicted embedding, shared rows / row clusters / row-level VQ, head KLT, better codes.
- **Screen (4,096 rows, EXL3 16×16 tiles):** embedding LDLQ against a sensitivity metric (encode-only, same format):
  M0 = layer 0's input projections with the norm gains folded in, F = measured diagonal residual Fisher at layer 0
  (27B `hdiag-dy`, qad), MF = (M0 + F)/2 (hedge, added after the first data); control H = I reproduces #35 (t1 0.2562
  vs 0.2570, t3 0.0166). H = M0: +0.21–0.30 bit in M0, −0.13–0.16 in I and F. H = F: +0.07–0.08 in F (×0.89–0.91),
  −0.02–0.05 in M0. H = MF: better than the control in both sensitivity metrics at every K (M0 +0.15–0.26 bit, F
  +0.02–0.04), −0.05–0.06 in I. **Rule (≥ 10% lower or ≥ 0.25 bit in BOTH M0 and F at ≥ 2 of K1/1.5/2/3): FAIL for all
  three.** M0 and F disagree about which directions matter; the CPU proxy cannot pick the metric.
- **Translation:** 1 bit/value on the embedding or head = 0.0473 bpw; by knap4 v3e's size curve 0.1 bit is worth
  −1.5% KLD at 1.25 bpw, −1.1% at 1.5, −0.9% at 2.0. The metric question is therefore worth −0.3% (MF, if F is right)
  to −4.4% (M0, if M0 is right) at 1.0–2.0 bpw, free at runtime.
- **Decision / next (lead):** (1) GPU ~1–2 card-h: the full 5120² residual-gradient Fisher at the embedding output plus
  Σ‖g‖² per input token id (fit pool, collect.py's sampled labels) — decides the metric (then `embldlq.py` with H =
  F_full, minutes) and gives measured per-row sensitivities for a floored row allocation (IDEAS #2, the IQR-V4 head
  rule on the embedding; avoids #32's frequency trap). (2) CPU: low-budget slab heads (budgets 2.125–3.5, floor ~1 bit
  below, `v3/yaqa/data/lcurve.json`: 3.125/f3 −46% head KL at equal bits with the guard also lower) as knap5 options at
  1.0–2.0 bpw (today KN_HEADSLAB = K5–6 only; head at uniform K3 at 2.0): est. −2…−3% at 2.0, gated by #34-02.
  (3) The real-KLD t0.5 / t1 / t1.25 embedding swap (TODO, open) prices more bytes at ≤ 1.5 bpw than any representation
  change could. No format or embed_tq change is warranted by this study.
- **Runtime:** none (CPU analysis; the screened candidate is encode-only).

### 2026-11-08 5:59 AM CST · #105 · w78 closed: fixes, raw data, what a per-mix asym can reuse
- **Raw s1 results:** `data/asym27/` (pair, score summary, mkdenoise stats, logs).
  - ‖WF‖/‖W‖ medians: down 1.018, GDN out 1.015, o_proj 1.008, all above 1 (0.8B: below 1).
  - hsub medians: 0.45 / 0.35 / 0.38.
  - Dose: 0.53–0.81.
  - Base: stopped by the user at moments layer 24; no pair.
- **Fixes to merge:** both are in `v3/yaqa/asym27/`, with tests in `test_w78.py` and details in `dbg/README.md`.
  - `sibmom27-w78.patch` → `v3/sib27/sibmom27.py`: no expandable segments on ROCm, row-block saves, open shards
    dropped.
  - `mkdenoise-w78.patch` → `proxy2b/exp/w07-denoise/mkdenoise.py`: blocked Cholesky solve on the lean path.
- **Same allocator setting elsewhere on gpubox:** `c9w27.sh`, `cd27.sh` and `gpucheck.sh` also export
  `expandable_segments`. Their quants / scores ran clean, so it is not changed there.
- **Disk:** `/data/asym27` holds 774 GB (s1 moments 245 GB + dn 135 GB, base partial moments 190 GB, mixes). These
  moments are not reusable for a per-mix run (they go through K2.5-only siblings): delete once the lead agrees.
- **Runtime:** none.

### 2026-11-08 6:35 AM CST · #— · Search slices become two-part, calib-sparing (user)
- golden-search27 retired for decisions (overused, audit above). New 27B search slices: ~65% fresh xcorp (redrawn each
  round) + ~35% fresh calib search-pool docs (+ optional own-trace chunks, our cards only); pooled decision, same sign
  required on the xcorp and calib parts. First use: golden-search27b for #117 (w84). `docs/design/eval-data.md`.

### 2026-11-08 6:20 AM CST · #— · Overfit audit of the last 48 h (user)
- **Clean:** public untouched (the 6 mentions are repo publicity). Fitting / tuning stayed on fit or held-out fit
  halves (#114, K0.125). Drops (A4, #105, #106, #113, #117-proxy) cannot overfit. **swj** K2.5 confirmed on verify
  (0.935 / 0.932, = search); **C9** keeps 0.96 of its gain on verify; **H_out K4** seed 0 holds on verify (0.965 vs
  search 0.936–0.961; no verify rows for seed 1).
- **Risks:** (1) **golden-search27 overused**: the same 103 docs decided every 27B pair since 11-04; its document
  error (~0.010 SE a pair mean, #116) is as large as the seed noise and does not average over seeds, and seed luck
  replicated only partly on verify (s1/base 0.975 search vs 0.991 verify): the search slice exaggerates by about its
  document error. (2) **H_out at every level was adopted after the rule missed** (geo 0.9801 vs ≤ 0.98, a post-hoc
  standard), and **K2.5 has no verify rows**. (3) #116's lower pooled σ must only be used in rules written from now
  on, never to re-read old verdicts; future rules include the document error.
- **Actions:** #117 is read on a **fresh search slice** (golden-search27b, search pool, new seed; w84); one routine
  verify check of the K2.5 H_out arms on golden-verify27 (stands if verify geo < 1; once). golden-verify27 now has 4
  routine uses (swj, C9, H_out K4, K2.5): draw a fresh verify-pool slice for the finished V3 mixes.

### 2026-11-08 6:10 AM CST · #117 · residual-stream permutation: 27B KLD pairs (user)
- w82's proxy miss (0.982 / 0.984 vs ≤ 0.98; ~−1.5% parameter-weighted, every tensor better) is not the decision:
  the user wants real mixes. **w84**: conversion tool + bit-exact bf16 gate (and the per-position sign trap), then two
  seed pairs at K2.5 on card 0 after the current queue; rule (user's H_out standard): adopt if both pairs < 1. Zero
  build-time and runtime cost; must land before the freeze.

### 2026-11-08 5:45 AM CST · #118 · Knapsack pricing: swj and H_out make the one-sided proxy 12% / 7.6% pessimistic; V2 anchors fine per level, attention overpriced (w82)
- **Status:** done (report; nothing adopted) · **Machine:** iceman CPU (2(a): 3 single-thread jobs, nice 19, no GPU, no
  locks, 10:15–11:34 UTC), VM · **Code:** `v3/e3/settle/w82/` (`pw82.py`, `pwreport82.py`, `price82.py`); rules
  `v3/e3/settle/w82.md` item 2, written before any item-2 number
- **2(a) swj'd vs plain gate / up, one-sided proxy (decoded levels, calib H_in), 64 layers:** ρ1 = **1.122 at K2.5**
  (base / s1 / bn / bns1 1.120–1.124; gate 1.20, up 1.05) and **1.056 at K4** (base4 / bn4), while KL is 0.935 / 0.932
  (K2.5) and 0.990 (K4). The levels' svh sit at the proxy's per-row LS optimum (refit = plain × 0.99999); swj moves gate
  rows +3.4% (K2.5) and the proxy rises quadratically. The Σdy²-weighted proxy is no better (ρ2 1.13 / 1.08): swj's gain
  is the SwiGLU product. → Correction if swj'd proxies are ever read: gate / up × **0.891 (K2.5), 0.947 (K4)**; K2 /
  K3 not measured (no swj'd 27B level; rule values 0.891 / ~0.909). **knap5 needs none today:** it reads plain-level
  proxies (ourq's proxy.jsonl is pre-swjapply) and prices swj as a KL factor.
- **Same for H_out (now at every level), from the levels' proxy.jsonl:** H_out level / plain twin = **1.076** (K2.5, 3
  pairs) and **1.078** (K4, 2 pairs) one-sided, while KL is 0.97–0.99 / 0.94–0.96 (GDN qkv 1.17, z 1.22, attn_qg 1.13,
  gate / up 1.07, writers 0.99). Same at both K, so the ladder's `KN_LEVELS` positions stay unbiased; absolute values
  must not replace anchors undivided. H_out is unpriced in knap5 (a KL factor like swj would be consistent).
- **2(b) V2-based anchors vs measured levels:** knap5's prices equal the 11-05 ladder (`data/v3/levels`: m 1.013 at every
  K, writers exactly 1.000; gate / up 1.075 = geo(gate, up) vs V2's fused entry, cancels in ratios). Today's plain
  levels are **4.5% (K2.5) / 4.9% (K4) below** them, uniformly → rule: **no per-level correction**. Per family it is
  large and K-independent: attn_k 0.74 / 0.72, attn_v 0.83 / 0.80, attn_qg 0.93 / 0.92, ssm_qkv 0.95, rest 0.97–0.99:
  knap5 overprices attention q / k / v error by 8–28% (lead: per-family factor, or today's ladder as anchors).
- **2(c) two-sided proxy in kfit2 CV: not done** (needs every V2 container × tensor row-weighted: a card hproxy run or
  ~30+ CPU-h on iceman; over the 2 CPU-h bar).
- **Runtime:** none (pricing). **Files:** `v3/e3/settle/data/w82/pwreport82.txt`, `price82.txt`, `pw/`, `proxy/`.

### 2026-11-08 5:40 AM CST · #117 · Residual-stream 128-block permutation (block act order for every reader): FAIL on the CPU proxy (w82)
- **Status:** fail · **Machine:** colo + Thor CPU (108 settle jobs, 0 failed, 10:06–11:33 UTC, nice 19) · **Code:**
  `v3/e3/settle/w82/` (`btrace82.py`, `agree82.py`, `settle82.py` = settle64 + mode `rperm`, `q82.py`, `report82.py`);
  rules `v3/e3/settle/w82.md` item 1 (written before data)
- **1(a) block-trace agreement (published calib H_in + weights, 272 reader tensors):** median Spearman vs the
  leave-own-layer-out consensus 0.775, 98.5% ≥ 0.5, family medians ≥ 0.73 → **agree** (gate passed). But the
  exploitable spread is small: block-trace p90/p10 1.3 (up) – 1.5 (gate) – 1.9–2.4 (GDN) – 2.2–3.2 (attention); o_proj
  had 2.83.
- **1(b) settle64 (exllamav3's order), 27 readers × 2 samples, consensus order stored heaviest-last:** median
  rel / natural **0.9822 (K2.5), 0.9835 (K4)** vs the rule's ≤ 0.98 at both → **FAIL**. Every tensor < 1 (geo 0.982,
  parameter-weighted 0.985); attn q / k 0.96–0.97, GDN 0.985, gate / up 0.99. The per-tensor oracle order is no better
  (0.983 / 0.985): the small spread is the limit, not the shared order. Storing heaviest-first costs +6% (attn_qg
  ×1.6–2.3). With #106-01 (−9% proxy, no KLD) a −1.5% proxy gain is not worth a 27B pair. No ladder change.
- **Conversion list** (what an exact residual permutation must touch: readers' input columns incl. in_proj_a / b,
  writers' rows with C9 / asym, norms, embedding / head, vision merger fc2, MTP, DFlash2 taps and residual dim; readers'
  Σdy² diagonal unchanged; sign vectors must move too or a 27B pair is also a reseed): `v3/e3/settle/w82.md` 1(b).
- **Runtime:** none (a permutation folded into the stored tensors). **Files:** `v3/e3/settle/data/w82/` (`agree82.txt`,
  `order82.json`, `report82.txt`, `bt/`, `rperm/`).

### 2026-11-08 4:10 AM CST · #105 · s1 pair: asym writers +0.7% (z +0.6) at 27B; PASS already impossible (w78)
- **iceman downtime:** iceman was down 8:28 PM – 11:34 PM CST 11-07 (the upgrade: 9950X, 64 GB), and its container
  again 11:37 PM – 1:11 AM CST. w78 resumed the queue at 1:13 AM (moments kept, mk chunks 0–32 kept).
- **mkdenoise:** 128 groups.
  - clip > 0 in 126 groups (max 2.9e-3);
  - hsub median 0.399 (0.8B: 0.136);
  - F_dev 0.033–0.112;
  - **‖WF‖/‖W‖ 1.0012–1.0321.** All above 1, where the 0.8B's were 0.89–1.00 (`data/night1103/w07-denoise.txt`).
- **Dose** (reported, not gating): ‖F_N/2 − F_N‖ / ‖F_N − I‖ = 0.53–0.81 on layers 0 / 3 / 31 / 63, over the rule's
  ≳ 0.5 mark. So F at 1M tokens is still mostly sampling noise; per the rule, a 2M collection is the follow-up.
- **No-op gate PASS:** 512 / 512 tensors bit-identical to `q-s1/K2.5`, so the control is `s1` itself. q-noop took
  16 min, q-asym 27 min.
- **Pair** (golden-search27, 103 docs, 123,460 positions; mixes both 2.6244 bpw): s1 0.057570 → s1a 0.057957,
  **r 1.0067**, z +0.59 (SE 0.00066).
  - The proxy predicted −14% (K3) / −18% (K2).
  - All positions (unscored included): 1.2833 → 1.2673, not the rule's metric.
- **What the rule allows now:** PASS needs both r < 1, so it is already impossible. The pre-registered third pair
  needs both r < 1 too, so it will not run. The base arm decides between FAIL (g ≥ 1.00, i.e. r_base ≥ 0.993) and
  INCONCLUSIVE.
- **Base arm:** moments running since 3:52 AM CST; verdict ~9:30 AM CST.
- **Runtime:** none. Raw: `/data/asym27/s1a.json(.rows)`, `/data/asym27/s1/pair.json`; repo copies with the verdict.

### 2026-11-08 3:50 AM CST · #116 · The 27B noise floor: the rule's σ is a cross-seed SD; same-seed pairs are ~half as noisy; document error is in neither (w81)
- **Status:** done (report; nothing re-decided) · **Machine:** gpubox CPU (rows read-only), VM · **Code:**
  `local/workers/w81/noise27.py` (per-document sums, one joint document bootstrap for every arm), `noise_report.py`
- **Rows:** golden-search27 (103 docs, 123,460 positions): base / s1 / s2, bn / bns1 / bns2, dy, xbn / xbns1, base4 /
  base4s1 / bn4 / bn4s1, base4v4 / bn4v4, the swj and C9 arms; golden-verify27 (71 docs): base, s1, base4 / bn4,
  base4v4 / bn4v4, swj, C9. #105 (asym) has no 27B score yet (s1a mixed 09:25 UTC).
- **What each estimate measures:** H27 amendment 5's σ = 0.0178 = RMS |log| of base / s1, base / s2, s1 / s2: the noise
  of comparing two *independently seeded* quants (√2 × a per-quant seed SD of 0.0126; it also holds the doc-sampling
  error of those contrasts, SE ~0.010 each). The rule tests the mean of same-seed method pairs log(method_i /
  control_i); the right noise for that is those pairs' own spread across seeds (a paired t on seeds). Pooled over
  H_out K2.5 (SD 0.0109, df 2), H_out K4 (0.0043, df 1), xcorp (0.0065, df 1): **0.0086, df 4** (95% interval
  0.0052–0.0248, so 0.0178 is not excluded). Reading: arms sharing a rotation seed share most of its luck (implied
  ρ ~ 0.76, anything ~0–0.9 fits df 4); bn's own seed pairs are tighter too (RMS 0.0075 vs 0.0178). The pair SD is the
  right estimate for same-seed pairs, used with t quantiles at its df (pooling assumes one pair noise across methods);
  0.0178 is right only for cross-seed comparisons.
- **Missing from both:** every pair is scored on the same slice, so its document error does not average over seeds:
  joint-bootstrap SE of the 3-pair K2.5 mean 0.0104 (single pairs 0.011–0.015), K4 2-pair mean 0.0104, C9 0.0018.
  It is as large as the seed term (0.0086/√3 = 0.005) and matters for transfer to the private corpus.
- **Verdicts under the pooled SD:** K2.5 H_out (geo 0.9801): noise test |log geo| 0.0201 vs 2·0.0086/√3 = 0.0100 →
  pass (rule's σ: 0.0206 → miss); with the document error added, SE 0.0116, z −1.74. The geo ≤ 0.98 effect bar
  does not depend on σ and still misses by 0.0001: "real but below the bar" instead of "missed both by a hair". The
  user's 11-08 adoption at every level stands. **C9 #112-02:** fixed codes, no seed-noise test in its rule (per-pair
  z ≤ −2 on the doc bootstrap, g ≤ 0.99): unchanged PARTIAL (geo 0.9925, z −2.2 / −5.5, pair spread 0.0019); adopted 11-07.
- **Review item 4 (stored verify rows):** (a) K4 H_out head-matched (V2 4.00 head), seed 0: search 0.9360 (z −3.6),
  **verify 0.9647 (z −2.9)**; with the 3.00 head search 0.9613 / verify 0.9703 (the 3.00 head's larger error dilutes
  the trunk gain). No verify rows exist for the seed-1 pair. (b) seed luck s1 / base: search 0.9753 (z −2.6),
  **verify 0.9911 (z −1.1)**: same sign, smaller; the difference 0.016 ± 0.0125 (z 1.3) fits one common value
  ≈ −1.5% (inverse-variance pool). Luck replicates partly; the search slice exaggerated it, by about its doc error.
- **Runtime:** n/a (statistics). **Files:** `local/workers/w81/noise-readout.md`, `data/noise27.json`.

### 2026-11-08 3:40 AM CST · #115 · Block-diagonal two-sided LDLQ for a diagonal H_out: built, CPU check PASS, card-0 gate queued (w81)
- **Status:** running · **Machine:** VM (CPU check), gpubox card 0 (gate, under card0.lock + ram.lock shared, between
  the queue's steps) · **Code:** fork patch 0008 + rxq / ourq patch, `local/workers/w81/` (not applied in place)
- **Question:** a diagonal H_out becomes exactly block diagonal (128 × 128) after the output Hadamard, so is its block LDL
  factor; ldlq_2hess's dense feedback (`bmm(dE, Ln[cols][:, :, :n_hi])`, `F[:k_hi, :n_hi] += ...`) multiplies exact
  zeros. Restrict it to each tile's 128-column group: same objective, how much faster (review 11-07 item 1, PROFILE T1).
- **Built:** `prepare_H_out_blocks` (the (n/128, 128, 128) blocks and their batched block-16 LDL, no n × n matrix),
  `ldlq_2hess_blocks` with two schedules: `diag` = the dense path's anti-diagonals with the update per group;
  `group` = anti-diagonals inside every group at once (tile (a, c) at step a + c mod 8: tiles_k + 7 steps of up to
  tiles_n tiles, one (k_hi × 128) @ (128 × n) GEMM a step). Dispatch: a 1-D H_out = the diagonal → blocks;
  `quant_args["hout_dense"]` densifies (reference); a 2-D H_out is the old path unchanged. rxq: `inputs.hout.ldlq =
  "blocks" | "dense"`, `inputs.hout.schedule`; tile CD accepts the blocks; ourq `--hout-blocks group|diag` (unset = dense).
- **CPU check** (`w81/check_blockdiag.py`, stub trellis, 4 shapes): prep blocks = the dense matrix's blocks (max diff
  1e-7 rel; the dense factor is exactly 0 off the blocks); with a smooth stand-in quantizer Q_blocks = Q_dense to
  ≤ 4.8e-6 for both schedules (the same tile inputs in the same dependency order); a rounding stand-in agrees except
  near-ties; L_out = 0 moves Q by 1.3 (the check sees the feedback). PASS 17/17.
- **Gate (task w81, written before GPU data):** 27B layers 10 (GDN + MLP) and 39 (attention + MLP), K4 i8, bn H_out
  (`/data/swap27/hout-bn`), ladder settings, tile CD and in-job C9 off: blocks (group, diag) vs dense, per tensor
  f2 = the stored weight's two-sided objective in the original basis (Σ h_n e_n / Σ h_n r_n from `--save-wq` rowerr; the
  PROFILE T1 judge) within **1e-4 relative**; time per tensor both ways. **Pre-registered reading if that fails:** a
  float-order change can flip near-tie trellis decisions and the feedback then spreads the change (codes diverge,
  not the objective). Then the gate reads "same objective" if the CPU check holds (it does), the dense path's own
  run-to-run spread (arm D2) is reported beside it, and every tensor is within ±0.5% with |geo − 1| ≤ 0.1% (PROFILE T1's
  "≤ reference × 1.005" judge). Also run: N (no H_out, one-sided time), tile CD on L39 q_proj blocks vs dense, the i4
  path (layer 10, i4 K0.125 / 0.5 / 1 / 2 / 3, blocks vs no H_out; K0.125 also dense): runs, two-sided objective < no H_out.
- **Runtime:** none (quantizer only; same objective, format and kernels).

### 2026-11-08 5:15 AM CST · #105 · asym at 27B: first pair s1 → s1a 1.0067 (no gain)
- 27B writers asym ρ0, K2.5, golden-search27: s1 → s1a KLD 0.05757 → 0.05796, ratio **1.0067** (z +0.59, se 0.00066,
  123k positions). Rule (ASYM27.md): both pairs < 1 and geo ≤ 0.97 → the base arm must land ≤ 0.941 to pass; very
  unlikely. 0.8B gave ×0.84 (#87): no transfer so far. Base arm running (moments layer 23/64, ~6.5 h to its pair);
  continue vs stop is with the user. Runtime none.
- **5:25 AM: stopped (user).** ~6.5 h of card 0 for an arm that can hardly pass; per-mix healing will cover much
  of this ground. **#105: per-level asym not adopted** (one pair 1.0067, 95% CI ~0.985–1.03: at most a small hidden gain, vs the
  0.8B's ×0.84). **Per-mix asym stays an optional post-processing candidate** (no ladder impact): one arm asym → heal
  vs heal alone in the 27B healing pilot on a finished mix (~5 card-h) decides it (user 11-08).

### 2026-11-08 5:00 AM CST · #90 · sub-1 levels: kernel kept, levels built (user)
- User: the sub-1 decode support stays (~25–30 SLOC: the salted 0-bit read in the plan, decoders and CPU reference;
  ~40 test lines), and the 7 sub-1 levels (i4 K0.125–0.875) are built for V3: ~22 card-h with H_out as implemented
  (~6 with w81's fast H_out, ~4.5 plain) is acceptable. They stay last in the build order (after pass 2); the knapsack
  picks them only below ~1.3 bpw (#90: −1.8% at 1.25, −0.07% at 1.5, 0 at ≥ 2.0). `v3.toml` unchanged (by_level keeps
  i4 K0.125–1; H_out applies, K0.125 σ 0.2).
- Sizes (user): offer **1.0 / 1.25 / 1.5 / 1.75 bpw** as high-quality low-bpw sizes (supersedes 11-05's "1.25 dropped");
  the expensive per-mix post-processing (asym, healing) on them is optional, decided later.

### 2026-11-08 3:45 AM CST · #114-02 · σ 0.2 at K0.125 adopted (user)
- User: the lowest level is extreme and needs the regularization. `v3.toml` `quant.lowest_level_damping` (k_max 0.125,
  σ 0.2 on both groups); `rxq/config.py` resolves it (K0.125: sigma_reg = sigma_reg_in = 0.2; K0.25 unchanged).
  check_grouphead 5/5 PASS. Caveat: measured without H_out, which now applies at every level. C9 confirmed (user:
  practically free). Runtime none.

### 2026-11-08 3:30 AM CST · #33-28 · H_out adopted at every level (user)
- User: H_out lowers KLD on average (K2.5: every pair < 1, 0.972 / 0.992 / 0.976, geo 0.980; K4 PASS geo 0.964;
  0.8B positive at K2 / K3); noise limits confidence, not the direction, and a gain at high K should hold at low K.
  **Adopted at every level** (`v3.toml` apply_k_min 4 → 0). Supersedes "K ≥ 4 only". Cost (H_out levels ~5× slower
  to quantize) is the open problem: **w81** builds the block-diagonal two-sided LDLQ (same objective, ~4× faster
  expected) and re-checks the noise floor the rules use (0.018 vs a pooled 0.0086). Runtime none.

### 2026-11-08 2:50 AM CST · #38 / pipeline · gpucheck27 i4 (2) FAIL was the test's reference, not the damping; rerun PASS 3/3 (w80)
- **Cause:** the i4 unit's reference was `ourq --only {k, v, o}` of layer 39. ourq skips q_proj, so k_proj is the first
  tensor to finalize the shared input Hessian. exllamav3 then draws the group's sign vector su under k_proj's seed;
  v_proj reuses that draw. The ladder runs whole layers and draws su under q_proj's seed, and so does rxq: its
  per-tensor jobs pre-finalize under the group head's seed. That explains suh differing in 99.6% of entries and the
  different proxy_err / g_scale. o_proj is a group of one, hence identical. The damping is the same on both sides:
  both logs record σ 0.05 for k / v, and 0.025 for o_proj.
- **Neither side is wrong against the rule:** rxq matches the ladder. ourq's `--only` without a group's first tensor
  gives a tensor the ladder never makes. This affects only `--only` lists that hold a j > 0 member (k / v, GDN z, up)
  without its head. The ladder is unaffected, and so are the repo's `--only` users checked (asym27 / sib27 / g64 /
  test08 writers, g75 o_proj, cd27 / tilecd q_proj, damp3 / damp4 / shab / cbab full groups, s1oscale gate + up).
  Live `/root/ourq` needs no change and was not touched.
- **Fix (pipeline only):** `tools/gpucheck27.sh` i4 reference now quantizes q_proj too (`i4-ref-keys.txt`).
  `tests/check_quant.py` prints BADREF and counts it a fail when a reference lacks a key's group head
  (`rxq.qwen35.missing_group_heads`); on the 11-08 outputs it flags k / v.
  Regression test `tests/check_grouphead.py` (CPU, the fork's real finalize_capture_H): 5/5 PASS on the VM fork and
  on gpubox's `/data/exllamav3`. It checks:
  - rxq k / v su == ourq group run;
  - `--only {k, v}` su differs in 52% of entries;
  - su does not depend on σ;
  - the settings rule at K0.5 / K1 per family;
  - the guard;
  - the source lines.
  A mutated rxq (head seed := own seed) fails it.
- **Rule wording:** sub-1 σ 0.05 is the layer-input group only (attn q / k / v, GDN qkv / z), as ourq's
  `--sigma-reg-in`, `ladder2.sh`, v3.toml and rxq implement it. The 9:25 AM entry's "gate/up" is wrong: gate / up
  read the post-attention Hessian and stay at 0.025. #114 (w77) measured that group's sub-1 curve flat (0.01–0.2
  within 1%), so the implemented rule stands.
- Also fixed in gpucheck27: the c9 unit listed `layers.11.linear_attn.out_proj`, which does not exist (layer 11 is
  full attention). Its rxq job stopped after 2 of 4 writers, so c9's PASS 2/2 covered o_proj 3 and out_proj 10 only.
  Now out_proj 12. The c9 CPU line ("FAIL 0/2, max |dc|/c 1.2e-7 / 6.0e-8") is one f32 ulp of c (f32, values
  around 1). That is f64 summation order, CPU vs GPU, expected, and harmless: the folded svh is f16, whose ulp is
  8192× larger. The bit-identity verdict is meant for the same device only.
- **i4 rerun (card 0, 2:45–2:47 AM CST, card held 150 s; code restaged with `store/`): i4 (1) PASS, i4 (2) QUANT
  IDENT PASS 3/3** (k / v / o: tensors and log fields equal; rxq's proxy values unchanged from the failed run: the
  corrected reference now matches rxq), **i4 (3) PASS**. Raw: `pipeline/data/w80-2026-11-08/` (gpu/, prev-1108a/ = the FAIL plus the
  BADREF re-read).
- **Runtime:** none (test and docs only; no quantizer output changes).

### 2026-11-08 1:55 AM CST · #114-02 · K0.125 damping: the optimum is σ ≈ 0.2; above it nothing (w79)
- colo, 162 / 162 jobs (0 fail), w77's code / Hessians / scales. i4 K0.125 at σ 0.3 / 0.5 / 1.0, joined with w77's
  rows (648 = 54 tensor × samples × 12 σ). Rule (written before data, `v3/e3/settle/w79.md`): w77's pick-on-B1 /
  judge-on-B2.
- **Both group curves turn up after 0.2.** Held-out geo r_B vs current, inputs: 0.2 0.981, 0.3 0.984, 0.5 0.992,
  1.0 1.024. Rest: 0.988 / 0.989 / 0.999 / 1.023.
- Group picks stay at 0.2: inputs +1.97% (14/15), rest +1.28%. No cell gains ≥ 1% over 0.2: ssm_z picks 0.5
  (+0.3% vs 0.2), attn_k 0.3 (+0.1%), ssm_out 0.3 (−0.3%). 49 of 54 tensor × samples are worse at 1.0 than at 0.2.
- **Answer:** "σ = 0.2 at K0.125" is the candidate per-level setting for the lowest level (bound / 1.5 bpw research
  point). Held-out tensor error (proxy): inputs −1.9%, rest −1.2%; attn_qg −5%, attn_k −4%, ffn_down −3%.
  For the lead / user: adopt into `v3.toml` or not. Not adopted here.
- **Runtime:** none. **Files:** `v3/e3/settle/data/w79/report.md`, `v3/e3/settle/w79.md`. Ledger: same B1 / B2
  fit-pool docs as w77 (`w77-hb`), no new documents.

### 2026-11-08 1:35 AM CST · #89-06, #114-02 · k45lr read out: K4 at the line, K5 fails, a K5 fold anomaly; K0.125 grid extension running (w79)
- **#89-06 readout (d1105 k45lr, seed 0 only; seed 1 never ran):** vs each arm's own stack on g08-search1:
  K4 lr 3e-4 **0.9494** (z −14.4; held-out −5.4%), K4 lr 1e-4 0.9605 (held-out −4.4%), K5 lr 1e-4 **1.0119** (z +2.9;
  held-out −2.4%), K5 lr 3e-5 **1.1441** (z +22; held-out −0.8%). Against the #89-06 rule (≤ 0.95 on both seeds):
  **K5 fails** (its held-out-best lr is worse than its stack). **K4 is inconclusive**: one seed, right at the line.
- **K5 anomaly, unverified:** at lr 3e-5 (the smallest multipliers, max |log| 0.0046), held-out and golden disagree by
  15 points. Checks so far: the stack's score reproduces exactly. The arm reproduces at 1.144; with the norms not
  folded it is 1.130, so the linears carry it. The folded files equal stack × e^c × e^r to f16 rounding on all 150
  tensors. The loss is spread evenly over documents and sources. Not checked: whether heal.py's in-memory student
  scores the golden like the fold (every weight is re-rounded f16 → bf16 in evalq, which is larger here than the
  healing change). Step 0 of the screen below tests this. It ran 1:17–1:19 AM CST; the second half OOM'd beside the
  user's vLLM at 1:22.
- **#89-06c rule (written before data, `proxy2b/exp/w79-heal45/README.md`):**
  - Step 0: student-path vs fold-path check, plus jitter folds at K4 / K5.
  - K4 seed 1 at lr 3e-4 and 1e-4. The lr is picked by mean held-out gain; the picked lr's two seeds are read on
    search and checked once on g08-verify1 + verify2 pooled.
  - **Heal K4** if search ≤ 0.95 on both seeds and verify ≤ 0.97 on both. "Small" if some seed is ≤ 0.97 on search.
    Otherwise do not heal.
  - K5 is re-read only if step 0 flags the fold path.
  - Cost ≈ 2–2.5 h of the 4070.
  - **On hold:** the 4070 is off limits again (lead, 1:20 AM CST). Driver and queue file are ready, not installed.
- **#114-02 launched (colo, `v3/e3/settle/w79.md`, rule written before data):** i4 K0.125 at σ ∈ {0.3, 0.5, 1.0} on
  w77's code snapshot, inputs and Hessians: 162 jobs at 1:19 AM CST, ~1–5 min each, colo only (Thor not needed).
  Same pick-on-B1 / judge-on-B2 arithmetic as w77, plus the gain over σ 0.2.
- **Runtime:** none (both).

### 2026-11-07 6:50 PM CST · #105 · s1 moments done; mkdenoise's f64 solve fixed for ROCm (w78)
- **s1 moments:** DONE at 6:15 PM CST, 128 groups, 1,002,035 tokens, 170.8 min, peak GPU 15.6 GiB, peak RSS 3.9 GB.
  No GPU fault.
- **Next blocker:** the mk step. `torch.linalg.solve` at k 17408 f64 hits rocBLAS's degraded TRSM on the R9700.
  - The first down group ran > 20 min; a bench run alone on the card took > 6 min. ~20 h for 64 groups.
  - The queue was stopped at 6:37 PM CST.
- **Fix:** `v3/yaqa/asym27/mkdenoise.py` + `mkdenoise-w78.patch`. The lean path (k > 8192) now uses a Cholesky factor +
  blocked triangular solves (GEMMs + 1024-row TRSMs), with an LU fallback.
  - **Speed:** 16 s for the solve at 17408 (bench); 112 s for a whole real down group, eigh clip included.
  - **Real s1 layer 0:** hin bit-identical; wpre within last-bit f32 rounding (1,069 of 37.7M values, max rel
    2.2e-10); solve residual 4.4e-15.
  - **Unchanged:** groups ≤ 8192 take the default path as before (0.8B G1 unaffected). `test_w78.py` I: PASS.
- **Queue restarted 6:50 PM CST** (mk from chunk 0, the moments kept).
- **Reported, not judged:** layer 0 down hsub 0.585.
- **Runtime:** none (quantizer side). Details: `v3/yaqa/asym27/dbg/README.md`.

### 2026-11-08 2:25 AM CST · pipeline · w76's GPU checks on card 0; cdhout27 restaged
- gpucheck27 all (07:35–08:07 UTC): **tile CD GPU gate PASS 3/3**, two-sided proxy CD/off geo 0.9993 (H_out on);
  **i4 (1) qtf2 i4 == exl3cb PASS**, **i4 (3) K2 qtf2 vs exl3cb PASS**; **c9 in-job PASS 2/2** (~1.07 s / writer on
  card); c9 CPU reference max |dc|/c 1.2e-7 (report only). **i4 (2) FAIL 1/3:** ourq vs rxq at K0.5 i4 sub-1 damping
  differ on k_proj / v_proj (group members j > 0), o_proj identical → w80 fixes before the freeze.
- cdhout27 (tile CD + H_out K4 pair) failed at step 0: the staged code bundle lacked `store/` (lead's staging error);
  restaged, rerun from 08:18 UTC.

### 2026-11-08 1:40 AM CST · #9 · iceman upgrade: TP2 prefill ×1.7 (user)
- iceman upgraded: Ryzen 9 9950X, 64 GB, both R9700s on PCIe 5.0 x8 (x8/x8 CPU split; 32 GT/s under load), ReBAR on;
  card 1 moved from x4 (`06:00.0`) to x8 (`07:00.0`). User measured **×1.7 TP2 prefill** on llama-swap (now a bit over 5k tok/s). #9 ("TP2 slow
  over card 1's x4 link") no longer holds: re-time TP2 for our long-context / whole-corpus evals when both cards are
  ours. Card 0 itself went x16 → x8 (loads only; not yet measured).

### 2026-11-07 9:50 PM CST · #114 · damping per (level, family): null — keep the current setting
- w77, colo, 486 sweep jobs (4,374 rows, 0 fail), order-fixed proxy, scored on held-out input Hessians (hB1 / hB2,
  168 fit-pool docs, ledger `w77-hb1/2`). Rule (≥ 2% held-out gain, ≥ ⅔ of tensors agree): **no ladder cell of 66
  passes**; the largest gain anywhere is +0.63%. Inputs' best σ falls with K (0.1–0.2 at K0.25 → ≤ 0.005 from K1.5),
  but the current values (inputs 0.05 below K1, 0.005 from K1; rest 0.025) are within ~1% everywhere; the halves
  agree on σ* in only 34 / 66 cells (flat curves). 0.001 ≈ 0.005; #38's K1.5 / K2.5 keep holds held out.
- **K0.125 (bound, not a ladder cell):** wants more damping than the grid — 28 / 54 runs pick 0.2, the top; at 0.2
  attn_qg +5.1% (mostly one compressible block), attn_k +4.0%, ffn_down +2.7%. But the GPU search there is worse
  than all-zero (err 1.014; the proxy is ~13% optimistic), so the cell is unreliable either way. Open: extend σ > 0.2
  at K0.125 only. Report `v3/e3/settle/w77.md`, `data/w77/report.md`.

### 2026-11-07 8:35 PM CST · #46-03 · A4 speed bound: even a free A4 kernel doesn't pay
- knap5a with A4's speed over A8 forced to ×2 (colo) and ×1000 (≈ free; Thor), 144 solves each, step 4 vs noa4:
  **A4 is picked on no tensor at equal KLD at 2.0 / 2.5 / 3.0 in either arm** (time −0.0%, bytes fail). Only by
  giving up KLD does it show: ×2 buys 0.6–0.9% at +1% KLD (10.8% at +3% at 2.0 only); free A4 buys 0.9–7.3% at +1%,
  7–14% at +3%. The accuracy cost, not kernel speed, keeps A4 out: **no kernel work can save it.** Confirms the drop.

### 2026-11-07 3:55 PM CST · ops · Haiku → the user's local Qwen3.8-27B (mining, watchdog)
- User: the local Qwen (llama-swap, gpubox card 1; radiance, IQR-V4 S) may be used without limit at ~2 requests ×
  50k tokens. **Mining extractor** now runs 24/7 on it (`local/mine/minq.py`: 20k-token chunks + a 9k-token digest of
  the plan docs, no tools, extractor role only, shared mining ledger; 336 chunks queued). **Watchdog** cycles run on it
  too (`watchdog.js`, Haiku fallback on a failed / off-format cycle; user: no prompt-cache misses). `claude-usage.sh`
  stays on Haiku (it reads the Claude plan's rate-limit header). pi's `local/qwen38-27b` context set to 120k.

### 2026-11-07 3:35 PM CST · ops · Thor back as a CPU box; A4 speed bound split
- **User: Thor is fine for CPU work again** (it is still the VPN host, so it can vanish: every Thor job must be
  restartable and colo alone must be able to finish). AGENTS.md and the worker context updated.
- The #46-03 A4 speed bound (144 solves at ×2 and 144 at ×1000 ≈ a free A4 kernel) is split: **colo runs the ×2 arm**
  (`jobs-a4x-colo.txt`, 6 at a time, nice 19, beside w71 / w77), **Thor the ×1000 arm** (`jobs-a4x-thor.txt`, 10 at a
  time, nice 10); w72's tree copied to Thor `/data/work/w72` (692 KiB). Watch rules a4xc / a4xt.

### 2026-11-07 3:30 PM CST · #105 · asym27 moments fault = expandable_segments allocator; fixed, both arms running (w78)
- **Cause:** `asym27.sh` exported `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` for every card step. With it,
  `sibmom27.py` faults on card 0 in random **stock** kernels 30 s – 2 min into the run. Seven runs faulted (s1 ×3,
  base ×3, w78's repro A). The faults: HSA illegal instruction in the RMSNorm mean reduce, the CK conv of the
  reference `causal_conv1d` and hipBLASLt bf16 / f32 GEMMs; one aperture violation; then "illegal memory access" at
  `y.cpu()`. The level decode, int overflow and fla were not involved.
  - Without the allocator setting: no fault in 1.2M layer-tokens of tests (C, E, F).
  - With `AMD_SERIALIZE_KERNEL=3`: no fault either (D), so the serialized run gives no stack.
  - The gate (layers 2–3, ~1 min) always passed, which is why it never caught this.
  - Why the mode faults here: not checked (unverified idea: VMM map / unmap racing queued kernels).
- **Second bug, would have killed it next:** host RAM at the end of layer 0. The memory guard killed two test runs at
  the moment save.
  - An open checkpoint shard kept the 2.5 GB embedding host copy through layer 0.
  - `save_lean` copied each 17408² f32 to the host whole, plus ~3.6 GB of dirty pages.
  - iceman has ~9 GB available with nothing of ours running.
- **Fix** (`v3/yaqa/asym27/`, live in `/data/asym27/code/`):
  - `asym27.sh` drops and unsets the allocator setting;
  - `sibmom27.py` (`sibmom27-w78.patch`): refuses expandable segments on ROCm, saves in 64 MB row blocks
    (fdatasync + DONTNEED), drops open shards after each load;
  - **peak RSS 6.3 → 3.5 GB**, moments bit-identical (E = F, `cmpmom diff` 0), files byte-identical to the old
    `save_lean` (`test_w78.py`: ALL PASS).
- **Queue logging fixed:**
  - `/data/q1106/after-asym27.sh` logs each arm's exit status. It writes "asym27: done" only when both arms have
    `pair.json` and the verdict printed; otherwise "asym27: FAIL …".
  - `asym27.sh` no longer logs "DONE" for a DRY run (the 04:59 UTC bogus DONE), a STOP (now exit 3) or a missing pair.
  - Entry point `/data/asym27/go.sh`, resumable; status `/data/asym27/queue.log`.
- **Running since 21:24 UTC (3:24 PM CST):** s1 layer 0 done in 165 s (6,059 tok/s, peak RSS 3.5 GB). Moments ETA
  ~6:20 PM CST, the s1 arm ~9 PM CST, base + verdict ~2–4 AM CST 11-08.
- **Runtime:** none (quantizer side). Details: `v3/yaqa/asym27/dbg/README.md`, logs `dbg/logs/`.

### 2026-11-07 3:30 PM CST · #46-03 · A4 dropped (user); kernel-speed bound queued
- User: kill A4 (kernel stays merged in qc-exl3-v3 for now; may be reverted for the final PR). To answer "could a
  faster kernel save it": knap5a rerun with the A4 speed ratio forced to ×2 and ×1000 (≈ free) instead of the measured
  ×1.31 / ×1.51, same 144 jobs, colo (`/data/work/w72/jobs-a4x.txt`, nice 19, ~2 h). If a free A4 kernel still fails
  step 4, no kernel work can make A4 pay.

### 2026-11-07 3:25 PM CST · #46-03 · knapsack step 4: A4 stays out of V3
- knap5a with the measured A4 costs (g64 and the final table incl. K1 / K3) vs the A8-only fast variant, rule from
  `v3/e3/a4/README.md` step 4 (≥ 10% lower predicted gemm_us at equal KLD, or ≥ 0.03 bpw smaller): **fails at 2.0 /
  2.5 / 3.0** — the knapsack picks A4 on no tensor (time −0.0% at equal KLD). Even allowing +1% / +3% KLD buys only
  0.6–1.1% time. Same with the older E1c proxy costs. The per-tensor A4 penalty (cheapest at K1, ×0.39 of K3 on
  matched cells) is real but the speed it buys on these tensors is too small. **A4 out of V3; the int4-plane kernel
  is not built; the i4 levels stay for the record / very-fast variants as decided.** Tables
  `v3/e3/a4/cost/data/step4-final.txt`, `analysis-final.txt`.
- Ops: watch rule a4knapfail dropped (it matched "(fail)" inside the verdict text).

### 2026-11-07 3:20 PM CST · #105 · asym's moments step never ran; GPU fault → w78
- The 04:59 UTC "ASYM27 s1 DONE" was a **DRY run** (mom.log `DRY:`, 0 min). Real runs (16:04, 20:52, 21:01, 21:02
  UTC) pass pre (oracle bit-exact) + gate, then `sibmom27.py` dies within ~30 s: HSA illegal instruction / illegal
  memory access. Ruled out: venv (unchanged since 11-05), shared JIT dir and Triton cache (private dirs: same fault),
  the card (plain torch OK). → **w78** debugs with serialized kernels, fixes, runs both arms. Also: the queue logged
  "asym27: done" after a failed run (exit status ignored); w78 fixes that.
- w76 told to stage + launch its card-0 checks (tile CD + H_out no-harm, i4 sub-1) now, sharing card 0 via the locks.

### 2026-11-07 3:10 PM CST · #113 · random-table bound: a better 20-bit code is worth ~3–4% with exact search
- g5 finished on colo + Thor (Thor's half completed 18:58 UTC, before the outage). Error vs today's V16, real tensors:
  **exact Viterbi on an iid random 2^20 table (V20r): K2 0.974 (n 54, 0/27 worse), K3 0.970 (n 49), K4 0.963 (n 15)**;
  beam 4096 on it (B20-4096r): 1.057 / 1.032 / 0.998 — the beam throws the gain away. Our hash at exact V20: K2 0.990.
- Reading: a good 20-bit code would buy ~2.5–3.7% (matches the iid synthetic 0.975 / 0.962 / 0.944 at K2/3 and
  under-shoots at K4), but only with search near exact (×13–46 encode time) — a 4096 beam is worse than today.
  **V3 unchanged (16-bit window). V4 note:** needs a better computed code AND a search far cheaper than exact
  Viterbi at 2^20 states; plus +10–28% dequant cost. Report `v3/e3/settle/w71/report-final.md`.
- **Thor sandbox back (~3 PM CST); idle; no CPU work goes there (user).**

### 2026-11-07 1:40 PM CST · #112 · C9 α on the writers ADOPTED (user) · #114 blocked on Thor
- **#112 adopted by the user** despite PARTIAL (geo 0.9925 vs ≤ 0.99; verify keeps 0.96 of the gain): attn_o α −1,
  gdn_out α 3, down 0, plain writers. Runtime none (scales). Quant time in w75's 27B implementation: 859 s a K2.5 level
  as a separate pass; w76 folds it into the per-tensor job (a per-row quadratic form on the 64 writers).
- **#114 (damping, w77):** colo q77 at 222 / 405 jobs, the remaining 183 sweeps wait on the second held-out Hessian
  half (H_B2), which Thor was building; Thor unreachable → w77 resumed with a note to rebuild H_B2 on colo.
- w76 and w77 resumed after the 16:0x UTC rate-limit kills; w67 held (no card / Thor available for it now).

### 2026-11-07 1:30 PM CST · #112-02 PARTIAL · #46-03 done · #105 never ran · VPN outage
- **#112-02 (C9 α on the writers, 27B, 2 seed pairs):** search 0.9939 / 0.9911 (z −2.0 / −4.2), geo **0.9925** vs the
  rule's ≤ 0.99; verify 0.9920 / 0.9936 (keeps 0.96 of the gain); no flags → **PARTIAL** (real, ~−0.75%, just short of
  the bar). Lead / user decision; no runtime cost.
- **#46-03 (A4 costs, w72's queue):** `A4COST VERDICT: 28 of 144 cells`. Summed A4-g64 cost over the measured cells:
  K1 +56.9e-4, K2 +88.1e-4, K3 +72.0e-4 (g128: +91.0 / +72.9 / +83.8e-4). Per-cell ratios dK1/dK3 g64 ×0.39 (DEP:
  the cost depends on K, cheaper at K1), dK2/dK3 ×0.82 (independent). Knap step-4 verdict pending (w72's knapwait).
- **#105 asym (27B) never ran:** both arms stopped at 16:04 UTC on asym27.sh's `MemAvailable < 6 GB before oracle`
  pre-check (the same pre-lock memory gate that stuck c9w27). Relaunch with MINMEM_GB=4 once iceman is reachable:
  `MINMEM_GB=4 setsid nohup bash /data/asym27/code/asym27.sh >> /data/asym27/run.out 2>&1 < /dev/null &`.
- **19:25 UTC: iceman and Thor unreachable** (the user's VPN host is down for maintenance). colo is up (#113 g5 /
  #114 running). Card 0 had gone idle at 19:03 (Q2, A4COST, C9W27 all done). Queue on reconnect: asym27 (#105), then
  headwm.sh (slab head), cdhout27 (tile CD + H_out no-harm, w76), qtf2i4check (w76).

### 2026-11-07 10:00 AM CST · #46-03 interim (w72) · ops: c9w27 pack gate
- **A4 cost table, i4 K2 (2.17 bpw), real A4 kernel, search slice, baseline all-A8-g128 KLD 0.1277:** A8-g128 vs f16
  ×1.0001 (free). A4-g64 on **gate_up** by 16-layer band: +0.00056 / +0.00094 / +0.00123 / +0.00323 (z 3.8-9.6);
  cost rises ~6x from the first band to the last. `A4COST VERDICT: 4 of 24 cells; K2 g64 sum +59.6e-4` (the other
  families and g128 still running). **Level order changed to K1 -> K3 -> K2 (user 11-07: K1 is the decisive point,
  where A4 should be cheapest relative to the weight error);** K1 quant started 15:33 UTC.
- **Ops:** c9w27's pack waited on `MemAvailable >= 8 GB` **before** taking the exclusive ram.lock -- circular (the
  memory only frees once the lock is held), stuck 40 min and heading for a 3 h FAIL. Relaunched with PACKMEM_GB=3:
  the exclusive `conv.lock` + `ram.lock` are the real guard, the pre-lock poll only needs to clear memguard's 2.5 GB
  floor. Rule for every queue: gate packs on the lock, not on a memory poll.

### 2026-11-07 9:25 AM CST · #38 / ladder · user decisions: sub-1 input damping 0.05 ON; i4 sub-1 qtf2 patch applied
- **Sub-1 input damping (#38):** inputs (q/k/v, gate/up, GDN in) below K1 use σ 0.05 (others 0.005 / 0.025 unchanged);
  adopted by the user as #38 measured it. Implemented in the pipeline by w76.
- **i4 sub-1 (w53's `qtf2-i4.patch`):** applied to ourq / rxq by w76 (oracle-exact at every rate in w53's checks;
  `qtf2i4check` re-run on card 0 by the lead). The ladder's i4 K0.125–0.875 levels depend on it.
- w76 also ports the level passes and S6–S8 into `pipeline/`, H_out at K ≥ 4 (`apply_k_min`), tile CD with diagonal
  H_out (user: write it), C9 / asym switches (off until verdicts), swap-pair hygiene.

### 2026-11-07 9:20 AM CST · #112-01 / #112-02 / #106-01 (w75) · #46-03 interim (w72) · RULER harness (w66)
- **#112-01 (0.8B):** gdn_out α 4–4.75 beats α 3 by only 0.1–0.4% (geo 0.9987 / 0.9961 / 0.9987 / 0.9973, st K3 / K2,
  pl K3 / K2; verify agrees) → misses 0.995: **keep α = 3**, no second 27B arm; multipliers ≤ 1.9, no clamp needed.
- **#112-02 (27B, fixed codes, swj K2.5 levels; identity gate: 723/787 weights byte-identical, only the 64 writers' svh
  differ):** seed-1 pair **search 0.9911 (z −4.2), verify 0.9936 (z −3.1)**, p99 and top-1 better. Base pair ETA
  ~11:00–11:30 AM CST; rule: geo ≤ 0.99 over both pairs.
- **#106-01 (0.8B real quantizer):** heaviest-heads-last cuts o_proj quantization error 9% (every seed) but KLD does
  not follow: K3 0.9978 (mixed), K2 1.0033 (every seed worse), verify agrees → **#106 dropped.** bf16 gate amended
  (w75 amendment 1, accepted): permuted checkpoint bit-identical except o_proj summation order. Side finding: that
  order change alone moves the 0.8B's bf16 logits by mean KL 7e-3 (noted in `proxy2b/common.py`); paired same-path
  comparisons unaffected.
- **#46-03 (w72, interim):** measured with the **real A4 kernel** (amendment 1 in `v3/e3/a4/README.md`, accepted: the
  E1b hook never sees mul1i4 weights). i4 K2 (2.17 bpw) mixed into V2 3.00 (non-trunk byte-identical): baseline A8-g128
  0.1277; **A8-g128 vs f16 ×1.0001 (free)**; A4-g64 on gate_up per 16-layer band +0.00056 / +0.00094 / +0.00123 /
  +0.00323 (z 3.8–9.6): **A4 is not free at K2 and gets costlier with depth.** Reference knap5 with the old proxy costs
  never picks A4. Full table + knap step-4 verdict: ~7–9 PM CST (g64) and ~2–5 AM CST Nov 8.
- **RULER harness (w66):** `eval/ruler/` committed (sets git-ignored: the essays have no licence). Generators byte-equal
  to NVIDIA RULER on 27 cells 4k–768k; resumable runner; 27B fast probe (48 rows, ~11 min) passes; 64k / 128k 27B
  points not run yet.
- **Runtime:** #112 none (scales); #106 none; #46-03 is a prefill speed option.

### 2026-11-07 8:25 AM CST · #113 · trimmed to the random-table bound (user)
- K2 / K3 complete (54 pairs each): no arm passes at any K (B20-4096 1.0096 / 0.999 / K4 0.992 n 21; V18 0.994 /
  0.994 / 0.992; exact V20 K2 0.990 n 5). Every family within ±0.2% (attn_qg K4 0.979 rests on n 2: not followed up,
  too small a byte share to matter). **Verdict: not adopted, final for V3.** Remaining completeness jobs (K4 / K5 / K6,
  V20 on our hash) stopped at 14:24 UTC; only g5 runs (V20r / B20-4096r: an iid random 2^20 table, K2–4, colo sample 1,
  Thor sample 0) = what a better 20-bit code could reach. Reported only; a V4 format note if it shows ≥ 3%.

### 2026-11-07 6:55 AM CST · #33-28 / H27 · H_out at K4: PASS → adopted for levels ≥ K4
- K4 seed pair landed (bn4s1 vs base4s1): **0.9672** (z −2.54), with the first pair **0.9613** → geo **0.9643**;
  seed floor σ = 0.0062 (base4 → base4s1 1.006); amendment 5's rule (every pair < 1, geo ≤ 0.98, |log geo| ≥ 2σ/√n):
  **PASS**. With K2.5 not adopted (0.9801), **H_out applies at K ≥ 4 only.** Search pool slice, 103 docs. Mix
  structure identical across the four arms. Details `v3/yaqa/H27.md` "K4 result".
- **Cost:** H_out quantization ~3.2 h a level vs ~37 min (TODO: re-price the ladder's K ≥ 4 levels and donor hours).
  v3.toml needs an `apply_k_min` the pipeline engine reads (flagged in the file).
- **Runtime:** none (quantizer-side).

### 2026-11-07 6:50 AM CST · #113 · 2^20-state beam trellis: not adopted (provisional) · w71
- Order-fixed settle proxy (gates pass: L16 Viterbi = viterbi.c exactly; beam M = 2^16 = Viterbi), 27 tensors × 2
  samples. Error ratio vs today's V16: **B20-4096 (PR #364's setting) K2 1.0096, worse on all 27 tensors; K3 0.999**
  (36/54 pairs); B20-256 / 1024 1.07–1.18; exact **V18 0.994**. Rule (≥ 3% at 2 of K2/3/4) cannot pass: K2 is final,
  K3's remaining pairs would need ~0.93. K4 and the bounds (exact V20; a random-table L20 = what a better hash could
  reach) still run on colo / Thor (until ~8–11 PM CST), reported only.
- Why: M = 4096 of 2^20 is too narrow (synthetic: M ≈ 16384 needed), and our mul1 hash stretched to 20 bits is a weak
  code (iid, exact search: L20/L16 0.990 / 0.983 / 0.993 at K2/3/4; an ideal table 0.975 / 0.962 / 0.944, IQR's computed
  code about the same). Cost: encode ×3.7 / 6 / 9.8 at K2/3/4 (×19–33 at K5/6); decode estimate +10% dequant ALU
  (+28% at K4: the 16-bit packed-hash and v_perm paths break).
- **Keep the 16-bit window for V3.** Possible V4 note: a better 20-bit code (not our hash), if the bounds show a real
  ceiling. Code `v3/e3/settle/w71/`; rules / results `v3/e3/settle/w71.md`.
- **Runtime:** would have been decode + prefill (not adopted).

### 2026-11-07 6:00 AM CST · #112 · C9 α on the writers: PASS on the 0.8B (both stacks)
- 4070 g64 item 3 (rules `v3/e3/settle/w64.md`, written before data), 3 seeds, search deciding, verify agreeing:
  **st (asym writers) geo K3 0.9890, K2 0.9592; pl (plain writers) K3 0.9895, K2 0.9637**, every seed < 1, verify ≈
  search. The gain is almost all **gdn_out at α = 3 (the grid edge)**; attn_o −1 / 0, down 0. Rule → **27B fixed-code
  pair** (no requant, ~1 card-h). The pl result matters given #111: C9 on plain writers works too. Readout copied to
  `v3/e3/settle/w64/report-4070.md`. Next (w75): 27B pair on gpubox, α ∈ {3…6} on gdn_out (edge), 0.8B.
- **Runtime:** none (the factor folds into the stored scales).

### 2026-11-07 4:45 AM CST · #104 / ladder · config: swj at every level, mul1 K6 / K6.25 removed (w73 review)
- w73's consistency pass (docs 86e7871, paper rebuilt, 12 pages) flagged that v3.toml still had swj `k_max = 3.5` and
  mul1 K6 / K6.25. Both resolved by recorded user decisions: swj adopted at every level (SWJ27 amendment 2's K4 seed
  pair exists: head-matched 0.990 / 0.987) → `k_max = 8.0`; mul1_levels start at K6.5. Still open (flagged by w73):
  sub-1 damping 0.05 (#38 verdict vs ourq's 0.005; paper says "under review"), H_out K≥4-only scope (bn4s1 pending),
  agent32 incident wording (w67 owns it).

### 2026-11-07 3:45 AM CST · #109–#112, #106 / #103 re-run · w64 (settle proxy order fix + shortlist)
- **settle.py LDLQ order fixed** (`v3/e3/settle/w64/settle64.py`, `SETTLE_ORDER=rev`, exllamav3's last-block-first):
  matches w50's GPU tile-CD ratios within 0.5% on 9/9 blocks; the old forward order moved L3 attn_qg error +14…+26%.
  **Older settle.py results that depend on block order are biased** (#106's −7.3%, #77's proxy 2–4%).
- **#109 clipped i8: negative** — α never helps at K ≤ 7 (K7 i8/mul1 ×1.239 at α 1, worse for larger α), ≤ 3% above.
- **#110 bf16 (row,128) scales: fail** — ×1.12–1.18 vs plain K at equal bits; closed for V3.
- **#111 writers asym under knapsack mixes: fails** (0.8B, 2/3 seeds, z +12…+26, verify agrees): per-level asym writers
  next to siblings of another K are worse than plain — (3 on 2.5) 1.077, (3 on 3.5) 1.229, (2 on 2.5) 1.239. A per-mix
  writer pass keeps the gain (0.81 / 0.90 / 0.88). So asym (if #105 passes at 27B) needs a per-mix writer pass
  (~4–5 card-h a mix at 27B), writer K tied to sibling K in the knapsack, or plain writers — **user decision**.
- **#106 re-run:** o_proj head act order still passes, −3% (was −7.3%); GDN out_proj marginal pass 0.980/0.978/0.980.
  Converter must store the heaviest heads **last**. **#103:** still fails (0.990).
- #112 (C9 α after asym) running on the 4070 (`/data/yb/w64/`, ETA ~6:15 AM CST). Rules / results `v3/e3/settle/w64.md`.
- **Runtime:** #109 / #110 would have been decode / prefill (not adopted); #111 / #112 / #106 none.

### 2026-11-07 3:30 AM CST · #— · API rate limit killed five workers; resumed
- The 5-hour API limit (reset 09:10 UTC) killed w64, w67, w70, w71, w72 at ~08:55 UTC and the watchdog's cycle. Their
  machine queues kept running (they are separate processes). All five relaunched at 09:28 UTC with a resume note; the
  watchdog relaunched; w73 started a docs + paper consistency pass. Mining + ideas runs stay armed for 13:00 UTC
  (they wait for the cap themselves).

### 2026-11-07 2:45 AM CST · #113 · Outside entry PR #364 reviewed; bigger trellis queued
- Radiance PR #364 (Perire, "KL-guided IQR quants", Small 5.4988 bpw, not yet scored; IQR-V4 base): KL-weighted Gram,
  quarter-step widths by predicted KL, **2^20-state trellis with beam search**, row slabs, row-weighted search, vocab
  head, one bf16 (row,128) scale, hi/lo int8 activations. Everything but the 2^20-state beam trellis we have or measured
  (H_out ~0.98 at 27B; #96 slabs small; #95 bf16 scales mostly negative). At 5.5 the board has IQR-V4 0.00154 vs V2
  0.00153. → #113 (w71, colo): L = 16 / 18 / 20-bit state windows with beam M 256–4096 on the order-fixed settle.py.

### 2026-11-07 2:15 AM CST · #— · rqah PRs #17 / #21 / #22 / #23 merged locally; agent32 golden broken input
- rad-quant-at-home: #19 / #20 merged by the user (GitHub UI); #17, #23, #22, #21 merged locally as merge commits on
  main (doc conflicts: TESTS history lines kept, README count fixed after; #22 × #16 in `Queue.upload`: #22's byte
  accounting + #16's truncate, truncated bytes taken back off the spool counter). Follow-ups (user): w68's patch 0003 (worker re-syncs its upload offset after a connection error), a 1 h `--refail-s` window on #23's exclusion (which also covered expired leases, i.e. donors switching machines off; e2e check added, 119/119), and GitHub Actions CI for the no-GPU suites. All suites green: e2e 118/118 ×2,
  score 31, statuspage 19, checker_nan 12, sched 26, bandwidth 22. 18 commits for the user to push.
- gpubox insert-60 (w67's agent-search32 golden): killed at 07:56 UTC after stage 2 printed ppl 57,304 on doc 1 — a
  broken input, not the model; w67 is debugging it and will rerun as a new insert. bn4s1 (H_out K4) unaffected.

### 2026-11-07 1:50 AM CST · #33-28 · H_out (bn) at K2.5: third pair lands; misses the rule by a hair
- s2 → bns2 **0.9762** (z −2.2). The three K2.5 pairs: 0.972 / 0.992 / 0.976, every pair < 1, **geo 0.9801** (rule
  ≤ 0.98: misses by 0.0001); |log geo| 0.0201 vs 2σ/√3 = 0.0207 (σ = RMS |log| of the three base seed pairs, 0.0180):
  also misses by a hair. **By H27 amendment 5: not adopted for levels ≤ K3.5.** Real-looking (every pair < 1, −2%),
  but at the noise line, and H_out makes a level ~5× slower to quantize (two-sided LDLQ). K4 pairs (bn4s1, ~4:30 AM
  after the agentic-golden insert) decide ≥ K4; one K4 pair so far 0.961.

### 2026-11-07 1:35 AM CST · #— · rqah PR batch 2 reviewed (w68)
- #19–#23 all correct; merge order on main: #19 → #20 → #17 → #23 → #22 (needs a rebase: conflict with merged #16 in
  `Queue.upload`) → #21. Simulated in a throwaway clone with all suites after each step: green (e2e 118/118), except a
  pre-existing worker bug in main (after a crash mid-upload the worker re-sends at a stale offset; the server's 409
  arrives as a broken pipe) that #22 makes frequent: w68's patch 0003 (resync the offset after a connection error).
  #23 excludes a worker from a job it failed until `admin requeue` (stricter than w62's 1 h). w62's patches 0001–0002b
  are superseded. Still open: #4, #7, #8, #9, #11, #12 (decisions recorded in TODO). The 4070 went down 07:31 UTC
  (the user took it; preemption handling reported it as preempted).

### 2026-11-07 1:05 AM CST · #— · rqah setup wizard (w69)
- `rqah setup` on local branch `setup-wizard` (c095878, on main 2ac3b3a; new files only): read-only hardware / Docker
  detection, curses / plain / scripted modes, writes `~/.config/rad-quant-at-home` (worker.env 0600, compose.yaml,
  docker-run.sh), `--check`, re-run edits; card matrix (4 GB refused, 8 GB layers to K6, 12 GB all trunk jobs, 16 /
  32 GB evals). Tests 125/125 on fake machines (VM), 123/123 on the 4070 box. Not run on a real AMD card or a published
  image yet. Hook-up patches (CLI registration, README) wait for PRs #17 / #19–#23.

### 2026-11-07 1:00 AM CST · #106, #108, #103 · o_proj act order passes on a biased proxy; GDN refits fail in the stack
- **#106 (w46, settle.py; rule in `v3/e3/actblk/README.md`):** o_proj input-block order via a head permutation (free
  at runtime): 0.928 / 0.927 / 0.927 at K2 / 3 / 4 (n 6, no tensor > 1), all 16 layers 0.94, flat K1.5–6 → **PASS by
  the rule**; GDN out_proj 0.986–0.988 (fail). **Caveat: settle.py runs LDLQ over input blocks first-to-last,
  exllamav3 last-to-first, and #106's reversed-order arm cost +5%: the result may shrink or flip on the real order.**
  Re-run on the fixed proxy (w64 item 5, with #103 too) before a 27B pair. Was unlogged since 11-06 6 PM (my miss).
- **#108 final seeds (d1113, full 0.8B stack, search):** K3 geo 0.999 (5 seeds), K2 0.985 (4 seeds, all < 1); verify
  ~0.994 / ~0.985. Final (D1113 DONE 07:14 UTC): K2 seed 4 0.988 / verify 0.986 → K2 geo ≈ 0.986 over 5 seeds. Fails the rule (≤ 0.98 at K3 and K2): GDN refits overlap with H_out / asym in the stack. Not ported
  to the 27B unless H_out and asym both fail there (user: a barely-winning 0.8B result isn't worth a 27B port).

### 2026-11-07 12:40 AM CST · #— · D-LSA on window residency (w65): built, host / emulator tests pass, no card run yet
- Branch `lsa-residency` (`/workspace/forks/radiance-lsa-res`): #412 cherry-picked (author kept; ABI: #412 keeps 38, our
  D-LSA moves to 39), bridge ops `lsa_page_gather` / `lsa_page_scatter` (no change to our list-attention kernel),
  qwen35_bf16 wiring. Host + emulator tests pass (`r4d_emu_lsapage` 1853/1853: selected K / V bytes equal the
  all-resident run, bf16 and fp8). Design `docs/design/lsa-residency.md` (in that worktree).
- Findings (computed, not measured): #412's single `residency_window` ties the resident tail to the scratch pages, so
  D-LSA must declare a 16,384-token tail (+0.5 GiB fp8 / +1 GiB bf16 per sequence); residency does **not** raise the
  longest single prompt (prefill is dense and pages the middle back: ~530k fp8 / ~275k bf16 at 4.0 bpw stays the
  limit) but cuts a sequence's resident VRAM after prefill (128k fp8 4.3 → 1.06 GiB; 512k 17.2 → 1.18 GiB): a win for
  several long sessions per card. To measure on the card: worst-case page-in ~512 MiB / token (10–21 ms vs ~25 ms
  weights), pinned host store ~16.5 GiB at 512k fp8 (gpubox has 30 GB). Not on gfx1201 yet (w65 stayed off gpubox).

### 2026-11-07 12:00 AM CST · #— · Shortlist items 1–4 queued (w64); LSA on window residency (w65)
- w64 runs w61's items 1–4 (asym under knapsack mixes, per-rate int8 factor at K6.5–8, C9 writer gain refit after
  asym, bf16 (row, 128) scales from K5) on the CPU boxes (4070 only where a 0.8B KL is required), after fixing
  settle.py's LDLQ block order. Rules before data in `v3/e3/settle/w64.md`.
- LSA: Code000's LSA is not public (none in his radiance fork or RadianceDev). His #412 (window residency for sparse
  attention, built for QSA) is the natural partner for our D-LSA; w65 maps it onto our LSA and prototypes
  `lsa-residency` (long context on one card for the dense 27B). `pr3-isa-occupancy` is unrelated (an occupancy-script fix).

### 2026-11-06 11:55 PM CST · #— · Ladder re-price corrected (w63); ideas shortlist + mining brief (w61)
- **w63 (corrected swj curve ×0.91 K2, ×0.935 K2.5, ×0.989 K4, 1.0 from K5):** predicted KLD vs w58: −2% at 2.0, +1% at
  2.5, +7% at 3.0, +20% at 4.0, +22% at 5.0, +16% at 6.0, +9% at 7.0. The 11-06 projection holds at 2.0–2.5 (−27% vs V2's
  menu at 2.0) and 3.0 (−14% vs −16%); w58's "menu + swj alone reaches the freeze expectation from 3.5 up" is withdrawn
  (3–13% short at 2.5–6.0; the rest must come from H_out, asym, healing, search). Level usage unchanged (17 i4 levels
  never picked; pass-1 levels all used); turboderp tensors return for 10–18% of params at 5.0–7.0.
- **w61 shortlist (`local/mine/shortlist-2026-11-07.md`) and mining brief (`brief-2026-11-07.md`):** top pre-freeze
  items: writers' asym under knapsack mixes (it only works on the exact sibling codes; #105 tests uniform levels);
  per-rate int8 factor ("clipped i8") at K6.5–8; C9 gain refit for the writers after asym; bf16 (row, 128) scales from
  K5 (our #95-02 rejection used the biased proxy); massive-channel handling for layers 10–11; conv-aware H_in for GDN.
  Caveat: its rationale cites swj's withdrawn K4 −21%; the ranking still stands on the K2.5 result.

### 2026-11-06 10:25 PM CST · #33-28, #108, #— · s2 control; GDN refits shrink in the full stack; ladder re-priced
- **#33-28 control:** base → s2 0.985 (z −1.3), s1 → s2 1.010 (z +1.1): seed-to-seed spread at K2.5 is ~±1.5%, as
  before. bns2 (the third H_out pair) quantizing, done ~1 AM CST.
- **#108 full stack (d1113 stage 2, vs the adopted 0.8B stack; search / verify):** K3 s0 1.002 / 1.002, s1 0.995 /
  0.996; K2 s0 0.987 / 0.992, s1 0.979 / 0.974. In the full stack (H_out dy + swj + writers asym) the GDN refit's
  screen gain (0.98–0.99 K3, 0.95–0.96 K2) mostly overlaps what H_out / asym already fix: on track to **fail at K3**,
  borderline at K2. Seed 2 and the verdict pending (~1:45 AM CST).
- **Ladder re-pricing (w58, knap5 = knap4 + measured i8 / i4 / mul1 ratios, two-pass A8 times; `v3/levels/reprice.md`):**
  menu-only predictions move 0 to +3.6% (the i8-K7 re-pricing is small at most sizes); accurate variant uses i8 to
  K5.75 and mul1 from K6; fast variant is all A8; **fast 7.0 predicts ~0.00052, worse than V2 7.00 (0.00045)**; the
  17 i4 levels are never picked at 2.0–7.0 (A4 not picked on E1c costs either); mul1 K6 / K6.25 near-duplicates;
  pass 1's 25 levels come within 0.2–0.8% of all 73. **Its swj factor used the withdrawn K4 ×0.78**, so every
  "menu + swj" number above K2.5 is overstated → rerun with the head-matched swj (×0.935 K2.5, ×0.989 K4) queued.
- d1112 DONE 01:58 UTC (all 7 seeds; report on the 4070).

### 2026-11-06 8:55 PM CST · #77, #108 · Tile CD passes the 27B no-harm check; GDN refits hold at seeds 1–2
- **#77 (`v3/e3/settle/CD27.md`):** 16 attn_qg tensors re-quantized with cd3 in two K2.5 arms: s1 → s1cd 0.9994,
  base → basecd 0.9997 (z −0.3), geo 0.9996; both no-op gates pass (the private tree reproduces each level bit for
  bit). **PASS → cd3 on attn_qg adopted** (as expected, the whole-model effect is within noise; adopted on the per-tensor
  gain and no harm). Card 0 moved on to arm s2 at 02:40 UTC.
- **#108 screen, seeds 1–2 (fit golden, vs trunk + swj):** GDN arms reproduce seed 0: K2 gd 0.954, dw 0.945; K3 gz 0.987,
  gd 0.983, dw 0.983 (z −9…−14). Attention arms stay ≥ 1 (qk 1.006, at 1.006, aw 1.004 at K3). The rule's full-stack
  stage (search + verify) runs next.

### 2026-11-06 8:05 PM CST · #108, #— · Joint refits: GDN pays, attention doesn't (interim); rqah review
- **#108 (w60, d1113, 0.8B, seed 0 screen, ratio vs trunk + swj on the fit golden):** GDN junction refits pay: z gate
  only 0.9905 (K3) / 0.9666 (K2); qkv + z 0.992 / 0.964; with the loss after out_proj 0.991 / 0.960. Attention refits are
  flat or worse (qk 1.004 / 1.012, vg 1.001 / 1.003, q+k+v 1.003 / 1.013, loss after o_proj 1.000 / 1.010) even though
  they lower attention-output error on held-out text (q_norm / k_norm absorb per-head uniform scales; only within-head
  relative scales reach the logits). The fitter reproduces swj on the MLP (0.9996 / 1.0035), so the nulls are the
  junction, not the method. 3-seed screen and the full-stack stage (search + verify) are running; rule decides. Calib
  use: qad pool (moments.py windows, seed 11, first 600 windows) — ledger entry to add. 27B port if it passes: svh edits
  on in_proj_qkv / in_proj_z, swjapply-style, zero runtime cost.
- **rad-quant-at-home review (w62):** all 14 findings from the outside contributor are real on 0.4 (reproduced);
  PRs #16 / #18 merge as-is, #17 after a one-hunk rebase; w62 patches cover #5 / #6 / #10 / #13 / #14; open decisions
  #4 (eval_mix trust), #7, #9, #11, #12. `local/workers/w62/REVIEW.md` (draft replies included).

### 2026-11-06 7:50 PM CST · #104, #81-04 · CORRECTION: swj's K4 "−22%" was a head mismatch; A1 real but small
- **#104 amendment 2 (w59, fresh verify slice `golden-verify27`, ~100 verify-pool docs; rule before data):** the 6:10 PM
  K4 pairs compared mixes with **different head / embedding**: the swj arms were packed into V2 4.00's container, the
  base4 / bn4 arms into a head from V2 3.00. Head-matched controls (base4v4, bn4v4: the same trunks re-packed with 4.00's
  head) show the head swap alone is 0.761 / 0.757 (verify; 0.798 / 0.777 search). **Head-matched swj at K4: 0.9903 /
  0.9870 (verify), 0.9907 / 0.9869 (search)** → geo 0.989, every pair < 1 but ≥ 0.98: **NOT CONFIRMED** as a large gain;
  real and small (−1.1%). **K2.5 confirmed on verify: base → baseswj 0.9351 (z −15.3), s1 → s1swj 0.9316** (p99 0.96 /
  0.95, top-1 +0.4 pt). So swj: −6.5% at K2.5, ~−1% at K4 (the gain shrinks with K, as on the 0.8B). The 6:10 PM entry's
  "−21…−23% at K4 / adopt at every level" is withdrawn; swj stays adopted (it never hurts; size per K from these pairs).
  Lesson: every 27B pair must have identical non-trunk parts; the swap tooling now checks it (TODO).
- **#81-04 A1 (d1107, 0.8B, 2 seeds):** a1u vs base 0.984 / 0.971 / 0.970 at K2 / 3 / 4, every seed < 1: **real**, but the
  option rule (≤ 0.97 at K2 and K3) **fails** (K2 0.984, K3 0.971). a1ug (gate+up predictors, a second GEMM): 0.963 /
  0.951 / 0.959. Not runtime-free (one or two extra GEMMs on down_proj's input). User decides whether a1ug merits a
  kernel-cost check; default: not in V3.

### 2026-11-06 7:20 PM CST · #33-28 · K4 seed control
- base4 → base4s1 (K4, rotation seed only) **1.006** (z +0.4): the K4 seed floor is small (K2.5's was 0.975). The pack
  check passed (mix-base4s1 = mix-base4 byte size, 3.9811 bpw). Next: bn4s1 vs base4s1 (H_out at K4, second pair).
  w59's verify insert (insert-40) is running its scores; 4070 d1107 (A1) first pair k4a1u 0.971 (interim).

### 2026-11-06 6:40 PM CST · #81-04, #— · A1 rerun (it never ran); joint refits beyond the MLP; ideas sweep
- **#81-04 (A1, d1107):** the 11-04 run scored nothing: the sourced d1103 block reset `X`, so `a1q.py` was never found
  and every pair printed `no pairs`. Fixed in `run.sh`, registered in the 4070's `/data/queue.d/40-d1107.sh`, rerunning
  K2–K4 (the K5 bases are gone). Stale INDEX rows (#33-13, #58, #90, #90-02) closed.
- **w60 (4070, d1113):** swj-style joint scale refits at other junctions (q/k through attention, v / o, GDN
  projections through their gates, all on top of swj), 0.8B, rules before data.
- **w61:** sweep of every idea doc for untried quantizer-setting ideas + a brief for the next mining / ideas run.

### 2026-11-06 6:25 PM CST · #77 · Tile CD on attn_qg: user adopts pending the 27B no-harm check; check scheduled
- User: adopt cd3 on attn_qg (real per-tensor gain 0.5–1.2%, w50) if the 27B check passes. Code staged as a private
  tree on gpubox (`/data/cd27/code`, ourq + `ourq-tilecd.patch`; the live ourq untouched), dry run clean, insert
  `/data/q1106/insert-50-cd27.sh` queued (runs before the next q2 arm, after w59's verify insert if that lands first).
  Rule (`v3/e3/settle/CD27.md`, before data): pairs s1 → s1cd and base → basecd at K2.5 (16 q_proj re-quantized with
  cd3), pass = geo ≤ 1.000 and no pair > 1.01; a no-op gate first requires the private tree to reproduce each level
  bit for bit.

### 2026-11-06 6:20 PM CST · #100 · Hash codebook dropped (user)
- User decision on #100c: the IQR-style hash codebook is dropped; **mul1i8 stays the V3 codebook** (format decision
  closed before the ladder). Next for swj: confirm the K4 gain (0.79 / 0.77) on a verify-pool slice before quoting it.

### 2026-11-06 6:10 PM CST · #104, #100, #46-03, #— · swj at K4 and on H_out; hash codebook misses; kernels on qc-exl3-v3; rqah 0.4
- **#104 amendment 1 (gpubox, golden-search27, 103 docs):** swj on top of H_out bn at K2.5: bn → bnswj **0.9244**
  (z −14.4), bns1 → bns1swj **0.9210** (z −13.9): it stacks with H_out (the bn+swj vs plain+swj gap ≈ 0.978). **K4:**
  base4 → base4swj **0.7906** (z −5.0), bn4 → bn4swj **0.7667** (z −5.9): every pair < 1, geo 0.778 → **swj adopted at
  every level.** The K4 gain (−21…−23%) is 3× K2.5's; plausible (the scale error is a larger share of a smaller KL) but
  surprising, so check it on a verify slice before quoting it.
- **#100c hash codebook, real run (w48, `v3/e3/sub1/r100c.md`, rules before data):** 0.8B, 2 seeds, fake-quant
  scoring: KL ×1.026 / ×1.040 / ×1.016 at K2 / 3 / 4 on search (verify ×1.025 / ×1.029 / ×1.000) vs bar ≤ 1.01;
  ≈ +0.011…+0.025 bpw at equal quality. Speed on gfx1201: decode GEMV +15.6% / +17.6% / +12.6% (bar ≥ 15% at every K),
  prefill ±1%. Fails both bars, not a hard reject → user's call; recommendation keep mul1i8. Note: the per-tensor
  proxy says +0.8–0.9%, real KL 3–4× that: this codebook's cost is underpriced by the proxy.
- **Kernels:** w53's mul1i4 importer (byte-exact vs the quantizer's i4 decode at every rate, real 27B i4 tensors,
  gfx1201 test_a4 1093 PASS) rebased cleanly onto the A8 + A4 merge (docs-only difference); **`qc-exl3-v3` fast-forwarded
  to 42960d5 (+12 commits, no rewrite)**; the user's PR picks it up on the next push. #46-03 (A4 quality) unblocked.
  w53 ran 19 s on the 4070 without its lock (rule break, no harm seen).
- **rad-quant-at-home:** public commit 7df04f3 + GHCR names (f582e7d) + research kit (`research-kit/`: ROCm / CUDA
  images, LAN-blocking firewall, Tailscale ACL, selftest, VOLUNTEER / OPERATOR docs; firewall boot unit, R9700 image,
  real tailnet untested) + **0.4.0** (6ba5b12: per-direction Mbps caps, daily / monthly data budgets on the data volume,
  budget- and cache-aware scheduling; priority by level granularity 40 / 30 / 20 / 10 / 5, finish a level before the next).
- Design docs updated to 11-06 (w55); KV corpus: Infinite KV Cache + AutoQuant note (w49).

### 2026-11-06 3:00 PM CST · #101, #107, #77, #100, #—  · d1112 readout; tile CD on the GPU; pipeline, rqah v0.3, kernel merge
- **d1112 (w44, 4070, 0.8B, 3 seeds): xcorp H_in on top of the full adopted stack** (H_out dy + swj + asym; `k{K}st`
  vs `k{K}sx`). Search geo K3 0.991 / K2 0.988 (every seed < 1), verify 0.987 / 0.984 (keeps > 100% of the search
  gain), xcorp 0.962 / 0.976, **agentic 1.012 / 1.036** (flag > 1.02 at K2). K4 0.993, K5 0.991 (harmless). So it
  stacks on the 0.8B, but smaller than alone (−1% vs −1.6…−2.6%); moot now that the 27B pairs dropped it (2:15 PM).
  Repro check: `k3st` = d1104's `k3edyswjasym_r0` bit for bit on today's (w41-patched) code. The full stack over its
  trunk quant: 0.836 K3, 0.758 K2, 0.847 K4, 0.866 K5 (3 seeds each).
- **#107 fractional-K drift (d1112 item 3): null.** Interpolated (the default) vs exact-K lookup: K2.25 1.006, K2.75
  0.995, K1.75 0.980 (z noisy), K1.25 0.964 ± 0.028: no measurable effect. w43's worry (exact-K lookup, no correction at
  quarter K) was wrong: ourq already interpolates (drift 1.013 at K2.25).
- **#77 tile CD on the real GPU quantizer (w50):** behind `ourq --tile-cd 3 --tile-cd-families attn_qg` (off = today
  byte for byte, 9/9 shards and stamps). On 27B attn_qg, cd3 ÷ LDLQ = **0.987–0.996**, about a third of settle.py's
  2–4%. Cause: **settle.py runs LDLQ first-to-last over input blocks; exllamav3 runs last-to-first.** With the proxy's
  order reversed, all 9 GPU / CPU cases agree within ±0.33%. Family sweep at K2.5 on the GPU: attn_qg 0.989 / 0.995 still
  the largest; attn_k 0.991 / 0.994 (cheap, ~2 s / tensor); others 0.996–0.998. Time ×3.7 per tensor (q_proj K2.5
  8.1 → 30.1 s on the 4070), ~+5 min per ladder level. 27B no-harm check (`v3/e3/settle/CD27.md`, ~1.3 h) staged,
  not scheduled. **Open:** settle.py's order bias also touches w43's #38 / #39 (null verdicts, unlikely to flip) and
  w46's running re-validation of #90-02 / #100 (check before trusting it).
- **rad-exl3-quantizer (w52):** new side repo, package `rxq`: stages, a per-tensor job unit (group sign replay, < 2%
  overhead), `config/v3.toml` with every setting's experiment # and status; per-tensor quant, H_out two-sided LDLQ,
  asym writers, xcorp blend, H_out bn, swj math / apply, mkdenoise all bit-identical to the research scripts. Profile
  (4070, no H_out): trellis search 76–84% of quant time, scale search 7%, refit 4–8%; gpubox: 35 s / layer (37 min /
  level), **with H_out 178 s / layer** (two-sided LDLQ's error feedback = 72% of a gate_proj). Pack ~1,600 s CPU.
- **rad-quant-at-home 0.3 (w47):** renamed (command `rad-quant-at-home` / `rqah`), verify-by-score (server decodes
  codes, rel error in a per-tensor band from neighbouring results / anchors; 31/31 synthetic, 14/14 real 0.8B incl.
  wrong-K, lazy, perturbed, random codes all rejected; honest 27B seed spread ≤ 0.1%; scoring 18–30 s vs ~100 s to
  quantize a layer on the 4070). Small cards: every family but ffn_down fits ≤ 4.1 GB; ffn_down 6.2–7.0 GB (+~10 GB
  RAM). CUDA image build unfinished.
- **Kernels (w54):** two-pass A8 and A4 merged on `qc-exl3-v3-merged` (fast-forwards, no conflicts; tree = A4 tip);
  checks pass. Next: w53's i4 importer on top, then fast-forward `qc-exl3-v3`.
- w49: Code000's Infinite KV Cache paper and an AutoQuant integration note added to the KV corpus (KV parked).

### 2026-11-06 2:15 PM CST · #101 · xcorp H_in at 27B: too small, dropped
- **27B pairs (K2.5, H_out bn on both sides):** bn → xbn 0.9880 (z −1.2), bns1 → xbns1 0.9971 (z −0.2); pooled ~0.993 vs
  the rule's ≤ 0.98. The 0.8B's −1.6…−2.6% shrinks to ≲ 1% at 27B (as H_out's did). **User: not worth it.** The
  remaining xcorp arms (xbns2, xbn4, xbn4s1, ~11 card-hours) are cancelled by renaming `hi-X0.5/.done`; the bns2 / K4
  H_out arms keep running. The 0.8B recipe and d1112 stay recorded. Larger / broader corpora: future models (BACKLOG).

### 2026-11-06 10:20 CST · #33-28, #94, #38, #39, #77, #104 · Morning readouts; queues refilled
- **H_out bn at K2.5, second seed pair:** s1 → bns1 0.9923 (z −0.7). With base → bn 0.972: geo 0.982, every pair < 1
  but geo > 0.98 → **not yet a pass** under H27 amendment 5; the third pair (s2 / bns2) and K4 pairs are queued.
- **#94 rerun (w45, settle.py): negative, explained.** The GPU smoke used Gaussian tiles at sd 1, where codebook
  clipping hides int8 rounding; exllamav3's scale search picks ~0.71–0.83 sd at K6.5–8, where rounding dominates. On
  real tensors i8/mul1 = 1.018 K6.5, **1.27 K7** (~0.18 bit; 0.26 on attn_qg), 1.69 K7.25 … 4.48 K8; i8 levels top out
  near mul1 K6.85. No i8 level above K7; **the ladder's i8 K7 level needs re-pricing** (knapsack curves, not ×1.02).
  4070 real-quant confirmation queued (w46 item 5). `v3/e3/sub1/r94b.md`.
- **w43 settle (settle.py, 27 tensors × 2 samples; rules before data):** #39 gain per (family, K): **flat, keep 1.0**.
  #38 damping at K1.5 / 2.5 / 3.5 / 5.5: **keep** input 0.005, rest 0.025 (no alternative ≥ 1%). #77 tile CD per K:
  cd1 / cd2 / cd3 = −0.5 / −0.7 / −0.8…−1.1% at 1.7 / 2.5 / 3.2× LDLQ time, 54/54 tensors better; attn_qg −2.2…−3.7%
  (cd3). No adoption rule: user decides (option: cd3 on attn_qg only, a small family). `v3/e3/settle/`.
- **Queued:** gpubox insert `insert-20-swj-bn-k4.sh` (SWJ27 amendment 1: swj on bn / bns1 = stacking with H_out;
  base4 / bn4 = K4 transfer). CPU: w46 re-validates #90-02 (i4 vs i8) and #100 (codebook) on settle.py, new #106
  (o_proj / out_proj head-permutation act order), i8 real-quant check on the 4070.
- w40 (KV autoresearch task, branch `kv-task`) built; T1 blocked by gpubox host RAM (parked with KV).

### 2026-11-06 05:40 CST · #104 · 27B swj passes: −6.5% KLD at K2.5
- **Result (gpubox, golden-search27, 103 docs, 123k positions):** base → baseswj 0.9346 (diff −0.00386, z −12.5);
  s1 → s1swj 0.9346 (z −14.6). Rule (`v3/yaqa/SWJ27.md`, before data): geo ≤ 0.98 and both pairs < 1 → **PASS**.
- About 2.5× the 0.8B's swj gain at K3 (0.971) and in line with its K2 (0.906). Cost: one fit pass per level, no
  requant. Adopted for the stack; open: the K4 transfer check (no data yet), and whether it stacks with H_out bn
  (bns1 arm running) and writers asym (#105, queued after the trunk arms).
- 4070 d1112: `k3st` reproduces d1104's `k3edyswjasym_r0` bit for bit on today's patched code (w41 patches don't change
  results).

### 2026-11-06 04:50 CST · #101, #102, #103, #94, ideas 5 / 7 · Overnight readouts
- **#101 amendment 3 (d1111, 0.8B, 3 seeds, dygm stack):** ratios vs dygm, calib search K3 / K2 (verify; xcorp):
  X0.5 0.974 / 0.984 (0.975 / 0.981; 0.950 / 0.934); X0.75 0.974 / 0.985 (0.979 / 0.984); X3 (3M-token sample)
  0.979 / 0.975 (0.982 / 0.977); X1.0 0.981 / 0.972 but verify 0.990 at K3 = 53% of its search gain → **fails** the ⅔
  rule. Categories: code 1.03 / 1.04 (fail), prose 0.984 / 0.988, mt 1.006 / 0.981 (fail calib ≤ 1.005): none beats full
  mX. Passing arms are within 1% → tie rule: **full mX, w0.5** (the recipe the 27B arms use). Transfer: K4 0.982 (every
  seed < 1), K5 0.990. Agentic subset worsens: +4.3% K2, +1.8% K3, +0.7% K4 / K5. Next: does it stack with swj + asym
  + H_out (d1112, w44, 4070).
- **27B (#101 / #33-28):** first pair bn → xbn (K2.5, 103 docs) −1.2%, z −1.2 (not significant alone; amendment 4
  needs all pairs). Caveat: one tensor at calib H_in (see 02:05 entry).
- **#102 chained tail-biting (w38, colo): negative.** n=4 / n=1 median 0.9961 / 0.9963 / 0.9976 at K2 / 2.5 / 3 (rule
  ≤ 0.98); the n curve is flat from 2 to 32: chaining only amortizes the wrap guess, the free-start advantage is ~16−K
  start bits per tile the format can't store. New: tail-biting cost on real LDLQ tensors, free/TB 0.944 / 0.946 /
  0.951 / 0.959 at K2 / 2.5 / 3 / 4. `v3/e3/chain/`.
- **#103 down_proj block act order (w38): negative.** 0.9942 / 0.9970 / 0.9976 at K2 / 3 / 4 (rule ≤ 0.98). Never worse
  on 24 results; −2.4% L0, −4…−4.7% L63; gain follows the block-trace spread (p90/p10); census over 64 layers: −0.8%
  of down's proxy error, ≈ −0.2% trunk. o_proj (spread 2.83) / GDN out_proj (2.58) head-permutation half: untested,
  BACKLOG. `v3/e3/actblk/`.
- **Ideas 7 / 5 (w42, CPU): negative.** Per-block A8 exceptions for A4: k = 2 blocks cut gate_up's A4 error 6–11% (rule
  ≥ 30%); the error is spread, not concentrated. i4 decode plane on the A8 path at M 2048: +2.8% whole layer if the
  conversion hides (−4.9% if it serializes); not worth i4 levels for speed alone. `v3/e3/a4/blockx.md`.
- **#94 (cpu2 rerun, scale grid to 1.5): inconclusive.** Sanity fails (proxy K7 ×1.29 / K8 ×4.5 vs GPU smoke ×1.024 /
  ×1.347); the wider grid leaves mul1 unchanged. w43 found the old proxy's scale handling distorts results; rerun on
  its exllamav3-faithful proxy (w45, cpu2).
- **w43 (settle, CPU, running):** old #39 "best gain rises with K" looks partly a proxy artifact; items 1–3 land
  ~6–8 AM CST. Possible quantizer gap: `ldlq_drift(K)` is exact-K only, so fractional K gets no drift correction
  (w43 item 1b on CPU, d1112 item 3 on the 4070).

### 2026-11-06 02:05 CST · #101 (27B arm), #33-28 · Corrupt xcorp H_in for L43 o_proj stalled arm xbn; patched
- **What:** arm xbn (K2.5, H_out bn + xcorp H_in w0.5) wrote 63/64 layers; chunk 40–43 failed on L43 `self_attn.o_proj`
  with a Cholesky "not positive-definite" error every try, and swap27k.sh's chunk loop retried it forever (13
  tracebacks in `quant-xbn-c0.log` before the watchdog flagged it at ~1:56 AM CST).
- **Cause:** `/data/hin27/hinX/…layers.43.self_attn.o_proj.input` (mkhin27's xcorp collection) is corrupt: 500 entries
  > 1e10, max |H_ij| = 2.1e18 × max diag, eigenvalues ±3.6e31 (a Gram matrix cannot be indefinite). The published
  calib Hessian of the same tensor is fine (eig 5.2e3…3.7e8). A Cauchy–Schwarz + diagonal scan of all 257 hinX files
  finds only this one (30 residual-stream `.input` files have the usual 1e4–1e5 diagonal spread and pass C–S).
- **Fix (08:00 UTC):** hi-X0.5 uses the calib Hessian for this one tensor (xcorp weight 0 on 1/257 files); originals in
  `/data/hin27/corrupt/`, noted in `blend.log`. The running retry picked it up. The arm's other 63 layers are unchanged.
- **Pattern:** the bad values are one contiguous run of ~500 packed entries (row 3540–3541, columns 3541–4040), random
  magnitudes 1e18–1e29 with mixed signs: a damaged ~2 KB span (memory / transfer / read), not an arithmetic bug. The fp64
  checkpoints were already deleted, so where it happened can't be localized. Retry at 08:01 UTC: o_proj proxy 1.45e-2, normal.
- **Second fault, same arm:** chunk 60–63 crashed with HSA MEMORY_APERTURE_VIOLATION in `ldlq_2hess` (L61); the rerun
  was clean. Check: bn → xbn proxy error per tensor, 400/400 within ±10% (geo 1.022), so no silent damage in the arm.
  Two GPU memory faults on card 0 in one night: watch for more (watch-notes "Lead actions"); recheck after the upgrade.
- **Open:** before any reuse of mkhin27, add a write-time check (finite, Cauchy–Schwarz |H_ij|² ≤ H_ii H_jj, fail
  loudly) and, for the blend, the same check; consider recollecting this one tensor if xcorp H_in is adopted;
  the xbn pairs carry this caveat (one o_proj at calib H_in) — too small to move a whole-model ratio.
### 2026-11-05 11:00 CST · #91, #93 · Self-calibration readout; d1110: no overfitting, xcorp not a calib substitute
- **#93 (d1110), g08-verify1 (180 untouched verify docs), seed 0:** the adopted 0.8B stack keeps **91% (K3) / 96% (K2)**
  of its search-slice log gain (cumulative 0.698 / 0.521 vs 0.674 / 0.507); every step holds (H_out 0.921 / 0.930,
  swj 0.971 / 0.906, writers 0.866 / 0.839, healing 0.901 / 0.737). Seed floor base vs base_s1 1.0005 / 1.003. **Pass:
  no overfitting**; the 27B's smaller H_out gain (−2.8%) is a model-size effect.
- **#93 amendment 1, g08-xcorp1:** same sign everywhere, but kept gain H_out 52% / 71%, swj 63% / 70%, writers 91% / 91%,
  healing 75% / 77%, stack 75% / 79% (K3 / K2): outside ±30% for H_out and swj: **xcorp does not replace calib** for
  0.8B screening (calib-fitted statistics help most on calib-like text; real-world text keeps ~75–80% of the gain).
- **#91 (d1108), 0.8B, calib / own agentic text (g-agent) / calib's 27B traces, K3 / K2:**
  H_out self-gen blend w 0.25: 0.998 / 1.024 · 0.960 / 0.976; w 1: 1.085 / 1.122 · 0.956 / 0.965.
  H_in self-gen w 0.1: 1.010 / 1.011 · 0.947 / 0.942 · 1.11 / 0.956 (noisy); w 0.2: 1.022 / 1.015 · 0.958 / 0.939;
  w 0.5: 1.061 / 1.058 · 0.859 / 0.847 · 0.997 / 0.942; w 1: 1.45 / 1.43 · 0.854 / 0.821.
  Healing on self-gen / raw / turboderp text vs qad: calib +10% / +10% / +7%, agentic −1.4 / −2.2 / −1.8%.
  Per-family blends: no difference. **Verdict:** main rule and amendments 4 and 5 fail (w 0.1 H_in misses calib ≤ 1.01
  at K2 by 0.0008); on-policy text is a real but calib-costly effect (own text −5…−15%, off-policy agentic traces ~0).
  No 27B self-cal arm; statistics and healing stay on calib. The 27B self-gen x2 Σdy² (256 seqs) is kept, unused.

### 2026-11-05 01:55 CST · #— · Two-pass prefill (A8) drafted: host tests pass, gfx1201 build on gpubox; device run queued
- Worker w22, fork branch `qc-exl3-v3-twopass` (6488c96, 70fb1fa; `plugin/exl3i/TWOPASS.md`): new row
  `exl3i_gemm_nt_a8` (act a8t / a8g128–1024, codebook mul1i8): activation quant (rotated, int8, scale per row × 128),
  decode pass into radiance's i8a8 fragment plane, radiance 1.17's i8a8 prefill tile (copied, attributed) with EXL3's
  output-Hadamard + svh store; M < 512 stays on today's path (byte-identical). The f16 accurate path keeps T2 (already
  two-pass). Host test 55 checks PASS (plane byte-exact vs our C reference and radiance's own fragment order at all 64
  rates; A8 g128 vs exact 0.63% rel rms). Built on gpubox with ROCm (host ctest 2/2); `test_twopass` + timing queued on
  card 0 at the next night-chain gap (`/data/tp/cardrun.sh`, ~15 min). Open: separate vs fused decode, RB 32 spill.

### 2026-11-05 01:20 CST · #33-29, #90, #90-02, #94, #91 · Overnight readouts (part 1)
- **#33-29 hest** (3 27B layers, K2.5, quantizer proxy, 3 references): E_out/none under dy / B / xw refs —
  Σdy² ~0.936–0.942, Sketch B 0.930–0.937, xw 0.928–0.934. No estimator ≥ 2% better under all three: **no new arm**;
  the block-normalized Σdy² stays (xw ≤ 1% better under two refs, within noise).
- **#90 (w21):** rule 1 PASS (343 steps monotone, ×0.79 a 1/8 step; K1 anchor within 0.6%). Rule 2 **FAIL** at every
  size: knap4 with sub-1 options −0.07% at 1.5, −0.02% at 1.75, 0 at 2.0–2.5 (−1.8% at 1.25, reported only). Sub-1
  pays only ≲ 1.3 bpw. **Bigger:** K1.125–1.875 (1/8) in knap4's V3 ladder lowers predicted KLD −12% at 1.5–1.75,
  −6.7% at 2.0, −1.3% at 2.25; and knap4's 1 MiB buckets cost ~2% at 1.5 (256 KiB fixes it). Patch
  `v3/e3/sub1/knap4-sub1.patch` (KN_SUB1, KN_BK); data `data/e3/sub1/`.
  **Decision (user): sub-1 kernel work stays in V3** (the low-bpw research ladder below ~1.3 bpw is where it pays).
- **#90-02:** i4 replaces i8 at K ≤ 1.5 (1.0029 at K1.5); K2 1.0107 (just over 1.01); K3 1.067. Both ladders K1.625–3.
  **Decision (user, clarified):** i4 only at K ≤ 1; **both i4 and i8 at K1.125–3** (1/8 steps; i4 built unconditionally);
  i8 only above K3.
- **#94 invalid:** the CPU proxy's sanity check fails (K7 ×1.18 vs the GPU smoke's 1.02, K8 ×3.0 vs 1.35): every line on
  the top scale factor (grid capped at 1.15). Rerun with the grid to 1.5 before any decision.
- **#91 phase 1 (0.8B), H_out from self-gen blended into calib, calib golden:** w 0.25 0.998 (K3) / 1.024 (K2), w 0.5
  1.024 / 1.017, w 1 1.085 / 1.122 — **no gain, worse at higher weight**. The g-agent scores were lost (its kld.json
  lacked the radiance format tag: evalq refused it; fixed, re-scored after the H_in step).
- **#91 H_in (first arm):** hiC1 (self-gen input Hessians alone) K2: calib **1.37** (z +8.5), calib's agentic text
  0.99 (n.s.). Remaining arms running.

### 2026-11-05 12:00 CST · #34-02, #99 · Slab-head whole-model check and 8.0 floor split armed; two-pass kernels started
- `v3/yaqa/headwm.sh` (rules `v3/yaqa/HEADWM.md`): head Fisher + permutation rebuilt now (CPU, gpubox), head quants
  K3–K7 and the arms after the night chain: heads at equal bytes (bf16-dequantized) swapped into V2 4.00 (u5 / s5-3 /
  u6 / s6-3, + the true bf16 head as the plugin-path check), V2 8.00 ± bf16 head, the bf16 model on TP2 if card 1 is free.
- **Two-pass prefill kernels:** worker w22 (fork worktree `radiance-twopass`, branch `qc-exl3-v3-twopass`): decode
  pass to int8 / f16 planes for radiance's tuned tiles, A8 activation side, host tests + gfx1201 compile; GPU tests
  and timing driver ready for a card.
- TP2 recorder works with the HF checkpoint as is (`--max-num-seqs 4`): ~740 positions/s, so the 7.5M-position
  batch 1 takes ~3 h (not 1.5); batches 2+3 (15M) ~5.6 h.

### 2026-11-05 11:50 CST · #95 · IQR-V4's act order and bf16 (row,128) scales: both negative on the CPU proxy
- **95-01 act order** (16-col sub-blocks by H mass inside each 128-block): median ratio 0.9998 / 0.9994 / 0.9987 /
  0.9981 / 0.9996 at K2 / 2.5 / 3 / 4 / 5 (rule ≤ 0.99 at ≥ 2 K): **fail**. The 128-Hadamard flattens the sub-block
  mass (CV 0.17–0.58 → 0.01–0.02); across 128-blocks it still differs 10–150× (a block-level reorder would need its
  own rule and a runtime input permutation).
- **95-02 bf16 (row,128) scales refit after the trellis:** at equal bits ×1.27–1.29 attn_output, ×1.18 ssm_out /
  ffn_down (rule ≤ 0.97): **fail by far**; the per-block refit even hurts (LDLQ pushes error into later blocks), the
  joint solve gains only −0.5…−1.5%. Pre-trellis group scales (IQR's likely route) untested; #40 found them dead at low K.
- Files `v3/e3/trellis3.py`, `v3/e3/sub1/r95{.sh,report.py}`, `data/e3/r95/r95.jsonl` (worker w19). Runtime cost: none.

### 2026-11-05 11:20 CST · #96, #97, #— · Row-slab K fails; embedding menu extended; 4070 deadlock again
- **#96 (w20, cpu2):** row-slab K ceiling with Σdy² row weights, parameter-weighted median at S = 128: **0.036 bit**
  (input-only 0.044; FFN 0.031) vs the 0.10 rule: **negative**, no stage 2. Exception: ssm_qkv 0.33 / attn_qg 0.27,
  spread between GDN heads (one 128-row slab = one head); per-head K on those ≈ 0.04 bpw overall → BACKLOG.
  Files `v3/e3/rowslab2.py`, `v3/e3/sub1/r96*.py`, `v3/e3/data/rowslab2*.jsonl`.
- **#97 (w20, cpu2):** t0.25 / 0.5 / 0.75 and t6.25 … t8: error ×1.38–1.41 lower per quarter step, no knee; **all 11
  levels join the menu**; Lloyd dominated everywhere (e8 4.5e-5 vs t8 2.0e-5). Files `v3/e3/embtrellis2.py`,
  `v3/e3/emb/{run7.sh, r97hull.py}`, `v3/e3/data/embmenu2.jsonl`.
- **4070 lost ~3 h 45 min again (7:30–11:15 PM CST):** the VM's `autosync.sh` relaunched `hconv.sh` under an outer
  gpu.lock when the 1,920 snapshot synced (the same deadlock as 4:34 PM). Killed autosync for good (its job is
  done) and removed the outer flock from it. Also d1108 phase 1 had died at once: d1103's sourced block resets `X`
  (sggen.py looked in d1103/); fixed, restarted 11:17 PM CST. Every 4070 ETA moves ~4 h later.

### 2026-11-05 11:00 CST · #98 · Revised: 256 self-gen sequences in bf16 hi/lo (x2), to blend with the 1,920
- User: a small collection to *combine* with the existing one, and a test of the cheap recipe. tp2night stage 3 is now
  gate (x2) + 256 sequences on arm D's text (rule H27.md "#98"); ~2–3 h after the batch-1 goldens; TP2 goldens b2/b3
  after it, finishing ~9 AM CST.

### 2026-11-05 10:45 CST · #33-28, #91 · No more Σdy² collections; tonight's 27B self-gen collection skipped
- **User:** at ~−3% KLD (≈ 0.03 bpw) H_out does not justify another ~40-card-hour collection. The existing 1,920-sequence
  diagonal stays (paid, free at runtime). Stage 3 of tp2night (512 self-gen sequences) is skipped (`hdiag-D/SKIPPED`):
  the TP2 goldens follow arm D directly, card 1 is free ~5 h earlier. #91's H_out arms still run on the 0.8B (cheap)
  but cannot lead to a 27B collection; its **H_in** arm (forward-only statistics) is the 27B-relevant part.

### 2026-11-05 10:40 CST · #33-28 · Swap test K2.5: block-normalized Σdy² −2.8% (provisional); diagonals published
- K2.5 trunk swapped into V2 3.00, all arms the same size (2.6244 bpw), golden-search27 (103 docs, 123k positions),
  paired bootstrap: **bn 0.05737 vs base 0.05903 = −2.8%** (z −2.2, 95% upper bound −0.4%); dy −1.7% (z −1.6, n.s.);
  bn vs dy −1.1% (z −1.4, upper bound +0.5% < the +1% margin). H27 rule: bn provisionally chosen; "real" still needs
  the seed arm (noise ≤ 1.4%). **≈ a third of the 0.8B's −8…−9%**: proxy effect sizes shrink at 27B.
- **Published** (user go): HF dataset `Hero422/qwen3.8-27b-hout-diag` (merged layout: 63 files, 0.67 GiB; one
  safetensors per snapshot × estimator + block-normalized; `v3/yaqa/hf/unpack.py` restores ourq's per-tensor layout,
  round trip bit-exact). A first attempt created an empty *model* repo of the same name (to delete, user).
- Night chain: the TP2 recorder failed at startup (GDN state for 256 sequences did not fit); fixed with
  `--max-num-seqs 4`; arm D generating at ~100 tok/s; everything ~1.5 h later. speedcal at 250 W is not a power
  test (mean draw ~165 W, the cap never binds); card 0 reports sclk max 2,350 MHz vs card 1's 3,139 (check).

### 2026-11-05 10:15 CST · #33-28 · Night chain extended to ~6 PM CST without needing TP2 after 9 AM
- **User:** both cards until morning, card 1 may be taken back any time after ~9 AM; run until noon–6 PM.
- TP2 goldens (batches 2 / 3, 1M-token chunks) run to completion (~9:30 AM CST; user: no hard cutoff). Then a queue of 4-layer
  quant chunks (`v3/yaqa/swap27k.sh`): card 0 always, card 1 while it is ours (`stop2.sh` kills card 1's chunk, card 0
  redoes it): the seed arm s1 (K2.5) → **base4 → bn4** (K4 on V2 4.00, H27.md amendment 3) → paired comparisons.
  Est. done ~5–6 PM CST with card 1 helping until midday, ~midnight on card 0 alone. V2 4.00 downloaded beforehand.

### 2026-11-05 09:45 CST · #92, #94–#97 · Overnight CPU / 4070 plan; paper review applied; pool trace split corrected
- **cpu1:** #94 (i8 codebook above K7) running; #95 (act order, bf16 (row,128) scales) by worker w19.
  **cpu2:** #96 (row-slab K with Σdy² row importance) and #97 (embedding menu ends) by worker w20 (user approved #96).
- **4070:** hest → #91 phase 1 → d1110 (+ spare goldens) → **d1109 REAL-Q** (w17's implementation; Fisher, identity
  gate, GD smoke; arm set only if the gate passes; paper lr, stack writers) → #91 phase 2.
- **Paper:** w16 applied all 51 review items (49 fixed, item 16/51 dropped per the user, 50 n/a); 9 pages, builds clean.
- **Long traces per pool (checked on the corpus):** verify = express, lua, requests, rich; search = cobra, ripgrep;
  qad = attrs, click. AGENTS.md had the pre-11-01 split (corrected); pools-v3.json's `traces` field is stale too (the
  per-document `pool` list is right and authoritative; the file stays read-only).

### 2026-11-05 07:50 CST · #33-28, #91 · TP2 tonight: goldens, arm D, 27B self-gen collection (`eval/night1105/tp2night.sh`)
- **User:** both cards tonight, card 1 back tomorrow. d1108 split into phases (amendment 3) so #91's H_out answer
  comes ~1–2 AM CST.
- **gpubox chain (armed 7:45 PM CST):** (0) the swap test's seed arm is moved off card 1 (bn / dy answers unchanged,
  ~10 PM CST); (1) bf16 27B goldens by stock radiance 1.17 TP2 `--kld-record` (HF checkpoint served as is, else
  `rad-convert`): xcorp 8k 5.0M tokens, own traces 32k 2.5M; (2) arm D: the bf16 27B serves its own continuations
  of sgprep's first 640 train prompts (`sgserve.py`, sggen's output format); (3) Σdy² over them, 512 sequences,
  as #33-24 (fp32, gen-masked, snapshots 128–512), **unconditional** (the cards are free; used only if #91 passes,
  publishable either way); (4) the seed arm on card 0. Card 1 free ~5 AM CST, card 0 ~9 AM.

### 2026-11-05 07:40 CST · #— · No TP2 assumed: single-card goldens and collections; first golden corpora built
- **Constraint (user):** TP2 on R9700s is unreliable (the second card is often in other use), so nothing on the V3
  critical path may need two cards. 27B bf16 reference goldens: staged layer passes on one card (boundary hidden
  states to disk, ~10 KB a position; est. ~3–4 h for 7.5M positions on one R9700): `eval/gold27/mkgold27.py`, being
  written by worker w18 (correctness test vs `proxy2b/mkgold.py` on the 0.8B at 1–3 stages). Backward collections
  (the 27B self-gen Σdy² if #91 passes; REAL-Q's Fisher if #92 passes) can use the same boundary checkpointing
  (~2x the two-card wall time); none is needed unless those pass.
- **First golden corpora (cpu1, `kldcorpus.py`):** xcorp 8k pieces 5.00M tokens / 3,389 pieces (seed 2701);
  own-trace 32k chunks 2.53M tokens / 142 chunks (private, our machines only).
- **4070:** d1106's K1.5 arms dropped (moot after the K2 fail and #46-03); queue hest → d1108 → d1110.

### 2026-11-05 07:20 CST · #— · 4070 queue deadlock (lost ~2 h 40 min)
- `hconv.sh` had been launched under an outer `flock gpu.lock`; its first step waits until no `quantize/ourq.py`
  process exists, while hest.sh's `flock gpu.lock python3 ourq.py` (matching that pattern) waited for the lock hconv
  held. Deadlocked ~4:34–7:15 PM CST 11-04 (UTC on the box: 22:34–01:15); d1106 / d1108 / d1110 and an orphaned w13
  volunteer GPU test (`quant_e2e.sh`, its server still up) queued behind. Fix: killed the outer hconv and the w13 test;
  hest now runs. **Rule:** never wrap a script that takes `gpu.lock` itself (or waits on GPU processes) in an outer
  flock; workers must stop their GPU tests and servers before reporting. hconv's remaining fill (bn1664 layer 39)
  reruns later without the outer lock. Every 4070 ETA moves ~3 h later.

### 2026-11-05 03:00 CST · #89 · 27B healing deferred to the ladder phase; volunteers for search as well as quants
- **Decision (user), clarified 03:30 CST:** 27B frozen-code scale healing is **deferred, not dropped**: no 27B test
  before the settings freeze; work on it once the ladder generation starts (beside it), at the latest after V3; it can
  still go into V3's releases if ready. 0.8B healing tests continue now. V3's estimates vs V2 were made before healing, so they stand; the knapsack
  allocates without healing (a healing-aware reallocation is a V4 question).
- **Volunteers:** besides the ladder's per-layer quants, the search's candidate evals (`eval_mix`, ~1,500–3,000 evals
  ≈ 50–100 card-hours a pass) can run on donors in batches: public slices only; mixes assembled on the worker from
  cached ladder levels (not built yet). Parked until research is done (user: "a good long while").

### 2026-11-05 02:40 CST · #— · Machine names neutralized in the public repo
- Tracked files (not `local/`): iceman → `gpubox`, colo / R640 → `cpu1`, Thor → `cpu2`, the extra RDNA4 boxes
  unnamed; ssh helpers referenced as `ssh-gpubox.sh` / `ssh-cpu1.sh` / `ssh-cpu2.sh` (local symlinks to the old ones);
  9 files renamed (`data/a1/a1-cpu{1,2}.jsonl`, `data/e3/cd-cpu*.jsonl`, `v3/e3/cpu{1,2}chain.sh`,
  `proxy2b/exp/w10-longmine/run-gpubox.sh`). Git history keeps the old names (user: fine). Older entries in this log
  were rewritten the same way. The RTX 4070 and R9700 are named as hardware models.
- Private traces: the repo holds only the tools that render / redact / sample them and aggregate statistics, no
  trace content. cpu1's public IP prefix stays in the redactor (it will be public with the volunteer queue; user).

### 2026-11-05 02:10 CST · #— · ROADMAP and design doc brought in sync; research points 1.5 / 1.75
- **ROADMAP rewritten** (user: out of sync with `docs/design/v3.md`, and "small-model proxies after V3" was wrong):
  sections by release (V1/V2 released, V3 in progress, V3 extras, V4 = QAD / KV / LSA / MoE, later), "Research
  method and infrastructure (in use now)". Removed contradictions: 7.0–8.0 rebuilds, A4, sub-1 and the half-step menu
  listed as extras; QAD named V5.
- **Design doc:** "Goal and scope" rewritten as the current plan (speed targets vs IQR-V4, sizes incl. 7.0, variants,
  W/A, ladder, quantizer stack, two-pass kernels, radiance 1.17, eval data, hardware, not-in-V3); body statements
  that conflict are marked **[superseded date]** with the current text (YAQA dropped → diagonal H_out candidate;
  knap3 → knap4; Phase A/B → §6's current order; half-step menu → ladder; trunk row-slab reason; i8c; engine 1.17;
  turboderp's SC trace answered; QAD V5 → V4).
- **Privacy pass before commit (user):** no machine names, people, private-trace wording or local paths in the
  public docs; hardware described generically (the user's own line in the ROADMAP).
- **Research points (user):** 1.5 / 1.75, then 0.1 bpw steps below 1.5 (1.4, 1.3, ...) until KLD blows up; 1.25 is no
  longer a separate point.

### 2026-11-06 12:40 AM CST · #92 · REAL-Q fails every rule, healed and parallel arms included
- d1109 run2 (0.8B, seeds 0 / 1, paired vs the matching stack arm): REAL-Q + healing vs stack + healing K3 1.096,
  K2 1.037; parallel (bf16-prefix) REAL-Q K3 1.129, K2 1.017; parallel + healing K3 1.095, K2 1.031. Base rule
  (≤ 0.95 at both K) and amendments 1–2: **fail**. Our LDLQ + H_out + swj + writers stack already covers what
  REAL-Q's block-output Fisher step adds; no 27B test. Closed.
- Watchdog's first catch (cycle 0): `pair()` in `d1103/run.sh` clobbered its callers' `$b`, so d1111 run_v's
  xcorp / agentic / verify pair lines were empty; fixed (`local`), `repair_v.sh` recomputes them after run_v.

### 2026-11-05 9:30 PM CST · #33-28, #101, #92 · 27B seed floor ≈ H_out's effect; ukisai fails; night queue
- **#33-28 seed arm (K2.5):** base → s1 (seed offset 1, no H_out) = **0.975** (z −2.5); s1 → bn 0.997. The 27B
  quant-noise floor (~2.5% per pair) is as large as bn's −2.8%: **bn is not real by H27's rule** (needs ≥ 2× the floor).
  The per-document paired interval does not include quantization noise; single-pair 27B swap tests can't resolve
  effects under ~5%. **K4:** bn4 / base4 = 0.961 (z −2.5): passes amendment 3's letter, decision held for a K4 seed
  arm (H27 amendment 4). H_out not adopted for now.
- **#101 ukisai arm (0.8B, H_in w0.25 of public 27B agent traces):** calib 1.023 / 1.029 (K3 / K2), ukisai golden
  0.816 / 0.824, calib agent traces 1.026 / 0.995 (noisy): **fail**, the same trade as sessions and self-gen.
  xcorp w0.5 stays the only passing blend; its 27B arm (2 seed pairs, d1111 amendment 2) runs tonight.
- **#92 REAL-Q:** base rule failed (K3 ×1.12, K2 0.974); amendment arms (heal, rqp) still running on the 4070.
- **Queue (gpubox card 0, `/data/q1106/q.sh`, started 9:07 PM CST):** mkhin27 xcorp H_in (1–2 h) → blend w0.5 →
  K2.5 arms xbn, bns1, xbns1 → K4 seed arm base4s1 → slab-head retry (its K5 head quant died at 6:02 PM CST).

### 2026-11-05 4:15 PM CST · #101 · H_in blends: xcorp w0.5 passes, sessions fail, xcorp w0.25 misses
- d1111 (0.8B, dygm stack, K3 / K2 × seeds 0 / 1, paired; ratios = mean over seeds vs calib-only H_in):

  | blend | calib K3 / K2 | fresh xcorp K3 / K2 | own agentic K3 / K2 |
  |---|---|---|---|
  | xcorp w0.25 | 0.987 / 0.996 | 0.986 / 0.962 | 1.016 / 0.985 |
  | **xcorp w0.5** | **0.981 / 0.984** | **0.967 / 0.936** | 1.016 / 1.035 |
  | sessions w0.25 | 1.022 / 1.019 | 1.010 / 1.001 | 0.931 / 0.916 |
- **Rule** (calib ≤ 1.005 and xcorp ≤ 0.98 at both K): **xcorp w0.5 passes** (better on calib too); w0.25 misses on
  xcorp K3 (0.986); sessions fail on calib (+2%), the same trade as self-generated text (#91): own text −7…−8%.
- Seed spread is large (single-seed ratios differ by up to 4–6% on xcorp / agentic), so n = 2 is thin.
- Next: the ukisai arm (amendment 1) is running; then a 27B arm for xcorp w0.5 (forward-only H_in over ~2–3M xcorp
  tokens, blended into the published calib Hessians; one K2.5 swap arm) — needs the user's go and a card window.

### 2026-11-05 01:40 CST · #33-28, #91, #46-03 · Swap-test restart; discussion record (self-calibration, variants, 7.0 / 8.0)
- **#33-28 swap test failed at its first quant (5:34 PM CST 11-04):** `ModuleNotFoundError: exllamav3`. Since the
  gpubox rebuild the patched exllamav3 (a3bfbfd0 + 0001–0007, editable install) had never finished building (the 1 AM
  CST `exl3build.sh` log stops mid-compile) and the kernel sources next to ourq.py (`exl3frac`, `exl3cb`,
  `exl3frac2`) were missing. Copied the sources from the repo, restarted `exl3build.sh` (5:52 PM CST);
  `/data/swap27/afterbuild.sh` reruns `swap27.sh run` (NSEQ=1920) when it reports done. Base score and `hout-bn`
  (mkbn: 400 tensors, 15 blocks) kept. Second failure after the build (6:09 PM CST): ourq's `write_index` template
  (`RXM_TEMPLATE`, turboderp's `SC_4.00bpw_H5/config.json`) was gone too; refetched from HF (identical to the ladder's
  config apart from the bit fields ourq writes). Restarted 6:17 PM CST; ~4 min a layer.
- **Base score check:** radiance's "all" mean (1.27, p95 10.5) includes unscored positions, where the golden's rows are
  not a real reference (unscored mean 4.37). Scored positions: **0.059** (V2 2.5: 0.0666). `cmp115.py` pairs on gen.u8
  only, so the arms compare correctly; quote scored-position means only.
- **Self-calibration plan, 27B side (user):** if #91 adopts self-generated statistics at weight w*, collect ~500
  self-generated sequences on the 27B (~4–5 h of both cards; the collection converges by ~384–512, #33-27) and blend
  with the existing 1,920-sequence calib collection at w* (`blend.py hout`, sums simply add). Arm D (27B continuations
  of the same boundaries, served at low priority) after the swap test; also gives the agentic eval golden. Extra
  knob: upweight calib's own 27B traces (6% of qad scored rows vs ~25% of search / verify). turboderp's cal_trace
  is ready-made 27B self-generated general text for an input-Hessian / H_out blend (~20 min forward on gpubox).
- **Subagents:** several `pi-capped` agents may run at once (about three, by free RAM); queue only sequential work
  (user; AGENTS.md). w13b, w14 and w15 (paper accuracy check, read-only, `local/paper/review-w15.md`) run in
  parallel. The paper was reviewed by the user ("generally looks good").
- **W/A spectrum** written up as one table (`docs/design/v3.md` §1). Decode stays on the f16 path because it is
  memory-bound; the M ≥ 512 threshold is unmeasured (M = 16–511 sweep with the kernel timing). Decode kernels later.
- **Variants:** accurate + fast everywhere; a third (A4) at ≤ 3 bpw only if ≥ 15% lower `gemm_us` than the fast one.
- **7.0** adopted (user): expected accurate ~0.00040–0.00043, fast ~0.00045–0.00050 at ~2x V2's prefill speed; not
  ~0.0003, because V2's 8–9 bpw (0.00034 / 0.00028) sit near a non-weight floor (8-bit head, bf16 output stores,
  attention numerics). **8.0:** decided after the 7.0 and a floor split on V2 8.0 (bf16 head, f32 output stores,
  bf16 (row,128) scales; ~3–4 evals); built only if predicted ≥ 10% below 0.00034; small speed gain (K > 7 stays mul1).

### 2026-11-05 01:00 CST · #46-02, #46-03 · A4 returns as a per-tensor option on i4 weights; 7.0 proposed as a V3 size
- **#46-02 closed as fail** for A4 as a global setting (all MLP tensors, K2 ×1.14 / ×1.16, ≈ 0.07 bpw for an est.
  +25–30% prefill).
- **Decision (user):** W4A4 (i4-grid weights, K ≤ 3 × A4-g64) is a per-tensor knapsack mode for the fast variant at
  ≤ 3 bpw (#46-03, rule `v3/e3/a4/README.md`: 27B per-family costs, time table, knap4; the int4-plane kernel only if
  ≥ 10% lower predicted `gemm_us` at equal KLD, or ≥ 0.03 bpw smaller). Decode-side work (GEMV, trellis decode of the
  weights) stays for later (BACKLOG "V3 decode kernels after radiance 1.17").
- **7.0 bpw:** IQR-V4's IQ6R-L-V4 is 6.939 bpw / 0.000677 KL / 4,006 µs vs V2 7.00 (6.995 bpw) 0.00045 / 8,105 µs; the
  ladder already holds every level a 7.0 needs (i8 to K7, mul1 K6–K8), so a V3 7.0 costs ~3 h a variant (knapsack,
  pack, eval). Proposed to the user as a V3 size after the focus sizes and 6.0 (TP2 R9700 users). **Adopted (user, 01:20 CST).**
- **Runtime cost:** A4 is a speed option; 7.0 fast variant targets the 2× prefill gap.

### 2026-11-04 18:45 CST · #91, #33-28, #46-02 · Self-calibration test started (0.8B); swap test on 1,920 sequences; calib is partly SC already
- **#91 (d1108) started** (rule `proxy2b/exp/d1108/README.md`, written before data): do the model's own continuations of
  the user's agent sessions make better Σdy² H_out, input Hessians or healing data than calib qad or the raw sessions,
  and at what blend weight with calib. Arms: C (0.8B's own turns, 2,304 prompts of 3,072 tokens), B (raw sessions,
  scored on the other model's turns), T (turboderp's SC calibration trace, 27B-generated general chat / reasoning,
  217 rows), D (27B continuations) later on gpubox. Agentic eval golden from 256 held-out-session prompts. Tools
  `sgprep.py`, `sggen.py` (batched, no padding, 64 tok/s at batch 4 smoke), `sgbuild.py`, `blend.py`; heal.py
  `--data meta`. Prep: 3,578 train / 888 eval boundaries available (4:1 session split).
- **#33-28:** collection stops at the 1,920 snapshot (H27.md amendment 2, before data) to free cards earlier; the swap
  test starts on it automatically (`/data/swap27/stop1920.sh`).
- **Calib already contains 27B self-generated text** (dataset card: the 8 agent traces were "generated for this challenge
  by Qwen3.8-27B itself (the official FP8 checkpoint, its default sampling settings)"). Share: 19.3% of calib tokens;
  qad pool 18% of 2,048-token windows but 6% of scored rows; search 25%, verify 24% of scored rows. The private set is
  drawn the same way, so ~1/4 of what is scored is on-policy 27B text.
- **Earlier evidence that self-generated statistics help:** in the V2 fit, turboderp's plain tensors cost **1.23x his SC
  tensors' real KLD per unit proxy error** (same quantizer, same tensors; SC = LDLQ Hessians, measurement and recipe
  on a self-sampled trace, exllamav3 `doc/optimize.md`). Our tensors (calib Hessians) cost 1.25x his at K4–K6 and 1.58x
  at K7–K8 per proxy unit, though part of that is our proxy being in-sample for ours (#33, ~9%). Suggestive, not a test.
- **turboderp's main branch saved** to `/workspace/references/turboderp-qwen38-exl3-main/` (cal_trace with token ids,
  eval trace, kld_table, measurement, noise_attrib, recipes).
- **#46-02 (d1106, A4 on the MLP at low K) interim:** K2 A4 on MLP 1.139 unhealed / 1.161 healed (2 seeds, z > 24) vs the ≤ 1.10 rule:
  **fails at K2** whatever K1.5 shows; A8 reference arm failed to run (to diagnose). Full entry when the run ends.
- **Runtime cost:** none (statistics / data choices).

### 2026-11-04 17:50 CST · #— · Evaluation data for V3: xcorp + own traces for search, calib for finals
- **Decision (user):** `docs/design/eval-data.md`. Search accept/reject draws fresh slices from xcorp (8k chunks) and
  the user's traces (32k chunks, bootstrap by session; our machines only, never donors); calib fit / qad / verify keep
  their roles; final picks and published numbers on calib verify; long-context guards from calib's long traces, 2–3
  whole own sessions before release. AGENTS.md and `docs/CORPORA.md` updated.
- **Goldens:** stock radiance 1.17 `--kld-record --kld-corpus --kld-top 512` (no custom recorder needed; gen masks
  added by our tools). `xcorp/kldcorpus.py` builds the corpora (tested on cpu1: 200k tokens, 141 pieces, 0 round-trip
  failures; xcorp's tokenized meta contains no private traces). First batch ~5M xcorp + 2–3M traces on the first free
  cards after the swap test.
- **Publishing (pending the user's go per repo):** xcorp + goldens (`Hero422/xcorp-v1`, the challenge's licensing
  model) and the 27B H_out diagonals (`Hero422/qwen3.8-27b-hout-diag`, after the swap test).

### 2026-11-04 16:45 CST · #33-28 · 27B swap test prepared (K2.5, plain vs block-normalized H_out)
- **Pass rule:** `v3/yaqa/H27.md` amendment, written before data (K2.5: the K3 level was lost; arms base / bn / dy / s1).
- **Setup:** golden `golden-search27` (103 search-pool docs, 123,460 scored positions, 8k cap, seed 27) cut from the
  published calib golden; radiance 1.16.0 + our plugins; each arm's trunk imported as exl3i (mul1i8, rate 2.5) and
  mixed into V2 3.00; block-normalized diagonal from `v3/yaqa/mkbn.py` (checked on the 1,664 snapshot: totals equal,
  layer-0 gate max/mean 490 vs plain 3,739). Driver `v3/yaqa/swap27.sh`: card 0 base + bn, card 1 (user: tonight)
  dy then s1. First end-to-end use of the exl3i import path on a full model.
- **Runtime:** none.

### 2026-11-04 17:05 CST · #33-25, #33-27, #89-06 · Block-normalized Σdy² wins; healing at K4 with lr 3e-4
- **#33-27:** hconv (layers 10/39/50, K2.5, 19 tensors) with block-normalized training means (every 128-seq block the
  same per-tensor total): train 128…768 seqs 0.932 / 0.925 / 0.921 / 0.924 / 0.923 / 0.921 vs plain 0.932 / 0.925 /
  0.922 / 0.945 / 0.940 / 0.936 (plain reference); against the block-normalized reference the same ordering
  (bn1408 0.927 vs dy1408 0.942). The robust estimator removes the outlier blocks' damage and is ~1.6% better at
  the full sample; its curve plateaus from ~384 training seqs. **Decision:** the 27B swap test uses the
  block-normalized diagonal (plain as a second arm).
- **#89-06 (k45lr):** K4 seed 0 at lr 3e-4: held-out −5.4%, search golden 0.949 vs its stack (z −14.4) — at the
  ≤ 0.95 rule; seed 1 and K5 pending.
- **Files:** `data/hconv/result-2026-11-04.txt`

### 2026-11-04 15:10 CST · #90 · Sub-1 K moves into V3; error-curve sweep started
- **Decision (user):** sub-1 K (K0.125–0.875 at 1/8 steps) is in V3 (was V3.5): quantizer + decoders in the kernel
  pass, offered by knap4 at every size. Research sizes 1.25 / 1.5 / 1.75 added to the V3 ladder.
- **Run:** `v3/e3/sub1/run.sh` (rule in its README, written before data): 49 tensors × K 0.125…0.875 + 1 / 1.25 /
  1.5 / 2 anchors, i8c, 128 rows, scale factors 0.5–1.15. cpu1 part a (GDN layers 2/10/30/50/62), cpu2 part b
  (attention layers 3/19/43/63). Question the user raised: does knap4 pick K0.875-ish levels for some tensors even
  at 2 bpw?
- **Runtime:** P+D (one salted codebook read a 0-bit step; measured with the kernel pass).

### 2026-11-04 14:50 CST · #— · Decision: V3 release sizes
- **User:** V3 ladder = 2.0, 2.5–4.5 (the focus of all quality work), 5.0 and one 6.0 to show the top; nothing above 6
  (V2 covers it). Rationale: above ~5 bpw KLD differences are below fp8-KV and serving noise; real-use thresholds
  ~0.01–0.04 KLD fall at ~2.6–3.6 bpw; VRAM tiers 12 / 16 / 24 / 32 GB. Added (user): 1.25 / 1.5 / 1.75 as
  research points (1.25 needs K1 / K1.25 menu options).

### 2026-11-04 14:35 CST · #81-03 · A1 on all 64 layers: done
- **Result** (CPU trellis proxy, ffn_down, 256 sampled rows, i8c; coded error vs D): ug/D mean over 64 layers K1.5
  0.849, K2 0.838, K2.5 0.833, K3 0.830, K4 0.828; up-only u/D 0.879 / 0.870 / 0.865 / 0.863 / 0.862. Up-only keeps
  81% of ug's reduction at K2.5 (rule ≥ 75%): **up-only preferred** (one extra GEMM). By layer (ug, K2.5): 0.94–0.96
  in layers 0–4, ~0.88 in 6–16, **0.68–0.75 in 23–36**, ~0.80 in 37–50, 0.91–0.98 in 53–62.
- **Size of the prize:** 0.833 ≈ 0.13 bit on down at equal error (≈ 2^(−2ΔK)); down is ~23% of trunk parameters, so
  ≈ 0.03 bpw whole-model if applied everywhere; as a per-layer knapsack option on layers 23–36 ≈ 0.25 bit on those
  down tensors. Runtime P+D (+1/3 MLP work, up-only). Proxy error only: KL unmeasured.
- **Decision / next:** modest; a per-layer option for the middle layers if the V3 kernels make the extra GEMM cheap.
  Price it with the two-pass kernel timings; no GPU test before then.
- **Files:** `data/a1/a1-cpu1.jsonl`, `data/a1/a1-cpu2.jsonl`

### 2026-11-04 13:55 CST · #89-04, #89-05, #89-06, #33-25, #33-27 · d1105 results; heavy tail in the 27B Σdy²
- **#89-04 PASS:** one pass over 2,048 fresh qad windows vs heal0 (2 × 1,024), equal compute: K3 0.9954 / 0.9950,
  K2 0.9970 / 0.9947 (every seed < 1). Adopt single pass; not data-limited (rule 0.98 not met).
- **#89-05 inconclusive:** xcorp 80% + own traces 20% vs qad single pass, on the search golden: K3 1.0126 / 1.0156,
  K2 1.0110 / 1.0163 (mean 1.014 both): gray zone of the rule. Expected direction (qad matches the calib-drawn
  golden); whether the mix helps real use is not measured by a calib golden.
- **#89-06 inconclusive:** K4 / K5 stacks built (vs base K4 0.775 / 0.782, K5 0.803 / 0.809), but healing at lr 1e-3
  never improved held-out (early stop at step 128, identity): the lr is too large for errors this small. Amendment
  in the README; k45lr screen running.
- **#33-25 hconv** (3 layers, 19 tensors, K2.5, E_out vs no H_out, plain reference): train 128 0.932, 256 0.925,
  384 0.922, 512 0.945, 640 0.940, 768 0.936: non-monotonic. (Corrected 11-04 17:20: training sequences = 128 × the
  number of even blocks; an earlier draft of this entry halved them.)
- **#33-27 heavy tail:** per-128-sequence block energy of the Σdy² diagonal vs the median block: layers 0 and 10
  carry 4–8× in block 6 (seqs 768–896) and 5.5–27× in block 9 (1152–1280), concentrated in few rows (layer-0 gate
  max/mean 6,366 vs ~100); layers 39 / 50 / 63 flat (0.8–1.5×). A few sequences dominate the early-layer mean,
  as IQR-V4 reports for per-token gradient energy. Block 6 is a training block (the jump at 448 train seqs), block 9 in
  the reference. Next: block-normalized arms (each block the same per-tensor total) scored against plain and
  normalized references (hconv.sh, running); identify the outlier sequences (collector does not log per-sequence
  energy yet). The fixed 0.5% stop rule is not meaningful under this tail (user).
- **Runtime:** none.
- **Files:** `data/d1105/result-2026-11-04.txt`, `data/hconv/result-2026-11-04.txt`

### 2026-11-04 03:55 CST · #89-04…06 · d1105 queued (healing recipe for the 27B, 0.8B, 4070)
- **Pass rules:** `proxy2b/exp/d1105/README.md` (written before data). Arms at equal compute (512 steps): one pass on
  fresh qad windows; one pass on xcorp 80% + own traces (pisess-main) 20%, assistant turns scored
  (`heal.py --data mix`, mask checked by decoding: `<think>` … `<|im_end|>`); healing at K4 / K5.
- **Why:** d1104 held-out curves were still falling at the end of epoch 1; the user allows xcorp and own traces for
  healing (real-world use is the goal). The GPU is shared with hconv through `/data/yb/gpu.lock`.
- **Runtime:** none.

### 2026-11-04 03:15 CST · #89-03 · K2 seed 0 rerun: PASS
- **Result:** k2edyswjasym_r0_heal vs its stack 0.721 (z −16.4), vs base 0.507; with seed 1 (0.708) both ≤ 0.90:
  healing stacks on edy → swj → asym at K2 and K3. Held-out curves: still falling ~0.6% / 32 steps at the end of
  epoch 1, epoch 2 adds < 1% (LR already halved by the cosine); next arms test one pass on fresh data (xcorp + own
  traces allowed for healing, user 2026-11-04).
- **Files:** `data/d1104/result-2026-11-04.txt` (seed 0 lines in the 4070 log `/data/yb/q1104b.log`)

### 2026-11-04 03:00 CST · #85-03, #87-08, #89-03, #33-26 · d1104 results (0.8B, 4070); chunk floor
- **#85-03 PASS:** swj alone on edy, 3 seeds: K3 0.969 / 0.972 / 0.970, K2 0.910 / 0.914 / 0.912 (every z < −17).
- **#87-08 PASS:** edy → swj → moments on refitted siblings → writers asym ρ 0 beats edy → writers ρ 0: K3 0.993 /
  0.980 / 0.981 (mean 0.985), K2 0.969 / 0.977 / 0.967 (0.971). The whole stack vs edy: K3 0.836, K2 0.758. Ship
  order H_out → swj → writers confirmed.
- **#89-03:** healing on that stack, K3 seeds 0/1: 0.886 / 0.891 vs the stack (PASS ≤ 0.95), 0.674 / 0.668 vs base.
  K2 s1 0.708 vs stack (0.500 vs base); s0 crashed on the full disk, rerun queued (q1104b.sh), rule needs both seeds.
- **#33-26 chunk floor (pass rule: none, a measurement):** bf16 0.8B vs its own whole-document golden, search slice
  166k positions: chunk 2048 identical to whole; 512 → 0.00071 mean KLD (KL99.9 0.0075, top-1 98.79%); 64 → 0.00069.
  Chunked prefill alone moves KLD ~0.0007, ~35× below the 0.025 reported on Discord: that floor is mostly elsewhere
  (QSA / sparse attention, batching).
- **Runtime:** none (quantizer-side); healing none (scales only).
- **Files:** `data/d1104/result-2026-11-04.txt`, `data/d1104/chunkfloor.jsonl`

### 2026-11-04 02:30 CST · #— · Decisions: fast + accurate variant per size; two-pass prefill kernels, built now
- **User:** publish a fast and an accurate variant at each size (knap4 with per-tensor compute modes, a speed budget for
  the fast one). Build the kernels now: the two-pass design (decode once into an int8 / f16 plane + radiance's tuned
  tile) does not depend on the quantizer experiments still running.
- **Order:** (1) radiance 1.16.0 rebase of our plugins (the kernels call radiance's tile; API per version); (2) decode-
  to-int8 and decode-to-f16 passes for EXL3 tiles + routing to radiance's i8a8 / 16-bit tiles; (3) timing at the
  board's shapes (needs an gpubox card: after #33-24, or a pause); (4) knap4's speed table from those timings.

### 2026-11-04 02:00 CST · #— · IQR-V4 published (main competitor)
- **Notes:** `docs/research/iqr-v4.md`. Board: V4 leads our V2 by 7–13% at 2.5–5 bpw, level at 5.5, behind at 6+
  (35–50% at 7–8); prefill 1.02–1.08× FP8 vs our V3 target 1.4×.
- **Matters for us:** their YAQA-style / Fisher reweightings and post-GPTQ scale refits all failed, against our 0.8B
  H_out and #85 wins: the 27B swap test is the deciding check. Candidates: act order in the trellis, bf16 (row, 128)
  scales on attn_output / ssm_out at ≥ K5, their vocabulary-weighted head (= our slab head's direction). BACKLOG rows.

### 2026-11-04 01:20 CST · #— · Tail metrics (KL99, KL99.9) and top-1 on every paired comparison; the wins rechecked
- **Rule (user):** `proxy2b/pcmp.py` now reports KL99 / KL99.9 ratios (document bootstrap 95% CI) and top-1
  agreement (diff ± SE) beside mean KLD; mean decides, tails are a guard (`README.md` Rules).
- **Retroactive check** (0.8B, search golden, seed 0, from the stored per-position rows; no new evals), mean / KL99 /
  KL99.9 / top-1: H_out K3 0.911 / 0.959 / 0.924 / +0.7 pt; #87 on edy K3 0.876 / 0.892 / 0.966 / +0.8 pt, K2 0.850 /
  0.904 / 0.969 / +2.0 pt; healing on edy + asym K3 0.873 / 0.864 / 0.910 / +0.9 pt, K2 0.665 / 0.705 / 0.891 /
  +5.1 pt; swj K2 0.914 / 0.941 / 0.977. **No tail regression anywhere** (no interval above 1); tails improve less than
  the mean for most methods (the gain sits in the bulk), healing improves them about as much.
- **Files:** `data/tails/wins-2026-11-04.txt`, `data/tails/tails.sh`

### 2026-11-03 23:40 CST · #33-24 · 27B diagonal H_out collection started on the rebuilt gpubox
- **Status:** running · **Machine:** gpubox (rebuilt 11-03: OS drive lost in a linear LVM; new 4 TB data drive),
  both R9700s (`ROCR_VISIBLE_DEVICES=0,1`: the container also exposes the iGPU)
- **Setup:** `hdiag27.sh 2048` via `local/bin/gpubox-day27.sh`: fp32, gen-masked loss (`YAQA_GENMASK=1`, #83-02), long
  qad windows (pools-v3: 1,350 windows of 2,048), Σdy² + Sketch B + xw, snapshots every 128 sequences, resumable.
  Fresh venv: torch 2.15.0+rocm7.14, transformers 5.22.1, accelerate 1.15.0, fla 0.6.0 (no causal_conv1d: reference
  conv, correct). Model Qwen/Qwen3.8-27B @ 1d4bf0f2ff60; challenge data @ 2dcf0061.
- **Gate fixes:** (1) card 0 OOM in the forward: the diag hooks kept every linear's fp32 input from the forward to the
  backward (non-reentrant checkpointing runs the first pass with grad on; ~0.27 GB per 27B layer) → `YAQA_XOFF` parks
  them in pinned host memory; (2) the fp32 copy (18 GB pinned) failed to allocate on the 30 GB host → `YAQA_XOFF=bf16`:
  Σdy² and xw bit-identical, Sketch B within 9e-4 rel L2 (0.8B, `proxy2b/xofftest.sh`; 73cafe7).
- **Gate:** chunked vs whole fp32 G exact; seq 36.6 s (fwd 10.2, bwd 26.4); peak 28.4 / 28.4 GiB; 400 linears.
  → 2,048 sequences ≈ 21 h (512 ≈ 4:40 AM CST, 1,024 ≈ 10 AM CST, 2,048 ≈ 9 PM CST 11-04).
- **Pass rule for the swap test:** `v3/yaqa/H27.md`. Runtime cost: none.

### 2026-11-03 23:20 CST · #89-02 #87-07 #87-06 · Healing on the stack, sibling transfer, 27B collector GPU test
- **Status:** done · **Machine:** 4070 (`proxy2b/q1104.sh`) · **Files:** `data/q1104/`
- **#89-02 healing on edy + asym, re-scored with the trained norms** (w11 healfix): vs edy + asym K3 0.873 / 0.858,
  K2 0.665 / 0.662 (seeds 0 / 1, z −16…−22). The norm gains themselves are worth ≤ 0.2% (healn vs heal 0.998–1.000):
  the gain is the folded per-row / per-column scales. **Done: healing stacks on H_out + asym** (needs the user for 27B).
- **#87-07 sibling transfer** (w11 xseed: seed-0 asym writers on seed-1 siblings vs seed-0 plain writers there):
  K3 1.049, K2 1.026 (worse than no correction) while the matched pair gives 0.848 / 0.815. **The correction does not
  transfer**: on the 27B the writer requant runs last, on the exact shipped siblings (any later change to a sibling,
  incl. healing / scale refits, needs the writers redone or the order fixed: siblings → writers → healing of writers'
  scales only).
- **#87-06 sibmom27 GPU equivalence:** moments bit-identical to moments.py (H, C, N max rel 0); lean mkdenoise 96 / 96
  files bit-identical; end to end with level-decoded siblings: KL / w07 k3asym 0.992 (z −1.7), / k3base 0.852. **PASS.**
- **Runtime:** none.

### 2026-11-03 22:25 CST · #87 #89 #85 · w11 code review; w12 builds #87 for the 27B
- **w11** (`proxy2b/exp/w11-review/`): no leakage (goldens, moment windows, heal early-stop set and fit slices are
  disjoint, checked by document); mkdenoise's G / F recomputed independently (3e-8); `--wpre` in the right basis;
  every saved weight lands on the pack's i8 grid (eval = pack); wqall assembly and seed pairing correct; healing's
  parameters all store in a real pack (suh / svh / checkpoint norms). Findings: **A1** d1103's heal arms scored
  without their norm gains (fixed: `d1103-heal-norms.patch` merged; `healfix.sh` re-scores); **B1** #87's correction
  is specific to the exact sibling rounding (F − I cosine 0.02–0.10 across sibling seeds): on the 27B the writer
  requant must run **last**, on the final shipped siblings (`xseed.sh` tests transfer); **B2** run2's `full` order is
  wrong (above).
- **w12** (`v3/sib27/`): layer-streamed 27B sibling-moment collector (bit-exact vs moments.py on the 0.8B, CPU),
  pure-torch decoder of ladder shards (no `--save-wq` needed), lean mkdenoise for 17408² groups (merged; bit-
  identical on the 0.8B), driver `sib27.sh` (128 writers; ~5–7 card-hours per level at 1M tokens; ~450 GB disk).
  GPU equivalence test `test_gpu.sh` queued on the 4070. To set before the 27B run: ρ 0 (#87-05).

### 2026-11-03 22:20 CST · #87-04 #87-05 #88-02 #78-04 #33-23 #85-02 #89-02 · d1103 results
- **Status:** done · **Machine:** 4070 · **Files:** `data/d1103/result.txt`
- **#87-05 ρ 0 (full clean-output target):** vs ρ 0.5, plain K3 0.945 / K2 0.916, on edy K3 0.964 / K2 0.924, every
  seed (3). **Pass: ρ 0 replaces ρ 0.5.** ρ −0.25 (fit screen) is worse than ρ 0 (K2 1.019): ρ 0 is the optimum.
  Net vs edy alone: about 0.845 (K3) and 0.78 (K2).
- **#87-02 seeds 3–5:** edy + asym vs edy over 6 seeds K3 0.877 (mean log −0.132 ± 0.003), K2 0.848, every seed.
- **#87-04 K4 / K5:** asym vs base K4 0.868, K5 0.888 (2 seeds, every seed): **holds at high K.**
- **#88-02 asym on all 150 tensors** (moments through the whole quantized model): K3 1.53×, K2 2.31× worse.
  **Negative**: the target only works on the writers with sibling inputs.
- **#78-04 i8c on top of H_out**, 6 seeds: K3 mean log −0.007 ± 0.007, K2 −0.002 ± 0.008. **Null**: i8c adds
  nothing to H_out quants (its −1.7% on plain quants stands); the trunk check needs no i8c arm if H_out ships.
- **#33-23 Σdy² 512 vs 1,024:** K3 512 is +2.2% worse (≈ 2.2 SE), K2 −0.7%: 512 is not enough; supports the 27B going
  to ≥ 1,024 (the user's 2,048 stands).
- **#85-02 scale combos:** C9 α\* applied on top of C10 swj is broken (k3base_swj_astar 0.122 vs base 0.099;
  k3edy_swj_astar 0.111 vs 0.090), and the pairs of that step failed (a tag bug). The `full` stack (edy + asym + swj +
  α\*) is a mismatched order (w11 B2: the writers' F is stale once siblings are refitted; refitq measures α\*'s
  error against W, not W·F): K3 1.29× worse, K2 0.90. **Inconclusive**; redo as swj alone on edy, and any stack in
  the order "sibling refits first, then moments and writers".
- **#89-02 healing on edy + asym:** held-out KL −13% (K3) / −32% (K2) in training, but the search scores were taken
  without the trained norm gains (w11 A1: `ev()` used the stock checkpoint); re-scored by w11's healfix.sh (queued).
- **Runtime:** none for all.

### 2026-11-03 10:15 CST · #87-02 #87-03 #83-02 #87-05 · Sibling asym stacks on H_out; ρ 0 leads the screen; gen-mask holds for Σdy²
- **Status:** #87-02 done, #87-03 done (screen), #83-02 done (K2 seed 2 pending), #87-05 planned · **Machine:** 4070 (d1103)
- **#87-02** (edy + asym vs edy, search, seeds 0–2): K3 0.876 / 0.880 / 0.886 (mean log −0.127 ± 0.003), K2 0.850 /
  0.840 / 0.843; vs asym alone K3 0.937, K2 0.961, every seed. **Pass**: the two stack (H_out adds ~6% / 4% on top).
- **#87-03** (ρ screen, fit, seed 0, vs ρ 0.5): ρ 0 K3 0.947 / K2 0.911; ρ 0.25 0.954 / 0.949; ρ 0.75 1.062 / 1.073.
  Monotone: the full clean-output target is best. → #87-05 (run3.sh): ρ 0 on search × 3 seeds, plain and on edy;
  ρ −0.25 screen. Rule in `proxy2b/exp/d1103/README.md` "run3".
- **#83-02** (gen-masked Σdy² vs edy): K3 0.987 / 0.970 / 0.981 (mean 0.979, every seed), K2 0.964 / 0.955.
  **Pass** (K3 ≤ 0.985 every seed; K2 ≤ 1.00) → the 27B collection runs with `YAQA_GENMASK=1` (night27.sh, db8a3f1).
- **27B prep:** night27.sh now gen-masked, stops at 1,024; swap-test rule `v3/yaqa/H27.md` (written before data).
  Workers started 15:58 UTC: w11 reviews the #87 / #89 / #85 code for leakage and bugs; w12 builds the 27B #87
  pipeline (`v3/sib27/`).
- **Runtime:** none (all quantizer-side).

### 2026-11-03 09:35 CST · #— · Rule: every experiment states its runtime cost (prefill / decode)
- **Rule (user):** INDEX has a `runtime` column (legend in its header), LOG entries a **Runtime** line (template in
  `README.md`), each experiment README a "Runtime cost" section; `/workspace/AGENTS.md` carries the rule. The user
  cares about decode even though the board scores only prefill `gemm_us`, and may run decode tests themselves.
- **Current experiments:** quantizer-side, no runtime cost: H_out (#33 line, #82–#84), #85 scale refits, #86–#88,
  #87 asym, #89 healing (if folded into svh / suh / norms: verify in a real pack), CD (#77), the knapsack work.
  With a cost: **A1 (#81, P+D:** +1 transposed GEMM over up per MLP, ~+1/3 MLP work in prefill and ~+1/3 MLP bytes in
  decode), **i8c (#78, P+D?:** 2 FMAs per weight in the codebook decode), **slab head (#34, D:** head GEMMs + gather on
  every decoded token; prefill only at the last position), fractional K / qtf2 and A8 (priced in knap4's speed table).
- **Caveat:** 0.8B KL runs use the f16 path; at 5–6 bpw A8's ~+0.00016 constant makes 27B relative gains smaller.

### 2026-11-03 09:10 CST · #81-02 #81-03 · A1 control passes; whole-model A1 run under a memory-aware scheduler
- **Status:** #81-02 done (partial, 7 layers so far), #81-03 running · **Machine:** cpu2 + cpu1 (`v3/e3/a1/a1sched.py`)
- **#81-02 result:** permuted predictors give D's error exactly (ugperm / uperm ÷ D 0.999–1.001 on L2 / L10 / L30 / L50,
  K2.5 and K3) while the real predictors give 0.72–0.94. **The A1 reduction is real prediction, not an artifact of
  coding a residual.** (The 4.5% energy share vs 17% error gap stays unexplained; noted for the GPU step.)
- **Scheduler:** a1ctl's fixed-width stages left most CPUs idle (down jobs ~15 GB each). a1sched admits jobs by
  cgroup anon memory and CPU count; the down arms were trimmed (no normalized-H copy). Queue: remaining control arms;
  the other 48 layers at K2.5, K2, K3 (D / ug / u); then all 64 layers at K4 and K1.5 (K-dependence). Memory-bound:
  ~12–20 CPUs busy on cpu2, ~20–34 on cpu1; est. 12–16 h in all. Stop: `touch /data/work/a1/STOP`.

### 2026-11-03 08:57 CST · — · Saturation-aware allocation check (ModelOpt / arXiv 2607.12266) on the V2 knapsack data
- **Status:** done · **Machine:** cpu2 (`search/kfit2.py` `KFIT_SAT=1`: KLD = C (1 − exp(−P/C)), C fitted)
- **Pass rule (written with the BACKLOG row):** saturation matters if it fits measured mix KLD better than the
  additive sum, esp. at 1.75–2.5 bpw.
- **Data:** the V2 fit measurements whose mixes are still on disk (671 of 1,297; the r9 / other mix files live on
  gpubox only), same set for every variant; CV = 10-fold.
- **Result** (whole mixes, CV log(pred / measured) mean / rms):
  | model | < 1.6 bpw (13) | 1.6–2.6 (80) | 2.6–3.6 (66) | 3.6–5.1 (82) | pair ranking |
  |---|---|---|---|---|---|
  | additive (κ 0) | −0.28 / 0.32 | −0.025 / 0.136 | +0.060 / 0.092 | −0.003 / 0.094 | 607 / 778 |
  | additive + saturation | −0.28 / 0.31 | −0.025 / 0.131 | +0.059 / 0.090 | −0.002 / 0.089 | 612 / 778 (C → ∞) |
  | kfit2 (κ 3 knots, amplification) | +0.07 / 0.20 | +0.003 / 0.098 | +0.026 / 0.079 | −0.005 / 0.079 | 627 / 778 |
  | kfit2 + saturation | +0.07 / 0.20 | +0.011 / 0.100 | +0.021 / 0.079 | −0.008 / 0.080 | 628 / 778 (C 1.8) |
- **Reading:** **null, and the opposite sign.** On this dense 27B our errors compound (the additive sum *under*-predicts
  the low sizes by ~25%; kfit2's downstream amplification, κ ≈ 38 at layer 0, fixes that); a fitted saturation
  ceiling goes to infinity (additive) or to C ≈ 1.8, far above every operating point (no effect). Their gain was on
  MoE models at their whole-model damage ceiling. Not for V2 or V3; knap4 keeps kfit2's model.
- **Files:** `data/knap/sat/*.txt`

### 2026-11-03 08:52 CST · — · Literature: NVIDIA Model Optimizer (user pointer)
- **Status:** done (reading) · **Notes:** `docs/research/modelopt.md`
- **Reading:** their AutoQuantize sensitivity is our Σdy² estimator (used for allocation only); saturation-aware
  allocation (coverage model, Aumann–Shapley) is new to us and targets our low sizes; Local-Hessian / scale learning
  match our scale results (#85, #89); their layerwise PTQ is the sym form of #87 (null for us) without the asym target;
  they publish `nvidia/Qwen3.8-27B-NVFP4` (MLP NVFP4 W4A4, attention + GDN FP8; GPTQ hurt on this model).
- **Decision / next:** five BACKLOG rows ("NVIDIA Model Optimizer overlap"); the saturation check is the cheap first one.

### 2026-11-03 08:48 CST · E10 · Dead-unit census of the writer inputs (27B calib Hessians)
- **Status:** done · **Machine:** cpu1 (`v3/e3/e10/census.py`)
- **Result:** down_proj (64 × 17,408 units): none below 1e-2 × median (min 0.019); out_proj (48): 5 units below 1e-4,
  131 below 1e-2; o_proj (16): 22 below 1e-4, 160 below 1e-2. Bottom 10% of units hold ≤ 6.7% of input energy.
- **Reading:** null: no dead units worth a format feature (~0.03% of the writer columns near-dead). Faraway E10 closed.
- **Files:** `data/e10/census.jsonl`

### 2026-11-03 08:45 CST · #87-02 #87-03 #87-04 #83-02 #85-02 #88-02 #89-02 #78-04 #33-23 #81-02 #81-03 E10 · Day queue (gpubox still down)
- **Status:** planned → running · **Machine:** 4070 (`proxy2b/exp/d1103/run.sh`, ~10–12 h), cpu2 + cpu1 (`v3/e3/a1/a1ctl.sh`, ~5 h), cpu1 (E10 census, minutes)
- **Pass rules:** `proxy2b/exp/d1103/README.md`, `v3/e3/a1/README.md` (#81-02 / #81-03), all written before data.
- **4070 order:** #87 sibling asym stacked on edy H_out; ρ screen; gen-masked Σdy² (the 27B estimator); scale refits (B1 full, C9 on C10, both on edy); asym for all 150 tensors with own-seed quantized upstream; asym at K4 / K5; healing on the edy + asym stack; i8c + edy seeds 3–5; Σdy² 512 vs 1,024.
- **CPUs:** #81-02 permuted-predictor control for A1 (the 4.5% energy vs 17% error gap); #81-03 the other 48 layers at K2.5 only if the control clears.

### 2026-11-03 08:35 CST · #33-22 #78-03 #80–#89 · Night queue results (0.8B proxy, 4070; A1 and #80 on CPUs)
- **Status:** running → done (all steps rc 0) · **Pass rules:** each folder's README, YAQA08.md amendments 15/16, `v3/e3/a1/README.md`, `v3/e3/cd/README.md`
- **#78-03 (i8c on top of edy H_out), K3:** ratio 1.000 / 1.015 / 1.005 (seeds 0–2); K2 1.003 / 0.973 / 0.996. **Does not stack at K3**: the codebook's −1.7% (on plain H_in) vanishes under H_out. i8c adoption needs the 27B check to include an H_out arm.
- **#33-22 (dose):** dy1350 / edy K3 1.013 / 1.016 / 1.008 (every seed worse), K2 1.040 / 0.997 / 0.994; dy2700 / dy1350 K3 1.004 / 0.996 / 1.000, K2 0.96 / 1.03 / 1.03. **Fails both:** past 1,024 windows nothing improves (the +326 windows even cost ~1% at K3). Contradicts #33-21's 640 → 1,024 gain; reading: saturation by ~1K, the last 326 windows may be atypical. 27B: 512 → 1,024 is the useful range, no need to cycle to 2,048.
- **#82 (w10, attribution):** K3 gain shares gdn_in 0.39, mlp_in 0.44, attn_in 0.13, writers 0.03; depth thirds 0.32 / 0.35 / 0.38; additive (Σ/full 1.00–1.04). edy gains hold at K4 (−8.7%) and K5 (−7.5%). Verdict: two-sided everywhere except writers (one-sided is enough there).
- **#83 (w01, gen-masked loss):** masked / unmasked K3 0.971 (every seed better, z −2.4…−4.5), K2 0.950 (every seed). **PASS** → the 27B collection should use `YAQA_GENMASK=1`.
- **#84 (w03):** A9 analytic SwiGLU diagonal fails its cosine gate (median 0.86); gate/up's share of the B gain at K3 is 0.0030 (2.9 base SDs).
- **#85 (w02, fixed-code scale refits):** C9 per-family exponent α\* (fit slice) K3 0.973, K2 0.911 on search, transfers to seed 1 (0.973 / 0.914) **PASS**; C10 swj (joint scale on SwiGLU pair) K3 0.969, K2 0.914, also on top of bdiag (0.970 / 0.912) **PASS**; B1 h6 on bdiag K3 0.993 / K2 0.975 (seed 0 screen, 3-seed run pending). Single seed each (codes fixed, paired z −14…−22). Overlap of C9 and C10 untested.
- **#86 (w07-genh, C13):** K3 0.998 (inconclusive), K2 0.976 (every seed, z −4…−5; bar 0.97 missed) → inconclusive.
- **#87 (w07-seqsib, C6 asym):** K3 0.857, K2 0.824 (every seed, control SD 0.8% / 0.3%) **PASS, very large**: 48 writer tensors quantized with sibling-quantized inputs (λ 1, ρ 0.5). Beats edy H_out (K3 0.0845 vs 0.0899). Needs a stacking test with H_out and a sanity review before belief.
- **#88 (w07-denoise, C1):** K3 1.031, K2 1.10 → **FAIL**.
- **#89 (w06-heal):** jitter 1.0044 ok; frozen-code healing all-layer K2 0.619, last-only 0.845, oracle 0.377; K3 on bdiag 0.832; mixed-K (F/D) 0.65–0.74; gradcos gates pass. H1(a) ceiling 62% → by the README rule **build S27 (27B healing)**; that is a QAD-class stream and needs the user.
- **#80 (cpu1, CD replicate, row-seed 1):** cd3 / ldlq 0.993 (K2.5), 0.993 (K3), 16 / 16; cubic / mul1 0.989 / 0.985, 16 / 16. **PASS**: #77 and #78 replicate on other rows.
- **#81 (A1, cpu2 + cpu1):** ug / D mean K2 0.838, K2.5 0.832, K3 0.829 (L30 0.73 / 0.71); up only 0.871 / 0.866 / 0.863 (keeps 80% of the reduction); oracle 0.825. **PASS** (L30 ≤ 0.85, mean ≤ 0.90), up-only preferred. Caveat to check first: the predicted share of H-weighted energy is only 4.5%, yet the coded error drops 17%; that needs explaining (shape effect or a bug) before any GPU work.
- **Files:** `data/night1103/*.txt`, `data/yaqa08/result-fact1{2,3}.txt`, `data/a1/a1-{cpu2,cpu1}.jsonl`, `data/e3/cd-cpu1-rs1.jsonl`

### 2026-11-02 20:05 CST · #79 · Coordinate descent stacks on the cubic codebook (CPU proxy)
- **Status:** running → done · **Machine:** cpu2 · **Pass rule:** `v3/e3/cd/README.md` (#79)
- **Result** (cd3 / ldlq, both on cubic0.02_i8; mean / param-weighted / tensors ≤ 1): K2.5 0.992 / 0.993 / 16 of 16;
  K3 0.992 / 0.993 / 16. Combined (cubic + cd3) vs ldlq mul1_i8: K2.5 0.979, K3 0.974 (means).
- **Reading:** passes: CD's −0.8% is unchanged on top of the codebook (the two gains add).
- **Decision / next:** #77 stays "real, small"; with #80 (other rows) it decides whether CD gets a GPU version.
- **Files:** `data/e3/cd-cpu2-stack.jsonl`

### 2026-11-02 20:35 CST · #33-22 #81–#89 · Night queue (no gpubox): worker experiments on the 4070, A1 on the CPUs
- **Status:** planned → running · **Machine:** 4070 (`proxy2b/night4070.sh`, one after another), cpu2 + cpu1 (`v3/e3/a1/a1run.sh`)
- **Pass rules:** each folder's README (the workers' draft rules, committed unchanged before any data, f443a64 /
  146a4f3); #33-22 `proxy2b/YAQA08.md` amendment 16; #81 `v3/e3/a1/README.md`.
- **4070 order:** #82 w10-hout-attrib (where the H_out gain comes from: family / depth third, K4 / K5), #83
  w01-genmask (H_out loss on scored positions only; patch merged into collect.py by hand: the SHORT edits had moved
  it), #33-22 (dose past 1,024, repeated windows), #84 w03 (gate/up share of the gain, noGU), #85 w02 (fixed-code
  scale refits B1 / C9 / C10), #86 w07-genh (gen-weighted H_in, C13), #87 w07-seqsib (sibling-sequential Hessians,
  C6), #88 w07-denoise (activation-denoising target, C1; `ourq --wpre` merged), #89 w06-heal (frozen-code scale
  healing H1 / H4). Estimated ~14–17 h in all.
- **CPUs:** #81 faraway A1 (down_proj as a residual after predicting from decoded up / gate), 16 layers × K 2 / 2.5 /
  3, after #79 / #80 finish; ~8–10 h.
- **Files:** results land in `/data/yb/<exp>/result.txt` (4070) and `/data/work/a1/a1.jsonl` (CPUs); copied into
  `data/` with each experiment's result entry.

### 2026-11-02 19:55 CST · #78 · Cubic int8 codebook (i8c) adopted (user: a win)
- **Status:** running → done · **Machine:** 4070 · **Pass rule:** `proxy2b/YAQA08.md` #78 section, amendment 15
- **Result:** with seeds 4–5 (#78-02, which had already started): K3 0.983× (6 seeds, z −3.7), K2 0.973× (5 seeds,
  z −2.5; one seed worse); K4 0.977× (2 seeds). Real at K2, K3 and K4.
- **Decision (user):** a win; no further confirmation. i8c (`EXL3_I8_CUBIC 0.02`) becomes the V3 trunk codebook,
  with the 27B confirmation queue's check when the trunk is requantized. #78-03 (stacking with H_out) runs to the end.
- **Files:** `data/yaqa08/result-fact10.txt`, `result-fact12.txt`

### 2026-11-02 19:55 CST · #33-21 · Short qad documents in the H_out collection: neutral; more long windows help
- **Status:** done · **Machine:** 4070 · **Pass rule:** `proxy2b/YAQA08.md` amendment 14
- **Result** (Σdy², seed-paired, 3 seeds; 1,600 sequences = ~640 long windows + ~960 short docs): mixed / long-only
  from the same run K3 1.000×, K2 0.997× (SE 1.6%); both vs #33-18's `edy` (1,024 long windows) K3 1.015× (z 7.9 /
  8.3), K2 1.007× / 1.010×.
- **Reading:** short documents add nothing (not real → not adopted by the rule), and the long-window count still
  matters at 640 vs 1,024 (−1.5% at K3).
- **Decision / next:** the 27B night collects long windows only (`hdiag27.sh` YAQA_SHORT=0): 512, the K3 quants, then
  on through all 1,350 windows to 2,048 (repeats with fresh labels), a snapshot every 128 for the dose curve.
- **Files:** `data/yaqa08/result-fact11.txt`

### 2026-11-02 19:10 CST · #77 / #78 · CPU proxy sweep, final read (16 of 16 tensors)
- **Status:** #77 running (replicate + stacking), #78 passes · **Machine:** cpu2 + cpu1 · **Pass rule:** `v3/e3/cd/README.md`
- **Result** (ratio to LDLQ mul1_i8; mean / parameter-weighted / tensors ≤ 1):

  | | K2 | K2.5 | K3 | K4 |
  |---|---|---|---|---|
  | cd3 (#77) | 0.990 / 0.991 / 15 of 15 | 0.992 / 0.993 / 16 | 0.993 / 0.995 / 16 | 0.991 / 0.993 / 16 |
  | cubic0.02 (#78) | 0.991 / 0.994 / 14 | 0.987 / 0.989 / 16 | 0.982 / 0.986 / 16 | 0.979 / 0.985 / 16 |
  | gmul1 (exact) | 0.991 / 0.993 / 14 | 0.986 / 0.988 / 16 | 0.982 / 0.986 / 16 | 0.977 / 0.984 / 16 |
  | cubic0.05 | 0.996 / 1.001 / 8 | 0.992 / 0.996 / 10 | 0.985 / 0.991 / 16 | 0.977 / 0.987 / 15 |

- **Reading:** #78: cubic0.02 passes and equals the exact Gaussianization (no table needed). #77: fails the written
  0.98 bar but is real (every tensor, every K) and small (−0.7%); under the two-question rule it gets a replicate
  (#80, other rows) and a stacking test (#79, CD on the cubic codebook) before a GPU version is weighed against its
  2–3× quant time.
- **Files:** `data/e3/cd-cpu2.jsonl`, `data/e3/cd-cpu1.jsonl`

### 2026-11-02 19:10 CST · #78 · 0.8B KL test of the cubic int8 codebook (i8c): real at K3 and K4, K2 not yet
- **Status:** running (confirmation #78-02) · **Machine:** 4070 · **Pass rule:** `proxy2b/YAQA08.md` #78 section
- **Result** (seed-paired ratio i8c / i8, base settings): K3 **0.983** (4 seeds, SE 0.60%, z −2.9; per seed 0.971,
  0.986, 0.998, 0.978); K2 **0.980** (SE 1.06%, z −1.9; 0.979, 0.962, 1.010, 0.971); K4 **0.977** (2 seeds, 0.971,
  0.983). Matches the CPU proxy (−1.4% K2.5, −2.0% K3).
- **Reading:** real at K3 by the rule; K2 misses (z −1.9, one seed worse). Consistent → confirmation with seeds 4–5
  (#78-02) and the stacking test with diagonal H_out (#78-03), amendment 15.
- **Files:** `data/yaqa08/result-fact10.txt`

### 2026-11-02 19:10 CST · #33-20 · Sketch A refined weights (Σ(xᵀH_in x)dy²): as good as Sketch B, worse than Σdy²
- **Status:** done · **Machine:** 4070 · **Pass rule:** `proxy2b/YAQA08.md` amendment 13
- **Result** (seed-paired, 3 seeds): xh / eB K3 1.008 (z 2.2), K2 0.991 (z −1.2) → "as good as" (within one seed
  SD); xh / edy K3 1.020 (z 10.3), K2 1.015 (z 4.7); also edy / eB K3 0.988 (z −7.5), K2 0.976 (z −2.3).
- **Reading:** the refined weights buy nothing; plain per-token Σdy² is the best and cheapest estimator.
- **Decision / next:** the 27B collector keeps Σdy² as primary (Sketch B and xw ride along); no XH loading.
- **Files:** `data/yaqa08/result-fact9.txt`

### 2026-11-02 19:10 CST · #33-19 · H_out from our own agentic traces (pisess): worse; 50/50 mix no better
- **Status:** done · **Machine:** 4070 · **Pass rule:** `proxy2b/YAQA08.md` amendment 12
- **Result** (Sketch B diagonal, 1,024 windows, seed-paired vs #33-18 `eB` on qad windows): pisess K3 **1.035×**
  (z 5.7), K2 1.019× (z 1.0); 50/50 mix K3 0.999×, K2 0.977× (z −1.6). Cosine qad vs pisess per linear: median
  0.946, min 0.703 (layer 20 in_proj_z).
- **Reading:** neither arm passes; pisess is different and worse on g08-search1 (no re-check: the caveat applies
  only if pisess is not worse).
- **Decision / next:** the 27B collection stays on calib qad data.
- **Files:** `data/yaqa08/result-fact8.txt`

### 2026-11-02 17:05 CST · #— · Challenge author: the target is VRAM size; load-time expansion only as a stopgap
- **Decision (deadcode, asked by the user):** the challenge optimises VRAM size. Expanding weights at load time is
  acceptable only as a temporary workaround while a kernel is missing; the goal is minimal VRAM.
- **Consequence:** ideas whose storage is small but whose runtime form is larger (faraway A1 "down from up + gate",
  C5 entropy-coded storage, any load-time transcoding) only count once they run from the stored form: A1 needs the
  online-prediction kernel (faraway A2: transposed GEMMs re-reading the up / gate tiles, +1–2 GEMMs in the MLP), so
  its gain must be weighed against that speed cost from the start. A proxy test of A1 is still worth it (it decides
  whether the kernel is), but on the GPU (gpubox), not a CPU day.

### 2026-11-02 16:50 CST · #33-18 · Estimators and dose on 0.8B: Sketch A step 1 and Σdy² work too; 512 sequences suffice
- **Status:** running → done · **Machine:** 4070 · **Pass rule:** `proxy2b/YAQA08.md` amendment 11
- **Result** (3 seeds, base settings + diagonal H_out from 1,024 calib qad windows unless noted; base mean K3 0.0986,
  K2 0.5330):

  | diagonal | K3 mean | vs base | K2 mean | vs base | every seed beats every base seed |
  |---|---|---|---|---|---|
  | Sketch B (`eB`) | 0.0910 | −7.7% | 0.5119 | −4.0% | K3 yes, K2 no |
  | ‖x‖²dy² (`exw`, Sketch A step 1, the old 27B gates' weighting) | 0.0923 | −6.4% | 0.4988 | −6.4% | yes / yes |
  | Σdy² (`edy`, H_Z) | 0.0899 | −8.8% | 0.4996 | −6.3% | yes / yes |
  | Sketch B, 512 seqs | 0.0920 | −6.6% | 0.5072 | −4.8% | yes / yes |
  | Sketch B, 256 | 0.0927 | −6.0% | 0.5198 | −2.5% | yes / no |
  | Sketch B, 128 | 0.0918 | −6.8% | 0.5160 | −3.2% | yes / no |
  | (xcorp 8K Sketch B, #33-16) | 0.0906 | −8.1% | 0.5043 | −5.4% | yes / yes |

- **Reading:** A2: `exw` keeps the gain (83% at K3, more than Sketch B at K2) → the old 27B negatives were not the
  estimator's fault (they were full-matrix two-sided rounding from few sampled rows, #33-02…05); per the rule, a
  short 27B pilot swap comes before the full night. `dy` is as good as Sketch B or better (within / beyond one seed
  SD): the cheapest estimator, per-token, needs no x, works on short documents. A5: **512 sequences** qualify (K3 86%,
  K2 >100%, every seed beats base); 256 / 128 fail at K2.
- **Decision / next:** 27B collector: per-token Σdy² (plus Sketch B and xw in the same pass for a 27B check), 512
  sequences, then the pilot swap; YAQA08 amendment 10's stopping rule is replaced (see the next amendment).
- **Files:** `data/yaqa08/result-fact7.txt`

### 2026-11-02 16:52 CST · #77 / #78 · CPU proxy sweep, first read (13 of 16 tensors; ffn_down pending on cpu1)
- **Result** (ratio to LDLQ with mul1_i8, mean over tensors / parameter-weighted):

  | K | CD 3 sweeps | gmul1_i8 | cubic0.02_i8 | cubic0.05_i8 |
  |---|---|---|---|---|
  | 2 | 0.990 / 0.991 | 0.990 / 0.993 | 0.991 / 0.994 | 0.995 / 1.000 |
  | 2.5 | 0.991 / 0.992 | 0.985 / 0.988 | 0.986 / 0.988 | 0.990 / 0.996 |
  | 3 | 0.993 / 0.994 | 0.981 / 0.985 | 0.980 / 0.985 | 0.982 / 0.990 |
  | 4 | 0.990 / 0.992 | 0.975 / 0.983 | 0.976 / 0.984 | 0.974 / 0.985 |

  CD helps every tensor at every K but by ~1%, accepted tiles 25% / 14% / 9% per sweep.
- **Reading (provisional):** #77 C2 fails its 2% bar (K2.5 0.991). #78: gmul1_i8 and cubic0.02_i8 pass (≤ 0.995 at
  K2.5 and K3, ≤ 1 at K4) and are within 0.1% of each other → cubic0.02 (2 FMAs) by the rule; cubic0.05 fails at
  K2.5. Final after ffn_down. Consistent with the faraway lens's M6 (the weights are near-Gaussian, mul1's bytesum is
  not: matching the codebook to the weights helps).
- **Files:** `data/e3/cd-cpu2.jsonl`

### 2026-11-02 16:35 CST · #77 / #78 · Tile coordinate descent (C2) and Gaussianized mul1 codebooks (D3): CPU sweep started
- **Status:** running · **Machine:** cpu2 (20 jobs × 2 threads) · **Code:** `v3/e3/trellis2.py`, `v3/e3/cd/run.sh` @ 48e7b99
- **Pass rules:** `v3/e3/cd/README.md` (written before data).
- **Setup:** 16 real 27B tensors (GDN layers 10/30/50 × qkv, out, gate_up, down; attention layers 19/43 × q, o),
  256 rows, K 2 / 2.5 / 3 / 4, paired proxy error. Smoke test (one small tensor, 64 rows, K2.5): CD 2 sweeps
  −0.5%, gmul1_i8 −0.7% vs mul1_i8 (not a result).
- **Files:** (when done) `data/e3/cd.jsonl`

### 2026-11-02 15:55 CST · #76 · pisess v1: our own pi sessions as an agentic-trace corpus
- **Status:** done · **Machine:** cpu1 (CPU) · **Code:** `xcorp/pisess.py`, `xcorp/pisess_redact.py`,
  `xcorp/tokenize.py` @ 2cf7666
- **Setup:** active branch of every session, rendered with Qwen3.8's chat template (tools block from the tools each
  session used, thinking → reasoning_content, structured tool calls, tool results clipped to 16k chars head + tail);
  turns already emitted elsewhere skipped; print-mode runs' repeated attachments collapsed to placeholders (mining
  chunks of our own logs dropped); cut at turn boundaries into ≤ 250k-token pieces. Redaction: key prefixes, PEM,
  auth / password assignments, IPs, key-file names, emails, high-entropy non-hex tokens.
- **Result:** main + subagent sessions 89 pieces / 11.9M tokens (975 duplicate turns skipped); print-mode runs 450 /
  6.65M; **539 documents, 18.54M tokens**. Redaction hits: secrets 22, infra 2,162, email 774, entropy 532; integrity
  flags on 24 documents (unpaired tool call 17, unbalanced think 8), kept. Much of the Claude reasoning is absent from
  the logs (empty think blocks).
- **Decision / next:** training / statistics only (AGENTS.md); first use #33-19. Lives on cpu1
  `/data/work/pisess/v1` and the 4070 `/data/pisess/v1` (not in the repo: private content).

### 2026-11-02 15:58 CST · #33-19 · H_out on our agentic traces vs calib qad windows (0.8B): queued
- **Status:** planned · **Machine:** 4070 · **Code:** `proxy2b/yaqaFact8.sh` · **Pass rule:** `proxy2b/YAQA08.md` amendment 12
- **Setup:** 1,024-window diagonal on pisess v1 and the 50/50 mix with #33-18's qad diagonal; K3 / K2 × 3 seeds;
  starts when #33-18 finishes (~17:00 CST), ~1 h.

### 2026-11-02 15:30 CST · #— · Own pi session logs allowed as a training / statistics corpus (user)
- **Decision:** `/workspace/pi-sessions` (all our pi sessions: 117 main / subagent files, 459 print-mode runs, 396 MB)
  may be used for training and statistics (QAD, healing, H_out / Hessian collections), as a non-calib corpus like
  xcorp; not for selection or verification, never published (private material). AGENTS.md updated.
- **Census** (characters of message text; md5 dedupe of identical blocks): main + subagent sessions 58.7M (assistant
  22.1M, tool results 35.9M, user 0.7M), 42.4M unique ≈ 10–12M tokens — about the size of calib, and all agentic;
  longest session 16.8M chars (~4M tokens). Print-mode runs: 175.8M of user messages = repeated attached documents
  (low value), 19.7M of assistant reports. Private-looking blocks (IPs / key paths / emails): ~3,900.

### 2026-11-02 15:12 CST · #33-18 · Estimator check (Sketch B / ‖x‖²dy² / dy²) and dose 128–1024 on 0.8B: started
- **Status:** running · **Machine:** 4070 (≤ 7.2 GB) · **Code:** `proxy2b/yaqaFact7.sh`, `collect.py` YAQA_DIAG_EXTRA @ 39e3cd3
- **Pass rule:** `proxy2b/YAQA08.md` amendment 11 (critic items A2, A5)
- **Setup:** one fp32 collection with the 27B collector on calib qad windows (1,024 seqs, snapshots 128–1024), then
  6 diagonals × K3/K2 × 3 seeds = 36 quants vs the existing base seeds. ETA ~17:00 CST.
- **Files:** (when done) `data/yaqa08/result-fact7.txt`

### 2026-11-02 14:55 CST · #62 · Research-idea run done; published to docs/ideas/
- **Status:** running → done · **Files:** `docs/ideas/` (ranked.md + 10 lens reports + papers.md); raw in
  `local/papers/ideas/`
- **Result:** 10 lenses (composable, open, crossarea, lowbpw, measure, hout, kernels, heal, codes, redteam), 35–61 KB
  each; critic (13 min): ~950 ideas merged into 120, top 25, diagnostics, rejections, code and paper spot-checks
  (one lens misread of H_Z-PTQ corrected; code facts: collector loss over all positions, `ourq --hout` silently skips
  missing files, knap4 prices measured levels by the one-sided proxy). The codes agent ran heavy jobs on the VM
  (stopped six times; see AGENTS.md rule and `local/bin/pi-capped`).
- **Decision / next:** before the 27B diagonal collection, the critic's pre-run diagnostics on the 0.8B: A2
  (estimator check), A1 (gen-masked loss), A5 (dose below 1K), plus A3 instrumentation and A8 asserts.

### 2026-11-02 14:25 CST · #33-17 · 27B diagonal collector: split halves, one-GPU mode, tested on 0.8B
- **Status:** done · **Machine:** 4070 (≤ 7.2 GB, peak 4.8 GB) · **Code:** `v3/yaqa/collect.py` (even-sequence sums,
  halves.json per save, one-GPU and tied-head support, resumed sums moved to the card), `proxy2b/diag08test.sh`,
  `proxy2b/diaghalves.py`
- **Result:** gate: chunked diagonal = whole fp32 G (rel L2 0.0); 40 sequences straight vs 20 + resumed 20:
  **bit-exact** (max diff 0.0; the first resume attempt crashed on a CPU/GPU device mix, fixed); halves.json written
  (40 seqs: median cos 0.94). 0.8B reference halves of 1,024 seqs: median 0.995, p5 0.964, min 0.916; at 2,048:
  0.998 / 0.978; at 4,096: 0.999 / 0.990.
- **Decision / next:** stopping rule for the 27B run in `proxy2b/YAQA08.md` amendment 10. The collector is ready for
  `hdiag27.sh` once both gpubox cards are free.
- **Files:** `data/yaqa08/diag-halves-and-collector-test.txt`

### 2026-11-02 14:06 CST · #33-16 · Seed replicates: the diagonal H_out gain holds over 3 seeds; 0.8B quant noise measured
- **Status:** running → done · **Machine:** 4070 (≤ 7.2 GB) · **Code:** `proxy2b/yaqaFact6.sh` @ 5a5e596
- **Pass rule:** `proxy2b/YAQA08.md` amendment 9
- **Result** (KL, rotation seed offsets 0 / 1 / 2; base settings; bdiag = + 8K diagonal H_out):

  | K | arm | seed 0 | seed 1 | seed 2 | mean | seed SD |
  |---|---|---|---|---|---|---|
  | 3 | base | 0.0989 | 0.0994 | 0.0974 | 0.0986 | 1.05% |
  | 3 | bdiag | 0.0910 | 0.0906 | 0.0902 | 0.0906 | 0.44% |
  | 2 | base | 0.5354 | 0.5411 | 0.5225 | 0.5330 | 1.79% |
  | 2 | bdiag | 0.4928 | 0.5107 | 0.5093 | 0.5043 | 1.97% |

  Mean gain K3 −8.1% (10.2 seed-SDs), K2 −5.4% (2.95 seed-SDs; 5.1 SDs of the 3-seed mean). Every bdiag seed beats
  every base seed at both K. Identity H_out (earlier in this experiment): K3 +4.9%, K2 +2.1% vs base.
- **Reading:** rule 1 passes at K3; at K2 the "every seed beats every seed" part passes and the 3×SD part is
  borderline (2.95× the per-seed SD; the rule did not say per-seed or per-mean). Rule 2: the identity arm is worse
  than base, so the svh weighting explains none of the gain. **0.8B quant-to-quant noise (seed SD): ~1% at K3,
  ~2% at K2; best–worst of 3 seeds 1–2% (K3) and 3.5% (K2).** Paired z between single quants overstates confidence
  by about this much. The K2 gain from seed 0 alone (−8.0%) was lucky; the 3-seed figure is −5.4%.
- **Decision / next:** the diagonal H_out is worth the 27B collection (fp32, 1K sequences, ~8.5 h, both cards).
  From now on, proxy comparisons under ~3% (K3) / ~5% (K2) need seed replicates. Best-of-N seeds: 1–3.5% best–worst
  spread, mostly noise on the picking slice; not pursued (would need pick-on-search, confirm-on-verify).
- **Files:** `data/yaqa08/result-fact6.txt`

### 2026-11-02 13:58 CST · #62 · Research-idea run started: 10 Opus 5.5 xhigh lenses + critic
- **Status:** planned → running · **Machine:** VM (`pi -p`, shared cached prefix) · **Code:** `local/papers/ideasrun.sh`,
  `ideaslane{2,3,4}.sh`, `ideasqueue.sh`, briefs in `local/papers/run/`
- **Setup:** prefix = shared context, design/v3, experiment INDEX / LOG / BACKLOG / archives, ROADMAP, design/v2,
  YAQA08 / CALIBRATION, the 448-paper index, 187 digests (~250k tokens); one instruction file per lens last. Lenses:
  composable, open, crossarea, lowbpw, measure, hout, kernels, heal, codes, redteam; then a critic merges and ranks.
  Run in parallel lanes (up to 8 at once), one lock per lens; the critic waits for every lens.
- **Result so far:** 7 reports done (composable 61 KB / 94 ideas, open 56 KB, crossarea 59 KB, measure 43 KB, hout
  51 KB, lowbpw 49 KB, kernels 61 KB); codes, redteam, heal running; critic after them. Usage: ~one 5-hour window.
- **Decision / next:** critic → `ideas/ranked.md`; then publish to `docs/ideas/` (scrubbed, "untested").
- **Files:** `local/papers/ideas/`

### 2026-11-02 13:55 CST · #33-16 · Seed replicates and identity H_out on 0.8B: started; identity read
- **Status:** running · **Machine:** 4070 (allocator ≤ 7.2 GB) · **Code:** `proxy2b/yaqaFact6.sh`,
  `ourq.py --seed-offset` @ 5a5e596 · **Pass rule:** `proxy2b/YAQA08.md` amendment 9
- **Result so far:** identity H_out (base settings + H_out = I) is **worse** than base: K3 0.1037 (+4.9%, z 7.5),
  K2 0.5466 (+2.1%, z 3.9); vs the diagonal arm +14% / +11%. So the svh row weighting does not explain the gain: the
  measured diagonal does, and the 27B diagonal collection is needed. Seed 1: K3 base 0.0994 (+0.6% vs seed 0),
  bdiag 0.0906 (−8.8% vs its base, z −13); K2 base 0.5411 (+1.1%). Seed 2 and K2 bdiag pending.
- **Decision / next:** full entry with the 3-seed noise floor when done (~14:10 CST).

### 2026-11-02 13:21 CST · #33-15 · Split precision judged by KL: fails as written; the test shows quant-to-quant noise instead
- **Status:** done (negative as written) · **Machine:** 4070, capped at 7.2 GB (peak 3.4 GB)
- **Code:** `proxy2b/yaqaFact5.sh`, `prectest.py --save` @ b005ba3 · **Pass rule:** `proxy2b/YAQA08.md` amendment 8
- **Setup:** diagonals from the first 1,024 windows, yaqaB's targets, linear GEMMs x2 or bf16; quants on `base`'s
  settings + diagonal H_out, paired against the fp32 1K-diagonal arm (`kXbdiag1024`). Collection 1.79 (x2) /
  0.98 (bf16) s / sequence.
- **Result** (KL ratio vs the fp32-diagonal arm, z):

  | collector | K3 | K2 |
  |---|---|---|
  | x2 | 0.982 (−2.7) | 1.028 (+4.7) |
  | bf16 | 0.982 (−2.4) | 1.005 (+0.9) |

- **Reading:** x2 fails (K2 +2.8%). But the less precise bf16 collector does *better* than x2 at K2 and both beat
  fp32 at K3, so the KL differences are not about precision: a tiny change in the diagonal gives a different quant,
  and quant-to-quant spread is ~±2–3% at K2 and ~±2% at K3 on this slice, with |z| up to ~5. **The paired
  per-document SE does not include this rounding variance**, so 0.8B arm-vs-arm differences below ~3% are not
  trustworthy, whatever z says. The H_out gains (−8…−10% vs base) are larger than that spread; smaller effects in
  #33-10 … #33-14 (e.g. diagonal vs full at K3, 1K vs 2K) are within it.
- **Decision / next:** (1) the 27B collector stays fp32 (the KL test cannot separate x2 from fp32; fp32 is the safe
  choice, ~8.5 h). (2) New rule for proxy comparisons: estimate quant-to-quant noise with replicate quants (e.g. 3
  rotation seeds per arm) and require differences beyond it; write it into the proxy method before the next
  comparisons. (3) Re-check the H_out gain with seed replicates (base and bdiag, 3 seeds each at K3 and K2).
- **Files:** `data/yaqa08/result-fact5.txt`

### 2026-11-02 12:55 CST · #33-13 · Diagonal H_out dose on 0.8B: 1K sequences are enough
- **Status:** running (27B part pending) · **Machine:** 4070, allocator capped at 7.2 GB (peak ~1.1 GB)
- **Code:** `proxy2b/yaqaFact4.sh` @ f0fb687 · **Pass rule:** `proxy2b/YAQA08.md` amendment 6
- **Result:** identity check: `k3bdv8192` (diagonal as a `hout_diag` vector file) = `k3bdiag` exactly (KL
  0.0909878287 both, diff 0.0): ourq.py's new path is correct. Dose, base settings + diagonal H_out:

  | seqs | K3 KL | vs base (z) | share of 8K gain | K2 KL | vs base (z) | share of 8K gain |
  |---|---|---|---|---|---|---|
  | 1K | 0.0922 | −6.8% (−10.4) | 85% | 0.4933 | −7.9% (−13.0) | 99% |
  | 2K | 0.0911 | −7.8% (−11.9) | 98% | 0.4995 | −6.7% (−12.1) | 84% |
  | 8K (#33-12) | 0.0910 | −8.0% | 100% | 0.4928 | −8.0% | 100% |

- **Decision / next:** 1K passes the 80% bar at both K, so **the 27B diagonal collection is planned at 1K
  sequences** (fp32, ~8.5 h on both cards; the split-precision speed-up failed, #33-14). Run `hdiag27.sh 1024`
  when gpubox is back and card 1 is given; snapshots at 256 / 512 / 1024.
- **Files:** 4070 `/data/yb/result-fact4.txt`

### 2026-11-02 12:40 CST · #33-14 · Split-precision GEMMs on 0.8B: fails the pass rule (10× better than bf16)
- **Status:** planned → negative (as written) · **Machine:** 4070, allocator capped at 7.2 GB (peak 3.6 GB)
- **Code:** `proxy2b/prectest.py` @ 0ef4e99 · **Pass rule:** `proxy2b/YAQA08.md` amendment 7
- **Setup:** 16 xcorp windows, same sampled targets, trunk linear GEMMs fp32 / x2 / bf16, everything else fp32; the
  summed diagonal per linear vs fp32.
- **Result:** loss/token |diff| max: x2 8.2e-5, bf16 1.8e-3. Relative L2 error of the summed diagonal:

  | layers | x2 median | x2 max | bf16 median | bf16 max |
  |---|---|---|---|---|
  | 0–5 | 5.5e-3 | 2.5e-2 | 5.3e-2 | 1.0e-1 |
  | 6–11 | 4.5e-3 | 9.8e-3 | 3.4e-2 | 1.0e-1 |
  | 12–17 | 3.1e-3 | 5.0e-3 | 2.5e-2 | 4.3e-2 |
  | 18–23 | 1.2e-3 | 5.1e-3 | 1.3e-2 | 3.0e-2 |

  Time (4070, not informative for the R9700): fp32 1.60, x2 1.43, bf16 1.04 s / sequence.
- **Reading:** the products are ~1e-6 accurate (CPU self-test); the backward through 24 layers amplifies the
  difference, most in the early layers. Misses the median bound (1e-3) everywhere and the worst-linear bound in
  layers 0–5. Missing control: fp32's own spread under a harmless change (summation order), which would show
  whether 1e-3 was attainable at all.
- **Decision / next:** the 27B collector stays fp32 for now. Options if speed matters: the fp32 noise-floor control;
  a three-way split (~2^-24, 3 / 6 GEMMs); or judge x2 by its effect on KL (a quant from an x2-collected diagonal
  vs an fp32-collected one) instead of a matrix error bound — any of these needs a new pass rule first.
- **Files:** `data/yaqa08/prectest.jsonl`

### 2026-11-02 12:20 CST · #33-14 · Split-precision (bf16 hi/lo) GEMMs for the fp32 collector: prepared
- **Status:** planned · **Machine:** 4070 (check), gpubox (use) · **Code:** `v3/yaqa/bf16x.py`, `proxy2b/prectest.py`,
  `collect.py` YAQA_PREC=x2 @ 0ef4e99
- **Question:** can the 27B collector's fp32 GEMMs (~30 s / sequence) run as bf16 hi/lo splits on the matrix cores
  (2 GEMMs per product, 3 for the weight gradient; weights exact in bf16) without losing the fp32 gradient quality?
- **Pass rule:** `proxy2b/YAQA08.md` amendment 7.
- **Setup:** CPU self-test on random data (emulated): x2 vs fp32 relative error 2.5e-6 (products), 6e-7 (diagonal);
  bf16 1.7e-3 / 3.8e-4. The model-level test (`prectest.py`, ~10 min on the 4070) is not run yet.
- **Decision / next:** run prectest + `yaqaFact4.sh` when the 4070 is free; on gpubox, the gate run of `hdiag27.sh`
  with YAQA_PREC=x2 also reports one linear's precision against a whole fp32 G and the time per sequence. Then the
  two-stage pipeline (#33-13) if the GEMMs still dominate the profile.

### 2026-11-02 11:50 CST · #33-13 · 27B diagonal H_out: collector and 0.8B dose check prepared
- **Status:** planned · **Machine:** gpubox (both cards) + 4070 · **Code:** `v3/yaqa/collect.py` YAQA_DIAG,
  `v3/yaqa/hdiag27.sh`, `quantize/ourq.py` (`--hout` reads `hout_diag` vectors), `proxy2b/yaqaFact4.sh` @ f0fb687
- **Question:** how many sequences the diagonal needs (0.8B), then the 27B collection and a 27B swap test.
- **Pass rule:** `proxy2b/YAQA08.md` amendment 6 (0.8B dose); the 27B swap test's rule is written before it runs.
- **Setup:** 27B: per sequence and trunk linear, diag(G_b G_bᵀ) with G_b = Σ_t dy_t x_tᵀ, sampled targets, fp32
  backward (~30 s / sequence on both cards, measured in gate 2b), qad-pool 2048-token windows (pools-v3), snapshots
  at 256 / 512 / 1024 / 2048; accumulators ~10 MB. 27B H_out sizes for reference: full 206 GB fp32 (gate / up 155),
  without gate / up 51 GB.
- **Decision / next:** run `yaqaFact4.sh` when the 4070 is free (~12 min); the 27B run needs both cards (user's
  arrangement, e.g. a night): 1K sequences ≈ 8.5 h. Faster near-fp32 backward (bf16 hi/lo split of activations,
  weights exact in bf16) is a possible speed-up, to be checked for precision first.

### 2026-11-02 11:15 CST · #33-12 · H_out for 27B sizing: diagonal is enough; ~4K sequences
- **Status:** done · **Machine:** 4070 · **Code:** `proxy2b/yaqaFact3.sh` @ 86f1233
- **Pass rule:** `proxy2b/YAQA08.md` amendment 5 (written before data).
- **Setup:** all arms on `base`'s settings + H_out at output damping 0.01, i8, g08-search1.
- **Result** (vs `base`: ratio, z; vs `bhout` = full H_out, 8K):

  | arm | K3 KL | vs base | vs full | K2 KL | vs base | vs full |
  |---|---|---|---|---|---|---|
  | full (`bhout`, #33-11) | 0.0893 | −9.7% (−12.4) | — | 0.506 | −5.6% (−6.3) | — |
  | full, gate/up diagonal (`bmix`) | 0.0891 | −9.9% (−13.5) | −0.2% (−0.4) | 0.494 | −7.7% (−11.0) | −2.2% (−5.0) |
  | diagonal only (`bdiag`) | 0.0910 | −8.0% (−13.5) | +1.9% (2.4) | 0.493 | −8.0% (−13.1) | −2.5% (−3.0) |

  Dose, full H_out at K3: 1K 0.0950 (−3.9%, 40% of the 8K gain), 2K 0.0922 (−6.8%, 70%), 4K 0.0890 (−10.0%,
  103%; vs 8K z −0.4).
- **Reading:** the diagonal keeps the gain by the rule (≥ 2/3 at both K, worse than full only at K3 by z 2.4 < 3), so
  it sets the 27B collection: **diagonal only** (~10 MB of accumulators). The full gate/up H_out *hurts* at K2 (the
  noisiest factors, #33-09). The mixed arm is the best at K3 (+~2% over diagonal) for 25 GB of accumulators at 27B.
  Full H_out needs ~4K sequences (2K reaches 70%); the dose of the diagonal alone was not measured (its two halves
  agree at ≥ 0.997, #33-09, so likely fewer).
- **Decision / next:** plan the 27B collection: per-sequence weight-gradient row norms (diag of Sketch B's H_out)
  with sampled targets, fp32 backward, calib H_in unchanged; ~4K sequences of 2048 tokens from the qad pool; then a
  27B K3 / K2.5 swap test against the ladder's levels before anything changes in V3. Optional on 0.8B: dose of the
  diagonal (1K/2K on `bdiag`).
- **Files:** 4070 `/data/yb/result-fact3.txt`; 27B H_out sizes per type in this entry's commit message context
  (full 206 GB fp32, without gate/up 51 GB, diagonal ~10 MB).

### 2026-11-02 10:30 CST · #33-11 · H_out follow-ups on 0.8B: holds at K2 and stacks on our settings
- **Status:** done (positive) · **Machine:** 4070 · **Code:** `proxy2b/yaqaFact2.sh` @ 6496e9e
- **Pass rule:** `proxy2b/YAQA08.md` amendment 4 (written before data).
- **Setup:** 8K snapshot, i8, g08-search1. `hout` / `hdiag`: paper damping 0.01, no scaled Hessian. `bhout`: our
  `base` settings (scaled Hessian, damping 0.025 / input 0.005) plus full H_out at output damping 0.01.
- **Result:**

  | K | arm | KL | vs ldlq (z) | vs base (z) |
  |---|---|---|---|---|
  | 3 | base | 0.0989 | −5.7% | — |
  | 3 | hout (#33-10) | 0.0946 | −9.9% (−13.8) | −4.3% (−5.4) |
  | 3 | **bhout** | **0.0893** | — | **−9.7% (−12.4)** |
  | 2 | base | 0.535 | −3.7% | — |
  | 2 | hout | 0.537 | −3.4% (−3.6) | +0.2% (0.2) |
  | 2 | hdiag | 0.532 | −4.2% (−7.9) | −0.6% (−0.9) |
  | 2 | **bhout** | **0.506** | — | **−5.6% (−6.3)** |

- **Reading:** all three rules pass: H_out holds at K2; it stacks with the scaled Hessian and tuned damping (bhout vs
  hout: K3 −5.6%, K2 −5.8%); bhout beats base at both K with z ≤ −2, which meets the user's switch bar **on the
  0.8B**. At K2 the diagonal alone does as well as the full H_out without our settings.
- **Decision / next:** plan a 27B check; its cost is set by two 0.8B tests first: (1) a mixed arm on base's settings,
  full H_out except gate/up (diagonal only), and the all-diagonal arm — the full gate/up H_out is ~155 GB of fp32
  accumulators at 27B, the diagonal a few MB; (2) dose: 1K / 2K / 4K snapshots on base's settings.
- **Files:** 4070 `/data/yb/result-fact2.txt`

### 2026-11-02 10:15 CST · #33-10 · YAQA factorial on 0.8B: the H_in swap was the loss; H_out with calib H_in wins
- **Status:** done (positive) · **Machine:** 4070 · **Code:** `proxy2b/yaqaFact.sh` @ 053a640
- **Pass rule:** `proxy2b/YAQA08.md` amendment 3 (written before data).
- **Setup:** K3, i8, damping 0.01 both sides, no scaled Hessian, 8K snapshot, g08-search1; reference `ldlq`
  (calib H_in, one-sided) and `base` (our tuned quant).
- **Result:**

  | arm | H_in | H_out | KL | vs ldlq (z) | vs base (z) |
  |---|---|---|---|---|---|
  | ldlq | calib | — | 0.1049 | — | +6.1% |
  | hin | Sketch B | — | 0.1257 | +19.8% (16.5) | +27.1% (20.2) |
  | yb8192 | Sketch B | full | 0.1123 | +7.0% (5.1) | +13.6% (8.1) |
  | **hout** | calib | full | **0.0946** | **−9.9% (−13.8)** | **−4.3% (−5.4)** |
  | hdiag | calib | diagonal | 0.0972 | −7.4% (−11.1) | −1.7% (−2.1) |

- **Reading:** the earlier YAQA losses (#33-08) came from Sketch B's gradient-weighted H_in, not from H_out. With
  the calib H_in, the full H_out beats paper LDLQ by ~10% and our tuned one-sided quant by ~4%, even without the
  scaled Hessian and tuned damping `base` has; the diagonal alone gets about three quarters of the gain over ldlq.
- **Decision / next:** H_out passes. Before anything at 27B, on the 0.8B: (1) the same arm at K2; (2) H_out on top of
  `base`'s settings (scaled Hessian, tuned damping) — the arm that would actually ship; (3) dose (1K/2K/4K
  snapshots) to size a 27B collection; (4) whether H_out damping matters. 27B collection only after those.
- **Files:** 4070 `/data/yb/result-fact.txt`

### 2026-11-02 10:25 CST · #33-09 · Sketch B factors: structure and noise (CPU)
- **Status:** done · **Machine:** 4070 box CPU (niced) · **Code:** `proxy2b/houtstruct.py`
- **Question:** is the 0.8B Sketch B H_out noise-limited, how is it shaped, and how far is Sketch B's H_in from the
  plain calib input Hessian?
- **Setup:** all 150 linears, snapshots 1K–8K (each the mean over its first n sequences). Noise: cosine between the
  two independent 4K halves (H(4096) vs 2·H(8192) − H(4096)). Spectrum of H(8192): participation ratio and entropy
  rank as a fraction of d, diagonal share of ||H||²_F, spread (CV) of the diagonal.
- **Result** (medians by type):

  | type | d | PR | diag share | diag CV | H_out halves cos | diag halves cos | H_in vs calib cos |
  |---|---|---|---|---|---|---|---|
  | GDN in_proj_qkv | 6144 | 0.053 | 0.22 | 1.69 | 0.903 | 0.998 | 0.764 |
  | GDN in_proj_z | 2048 | 0.054 | 0.81 | 3.96 | 0.960 | 0.998 | 0.756 |
  | GDN out_proj | 1024 | 0.49 | 0.50 | 0.18 | 0.770 | 0.999 | 0.891 |
  | attn q / k / v | 4096 / 512 / 512 | 0.04 / 0.04 / 0.20 | 0.27 / 0.13 / 0.43 | 2.3 / 1.5 / 0.5 | 0.958 / 0.997 / 0.905 | ≥0.998 | 0.79 / 0.54 / 0.73 |
  | attn o_proj | 1024 | 0.37 | 0.41 | 0.20 | 0.656 | 0.997 | 0.866 |
  | mlp gate / up | 3584 | 0.28 / 0.22 | 0.63 / 0.47 | 1.0 / 1.0 | 0.797 / 0.845 | 0.999 | 0.69 / 0.81 |
  | mlp down | 1024 | 0.60 | 0.63 | 0.17 | 0.944 | 1.000 | 0.633 |

  By depth (median over types): H_out halves cos 0.69 (layers 0–5), 0.80 (6–11), 0.92 (12–17), 0.98 (18–23).
  Sketch B's H_in is itself stable (halves cos ≥ 0.96) but differs from the calib Hessian (cos 0.35–0.98).
- **Reading:** (1) the **diagonal of H_out is precise** at 4K (≥ 0.997) and very uneven for input projections
  (CV 1.5–4) but flat for residual writers (out_proj, o_proj, down: CV ~0.2); (2) the **full H_out is noisy in the
  early half and for residual writers / gate / up** (0.66–0.80 at 4K sequences per half): two-sided LDLQ there rounds
  partly against noise; (3) residual writers have near-isotropic H_out (PR 0.4–0.6), where two-sided ≈ one-sided;
  (4) the YAQA arms also **replaced the calib H_in** (10M tokens, x xᵀ) with Sketch B's gradient-weighted H_in from
  xcorp, which is far from it (cos ~0.5–0.9), and gradient-weighted H_in alone was +31…+44% worse on the 27B (#33-07).
  So the YAQA loss mixes two changes; which one costs the KL is untested.
- **Decision / next:** a factorial on the 0.8B (GPU, ~4 min an arm) separating the two: calib H_in + Sketch B H_out;
  Sketch B H_in + identity H_out; calib H_in + diag(H_out); pass rule to be written before it runs.
- **Files:** `data/yaqa08/houtstruct.jsonl`

### 2026-11-02 09:10 CST · #33-08 · YAQA Sketch B on 0.8B: 2K–8K negative, 16K stopped
- **Status:** running → negative · **Machine:** 4070 · **Code:** `proxy2b/yaqaB.py`, `yaqaBrun.sh`
- **Pass rule:** `proxy2b/YAQA08.md` (amendment 2); reference arm `ldlq`, our tuned `base` beside it.
- **Setup:** as the 1K read; snapshots 2K/4K/8K from one bit-exact resumed collection (8,192 sequences, 207.6 min).
- **Result** (KL on g08-search1; ratio to ldlq, z paired):

  | seqs | K3 yb | vs ldlq | z | vs base | K2 yb | vs ldlq | z | vs base |
  |---|---|---|---|---|---|---|---|---|
  | 1K | 0.1184 | +12.8% | 9.4 | +19.7% | 0.690 | +24.1% | 19.6 | +28.8% |
  | 2K | 0.1158 | +10.3% | 7.1 | +17.1% | 0.677 | +21.8% | 13.5 | — |
  | 4K | 0.1121 | +6.9% | 5.1 | +13.4% | 0.675 | +21.4% | 13.4 | +26.0% |
  | 8K | 0.1123 | +7.0% | 5.1 | +13.6% | 0.642 | +15.5% | 11.8 | +19.9% |

  References: K3 ldlq 0.1049, base 0.0989; K2 ldlq 0.556, base 0.535.
- **Decision / next:** fails the pass rule at every snapshot. K3 is flat from 4K to 8K; K2 still improves at 8K,
  but extrapolating it would at best approach ldlq, still far from base. 16K collection stopped by the user
  (15:07 UTC); the 4070 is free. Snapshots 1K–8K kept for H_out research and a later diagnosis run. YAQA is not
  adopted for V3; the V3 design stays.
- **Files:** 4070 `/data/yb/result.txt`, `/data/logs/yaqaBrun.log`, `/data/yb/snap{1024,2048,4096,8192}`

### 2026-11-02 06:20 CST · #— · Tracker reorganised
- **Decision:** the two old trackers (`EXPERIMENTS-new-format.md`, `EXPERIMENTS.md`) are archived verbatim in
  `archive/`; every experiment so far was numbered #1–#62 in `INDEX.md`; open plans and the backlog moved to
  `BACKLOG.md`. Old paths keep a one-line pointer. From now on: one numbered row per experiment, one templated log
  entry per event.

### 2026-11-02 05:50 CST · #33-08 · YAQA Sketch B on 0.8B: two-sided objective optimised but does not track KL
- **Status:** running (unchanged) · **Machine:** 4070 · **Code:** `proxy2b/objcheck.py` (CPU)
- **Question:** is the 1K loss a quantizer bug (two-sided objective not optimised) or the objective itself?
- **Setup:** 14 linears of layers 0/5/11/19/23; per linear, one-sided error tr(E Hc Eᵀ)/tr(W Hc Wᵀ) with the calib
  input Hessian, and two-sided tr(E Hi Eᵀ Ho)/tr(W Hi Wᵀ Ho) with the 1K Sketch B factors; K3 arms base, ldlq, yb1024.
- **Result:** yb1024 has a 1.4–8× lower two-sided objective than base on every linear, and a 1.4–3.5× higher
  one-sided error (e.g. layer 5 in_proj_qkv: two-sided 5.9e-3 → 7.6e-4, one-sided 3.0e-3 → 8.8e-3).
- **Decision / next:** the quantizer path works; at 1K sequences and damping 0.01 the two-sided objective is the
  wrong target. Wait for 2K–16K before diagnosing; keep every snapshot for H_out research (`local/TODO.md`).
- **Files:** 4070 `/data/logs/objcheck.log`

### 2026-11-02 05:25 CST · #33-08 · YAQA Sketch B on 0.8B: 1K read negative
- **Status:** running (unchanged) · **Machine:** 4070 · **Code:** `proxy2b/yaqaB.py`, `yaqaBrun.sh`
- **Pass rule:** `proxy2b/YAQA08.md` (amendment 2 written before any data); `ldlq` is the one-sided reference,
  `base` (our tuned quant) reported beside it.
- **Setup:** 0.8B, i8 codebook, K3 and K2; arms base / ldlq (σ 0.01, no scaled Hessian) / yb1024 (per-linear
  H_in, H_out from 1,024 sequences, σ 0.01 both sides); golden g08-search1.
- **Result:**

  | K | base | ldlq | yb1024 | yb vs ldlq |
  |---|---|---|---|---|
  | 3 | 0.0989 | 0.1049 | 0.1184 | +12.8% (z 9.4) |
  | 2 | 0.535 | 0.556 | 0.690 | +24.1% (z 19.6) |

- **Decision / next:** collection continues; 2K/4K/8K scored ~09:00–09:15 CST, 16K ~13:20 CST. The 2K–8K reads
  decide whether to stop before 16K.
- **Files:** 4070 `/data/yb/result.txt`

### 2026-11-02 ~04:25 CST · #60-03 · Our quantizer on CUDA (qtf2 builds, qtf2check PASS)
- **Status:** done · **Machine:** 4070 (CUDA sm_89) · **Code:** exllamav3 fork (built in place), qtf2, `qtf2check.py`
- **Result:** qtf2 builds after two CUDA-only fixes (include order; half2 bit casts via memcpy; HIP code unchanged). `qtf2check.py` PASS on CUDA: 30 rate × codebook rows bit-identical to the fork's kernel, 1/8-step round trips. 2B input Hessians over all calib (34.7 min, long docs as 8k windows); 0.8B (14.9 min). First 2B quant, uniform K3 i8 trunk: KLD 0.0685, top-1 88.9% on g2b-search1 (173 search-pool docs, 166k gen rows; about 2x the 27B's uniform K3). Whole-trunk 2B quant ~2.5 min, its eval 23 s.
- **Decision / next:** rerun the HIP identity gate before syncing to gpubox.

### 2026-11-02 ~04:20 CST · #60-02 · 2B: prefix run vs whole document
- **Status:** done · **Machine:** 4070 · **Code:** `proxy2b/lencheck.py`
- **Question:** why does a document's prefix run alone differ from the whole document?
- **Result:** not a bug. cuBLAS picks its matmul kernel by matrix size; kernels sum in different orders, and the bf16 output rounding then differs. One bf16 linear (6144x2048): 27% of outputs differ between M = 512 and the same rows at M = 3519; none at M = 64 or 2048. Layer 0 differs by ~1 bf16 ulp (3.4e-3 rel); 24 layers amplify it to ~7% hidden / KL ~3e-3 on the first 64 rows. Deterministic per shape (repeat = 0), same on the pure-torch path.
- **Decision / next:** golden and evals run each document whole, same cap, unbatched (mkgold / evalq; self-KL -2.8e-7). Never chunk or batch differently between them.

### 2026-11-02 ~04:05 CST · #61 · xcorp v1: 96.1M-token proxy corpus from the upstream sources
- **Status:** done · **Machine:** cpu1 (`/data/xcorp/v1`) · **Code:** `xcorp/`
- **Result:** the manifest's 58 sources at pinned revisions, with categories, kinds and caps, 12x the per-split budgets, rendered as the challenge (Qwen3.8 chat template: xhigh preamble, tools block, reasoning kept on every turn, structured tool calls). Not the 8 long agentic traces (unpublished). Head of each stream skipped (multi-file: second half of the files; single-file: 1.1x the records it read); a line-level check against calib + public rejected 2,205 documents. Shortfall: hermes-fc 0.8M of 1.9M (source exhausted). Build ~8 min.
- **Decision / next:** proxy research only. The private split is unpublished, so overlap with it cannot be checked; check the challenge's rules before anything trained on xcorp ships.

### 2026-11-02 ~04:00 CST · #60-01 · 2B proxy: bf16 vs fp32 inference
- **Status:** done · **Machine:** 4070 · **Code:** `proxy2b/halves.py`
- **Setup:** model in two halves so fp32 fits the 4070; golden g2b-fit1 = 87 fit-pool docs, 2k cap, 86.9k gen rows, bf16, fla + causal-conv1d.
- **Result:** control: the split run in bf16 reproduces the golden (KL -4e-7, top-1 1.0000). fp32 vs the bf16 golden: KL 3.75e-4, p99 1.8e-3, top-1 98.4%, flat by position (3.9e-4 at 0-128, 3.6e-4 at 1k-2k: per-layer rounding, not state build-up). GDN kernel-path differences on the same rows ~1.5e-4.
- **Decision / next:** goldens stay bf16 (the challenge's reference is bf16; quants run on the same path, so the rounding is shared, not noise). Matters only at high bpw, where 2B KLDs approach ~1e-3: there, check a screening result with an fp32 halves run.

### 2026-11-02 ~02:05 CST · #35 · Embedding run6 (scale search 0.5..1.0): null
- **Status:** null · **Machine:** cpu1 (run6 per the dead-end entry)
- **Result:** K1 rel 0.25700 -> 0.25642 (-0.2%, edge rows 0%); K1.25 0.18289 -> 0.18287. The scale search was never what limits the K1 / K1.25 embedding.
- **Decision / next:** menu unchanged.

### 2026-11-02 ~02:00 CST · #59 · exl3i serving path works on the GPU, up to the full model
- **Status:** running (GPU selftests pass; `d0.sh` not run) · **Machine:** gpubox GPU · **Code:** radiance-hero `qc-exl3-v3` 4edad1f..340165b
- **Result:** encoding `exl3i` (scheme exl3i, transform `<cb>/h128/k<rate>`, a u32 record TABLE plane of 16-row groups); C oracle `common/exl3i_decode.c` (= the reference decoder); importer `exl3quant`'s `exl3i` quantiser (real K2.5 / K3 tensors: H-weighted error = ourq's logged proxy_err, ratio 1.000, stacked gate_up included); kernel library `exl3i` (op `exl3i_dense`, exl3r4d's kernel with the decode as a rate policy). Selftest `exl3i_gpu_test`: decode bit-exact at all 64 rates x 2 codebooks; 256 GEMMs over every rate; 48 shape / dtype cases; integer K = exl3r4d bytes; 19 refusals. Mutation check: 6 of 10 mutants run (all caught) before gpubox went down. `qwen35_exl3` routes exl3i weights to `exl3i_dense` (host test).
- **Decision / next:** `d0.sh` (a level's trunk in V2 3.00, real KLD) not run yet.

### 2026-11-02 ~01:40 CST · #35 · Embedding run5 was a no-op
- **Status:** null (no-op)
- **Result:** run5 identical to run4 (K1 rel 0.25700, edge 0.2017). The edge rows sit at the LOW end: K1 826 / 4096 rows at factor 0.85 (the smallest), none above 0.95; K1.25 all at 0.95. run5 widened the top (to 1.75).
- **Decision / next:** run6 searches 0.5..1.0 (cpu1).

### 2026-11-02 — · #37 · Speed pass 2: profile and output-identical speedups (gated)
- **Status:** running · **Machine:** card (K2.5 layers 3-4, ladder settings) · **Code:** qtf2 (`v3/e3/exl3frac2`), `ourq.py`
- **Result:** profile: kernel 73% of wall, refit 11%, Cholesky / rotations / Hessians ~10%; the host thread waits on the GPU 86% of its samples (CPU is not a limit). Kernel time split (timing-only builds): traceback 3-5%, history writes 11-27% (K5.5: 45%). Tiles per call barely matters (64 tiles: 45k/s, 320: 50k/s). qtf2 v5 (packed history, one 16/32-bit store per thread per position) +5-11% at K1.x-2.x, +3-4% at K3.x; LDS-only step barrier ~+1%. v6 (packed minimum in the uncached path): K5.5 14.1k -> 31.1k tiles/s (x2.2); K6.5 / 7.5 barely. quantize_exl3: regularized-domain proxy and refit's e_before (verbose only) skipped; ourq caches the last Hessian (the sweep reloaded it 4x). e2e K2.5 L3-4: 52 tensors + 13 proxy rows byte-identical to the old code, 63 -> 55.5 s a layer (-12%).
- **Not done:** reusing K-independent work across levels (~7%, needs a tensor-major ladder); refit via Cholesky instead of LU (faster, not bit-identical); KA 6-7 lane-cooperative search; sub-1 0-bit-step fusion. Sub-1 (KA 0): 11.3k tiles/s at the ladder's batch before v7 (costs in global memory): ~3.9 card-h a level, ~27 for all seven.
- **Checks:** qtf2v5check (v6 vs v4 every instance), qtf2int6check, qtint5check, e2e5, speed2deploy.

### 2026-11-02 — · #37 · Speed pass 2: integer K1-4 on qtf2 (MASK 0)
- **Status:** done · **Machine:** card
- **Result:** bit-identical to both integer kernels; x1.9 / 2.1 / 1.9 / 1.3 (K1 67k, K2 81k, K3 85k, K4 93k tiles/s vs 36k / 39k / 44k / 69k). K5-8 stay on the integer kernel (LDS barrier there: +1-6%).

### 2026-11-02 — · #37 · Speed pass 2: v7, step 0 peeled
- **Status:** running · **Machine:** card
- **Result:** `first` made a compile-time constant. With the 4-way width switch the compiler had kept an `if (first)` branch on every candidate (55 KB of code, ~1,100 branches vs 15 KB / 14 at MASK 0). K1.x +70-80%, K2.x +25%, K3-4.x +50-65%, K5.5 41k, K6.5 17k, sub-1 +50%.

### 2026-11-02 — · #37 · Speed pass 2: race found and fixed (v8)
- **Status:** running (v8 gate passed; speed pass continues) · **Machine:** card
- **Result:** the LDS-only step barrier (v5) made KA 1 output nondeterministic (`qtf2race.py`: first run after load differed in v6 / v7; random runs in a byte-history build). The cost arrays are reached through generic pointers (flat loads / stores), which an LDS-scoped fence does not wait for. The earlier e2e passes were luck. Every barrier is `__syncthreads()` again (the barrier was worth ~1%); the same patch on the integer kernel was reverted before use. v8 also picks packing per rate (KA 1 and integer KA <= 3; byte history for mixed KA 2-5).
- **Checks:** v8 gate: exact vs v4 arithmetic at all 34 instances, 0 vs the fork, integer identical, race 34 x 4 runs: 0.

### 2026-11-02 — · #37 · Speed pass 2, later results
- **Status:** running · **Machine:** card
- **Result:** multi-level ourq (`-K a,b,c`) byte-identical to separate runs (K2.25 / 2.5 / 3: 52 tensors + proxy rows + stamps), only -5% time (259 vs 272 s): the K-independent work is small. Integer K6 / K7 on qtf2: identical, x2.0 / x3.0 (i8); K5 slower on qtf2 (30k vs 45k tiles/s), so it stays on the integer kernel. Two ourq per card on v8: identical outputs, throughput x1.14.

### 2026-11-02 — · #37 · qtf2 v9 (QTF2_HIOPT: inlined forward lambdas)
- **Status:** done (final build chosen) · **Machine:** card
- **Result:** identical at every KA 5-7 instance; levels out at ~25k (i8) / ~35k (mul1) tiles/s. Mixed KA 6: 15-18k -> 25k; KA 7.5: 4.3k -> 24k (i8) / 5.6k -> 33k (mul1). Integer K6 / K7 and KA 5 slower with it (unroll capped at 16 for KOUT >= 6).
- **Decision / next:** final build: HIOPT only for mixed KA 6-7 (own compile units, `qtf2_inst_a6f` / `a7f`). K5 integer stays on the integer kernel.

### 2026-11-02 — · #38 · damp3: damping held out across K0.5-7
- **Status:** running
- **Question:** does the best damping spread change with K, and integer vs fractional? (user)
- **Pass rule:** grid and rule fixed before the run in `v3/e3/damp3/damp3report.py`. Grid 0.025 / 0.01 / 0.005 / 0.0025 / 0.001 / 0.0003 / 0.0001, quantized on the split-half H_A, scored on H_B.
- **Result:** old ladder data (K2-4.5): winners integer 51 / 42 / 7% (0.001 / 0.005 / 0.025), fractional 55 / 37 / 8%; neighbour-level agreement ~53% for int-int, frac-frac and int-frac alike: no sign of a type effect. The 0.005-vs-0.001 pick is mostly noise.

### 2026-11-02 — · #34 · Head mean-row subtraction: head-only KL -30%
- **Status:** done (dry run; adopted for every head donor) · **Code:** `headremake.sh` step 1
- **Setup:** exact headsim, uniform heads, 32,768 held-out head inputs.
- **Result:** K3 0.00794 -> 0.00555, K5 0.000492 -> 0.000346 (geo ratio 0.70). Exact (softmax-invariant), free. Damping with mean-row: K3 best sigma 0.005 (-2.6% vs default), K5 best default (0.01 / 0.005 ~+1%): they disagree.
- **Pass rule:** written beforehand: every remade donor runs all three damping values and keeps the lowest headsim KL.
- **Decision / next:** adopted for every head donor. headsim's donor grouping (HS_MAXK) reproduced the earlier pricing exactly.
- **Files:** `v3/yaqa/data/head-settings.json`

### 2026-11-02 — · #35 · Embedding K1 / K1.25: wider per-group scale search (run5, run6) — null
- **Status:** null · **Machine:** cpu1 (`/data/work/embt/rows5`, `rows6`)
- **Setup:** run5 (factors to 1.75) and run6 (0.5..1.0) on 4096 rows, trellis, group 5120.
- **Result:** K1 rel 0.25700 -> 0.25642 (-0.2%); K1.25 unchanged. At K1 a fifth of the rows sat on the low edge of the old search, but moving the edge bought nothing: the error at K1 is the trellis's, not the scale's.
- **Decision / next:** menu unchanged.

### 2026-11-01 ~23:20 CST · #43 · Rounded-codebook quantizers pass; row-slab head works; knap4 vs V2 menu
- **Status:** done · **Machine:** card (GPU) · **Code:** `cbsmoke.py`, `headrun2.sh`, exl3cb
- **Result:**
  - Rounded codebook: int8 on grid, x1.000 K1–6, x1.024 K7, x1.347 K8. int4 on its 16 levels: x1.001 K1, x1.007 K2, x1.020 K2.5, x1.059 K3, x2.5 K4.
  - Row-slab head (#34; exact head-only KL on 32,768 held-out head inputs, paired vs uniform, 95% CI): K5 budget −40% (−48…−31), K6 −36% (−42…−29), K4 −9.6% (−23…+5, n.s.); 3 slabs = per-block; without the K3 floor 3–12x worse; rare-token guard KL x1.2–1.5 at K5–6. Head ≈ 1.5–3% of total KLD: ~0.5–1% at 3.5–5 bpw.
  - knap4 (V3 menu, quality only) vs V2 menu (#45): 1.75 −25%, 2.0 −21%, 2.5 −15%, 3.0 −10%, 3.5 −13%, 4.0 −10%, 4.5 −11%, 5.0 −6%. At 2.5: steps −9.6%, trellis embedding −5.4%, damping ~0, quarter heads 0. At 2.5 bpw nothing goes below K1.5; at 2.0 the embedding hits its K1.5 floor (extend to K1). Speed: 3.0 at x1.53 prefill for +0.6% KLD; 4.0 for +2.5%.
- **Decision / next:** levels settings = int8 codebook + per-tensor damping sweep on inputs. Gate 2e crashed twice more (profiler OOM, then expandable_segments at load); restarted at 23:05 without either.

### 2026-11-01 ~23:05 CST · #57 · GPU search's "4–6% gap" is the tail-biting constraint, not a flaw
- **Status:** done · **Machine:** CPU emulation (`f16gap.sh`) · **Code:** `f16gap.sh`, `viterbi_tiles_tb` (exllamav3 heuristic, NTRY 1)
- **Result:** fp16 costs change nothing (random tiles 1.000; real tensors with LDLQ median 1.000 at K2–6). The CPU reference Viterbi has a free start state, which EXL3's circular tiles cannot encode. A CPU tail-biting search reproduces the GPU exactly (free/TB 0.948 K2, 0.962 K4 vs GPU 0.949 / 0.960). Trying 16 wrap candidates recovers 12% / 8% / 4% of the gap at K1/2/3 (0.8% / 0.4% / 0.16% error): exllamav3's heuristic is near-optimal.
- **Decision / next:** not worth a kernel change. All earlier CPU trellis sweeps (E3b, fracsweep, gscale, damping) used the free start, so their absolute errors are 4–6% optimistic; ratios between arms are unaffected.

### 2026-11-01 ~21:20 CST · #36 · Quarter-step quantizer works on the GPU (`fracsmoke.py`)
- **Status:** running (smoke-test verdict fixed; levels unblocked) · **Machine:** card 0 · **Code:** `fracsmoke.py`
- **Result:** 8192 Gaussian tiles on card 0: pack/unpack round trips and decode exact at every menu K; quarter / half steps on the curve (0.993–1.003 to K4.75). The first verdict said FAIL only because the CPU-vs-GPU check expected agreement within 3%: the GPU search is 4–6% worse than the exact CPU Viterbi at every K, integer K (exllamav3's own kernel) included, and fractional K sits within 0.4% of its integer neighbours' offset. Check fixed.
- **Decision / next:** the levels (chain0) are unblocked. Overnight CPU chains queued: fine gain sweep on cpu2 after the seed test; damping at 6 more depths (K3/5/7) on cpu1 after the damping sweep. The 4–6% gap itself is open.

### 2026-11-01 ~17:45 CST · #37 · qtf2 v1 (`v3/e3/exl3frac2`): bit-identical to the fork's fractional kernel
- **Status:** running (v2 gate) · **Machine:** card · **Code:** `v3/e3/exl3frac2`, gate `qtf2check.py`
- **Result:** bit-identical to the fork's fractional kernel on all 15 old rates in both codebooks (0 index / value differences, tie-heavy and extreme tiles included). New rates (1/8 steps K0.125-2.875; sub-1 with the salted 0-bit read) pass the window-structure and pack round-trip checks. Speed: x1.27 at KA 2-4, x2.1 at K5.5, x1.03 at KA 1: still ~2x slower than the integer kernel (K3 43k vs 18.6k tiles/s as written; the order looks reversed, see Rewrite notes).
- **Decision / next:** v2 caches every codebook value a thread needs in registers (as the integer K2-5 kernel does) and keeps KA 1's costs in one LDS buffer (register-staged). Gate running.

### 2026-11-01 ~17:45 CST · #34 · Head donors at half steps work alone; K4.5 / K5.5 segfault fixed
- **Status:** running · **Code:** ourq (refit_scales)
- **Result:** head donors at half steps work alone (K3.5 25.6 GB peak). K4.5 / K5.5 segfaulted in refit_scales: a kernel's code object could not load at full VRAM (HSA_STATUS_ERROR_OUT_OF_RESOURCES). Fix in ourq (empty cache + preload the refit kernels): K4.5–6.5 then fine. K7.5 hung (50 min): skipped.

### 2026-11-01 ~17:45 CST · #34 · Slab-head curve is smooth except at the floor rule
- **Status:** running · **Code:** `headslab.py HS_FLOOR_ALL`, `floorcurve.sh`
- **Result:** with half-step donors every 1/8-bit budget step gains 7–10% head KL, but the fixed tail floor round(budget − 1.5) jumps K3 -> K4 at budget 5 (−63%) and K5 -> K6 at 7 (−45%). Between 4.25 and 4.875 the slab plans are worse than uniform half-step heads. Quarter-step donors not needed for smoothness.
- **Decision / next:** one plan per floor; headsim chooses.

### 2026-11-01 ~17:45 CST · #34 · Floor search fixes the slab head
- **Status:** running (whole-model check still to do) · **Code:** `fcurve-A/B.json`
- **Result:** with half-step donors and the tail floor chosen per budget by headsim, the curve is smooth (5–24% per 1/8 bit) and 43–54% below a uniform head at every budget 3.5–7.5 (exact head-only KL, 32,768 held-out head inputs). The best floor sits ~0.5–1 bit below the budget.
- **Decision / next:** whole-model check still to do (critique C3.5).

### 2026-11-01 ~17:45 CST · #74 · radiance 1.15.1 port works (`qc-exl3-115`)
- **Status:** done · **Code:** fork branch `qc-exl3-115` (ABI 32 pins + layout-fingerprint export)
- **Result:** all 4 plugins build; qwen35_exl3 synthetic tests pass. V2 3.00 on golden-ver1: stock 1.15.1 `--kld-ref` vs the frozen 1.13 recorder, gen-masked: 0.027122 vs 0.027122 (+0.002%, SE 0.001%; max per-row difference 0.026). Stock `--kld-ref` scores every row, so mask with gen.u8 (`eval/cmp115.py`), as mergerep does.

### 2026-11-01 ~17:45 CST · #48 · Output scales off on upstream-flagged gate/up: WORSE, +2.5% KLD
- **Status:** negative · **Code:** `v3/e3/osr/`, `v3/e3/osr/result-K3.json`
- **Setup:** K3 mul1; gate + up of the 36 layers with a flagged projection, 52 scales off; swapped into V2 3.00; golden-s2k: 583 search docs, 481k positions.
- **Result:** +2.5% KLD (z 2.1). Upstream's auto rule (ratio > 1.05) misjudges this model, like gate 2d's output-side model.
- **Decision / next:** the requant keeps output scales always on (rule fixed beforehand); the K5 repeat was dropped.

### 2026-11-01 ~17:45 CST · #34 · Low-K heads (`lcurve.json`)
- **Status:** running · **Setup:** donors K2 / K3 / K3.5 / K4, floors >= K2
- **Result:** uniform K2 head 0.0379 (x4.8 K3), but slab plans with a K2 floor are strong (budget 2.125: −48% vs uniform K2; budget 3.0: 0.0060 vs uniform K3 0.0079). Cliffs where the floor switches (2.0 -> 2.125: −48%; 3.0 -> 3.125: −41%) mean the donors are too coarse.
- **Decision / next:** the remake adds K2.25 / 2.5 / 2.75 (+ K1, K1.5); quarters at K2–3 if the cliffs remain.

### 2026-11-01 ~17:45 CST · #37 · qtf2 v3/v4 gate PASS
- **Status:** running · **Machine:** card
- **Result:** packed-min: x3.1–3.5 vs the old kernel at KA 2–4, faster than the integer kernel. Sub-1 table path is procedural with no speed gain: costs in global memory are the limit. `ourq --qtf2` end to end: 52 tensors byte-identical to the old path (K2.5, i8, damping sweep, layers 10–11).

### 2026-11-01 ~17:45 CST · #59 · exl3i oracle and decode plan (`v3/infer`)
- **Status:** running · **Code:** `v3/infer`
- **Result:** plain-C reference decode (fractional, sub-1 salt, mul1 / mul1i8; codebook exact on all 65,536 states) and the generalized per-lane plan of exl3r4d (host emulation: 409,600 lane-position cases, every V3 rate, 0 failures).

### 2026-11-01 ~17:45 CST · #36 · Extra trunk levels don't pay (knap4 `KN_FRACS_EXTRA`)
- **Status:** done (base list stands) · **Code:** `k4levels.sh`
- **Result:** K5.25 / 5.75 ≤ 0.7% (6.0 bpw); 1/8 steps at K3–5 ≤ 0.8% (4.0).
- **Decision / next:** base list stands.

### 2026-11-01 ~15:10 CST · #58 · Full uniform requant planned (user)
- **Status:** planned · **Machine:** card (~90 card-h at today's speed, after a quantizer-kernel pass)
- **Setup (plan):** K0.125–3 at 1/8 steps (sub-1 included as research), quarter steps to 4.75, K5 / 5.5 / 6.
- **Result:** CPU fracsweep below K1: rel error K0.5 0.227, K0.75 0.137, K1 0.089 (2.3–2.5x per bit), so sub-1 is a usable curve. Search slices capped at 2k positions, held-out 8k (`eval/capstudy.py`: ~27% less noise per GPU-second, calls agree 0.993–0.997). Damping sweep kept (per-tensor winners agree only 65–70% across levels). Upstream's output-scale test (`v3/e3/osr/osr.py`, all 64 layers, `v3/e3/data/osr.jsonl`) flags up_proj in 35 of 64 layers (nearly all of L26–63) and gate_proj in 17 (L49–63, L0, L2).
- **Decision / next:** a paired swap test decides the output-scale question before the requant.

### 2026-11-01 ~01:35 CST · #55 · Document-reuse bias test: no measurable flattering from fit-slice reuse
- **Status:** null · **Code:** `eval/biastest.py`; `data/biastest-2026-11-01.out` (existing scores only)
- **Setup:** 76 V2 mixes scored on a fit-pool slice (calA / fit2 / fit3, where V2's models were fitted and the mixes picked) and on ver1; each as a ratio to turboderp's untuned SC reference at the nearest bpw.
- **Result:** bias = ratio on the fit slice / ratio on ver1: median 1.024, weighted mean 1.014 [1.005, 1.023]. By slice: calA 1.027, fit2 1.009, fit3 1.033; by size < 2.5 bpw 0.97 (n = 8), above 1.02–1.03. If anything the mixes look ~2% WORSE on the slices they were fitted and selected on (slice-to-slice spread, not winner's curse).
- **Decision / next:** fit / search reuse costs at most a couple of percent here and is not the problem. The ledger's strict "spent" rule can be relaxed for non-trained documents; QAD-trained ones stay excluded until the same test is run after V4. Caveat: ver1 was the V2 monitor slice (no selection on it; mixes were queued there because of their fit-slice scores).

### 2026-11-01 ~01:25 CST · #46 · A4 timing was invalid (ntbench predates the T2 / NFQ struct fields)
- **Status:** running (rerun queued in cardzero.sh) · **Machine:** card zero · **Code:** `/data/bench/ntbench2` (rebuilt from `/root/ntbench`)
- **Result:** `/root/ntbench` (Oct 29) predates the T2 / NFQ struct fields, so both arms ran the fused NT kernel (identical hashes, no wdec kernels in the profile).
- **Decision / next:** rerun in cardzero.sh with a hash sanity check. The A4 drop (#46, 00:55 CST entry) rests on the matmul-core table, not on this slot.

### 2026-11-01 ~01:25 CST · #34 · Fractional-K heads do not fit the GPU box's RAM
- **Status:** running (integer donors only) · **Machine:** card 1 · **Code:** `headslab.py`
- **Result:** K3.5 / K4.5 OOM-killed; integer heads (~22 GB) fit alone. The slab-head curve uses integer donors K3–K8 (1/8-bit slab budgets still give a continuous curve). Head quants pause card 1's ladder; an OOM kill also took the card-1 K2.75 run once; it resumes per layer. (Superseded by the 17:45 entry: K3.5 works alone there.)

### 2026-11-01 ~01:25 CST · #37 · Ladder profile (py-spy, 150 s of K2.75)
- **Status:** running · **Machine:** card
- **Result:** 84% in ldlq, 77% of wall time waiting on the GPU (trellis search kernel), 9% g_scale search, 4% refit. No host-side speedup is possible; only the frac Viterbi kernel or fewer passes. The input-group damping sweep runs that group 3x: ~+40% time per level for a gain that D2 / damp2 put under 1%.

### 2026-11-01 ~01:20 CST · #53 · B10: the board's `bytes` is the tensor payload, not the file
- **Status:** done · **Code:** `store/sizedef.py` (4 V1 containers; HF files byte-identical to ours)
- **Result:** board vs sum of plane bytes −0.13 / −0.04 / +0.18 / +0.01 MB; vs file size −9.6 to −9.9 MB (header, directory, alignment). Real bpw = payload bytes x 8 / 26,895,998,464; "container bytes" overstates by ~10 MB = 0.003 bpw. radiance 1.15.1's Qwen3.5 plugins do not declare or load the `mtp.` block; whether an MTP drafter shipped in the container would count toward size is not knowable from the board data (none of the entries has one).
- **Decision / next:** ask the challenge author before planning on it.

### 2026-11-01 ~01:05 CST · #52 · B4 closed: KLD rises with position for every quant alike
- **Status:** done · **Code:** `eval/posbins.py`, golden-long2 (2 traces)
- **Result:** ratio to turboderp SC 4.00 per bin 0-2k / 2-8k / 8-32k / 32-64k / 64-128k / 128k+. The reference's own KLD doubles (0.0040 -> ~0.008 past 32k), but the ratios are flat: V2 4.00 0.76–0.84, V2 4.50 0.42–0.51, td SC 3.00 3.46–3.69, V2 3.00 2.5–3.3 (first bin highest). 2 docs only.
- **Decision / next:** no quant compounds error with context faster than another, so short-slice rankings hold at long context; the rise is a property of the text / position.

### 2026-11-01 ~00:55 CST · #45 · knap4 on the measured speed table
- **Status:** done · **Code:** `KN_SPEED=v3/bench/data/speedcal-stock.jsonl` (k4grid default); `KN_A4=a4time.jsonl` swaps in measured paths once A4 has run
- **Result:** also fixed: the LM head was timed at M 4096 (~77 ms, ~5% of the step) though prefill runs it on the last token only; flop shares now exclude it. Measured f16 core ~116 TF (not 135), iu8 ~2.2x. Same picks and KLDs as before at 2.5 / 3.0 / 4.0 (the int8 choice was already saturated). All-int8 speedup estimate x1.65 -> x1.88, at +0.9% (3.0) / +5% (4.0) KLD.
- **Decision / next:** decode is still the placeholder (0.006 ns/weight); A4 decides whether x1.88 holds.

### 2026-11-01 ~00:55 CST · #46 · A4 (int4 activations) not worth building at any size
- **Status:** negative · **Machine:** — (knap4, V3 menu + measured speed table) · **Setup:** sizes 1.75 / 2.0 / 2.5 / 3.0, lambda up to 1e-4
- **Result:** even under the strongest speed pressure the solver puts only 7–14% of FLOPs on A4, buying x1.88 -> x1.91–1.94 prefill for +2…+5% KLD. int8 already takes most of the time out; what is left is decode and the non-GEMM ops.
- **Decision / next:** no int4-codebook quants (they would need their own ladder: post-hoc int4 rounding costs x1.06 at K3, x2.5 at K4) and no iu4 kernel. Revisit only if the A4 timing slot (int8 with decode) comes out far worse than the matmul-core table.

### 2026-11-01 ~00:50 CST · #54 · B19 closed: our K1 decode is bit-exact with exllamav3's
- **Status:** done · **Code:** `v3/e3/b19k1.py`
- **Result:** the plugin's C oracle `exl3_trellis_decode` vs exllamav3's `reconstruct` on turboderp's SC 1.40 tensors: 4 K1, 4 K2, 4 K3 incl. lm_head, 0 differing f16 values. The +12% of our 1.40 vs his chart is not a decode bug.

### 2026-11-01 ~00:47 CST · #50 · D2 done: our proxy edge over turboderp is ~9% in-sample; the rest is real; damping holds
- **Status:** done · **Code:** `v3/yaqa/d2report.py`, data `v3/yaqa/data/d2-score.jsonl`
- **Setup:** split-half input Hessians (layers 3/10/43, 1,200 non-verify windows by document parity), ours quantized on H_A at K4 / K6, scored on H_A / H_B / published H.
- **Result:** ours / turboderp proxy, median of 17 tensors: K4 0.775 in-sample, 0.851 out of sample, 0.863 published; K6 0.774 / 0.852 / 0.862. So ~9% of the advantage is fitting to the Hessian sample at ~1.2M tokens per half; with the published 10M tokens it should be much smaller. Damping: the H_A-best value is also H_B-best for 13/17 (K4) and 14/17 (K6) tensors; the median tensor keeps 0.025 (gain small, as damp2).
- **Decision / next:** ours-vs-ours comparisons (fractional K, damping, codebook) are not affected.

### 2026-11-01 ~00:45 CST · #33-06 · D1 closed by reading the code: no auto-vs-always confound in gate 2d
- **Status:** negative (gate 2d's failure stands; alias D1) · **Machine:** — (code reading, no GPU run) · **Code:** exllamav3 `regularize`, `ourq.py`
- **Result:** output scales on auto differ from "always" only for tensors given an `out_sensitivity` (the gated-MLP path of convert.py); ourq.py never passes one. With a diagonal, the Hessian-derived scales replace the plain ones and output scaling stays on; every gate-2d tensor took them (proxy.jsonl `hout_diag_scales` true for 13/13 in dsamp-K2, dfull-K2, dfull-K3; dsamp-K3 only got 5 tensors).
- **Decision / next:** none needed; no GPU run.

### 2026-11-01 ~00:45 CST · #46 · A4 timing slot written (`v3/bench/a4time.sh`, `a4report.py`)
- **Status:** planned (queued after D2, `a4wait.sh`, card 0)
- **Setup:** ntbench on the NFQ build: f16 T2 vs the int8 path *with* its decode (nfq_act_i8 + nfq_wdec_i8 + nfq_gemm_i8), 5 TP1 shapes, K2–6, M 512 / 2048 / 4096, rocprofv3 decode share at M 2048.

### 2026-11-01 ~00:40 CST · #45 · knap4 wired to real options (quality only; V3 menu: frac + damp; measured speed table)
- **Status:** done (quality only) · **Code:** `KN_EMBT=2`, `KN_HEADSLAB=1`, `KN_LEVELS=/data/v3levels`
- **Result:**
  - `KN_EMBT=2`: trellis embedding priced from chain1's real KLD swaps (golden-ml increments scaled to the pooled corpus by the Lloyd points, 0.333; other K along the CPU curve from the nearest measured point). Result vs the CPU-priced menu (`KN_EMBT=1`): +0.0…+0.8% at 1.75–4.5 bpw, −0.8% at 5, −4.6% at 6. "The CPU-based embedding estimate holds on real KLD." Embedding still sits at its t1.5 floor at ≤ 2 bpw.
  - `KN_HEADSLAB=1` (#34): slab heads at K5 / 5.5 / 6 priced at the conservative end of headsim's interval (x0.69–0.71 head KL; K4 not offered: n.s. and rare-token KL x2.5). Picked at 3.5–5 bpw: −1.3…−1.5% total KLD at 4–5, ~0 at ≤ 3 and at 6. Needs hdslab-K5/5.5/6 containers before validation.
  - `KN_LEVELS=/data/v3levels`: measured fractional levels replace the midpoint interpolation per tensor once the level and both neighbouring integers exist (tested on synthetic data).

### 2026-11-01 00:17 CST · #33-07 · Gate 2e FAILED: gradient-weighted H_in is worse in both K
- **Status:** negative · **Machine:** card · **Setup:** gradient-weighted H_in from the 100-sequence snapshot, blocks 22/23, ver1, paired vs one-sided LDLQ
- **Result:** K2 +44% (z 10.3), K3 +31% (z 11.6).
- **Pass rule (fixed beforehand):** either K positive -> drop. So no blend test (`gate2e-blend.sh` / `hblend.py` were written and tested, unused). All backward-based Hessian ideas are closed (EXPERIMENTS.md). D2 started right after.

### 2026-11-01 ~00:10 CST · #56 · Held-out provenance of every golden
- **Status:** done · **Code:** `eval/pools/provenance.py` -> `eval/pools/provenance-2026-11-01.json`
- **Result:** verify pool: 845 docs; 317 never used by any slice (434k tokens), but none longer than 16k tokens and none agentic / long-agentic / long. Those categories (43.8% of verify tokens) are entirely spent (calB / calBbig / ver1 / ver2 / long2). All 10 verify docs >= 32k are spent; both 250k traces 17 times. Overlaps: ver1 shares 8 docs with calB and 32 with calBbig; ver2 shares 30 / 73. calA / fit2 / fit3 / ml are fit pool (ml 8–12 docs shared with the fit slices).
- **Decision / next:** a ver3 drawn from clean docs only is clean but unrepresentative (no agentic text; code / multilingual 2x over-weighted). Decision needed before building ver3 / ver4 (see TODO).

### 2026-11-01 — · #— · Stop rule for the V3 stream (user)
- **Decision:** a size band's search stops when its held-out (ver3) best-so-far stops decreasing (no gain beyond its paired SE over N ver3 reads, N fixed beforehand). No fixed eval budget (~1,500-3,000 expected). ver3 is read for ~1 in 10 evals only (a candidate that becomes its band's best on the search slice), as in V2.

### 2026-11-01 — · #— · V3 stream budget raised to ~1,200-1,500 evals (user)
- **Decision:** a per-band stop on the ver3 validation curve. Solver-model plan: an uncertainty-aware solve first (bootstrap fits, penalty per option, B11), targeted evals where the fits disagree, partial pooling of factors. The XGBoost / residual re-ranker is retried at the halfway point, with a pass rule fixed beforehand (beat kfit2 on grouped-CV pair ranking by more than the bootstrap noise and on a held-out search slice).

### 2026-11-01 — · #33 · YAQA scratch deleted on the GPU box
- **Status:** housekeeping (no result) · **Machine:** GPU box
- **Result:** deleted qfull, rows32, qdiag, full32 / full64, hout*, n16 / n32, q*, qe: 192 GB; 649 GB free. Kept: headslab / headslab2 (head-slab quants), hs (head Fisher), pylib, scripts, logs.

### 2026-11-01 — · #35 · chain1: real trellis-embedding swaps; ours vs turboderp K6 (results from ~23:50 CT 10-31)
- **Status:** done (trellis embedding and K6 comparison both measured) · **Machine:** card 1 (`chain1.sh`)
- **Setup:** V2 6.00 with golden-ml (multilingual, 67 docs); even-layer K6 swaps into V2 5.00 (fit3, 98 docs).
- **Result:** trellis t2 +0.0047 vs Lloyd 2-bit +0.0092; t3 +0.0012 vs Lloyd 3 +0.0021; t4 +0.00017 vs Lloyd 4 +0.00030; t2.5 +0.0022 ~ Lloyd 3. Trellis costs ~0.5-0.6x Lloyd at equal bits, ~0.5 bit saved at K2-3: confirms the CPU result on real KLD. The fit3 half is empty (no base6 score on golden-fit3). Both sets are on the fit pool, so not held-out numbers. K6 swaps: ours K6 -24.2%, turboderp K6 -23.1%; turboderp vs ours +1.3% (z 1.26, n.s.). Ours and his K6 are equal on real KLD (see #63). The low-damping arm (ours-K6-d005) did not quantize.

### 2026-11-01 — · #75 · tdverify: all 16 turboderp downloads byte-exact against our packs
- **Status:** done · **Result:** 337 tensors each, 0 problems. Each download has 8 MTP trellis tensors that exist only in the downloads.
- **Decision / next:** keep them (or pack them) before deleting.

### 2026-11-01 — · #38 · damp2 (CPU, cpu1): best lower damping vs 0.025, away from layers 11 / 43 / 63
- **Status:** done · **Machine:** cpu1 · **Setup:** layers 1 / 5 / 19 / 30 / 47 / 58, K3 / 5 / 7 · **Files:** `v3/e3/data/damp2.jsonl`
- **Result:** the win is much smaller away from layers 11 / 43 / 63: attention inputs 0.989–0.996 (first sweep: 0.94–0.97); GDN inputs median 0.992–0.998 but one tensor 0.94; everything else >= 0.993.
- **Decision / next:** the gain is tensor-specific, which is what the per-tensor sweep in the level ladder (`--sigma-in-sweep`) is for. Expect well under 1% total.

### 2026-11-01 — · #— · Sub-1 K moved to V3.5 (user)
- **Decision:** sub-1 K goes to V3.5, before QAD, since sub-2 quants are the natural QAD targets.
- **Also in this entry:** A5 selftest classes added to DESIGN §1c.4; D6 (GDN a/b merge) closed (0.27-0.4% share); B13 `ourq.py` code stamp committed (deploy and smoke test in the morning); B19 K1 bit-compare script queued after the gate 2e verdict.

### 2026-11-01 — · #33-07 · Gate 2e MARGINAL rule (written before the verdict)
- **Status:** running (verdict in the 00:17 CST entry) · **Pass rule:** both K negative (even at z ~ -0.5) -> a blend test alpha*H_grad + (1-alpha)*H_plain (quants + evals only, ~40 min) before any 200/400-sequence collection; either K positive -> drop.
- **Note:** this entry sits after the ~00:55 CT entry in the source, but it must precede the 01:17 CDT verdict. Dated as the source gives it (see Rewrite notes).

### 2026-11-01 — · #48 · Output scales by upstream's auto rule (dead end)
- **Status:** negative · **Setup:** exllamav3's out_scales_ratio test flags up_proj in 35/64 layers and gate_proj in 17; turning their scales off (K3, swapped into V2 3.00, paired on 583 search docs).
- **Result:** KLD 2.5% worse (z 2.1).
- **Decision / next:** output-side local sensitivity models keep failing on this model (cf. YAQA gate 2d). Not adopted.

### 2026-11-01 — · #33 · YAQA output-side Hessians from sampled positions, two-sided LDLQ (dead end)
- **Status:** dropped (every backward-based Hessian variant dropped by 2026-11-01; Sketch B not tested)
- **Result:**
  - (a) bf16 backward, 200 seqs: two-sided +23% (K2) / +28% (K3) worse increment. bf16 gradients are noisy toward early layers (down_proj per-row cosine vs fp32 p10: 0.999 at layer 63, 0.02 at layer 20); the bad rows are ~3x too large and dominate H_out.
  - (b) fp32 backward, 200 seqs: +17% / +19% worse at sigma_reg_out 0.025, +6% / +10% at 0.3 (z ≥ 2). More output regularisation keeps helping, i.e. the trend is toward one-sided.
  - (c) full-token fp32 H_out, 200 seqs: still worse (+30% / +21% at 0.025; +7% / +5% at 0.3); worse than sampled.
  - (d) diagonal output scales (gate 2d): K2 +10–15%, K3 +4% (no auto-vs-always confound: D1, code reading).
  - (e) gradient-weighted input Hessian (gate 2e, 100 seqs, one-sided LDLQ, 2026-11-01): K2 +44% (z 10), K3 +31% (z 12); far worse than the plain Hessian, so the pre-set blend follow-up was skipped.
- **Decision / next:** every backward-based Hessian variant is dropped (2026-11-01). Not tested: the paper's Sketch-B estimator with far more data (64K sequences), which we cannot afford. Code: `v3/yaqa/`, tracker 2026-10-31.

### 2026-10-31 ~22:45 CST · #33 · Upstream exllamav3 already has a YAQA collector (reference v1.6.3)
- **Status:** running (second suspect for gate 2) · **Code:** `util/yaqa_hessians.py`; `convert.py -hess / -h1 / -hreg / -sh`; `--out_scales yaqa`
- **Result:** differences from ours: upstream sums d dᵀ over **every** token on the GPU (layer ranges per run, for memory); we keep 32 sampled positions per sequence (1.6%) as rows on disk. It weights tokens by xᵀ H_in x (Sketch A), we by ‖x‖² (H_in = I); bf16 activations with fp32 sums. Its notes (Qwen3.5-2B): the benefit comes from the output side alone; H_out is "dominated by relatively few high-gradient tokens"; two seeds differ ~20% at 250 full 2048-token rows. So our 6,400 sampled rows likely miss most of the mass. turboderp's SC quants take their GPTQ Hessians from the self-sampled trace (`convert.py --cal_data`), i.e. model-written text only; ours cover every calib token, prompts included.
- **Decision / next:** if gate 2b (fp32, sampled) fails too: full-token sums for blocks 22/23 (≈ 6 GB of fp32 accumulators; no room on the cards beside fp32 activations, so on the GPU box's CPU, ~10 s a sequence) before giving up.

### 2026-10-31 ~22:30 CST · #33-06 · Gate 2d failed: diagonal output scales hurt
- **Status:** negative · **Setup:** one-sided LDLQ with exllamav3's sensitivity_out_scales from diag(H_out), blocks 22/23, vs one-sided.
- **Result:** sampled-H_out diagonal K2 +14.9% (z 7.1); full-token diagonal K2 +10.2% (z 5.9), K3 +4.3% (z 3.0). Output-side Hessian information hurts in every form we tried.
- **Decision / next:** only gradient-weighted H_in (gate 2e) is left of the backward-based ideas.

### 2026-10-31 ~22:30 CST · #42 · Shared-input siblings do not share `suh`
- **Status:** done · **Machine:** CPU (our K6 quant's dumped suh, every layer)
- **Result:** signs agree 100% (one seed per input-Hessian group), but the per-channel input scales differ: gate vs up median |ratio| 1.015 (p99/p1 spread 1.27), GDN qkv vs z 1.036 (1.19), attn q vs k / v 1.092 (1.92); no pair identical. The difference is exllamav3's per-tensor scale refit (`refit_scales`).
- **Decision / next:** on V2's quants each sibling needs its own rotate + int8-quantize prep pass (cheap: ~20 MB of activations at M = 2048), not one shared tile. If shared tiles pay in the time table, the Phase-B requant can tie su across siblings (joint refit), at a quality cost still to be measured.

### 2026-10-31 ~22:20 CST · #33-05 · YAQA gate 2c failed: two-sided YAQA dropped
- **Status:** negative · **Setup:** full-token fp32 H_out (200 seqs, every position) on blocks 22/23; KLD increment vs one-sided.
- **Result:** K2 +30% (sigma_reg_out 0.025) / +7% (0.3); K3 +21% / +5%, all z ≥ 2.9. Full-token is even WORSE than sampled at 0.025 (+15% K2, +10% K3), equal at 0.3. Truncated backward checked exact (0 difference). Sampling was not the problem. Collector profile (seq 1): fwd 8.8 s, bwd 13.8 s, CPU-sum wait 11.3 s with H_in on (the CPU sums, not the GPU, are the bottleneck once H_in is added).
- **Decision / next:** two-sided YAQA dropped. Remaining backward arms: gradient-weighted H_in (gate 2e: crashed on a profiler-path bug before collecting, fixed, not rerun) and diagonal output scales (gate 2d, evals running).

### 2026-10-31 ~22:20 CST · #41 · Best-of-N rotation seeds: dead
- **Status:** null · **Code:** `seedtest.sh` (cpu2) · **Setup:** 9 tensors x 4 seeds x 2 row samples, K2–4
- **Result:** seed spread 0.5%; A-vs-B correlation +0.22; best-of-4 picked on A, measured on B: −0.2% / +0.2% / +0.0%.
- **Decision / next:** dead. No seed pick.

### 2026-10-31 ~22:20 CST · #38 · Damping vs K (`dampsweep.sh`)
- **Status:** done · **Machine:** cpu1 · **Setup:** 16 tensors; error vs undamped H; ratio to 0.025
- **Result:** attention inputs (q/k/v) gain with less damping, more at high K: K4 0.970 (0.0025) / 0.961 (0.001), K8 0.938–0.955. GDN inputs, out-projections, FFN: 0.994–1.000 at every K. 0.05 is worse everywhere. Damping does not explain turboderp's K6–8 wins (attention inputs are a small share).
- **Decision / next:** settings 0.005 (or 0.0025) for attention inputs; 0.01 elsewhere is harmless. Depth extension running (cpu1).

### 2026-10-31 ~22:20 CST · #39 · Gain (global scale) vs K (`gainsweep.sh`)
- **Status:** running (GPU gaintest) · **Machine:** cpu2 · **Setup:** 16 tensors, RMS x f, f = 0.8–1.8
- **Result:** the best factor rises with K in every family: ~1.1 to K5, 1.2 at K6, 1.3 at K7–8. Out-projections (attn_output, ssm_out) more: 1.2 (K5), 1.3 (K6), 1.4 (K7), 1.5 (K8; range to 1.8). A fixed 1.1 costs x1.05 (K6), x1.2 (K7), x2 (K8) error; for out-projections x1.2 (K6), x1.7 (K7), x4 (K8).
- **Decision / next:** this is the CPU tool's RMS scale, not exllamav3's own g_scale search. The GPU gaintest (running) shows whether exllamav3's choice already tracks this. If it doesn't, it is a strong candidate for the high-K gap.

### 2026-10-31 ~22:10 CST · #36 · Fractional-K trellis works on real tensors (`v3/e3/fracsweep`)
- **Status:** done (on the curve; → qtf2, #37) · **Machine:** CPU box A · **Setup:** 14 tensors of layers 10 / 11 / 42 / 43, every family; 128 sampled rows, same rows at every K (`fracreport.py`)
- **Result:** measured proxy error over the log-linear interpolation of the integer neighbours, median: K1.25 0.990, K1.5 0.988, K1.75 0.990, K2.25 0.999, K2.5 1.000, K2.75 1.002, K3.5 0.998, K4.5 0.996 (all tensors within 0.95–1.02). Integer slopes match the hproxy table (K1→2 2.12, K2→3 1.99).
- **Decision / next:** the shift-pattern trellis costs nothing over the integer ones, so the −5…−7% menu estimate stands. Sub-1 (K0.5 / K0.75) can't be judged this way (K0 = dropped is not a curve point); they sit well below the line and need real swaps.

### 2026-10-31 ~21:50 CST · #44 · Sub-2-bit pricing on turboderp's tensors (CPU, V2 data)
- **Status:** done · **Setup:** ours, all 337 tensors; V2's fitted model
- **Result:** proxy error per bit (ours, median log2 ratio): K1→K2 2.11, K2→K3 2.00, then 1.97 … 1.88. V2's fitted model (q = 1.004, one factor for K1–K3) under-predicts the three K1-dominated band swaps by 20% in-sample / 36% CV (K2 5%, K3–K8 ±1–4%). A refit with its own K1 factor: K1 1.35 vs K2–K3 1.09 per unit proxy error; CV log-RMSE 0.870 -> 0.845, pair ranking 3,115 -> 3,119 / 3,786. Its solutions use far fewer K1 tensors (1.5 bpw 180 -> 105, 1.75 106 -> 52, 2.0 54 -> 21). The K1→K2 KLD slope is ~2.3–2.5 bits per bit, as turboderp notes for this model (~2.35).
- **Also:** what V2 takes from turboderp at high bpw: mostly the tensors his recipes raised to K7 / K8 (6.0: 37 of 49; 8.0: 36 of 42). The fit prices our K7–K8 at 1.58x KLD per unit proxy error (K4–K6 1.25x, his ~1.0x). His per-tensor S (`kld_table.json`) vs our fitted K4 cost: Spearman 0.71 overall; within families weak. Not worth a per-tensor prior.
- **Decision / next:** V3: K1 and each sub-2 level get their own factor and must be measured; K0.25 / K0.75 only if K0.5 earns picks. Thin evidence: 3 K1 swaps. Final kfit2 model dump: `data/v3/kfit2-model-final.json`.

### 2026-10-31 ~21:00 CST · #49 · radiance PR #362 (prefill softmax rescaled per row) on our eval
- **Status:** done · **Code:** `eval/pr362/run.sh` (golden-ver1, V2 3.00)
- **Result:** KLD −0.12% (unpacked) / −0.13% (packed), z ≈ −2.2; 6.00 −0.5% ± 1.0%. Packed vs unpacked differs by +0.007% today, −0.001% with the fix.
- **Decision / next:** real but ~10x below the ±1.5% decision noise: no V2 action; take it with the 1.15.x port (V3 re-baseline then).

### 2026-10-31 ~21:00 CST · #33-03 · YAQA: bf16 backward is too noisy
- **Status:** done · **Code:** `noise.sh`, `cmp.py` (same 8 sequences, positions and labels; rows kept in fp32)
- **Result:** per-row cosine bf16 vs fp32 for down_proj: layer 63 median 1.000 (p10 0.999), layer 40 0.998 (p10 0.66), layer 20 0.988 (p10 0.02), layer 2 0.92 (p10 < 0). The bad rows are 2.7–3.2x too large, so they dominate H_out. H_out diagonal cosine 0.88 at layers 48–63, 0.36 at 0–15. The bf16 forward itself diverges on 2/8 sequences (seq 7 loss/token 1.03 vs 0.83). fp32 mode: 30 s per sequence (5.3 s bf16) after a ~15 min warm-up, 28.4 GiB peak per card.
- **Decision / next:** likely cause of gate 2's failure (blocks 22/23, diag cosine ~0.5).

### 2026-10-31 ~21:00 CST · #35 · Embedding menu, token-frequency weighted (`v3/e3/embfreq.py`)
- **Status:** done · **Setup:** the sweep's saved per-row errors x search + qad token counts
- **Result:** the error curve is unchanged (x1.40 per quarter step; frequency-weighted error +1–3%). Per-row scale still wins to K4 (G1024 x1.011 / 1.006 / 1.003 at K2/3/4, frequency-weighted); from K5 G1024 is 0.5–1% better frequency-weighted. Caveat: 48% of the 4096 random rows never occur and the rest carry 1.2% of the tokens, so the frequency weighting rests on few rows. Group-scale recheck final (16 tensors): same as before.
- **Decision / next:** per-row only (a 1% embedding-error difference at K ≥ 5 is far below anything measurable in KLD).

### 2026-10-31 ~20:40 CST · #35 · Embedding menu sweep done (`v3/e3/emb/run3.sh`, 4096 rows)
- **Status:** done · **Machine:** cpu2 · **Files:** `v3/e3/data/embmenu.jsonl`
- **Result:** one scale per row wins at equal bits at every K: G1024 is x1.015 worse at K1.5–2.5, x1.008 at K4, x1.003 at K5, even only at K5.75; G512 x1.01–1.02 worse throughout. The quarter-step curve is smooth (x1.40 error per quarter step at every K, i.e. ~3.9x per bit). The best scale factor reaches the search edge (1.35) for 0.6–3% of rows at K5–6 only.
- **Decision / next:** embedding menu = quarter steps, one bf16 scale per row, no group options.

### 2026-10-31 ~20:40 CST · #40 · Per-group weight scales stay dead at low K, Hessian-weighted (`v3/e3/gscale.py`)
- **Status:** negative · **Machine:** cpu1 (12 of 16 tensors so far)
- **Result:** the raw gain at equal K is only 1.3–1.8% (ratio 0.982–0.987 at K1–2; at K0.5 groups are even slightly worse, 1.008–1.035). After the 8/G-bit scale overhead groups lose: x1.08 (G128) / x1.17–1.20 (G64) worse at equal bits, K1–2. Confirms the Oct-30 amgm null result where it was untested (Hessian-weighted, K ≤ 2).
- **Decision / next:** dead at low K.

### 2026-10-31 ~20:20 CST · #— · Overnight queue (TP2 to ~13:00 CT Nov 1), all on gpubox, ordered by locks
- **Decision / queue:**
  1. gate 2c (running) -> `gate2e.sh`: blocks 22/23 again with gradient-weighted H_in (`collect.py YAQA_HIN`); if 2c passed, also 400 sequences with snapshots at 100/200/400 and split halves (`hhalf.py`) = the sample-count test.
  2. quarter-step smoke test, gate 2d, head rerun (card 0 / both).
  3. `v3/levels/chain0.sh` (card 0): gain sweep (layers 3/10/43, K2/4/6) and high-K damping sweep (K6/8), then waits for `/data/v3levels/settings.env` (default `--sigma-reg-in 0.005` at 2:30 AM CT) and builds the levels K2.25…K4.75 (exports, trunk only).
  4. `v3/levels/chain1.sh` (card 1): real trellis embedding K2–4 vs Lloyd e2–e5 on V2 6.00 (fit3 + ml); ours vs turboderp K6 (even layers on an ours-K5 base), fit3.
- **Also:** store index done: 53 packs; his plain6.00 / 5.00 packs are uniform K6 / K5; 3–6 of his tensors per pack in some sizes are byte-identical to another of his packs (counted once).

### 2026-10-31 ~20:00 CST · #36 · High-K half steps land on the curve (`fracsweep-hi.sh`)
- **Status:** done · **Machine:** cpu1 · **Setup:** 18 tensors of layers 0 / 10 (GDN), 11 / 63 (attention), 43 (MLP); scale search 0.85–1.75; `v3/e3/data/fracsweep-hi.jsonl`
- **Result:** error / log-linear interpolation of the integer neighbours, median [range]: K5.5 0.997 [0.993, 1.042], K6.5 1.003 [0.980, 1.010], K7.5 1.006 [0.994, 1.021]. Integer steps stay steep: ×3.77 / ×3.67 / ×3.81 error per bit at K5→6 / 6→7 / 7→8. Best factor rises with K (1.15 to K6, 1.3 from K6.5–7, some 1.5 at K7.5–8). K0.25: proxy rel err median 0.36 [0.05, 0.59] (1.0 = dropped): judge only by real swaps. 3 jobs failed on a wrong tensor list (layer 43 is full attention, has no GDN tensors); not rerun, every family is covered.
- **Decision / next:** the K5–8 half steps of the menu are sound.

### 2026-10-31 ~20:00 CST · #34 · LM head sensitivity (Fisher diagonal, `v3/yaqa/headsens.py`)
- **Status:** running · **Setup:** exact Fisher diagonal, sum_t p(1−p), from the calib golden's top-512
- **Result:** ~400 of 248,320 rows hold 50% of the mass; 16–19k hold 90%; 68–90k hold 99%. The four pools agree (verify included).
- **Decision / next:** supports a row-slab head; slab test next (needs `ourq.py --save-wq` head quants).

### 2026-10-31 ~19:50 CST · #33-02 · YAQA gate 2 failed with the bf16 pilot
- **Status:** negative · **Setup:** 200 sequences, blocks 22+23 swapped into the V2 6.00 release, golden-ver1; base 0.000703
- **Result:** KLD increment over base: K2 one-sided +0.00290, two-sided +0.00357 (+23%); K3 one-sided +0.00067, two-sided +0.00086 (+28%). Suspects: bf16 noise (confirmed in 33-03), 6,400 rows for the 17,408-wide H_out, output regularisation 0.025.
- **Decision / next:** rerun in fp32 with sigma_reg_out 0.025 and 0.3 (`gate2b.sh`).

### 2026-10-31 ~19:45 CST · #— · Mining merge acted on (`local/mine/merged-2026-10-31.md`, via `mergep.sh`)
- **Decision:** DESIGN-v3 corrected (YAQA status and costs: 3.7M qad tokens, ~0.2 TB full H_out; head K priced by KLD + bytes; his S not a prior) and the merge's checks added to §6. Queued on gpubox behind gate 2c:
  - `gate2d.sh` (merge D1): diagonal-only output weighting, one-sided LDLQ with exllamav3's sensitivity output scales from diag(H_out) (`ourq.py --hout-diag`), blocks 22/23 K2 / K3, paired vs one-sided.
  - `headrun2.sh` (merge A2, replaces the invalid headrun): Fisher ranking from the qad pool only (top-set overlap 0.93–0.98; fit-pool Fisher captured in the top 1,280 rows 0.610 vs 0.613), floor K3, budgets K4 / K5 / K6, judged by `headsim.py` (exact head-only KL on the 32,768 calib-A head inputs; paired vs uniform, 512-position block bootstrap). Whole-model check for the finalist only, with the fixed plugin.

### 2026-10-31 ~19:15 CST · #34 · Head slab test INVALID (plumbing bug); rerun pending
- **Status:** blocked (rerun waits for YAQA gate 2c and the cards) · **Machine:** card
- **Result:** every spliced variant (u3 / u4, slabs, Lagrange) scored KLD ~15 with top-1 0.0002 (V2 3.00 itself: 0.027). The donor heads are right (rows vs the bf16 head: K4 rel err 0.076, cos 0.997; K3 0.15), so quantization and the un-permute work. Cause: `qwen35_exl3`'s fallback for an unquantized head used `gemm_nt`, whose output is the activation dtype (bf16), written into the f32 logits buffer; never exercised before (V1 / V2 heads are always EXL3). Fix: `logits_gemm` as in the bf16 arch (radiance-hero 9a156d0, local).
- **Decision / next:** rerun = dev build of the plugin + the 14 evals (~20 min, both cards) after YAQA gate 2c. Plan numbers (proxy) are unaffected.

### 2026-10-31 ~19:10 CST · #33-05 · YAQA gate 2c started (`v3/yaqa/gate2c.sh`)
- **Status:** running · **Setup:** full-token fp32 H_out for blocks 22/23 on the pilot's 200 windows and labels; backward truncated at layer 22 (gradients at/above unchanged; checked against an untruncated 2-sequence run), CPU sums async. Then two-sided K2 / K3 at sigma_reg_out 0.025 / 0.3 vs one-sided and vs gate 2b's sampled two-sided.
- **Result (build):** first fracsmoke build failed (torch's ROCM_HOME = pip runtime without hipBLAS / thrust headers); now ROCM_HOME=/opt/rocm/core-10.1 as for the exllamav3 build.

### 2026-10-31 ~19:00 CST · #36 · Quarter-step quantizer instances written (exllamav3 branch `rxm-frac-quarter`)
- **Status:** running (not yet built) · **Code:** worktree `/workspace/forks/exllamav3-frac`; `v3/e3/exl3frac`; `ourq.py --frac-ext SRC`; `v3/e3/fracrun.sh`; `fracsmoke.py`
- **Result:** 15 instances for the V3 menu (KA 1–4 x 0x8888 / 0xAAAA / 0xEEEE, KA 5–7 x 0xAAAA), one compile unit per KA; `frac_k` derives the canonical even mask for any K = KA + n/16 (n even); exllamav3's own decode refuses non-0xAAAA rates instead of misdecoding. The live exllamav3 is a prebuilt 159 MB .so (no build tree), so the fractional kernel is built as a side extension and swapped in (prebuilt pack / unpack already take any mask).
- **Decision / next:** `fracrun.sh` (gpubox, queued behind the head test) builds it and runs `fracsmoke.py`: every menu K on 8192 Gaussian tiles: on-curve vs integer neighbours, GPU vs CPU Viterbi error, pack/unpack round trip, decode check.

### 2026-10-31 ~19:00 CST · #33-01 · YAQA gate 1 passed (`v3/yaqa/collect.py`)
- **Status:** done · **Code:** `collect.py`, `fold.py`, `ourq.py --hout DIR`, `gate2.sh`
- **Result:** bf16 text model streamed tensor by tensor onto both cards (layers split 32 / 32; the transformers loader OOMs host RAM on the 30 GB box); fla 0.6.0 chunked gated-delta, gradient checkpointing, chunked sampled-label CE (labels drawn from the model's own distribution, as YAQA's Fisher). Per sequence records r_t = ‖x_t‖ dy_t at 32 random positions for all 496 trunk linears (≈233 MB bf16). 5.0 s per 2048-token sequence including writes; peak 26.9 GiB per card. The qad pool (1,819 windows, 3.7M tokens) is ~2.7 h and ~425 GB of rows.
- **Decision / next:** gate 2 (`gate2.sh`): 200-sequence pilot H_out (6,400 rows: rank-deficient for the 17,408-wide gate / up, which leans on sigma_reg_out), blocks 22+23 at K2 / K3, one- vs two-sided, swapped into V2 6.00, golden-ver1.

### 2026-10-31 ~18:55 CST · #36 · Fractional-K at the menu's ends (`fracsweep-hi.sh`, 21 tensors)
- **Status:** running (~1.5 h) · **Machine:** cpu1
- **Setup:** 21 tensors = the 14 of fracsweep + layers 0 and 63: K5–8 half steps and K0.25, K4.5 / 5 overlapping.
- **Result (while setting up):** `trellis.py`'s scale search {1, 1.15, 1.3} hit its top edge at K7.5 and its bottom edge (1.0) for most tensors at K ≤ 2 in the earlier sweep, so those low-K proxy numbers may be slightly pessimistic (relative K steps less affected; real quants search their own scale). Now `E3_FACTORS`, here 0.85–1.75.

### 2026-10-31 ~18:45 CST · #35 · Embedding menu sweep (`v3/e3/emb/run3.sh`, 4096 sampled rows, 47 runs)
- **Status:** running (~1 h) · **Machine:** cpu2
- **Result:** quarter steps K1.5–6 x {one scale per row, G1024}, + G512 from K4; per-row error / energy / chosen factor saved (tail rows, later token-frequency reweighting). Smoke test: at K5.75 9% of groups took the top factor 1.25, so the scale search is now 0.85–1.35.

### 2026-10-31 ~18:35 CST · #35 · Embedding scale group size (cpu2, 1024 rows, K2 / 3 / 4)
- **Status:** running (menu sweep pending) · **Machine:** cpu2
- **Result:** error vs G128 = 1.016–1.018 at G256, 1.02–1.025 at G512, 1.022–1.031 at G1024, 1.023–1.040 at one scale per row (G5120). Scale overhead 0.125 -> 0.003 bit/value; on the trellis curve (~4x error per bit) 0.12 bit is worth ~18% error, so per-row wins by ~15% at equal bits. Per-row vs G1024: 0.013 bit (~1.8%) saved against +0.1 / +0.2 / +0.9% error at K2 / 3 / 4.
- **Decision / next:** menu: per-row scale, G1024 where it is on the hull; G1024 likely wins only around K5–6 (the menu sweep settles it).

### 2026-10-31 ~18:25 CST · #33-04 · YAQA gate 2b failed too (fp32 backward)
- **Status:** negative · **Setup:** fp32 backward, 200-sequence sampled-row pilot, blocks 22+23 into V2 6.00, golden-ver1, paired vs the same one-sided quants.
- **Result:** KLD increment over base, two-sided vs one-sided: K2 +17% (sigma_reg_out 0.025) / +6% (0.3); K3 +19% / +10% (all z ≥ 2). bf16 gate 2 was +23% / +28%: fp32 helped, and more output regularisation helps further, but the trend runs toward one-sided (infinite regularisation), i.e. our sampled H_out still hurts.
- **Decision / next:** remaining suspect: 32 sampled positions a sequence (1.6%) miss the few high-gradient tokens that dominate H_out (upstream's note). Last test before dropping YAQA: full-token sums for blocks 22/23 (`collect.py YAQA_FULL=22,23`, fp32, CPU accumulators), ~1.5 h with quant + eval. Not run tonight (head test has the cards).

### 2026-10-31 ~18:05 CST · #35 · Trellis embedding vs V2's Lloyd tables (`v3/e3/embtrellis.py`)
- **Status:** running (menu sweep to follow) · **Machine:** cpu2 · **Setup:** 1024 random rows, H = I, same fwht128 + per-(row,128) scale · **Files:** `v3/e3/data/embtrellis.jsonl`
- **Result:** relative error at equal bits, trellis / Lloyd: 0.70 (1 bit), 0.56 (2), 0.51 (3), 0.53 (4), 0.51 (5). Trellis K matches Lloyd at K + 0.46–0.49 bits for K ≥ 2 (DESIGN's "trellis-3 ≈ Lloyd-4" was too optimistic: trellis-3 ≈ Lloyd-3.5). ~0.023 bpw saved at equal error, ~1% KLD at ~3 bpw by V2's fitted embedding costs.
- **Decision / next:** quarter steps adopted.

### 2026-10-31 ~04:30 CST · #32 · Vocab-tiered embedding dropped
- **Status:** negative · **Setup:** early data (embedding tiers, multilingual guard slice)
- **Result:** 30k@3 + rest@1: KLD 0.0140 vs uniform e4 0.0081 (bf16 embedding 0.0080). 30k@3 + rest@2 (2.25 b/value) 0.0094 vs uniform e3 (3.125) 0.0086. On the multilingual guard slice 30k@3 + rest@2 is +40% over bf16 (uniform e4 +4%). Rare-token rows matter more than frequency predicts.
- **Decision / next:** est. gain 0–3% at medium-low confidence, not worth the risk to international users. Remaining plans finish for the record; results go into EXPERIMENTS.md as a dead end.

### 2026-10-31 ~04:00 CST · #— · IQR-V3 published (competitor)
- **Status:** n/a · **Source:** whitepaper and HF `StillDeadcode/qwen3.8-27b-iqr-v3`; kernels in radiance 1.15.0, ABI 32. Board numbers unchanged (already scored).
- **Result:**
  - Convergence on EXL3's design: computed trellis code (3 ALU ops from a 16-bit state, like EXL3's mul1), L = 16, one f32 scale a row (EXL3: svh). Their code values are int8 (q(s) in [-126, 126]), so the Viterbi searches against the exact integers the int8 tile computes. Our E3 found mul1 rounded to int8 costs x1.25 at K7 / x4.5 at K8; an int8-native code removes that at the source.
  - Prefill fold: one scale a row (applied at the store) and a per-128 block factor shared by all rows: their int8 tile folds like FP8: 1.00–1.06x FP8 time vs 1.19–1.21x with per-(row,128) scales.
  - Head-fed projections (attn_output, ssm_out, 8% of trunk) need a larger scale gain from 5 bits (1.36–1.6 vs 1.16–1.24) and still lose to block scales at 6–7 bits.
  - GPTQ damping 0.05 -> 0.01: −3.5% KL on 8 layers at 3 bits. exllamav3 adds sigma_reg 0.025 x *mean* diag(H) to every channel, ≈ 22% of the median channel's curvature with massive-activation channels (mean ≈ 9x median): the same over-damping.
  - Their GEMM tables (decode GEMV vs FP8 per width, prefill per width) are V3's speed reference.
- **Decision / next:** check our K5+ quants of attn_output / ssm_out. Cheap test: one layer's tensors at 0.025 / 0.005 / 0.0025, hproxy error. If it holds, a requant of ours K2–6 (~6 h card time) is a V2.1 / V3 quality step.

### 2026-10-31 ~04:00 CST · #74 · radiance 1.15.0 / ABI 32 port: plugins must be rebuilt and re-validated
- **Status:** planned · **Code:** exl3r4d, exl3dense, qwen35_exl3 (PR #292 on 1.13.0)
- **Decision / next:** rebuild and re-validate (byte-identical KLD) before V2 ships; V2 should target 1.15.0.

### 2026-10-31 ~03:10 CST · #32 · Vocab-tiered head, step 1 (simulation; `quantize/vtsim.py`)
- **Status:** running (embedding part next) · **Code:** `mix/headdec`
- **Setup:** ranking from token counts on search + qad pools (7.16M tokens; top 30k ids = 94.4% of tokens, 51% of ids unseen). Head inputs: 32,768 rows of the head's input on calib-A (fit pool, disjoint from the ranking), all-K6 model, via the E1b hook's `EXL3_AQ_DUMP_ONLY=336` (dump run KLD 0.00101 = the plain run). Our head decoded at K1–8. Validate first: uniform-K sim deltas vs the measured whole-model head swaps (sc4.00-e4 base: K8 0.00764, K7 +0.00002, K6 +0.00009, K5 +0.0004, K4 +0.0016, K3 +0.0066; 0.047 bpw per K step).
- **Result:** head result (`data/vtsim-head-2026-10-31.json`): small. The sim matches the measured swaps once logits are f32 (the first run used f16 logits: a ~0.0002 KL floor). Uniform head-only KL: K3 0.0074, K4 0.0018, K5 0.00047, K6 0.00012, K8 0.000012. Tiering saves only 0.01–0.03 bpw at equal KL, or cuts head KL 30–45% at equal bytes (K4 budget: 0.0018 -> 0.0010, top-1 97.7 -> 99.3%). Against the head the mixes now pick (2.0 K3, 2.5–3 K4, 4 K5, 4.5–5 K6, 6 K8) that is ~1.5–3% of total KLD. The tail cannot go below K3 (tail K2 under 100k@K4: KL 0.0067 vs 0.0018).
- **Decision / next:** not worth a head kernel / format change now (a 2-tier head stays a backlog item, ~2% gain). Embedding next: measured swaps on the same base e1 +0.015, e2 +0.0033, e3 +0.0006, e4 0, e5+ noise; `quantize/embtier.py` (per-row tier embq) on a bf16-embedding copy of the base, 3 validation + 8 tier plans queued as `vt-emb-*` on calib-A. Validation: uniform 2-bit sim 0.01096 vs the real e2 container 0.01131 on calib-A (fake-quant ≈ real).

### 2026-10-31 ~01:50 CST · #19 · V2 side-experiments for V3: per-layer edge offsets and a learned residual re-ranker
- **Status:** negative (both) · **Code:** KFIT_EDGE (kfit2); `search/rr/rr.py`
- **Result:** per-layer edge offsets in the fit: no ranking gain. Learned residual re-ranker (ridge / XGBoost no better than kfit2; GP too slow). Both logged in EXPERIMENTS.md.
- **Decision / next:** a re-ranker is worth retrying in V3 with more data. Edge offsets only if V3's new terms (activation x depth) need per-layer freedom.

### 2026-10-31 ~01:50 CST · #21 · 47 single-layer probes: the sensitivity spike is the last layer
- **Status:** null (layer 0 hypothesis wrong) · **Setup:** 47 single-layer probes
- **Result:** the sensitivity spike is the last layer (L63 ffn_gate_up), not layer 0.

### 2026-10-31 ~01:50 CST · #— · V3 backlog from the probes: K0 / K0.5 / K1.5 / K2.5 and half steps 1–8
- **Decision / next:** backlog: fractional trellis levels (K0, K0.5, K1.5, K2.5 and half-steps 1–8). Not started.

### 2026-10-31 — · #32 · International-user guard for the vocab-tiered head and embedding (plan)
- **Status:** running (plan) · **Result (plan):** calib is ~94% English / code and only 51% of ids ever occur in it, so a calib-ranked low-bit tail could hurt other languages without the score showing it. Plan: (1) language-neutral ranking `quantize/vtguard.py`: min(calib rank, token id) (keeps 99.1% of calib mass in its top 100k; calib ranking 99.6%); (2) 2-bit tail floor as the default (`vt-emb-g-*` plans; 1-bit tail also measured); (3) guard slice `golden-ml` = all 67 multilingual docs of the fit pool (77k tokens; out of sample for the ranking, which used search + qad); every variant measured on it as `vt-ml-*`, with base-bf16 and e4 refs; (4) calib-ranked / 1-bit-tail plans ship only as an explicit "English/code-tuned" variant with an asterisk, unless the ml slice shows they cost international text about nothing.

### 2026-10-31 — · #29-01 · E3b results: int8 / int4 trellis codebooks (tsweep2) — topic summary
- **Status:** done (int8 codebook free through K6; int4 grid ×1.01 at K2, ×1.07 at K3, ×2.7 at K4) · **Machine:** CPU (`v3/e3/data`)
- **Setup:** tsweep2, all 11 tensors of layers 10 and 43, 256 rows; then the full set (layers 0 / 10 / 31 / 43 / 62 / 63, 33 tensors).
- **Result:** error relative to EXL3's float mul1 codebook (layers 10 / 43, 256 rows):

| codebook | K2 | K3 | K4 | K5 | K6 |
|---|---|---|---|---|---|
| mul1 rounded to **int8** | ×1.000 | ×1.000 | ×1.001 | ×1.002 | ×1.004 |
| Gaussian int8 table | ×1.008 | ×1.008 | ×1.108 | ×0.975 | ×0.969 |
| mul1 on the **int4** grid | ×1.010 | ×1.065 | ×2.72 | ×10.5 | ×39.6 |

  Full set (33 tensors): mul1 rounded to int8, K2–K6 ×1.000–1.004 (range 0.99–1.01); **K7 ×1.25, K8 ×4.5** (9 tensors). Int4 grid: K2 ×1.01, K3 ×1.07, K4 ×2.7. W4A4 keeps trellis quality at K2–K3; only the activation side costs there.
- **Decision / next:** int8 compute keeps all of EXL3's weight quality through K6. At K7–K8, 8-bit weight values cannot hold the code: V3's top end needs f16 compute (or a 9-bit-ish decode) for K ≥ 7 tensors. The earlier "K ≥ 5 requant" estimate was for post-hoc rounding of existing quants; the searched codes say K5–6 are fine at int8 when quantized with the rounded codebook.

### 2026-10-31 — · #41 · Best-of-N rotation seeds (dead end)
- **Status:** null · **Code:** `v3/e3/seedtest.sh`
- **Result:** pick each tensor's rotation sign seed from 4 by proxy. Seed spread 0.5%, not reproducible across row samples (corr +0.22); honest gain ~0.
- **Decision / next:** dead.

### 2026-10-31 — · #40 · Per-group weight scales, finer than EXL3's per-row and per-column scales (dead end)
- **Status:** negative (first test 2026-10-30; retest 2026-10-31)
- **Setup:** first test: EXL3's rotated domain (128-Hadamard both dims, random signs) after per-row / col RMS normalisation, 0.5·log2(AM/GM) of group variances, 23 layers x all projections + head (`quantize/amgm.py`, `logs/amgm.jsonl`). Retest (`v3/e3/gscale.py`): Hessian-weighted, real trellis + LDLQ, K0.5–2, G64 / G128 8-bit scales.
- **Result:** first test 0.000–0.0025 bit/weight (MLP ≈ 0, worst tensor 0.006). Retest: raw gain 1.3–1.8% at equal K, so groups lose 8–20% at equal bits. Dead at low K too.
- **Decision / next:** rotation + row / col scales already flatten the variance. Not tested: heavy tails.

### 2026-10-31 — · #22 · GPTQ damping for our quants (deferred, not a dead end)
- **Status:** superseded (by #38) · **Code:** `quantize/damptest.py`, `data/damptest-2026-10-31.jsonl`
- **Result:** sigma_reg 0.025 -> 0.005 changes per-tensor error (undamped H) by −2 to −2.7% on GDN in_proj_qkv / z, −3 to −6.5% on attention q/k/v (layer 3), −0.5 to −1.2% on o_proj, ~0 on ssm_out and the FFN (70% of weights). Est. ~0.5–1% whole-model KLD.
- **Decision / next:** not worth a V2 requant. Use 0.005 for attention / GDN inputs in the V3 requant.

### 2026-10-31 — · #19 · Per-layer edge offsets in kfit2 (KFIT_EDGE)
- **Status:** negative
- **Setup:** a free, shrunk log-offset per (group, layer) in layers 0–3 / 60–63 on top of the 8-layer depth curve, plus a variant with knots every 6. All 47 single-layer K2 probes in the data.
- **Result:** CV pair ranking 1614 / 1618 vs 1620 / 1898 for the plain model: no gain. The probe data itself (in the plain fit) carries the allocation signal, e.g. the L63 ffn_gate_up spike.
- **Decision / next:** code kept behind the flag.

### 2026-10-31 — · #20 · Residual re-ranker on top of kfit2 (`search/rr/rr.py`)
- **Status:** negative
- **Setup:** learn log(measured / kfit2 out-of-fold prediction) from 102 summary features; re-rank candidates. Grouped 5-fold CV on 519 mixes.
- **Result:** plain kfit2 log-RMSE 0.118 / pair ranking 79.9%; ridge 0.157 / 76.4%; XGBoost (all measurements) 0.212 / 80.4%; XGBoost (mixes only) 0.127 / 80.5% (+7 pairs of 1130: noise); a Gaussian process with ARD did not finish in 16 min. The residual (~±12%) is mostly slice / run noise, not learnable structure at this data size.
- **Decision / next:** revisit with V3's larger dataset (retry rule in BACKLOG).

### 2026-10-31 — · #32 · Vocabulary-tiered embedding (`quantize/embtier.py`; data in the new-format log)
- **Status:** negative · **Setup:** rows ranked by token frequency; top tiers at 3–4 bits, tail at 1–2 bits; fake-quantised on a bf16-embedding copy of sc4.00-e4; calib-A + a multilingual guard slice (67 fit-pool docs).
- **Result:** uniform e2 +38% / e3 +8% / e4 +2% calib-A KLD over a bf16 embedding (multilingual +64 / +20 / +4%). Best tiered, 20k@4 + 30k@3 + rest@2 (2.41 bits a value): +11% / +28%: better than uniform at equal bytes, but against the e3 the mixes use it saves 0.034 bpw for +3% KLD, net ~1–2% at 3–4 bpw, while multilingual text pays more (+28% vs +20%). 1-bit tails are bad everywhere (+29–76%; multilingual +60–145%): rare-token rows matter far more than their frequency. A language-neutral ranking (min(calib rank, token id)) did not help the multilingual slice.
- **Decision / next:** dropped (not worth hurting international text). The head version: simulated, ~2% (backlog).

### 2026-10-31 — · #47 · R1 residual-stream rotation for int8 / int4 activations (dropped)
- **Status:** negative (heading says dropped; measured worse) · **Code:** `v3/e1b/r1test.py`; data `v3/e1b/data/r1.jsonl`
- **Setup:** SpinQuant-style R1: one random orthogonal 5120² Q folded into the residual stream, the readers' norm gains (1 + w) folded into their weights. CPU proxy on the K6 activation dumps, all 208 residual-reader linears, output-error energy vs today's block Hadamard with the quant's own suh.
- **Result:** A8-g128 6.6x worse; A8 per-token 2.1x; A4-g64 5.7x (layers 0–15: 17x / 4.9x / 15x; layers 48–63: 1.3x / 0.66x / 1.3x). R1 does spread the massive-activation channel (top channel's share of input energy 65% -> 0.3% in layers 0–15), but this model's norm gains shrink that channel, and R1 must fold the gains into the weights, so the rotation spreads a much larger outlier over every 128-block. Only per-token A8 in deep layers gains, and per-token is ~5x worse than g128 there to begin with. A learned Q (SpinQuant's Cayley optimisation) would not close a 6.6x gap. Side result: the quant's learned input scales (suh vs signs only) lower the A8-g128 error 5%, per-token 35%.
- **Decision / next:** dropped.

### 2026-10-30 ~21:00 CST · #31 · First real int8 prefill kernels run end to end
- **Status:** running · **Code:** `v3/nf/kern/exl3r4d`, runner `/root/e1b/nfkrun.sh` · **Machine:** card
- **Result:** K6 all-int8: KLD 0.00178, eval 74 s vs 100 s (f16). Detail in the E4 summary.

### 2026-10-30 ~21:00 CST · #31 · Post-hoc int8 rounding of EXL3 quants costs at K ≥ 5; requant planned
- **Status:** running
- **Result:** the +0.0003 over fake-quant is the post-hoc int8 rounding of an existing EXL3 quant, matching the paper bound (K4 +4%, K5 +17%, K6 +60% weight error).
- **Decision / next:** requant planned (K5–8 with the int8-rounded codebook).

### 2026-10-30 ~21:00 CST · #30 · V0 validation: the weight model is off at 2 bpw; activation increments track
- **Status:** running · **Result:** ours-K1 / K2 overrated at 2 bpw; fixed by the r9 refit. Activation increments track.

### 2026-10-30 ~21:00 CST · #30 · K3: the new-format solver (`v3/nf/knap3.py`; topic summary)
- **Status:** running (V0 validation; later superseded by knap4, #45) · **Code:** `v3/nf/knap3.py`, model from `nf/kfit2dump.py`
- **Setup:** options per tensor: f16 (any EXL3 option or bf16); a8: ours-K1..8 trellis -> int8 x group {pt, 1024..64}; a4: ours-K1..3 on the int4 grid x group; shared-input linears share one mode + group. KLD: weights from kfit2's compounding model (551 measurements, dumped from round 9); activations additive, from E1c. Time: E2 TOPS at M = 4096 + a decode pre-pass (placeholder 0.006 ns/weight); non-GEMM work is 8% of today's step. Solve: min KLD + λ·time, bytes strictly under the target; layered DP with stage weights; ~2 s per point.
- **Result:** first frontier (prefill speedup vs today's all-f16 EXL3):

| target | all f16 | ×1.3 | ×1.5 | ×1.65 |
|---|---|---|---|---|
| 3.0 | 0.0256 | 0.0257 (+0.2%) | 0.0258 (+0.6%) | 0.0269 (+5%) |
| 4.0 | 0.00650 | 0.00657 (+1%) | 0.00665 (+2%) | 0.00707 (+9%) |
| 6.0 | 0.00063 | 0.00070 (+11%) | 0.00080 (+27%) | 0.00107 (+70%) |

  Readout: at 3–4 bpw, a8 is nearly free (90–95% of FLOPs at a8, a few sensitive tensors at f16); at 6 bpw the activation cost is a large share, so A16 matters; A4 is almost never chosen. Also at this point: the solver works; the V0 validation shows the weight model off at 2 bpw (fixed by the r9 refit); the bf16 head breaks mixes (disabled); the group size becomes a per-tensor option.
- **Decision / next:** caveats: the decode pre-pass time is a placeholder; g128 / g64 speeds are extrapolated. The KLD model is unvalidated for new-format mixes: it needs the E4 prototype to measure real mixes, then a refit.

### 2026-10-30 ~21:00 CST · #31 · E4: real int8 prefill kernels (`v3/nf/kern/exl3r4d`; topic summary)
- **Status:** running · **Machine:** card (build `/root/nfk`, runner `/root/e1b/nfkrun.sh`)
- **Setup:** kernels (per-token activation scales only so far): `nfq_act_i8` (quantizes the prep kernel's rotated rows to int8 in fragment order); `nfq_wdec_i8` (T2's trellis decode with int8 output, rint(v / 4 k_inv)); `nfq_gemm_i8` (iu8 WMMA core with T2's epilogue). Switched per tensor by `NFQ_MASK` ('8').
- **Result:** first end-to-end run (k0, ours-K6, calib-A, all 337 linears int8): **KLD 0.00178, top-1 98.13%, eval wall 74 s** (f16 100 s; fake-quant A8 per-token 0.00149). Correct first try; the whole eval 26% faster untuned (including load and scoring). The +0.0003 over fake-quant is the post-hoc int8 rounding of an existing EXL3 quant, matching the paper bound (K4 +4%, K5 +17%, K6 +60% weight error); E3b's ×1.004 needs the trellis searched with the int8 codebook.
- **Decision / next:** K ≤ 4 existing quants run as is; K ≥ 5 should be requantized with the int8-rounded codebook: a small change to exllamav3's codebook (same hash, rounded values), a ~6 h requant of K5–8 (or K1–8). Next: group scales in the real kernel, the requant, split-K for small N, and per-op timing (`--profile-ops`) for the real time table.

### 2026-10-30 ~21:00 CST · #31-01 · E4b: end-to-end prototype at ~4 bpw (topic summary; planned, no result yet)
- **Status:** planned · **Date:** the section is undated; placed at E4's date.
- **Setup (plan):** trellis -> int8, W8A8 everywhere except the A16 picks. Measure KLD and prefill / decode speed against ours-K4 EXL3.

### 2026-10-30 ~20:40 CST · #30 · Validation tooling and the first 12 validation evals
- **Status:** running · **Code:** `EXL3_AQ_GROUPS` (per-tensor group string in the hook); knap3 `KN_WRITE` (exports mix + masks); `nf/nfval.sh`
- **Result:** first 12 validation evals running.

### 2026-10-30 ~20:40 CST · #26-01 · Batches 4–7 in: A8 g256–g1024 plateau at +0.00015
- **Status:** done · **Machine:** card
- **Result:** A8 g256 / g512 / g1024 all about +0.00015–0.00018; only g128 and smaller go lower (+0.00005).

### 2026-10-30 ~20:40 CST · #26 · E1 / E1b: activation quantization cost (hook in `v3/e1b`; topic summary)
- **Status:** done (A8 g128 near-free; A4 g128 still a candidate at ≤ 3.5 bpw, pending E2) · **Machine:** card (`e1brun.sh`)
- **Setup:** our K6 EXL3 weights, scored on golden-calA (fit pool, 90.8k positions). Baseline KLD 0.00101, top-1 98.75%. Paired SE ±1.5%. The hook fake-quantizes the rotated GEMM input, one absmax scale per token row unless noted. No-op run byte-identical to the normal K6 score. Map: 337 linears in issue order.
- **Result, A8 (int8):**

| run | KLD | added | top-1 |
|---|---|---|---|
| all linears | 0.00149 | +0.00047 | 98.47% |
| all, **g128** activation scales | 0.00106 | +0.00005 | 98.71% |
| all, g32 | 0.00104 | +0.00003 | 98.68% |
| all, g256 | 0.00118 | +0.00017 | |
| all, g512 | 0.00119 | +0.00018 | |
| all, g1024 | 0.00116 | +0.00015 | |
| all, per-row, clip 0.9 | 0.00221 | +0.00120 | (clipping hurts A8) |
| per family, per-row: attn_in / attn_out / gdn_in / ssm_out | 0.00103 / 0.00100 / 0.00101 / 0.00105 | ≤ +0.00004 | |
| per family, per-row: gate_up / down / head | 0.00112 / 0.00103 / 0.00111 | +0.00011 / +0.00002 / +0.00010 | |
| all but down and head, per-row | 0.00124 | +0.00023 | 98.57% |

  A8 everywhere (per row) costs more than our best 7.5 bpw EXL3 mixes measure in total (0.00026–0.00035). IQR's 8.5 bpw I8A8 entry (0.00046) is consistent with this.

- **Result, A4 (int4):**

| run | KLD | added | top-1 | ~share of linear FLOPs |
|---|---|---|---|---|
| all linears | 0.150 | +0.149 | 83.9% | 100% |
| attn_out | 0.0085 | +0.0075 | 96.5% | ~2% |
| ssm_out | 0.0141 | +0.013 | 95.5% | ~5% |
| attn q/k/v | 0.0237 | +0.023 | 94.3% | ~5% |
| gdn_in (qkv, z) | 0.0241 | +0.023 | 93.9% | ~13% |
| down | 0.0262 | +0.025 | 93.7% | ~14% |
| gate_up | 0.0264 | +0.025 | 93.0% | ~28% |
| head | 0.0333 | +0.032 | 91.0% | ~10% |
| layers 0–15, all linears | 0.0295 | +0.028 | 93.8% | 25% |
| layers 16–31, all linears | 0.0199 | +0.019 | 94.6% | 25% |
| layers 32–47, all linears | 0.0305 | +0.029 | 93.5% | 25% |
| layers 48–63, all linears | 0.0407 | +0.040 | 91.9% | 25% |
| all linears, **g128** scales | 0.0312 | +0.030 | 92.3% | 100% |
| gate_up, g128 | 0.0100 | +0.009 | 95.8% | ~28% |
| gate_up, per-row, clip 0.85 / 0.7 / 0.55 | 0.0183 / 0.0146 / 0.0195 | +0.017 / +0.014 / +0.018 | | ~28% |
| gate_up, g128 + clip 0.85 | 0.0097 | +0.009 | 96.0% | ~28% |
| gate_up, g256 | 0.0127 | +0.0117 | | ~28% |
| gate_up, g256 + clip 0.85 / g512 / g512 + clip 0.85 | 0.0100 / 0.0144 / 0.0111 | +0.009 / +0.013 / +0.010 | | ~28% |
| gate_up g64 / g64 + clip 0.85 | 0.0092 / 0.0115 | +0.008 / +0.010 | | ~28% |
| down g64 / g64 + clip 0.85 | **0.0048** / 0.0052 | **+0.0038** / +0.0042 | | ~14% |
| down g256 / g256 + clip 0.85 | 0.0073 / 0.0064 | +0.006 / +0.005 | | ~14% |
| down g512 / g512 + clip 0.85 | 0.0088 / 0.0075 | +0.008 / +0.006 | | ~14% |
| down, g128 | 0.0070 | +0.006 | 96.8% | ~14% |
| down, per-row, clip 0.85 / 0.7 / 0.55 | 0.0205 / 0.0190 / 0.0365 | +0.019 / +0.018 / +0.035 | | ~14% |
| down, g128 + clip 0.85 | **0.0055** | **+0.0045** | 97.0% | ~14% |

- **Decision / next:** A8 with g128 scales is near-free for quality. A16 stays an option only for the very top of the Large tier, and only if E2 shows g128 iu8 is slow. A4 g128 stays a candidate at ≤ 3.5 bpw, pending E2 (iu4 at g128 measured 484 TOPS in the microbench). Batches 4–7: A8 g256 / g512 / g1024 plateau at about +0.00015–0.00018; A4 clipping helps at large groups and hurts at g64.

### 2026-10-30 ~20:15 CST · #29-01 · E3b full: int8 trellis codebook free through K6; int4 grid free at K2–3
- **Status:** done · **Code:** `v3/e3/viterbi.c`, `trellis.py`
- **Result:** the int8 trellis codebook is free through K6; the int4 grid is free at K2–3.

### 2026-10-30 ~20:15 CST · #30 · knap3 solver works; first frontier: ×1.5 prefill for +2% KLD at 4 bpw
- **Status:** running · **Code:** `v3/nf/knap3.py`
- **Result:** solver works. First frontier (target 4.0): ×1.5 prefill for +2% KLD (0.00665 vs all-f16 0.00650).

### 2026-10-30 ~20:15 CST · #30 · bf16 head breaks mixes in the EXL3 pipeline (r9 9.0 / 9.5); disabled
- **Status:** running
- **Result:** found that the bf16 head breaks mixes in the EXL3 pipeline (r9 9.0 / 9.5 options).
- **Decision / next:** disabled.

### 2026-10-30 ~20:15 CST · #29-01 · E3b: trellis decoding to int8 (tool; topic summary)
- **Status:** done (tool validated; int8 free through K6 on the full run) · **Machine:** CPU · **Code:** `v3/e3/viterbi.c`, `trellis.py`
- **Setup:** CPU reimplementation of EXL3 quantization on sampled rows: block LDLQ, 16-bit bitshift trellis over 16x16 tiles, per-row RMS scale. Simplifications: free start state (not tail-biting), no output-scale refit.
- **Result:** validation: layer 10 ssm_z, K4, 64 rows: rel 5.37e-4 vs our real ours-K4 4.16e-4 (×1.29). Slightly worse than exllamav3 but good for relative comparisons.

| tensor | K | float mul1 | mul1 rounded to int8 | int8 cost |
|---|---|---|---|---|
| L10 ssm_z (64 rows) | 4 | 5.37e-4 | 5.43e-4 | **+1%** |

  The paper bound is +4% at K4. tsweep1 covers K2–6 plus a Gaussian int8 table and an int4-grid codebook. Full run: int8 codebook free through K6; int4 grid free at K2–3.
- **Decision / next:** trellis quality survives int8 compute at mid bits.

### 2026-10-30 ~19:50 CST · #27 · E1c per-tensor activation costs done and calibrated (±30%)
- **Status:** done · **Code:** `aqcal.py`
- **Result:** per-tensor activation costs done and calibrated (±30%).
- **Decision / next:** these are the knapsack's per-tensor activation costs. Next: a depth term.

### 2026-10-30 ~19:50 CST · #29-02 · E3c row-slab K: ~0.05 bit on the big tensors → low priority
- **Status:** done · **Code:** `v3/e3/rowslab.py`, `rowslab.jsonl`
- **Result:** ceiling ~0.05 bit on the big tensors at 128-row slabs; the outlier tensors carry most of the gain.
- **Decision / next:** low priority.

### 2026-10-30 ~19:50 CST · #27 · E1c: per-tensor activation costs (proxy; topic summary)
- **Status:** done (calibrated, ±30%) · **Machine:** CPU box A · **Code:** `v3/e1b/aqproxy.py`, `aqcal.py`; data `v3/e1b/data`
- **Setup:** dump of 512 real input rows per linear (calib-A, K6 model). Scoring: every tensor x {A8: per-row, g1024…g32; A4: per-row, g512…g32} x clip, rel_out = Σ‖W·e‖² / Σ‖W·x‖², hook's exact quantizer and rotation; 10,784 rows; ~10 s per tensor.
- **Result:** calibration (refit on 45 runs after batches 4–7): log-RMSE 0.25, factors unchanged within 3%. First fit on 37 E1b KLD runs (KLD increment ≈ Σ_families k_f · Σ rel_out): per-family factors give log-RMSE 0.26 (≈ ±30%) over four orders of magnitude; one global factor gives 1.19. Factors: gate_up 0.020, attn_in 0.015, down 0.0097, attn_out 0.008, gdn_in 0.006, ssm_out 0.005, head 1.07 (the head's logit error counts ~100x more per unit of relative output error). Worst misses: tiny A8 runs near the noise floor. Depth: L48–63 under-predicted 1.7x (late layers more sensitive than the proxy): add a depth term. Group size matters very unevenly: A4 rel_out per-row -> g128 drops x2 on L0 gate_up but x28 on L3 q/k/v.
- **Decision / next:** these are the knapsack's per-tensor activation costs.

### 2026-10-30 ~19:50 CST · #29-02 · E3c: row-slab K (`v3/e3/rowslab.py`; topic summary)
- **Status:** done (low priority) · **Machine:** CPU · **Code:** `v3/e3/rowslab.py` · **Files:** `rowslab.jsonl`
- **Setup:** ceiling of re-allocating bits across output-row slabs at the same total error: ½·log₂(AM/GM) of the slab Hessian norms (input side only). 49 tensors over 9 layers.
- **Result:** param-weighted gain:

| slab | all tensors | gate_up + down only |
|---|---|---|
| per row | 0.29 bit | 0.18 bit |
| 128 rows | 0.12 bit | 0.05 bit |
| 256 rows | 0.11 bit | 0.045 bit |

  Outliers: a few small early tensors carry most of the gain: ssm_out L0 3.2 bit, attn_output L3 2.2, attn_qg L3 0.9, down L0 0.5 (at S128). These are residual writers with "massive activation" channels. For the big tensors ~0.05 bit at 128-row slabs.
- **Decision / next:** not worth kernel complexity now; low priority. Revisit with output-side importance (YAQA H_out), or "a few rows at bf16" for the outlier tensors.

### 2026-10-30 ~19:30 CST · #27 · Activation dump: 512 rows per linear, sent to CPU box A
- **Status:** running · **Machine:** CPU box A
- **Result:** 512 real input rows per linear, 3.3 GB. The dump run's KLD is byte-identical to the baseline. The proxy scorer is running.

### 2026-10-30 ~19:30 CST · #28 · E2 M sweep, and group scales in the real kernel
- **Status:** running · **Code:** `v3/bench/igemm.hip`
- **Result:** the group scales spill at the big tile (119 VGPRs): iu8 g128 73 TOPS at 128×256, unusable. At 128×128 (no spills) −12…−29%.

### 2026-10-30 ~19:30 CST · #28 · E2: int8 / int4 GEMM at real tiling (`igemm.hip`; topic summary)
- **Status:** running (decode pre-pass and scale tricks to do) · **Machine:** card 1 (first run, no tuning) · **Note:** this block has no heading in the source; it sits inside the E1c section.
- **Result, first run (M = 4096 unless noted):**

| shape | iu8 | iu4 | f16 T2 core |
|---|---|---|---|
| gate_up 34816×5120 | 5.60 ms (261 TOPS) | 2.96 ms (494) | ≈10.8 ms |
| down 5120×17408 | 2.91 ms (251) | 1.46 ms (500) | ≈5.4 ms |
| GDN in 10240×5120 | 1.71 ms (251) | 0.89 ms (485) | ≈3.2 ms |
| M = 512, 5120×5120 | 195 TOPS | 379 TOPS | |

  GEMM core: iu8 ≈1.9x our f16, iu4 ≈3.7x; 78–81% / 71–73% of peak. Outputs exact vs a naive reference.

- **Result, M sweep (TOPS):**

| M | gate_up iu8 / iu4 | down iu8 / iu4 |
|---|---|---|
| 512 | 237 / 385 | 189 / 357 |
| 1024 | 253 / 477 | 241 / 384 |
| 2048 | 259 / 493 | 248 / 482 |
| 4096 | 261 / 494 | 251 / 500 |
| 8192 | 264 / 495 | 255 / 510 |

  `down` at M = 512 is under-occupied (too few tiles): needs split-K.

- **Result, group activation scales (gate_up, M = 4096):** at the 128×256 tile the extra float accumulators spill (119 VGPRs): iu8 g128 73 TOPS, unusable. At 128×128 (no spills):

| | per token | g1024 | g512 | g256 |
|---|---|---|---|---|
| iu8 | 254 | 225 (−12%) | 220 (−14%) | 210 (−17%) |
| iu4 | 488 | 390 (−20%) | 375 (−23%) | 346 (−29%) |

- **Decision / next:** to do: the trellis -> int8/int4 fragment decode pre-pass; the activation quantize step fused into the rotation; an M sweep (incl. expert-sized shapes); tuning. Group scales: first version; to try prefetching the group scales and power-of-two scales applied as integer shifts.

### 2026-10-30 ~19:00 CST · #28 · E2 first run: iu8 ~255 TOPS, iu4 ~490 TOPS on real shapes
- **Status:** done · **Code:** `v3/bench/igemm.hip`
- **Result:** iu8 ~255 TOPS, iu4 ~490 TOPS on real shapes (1.9x / 3.7x our f16 GEMM).

### 2026-10-30 ~18:50 CST · #28 · Group-scale speed for iu8 and iu4 measured
- **Status:** done · **Result:** iu8 g512 −3.5%, g128 −18%.

### 2026-10-30 ~18:50 CST · #26-01 · A8 g256 +0.00017; batches 4–7 queued
- **Status:** running · **Result:** A8 g256 +0.00017. Batches 4–7 (g64 / g256 / g512 / g1024, A4 and A8) queued.

### 2026-10-30 ~18:50 CST · #30 · Group size becomes a per-tensor knapsack option
- **Status:** running (design) · **Result:** the group size becomes a per-tensor knapsack option.

### 2026-10-30 ~17:50 CST · #26-01 · A8 with g128 activation scales costs only +0.00005
- **Status:** done · **Result:** A8 with g128 activation scales costs only +0.00005 (per-row: +0.00047). The A8 floor was the scale granularity.

### 2026-10-30 ~17:50 CST · #26-02 · A4 g128: 5x better than per-row
- **Status:** done · **Result:** A4 g128: 5x better than per-row (all 0.031); gate_up g128 +0.009.

### 2026-10-30 ~17:50 CST · #29 · E3 sweep moved to CPU box B (44 jobs at once)
- **Status:** running · **Machine:** CPU box B · **Code:** `sweep2.sh`

### 2026-10-30 ~17:45 CST · #— · CPU work moves to CPU boxes A and B; sweep jobs rewritten
- **Decision / result:** CPU work moves to CPU box B (48 threads) and CPU box A (36 threads); the VM is downsized to 8 vCPU / 16 GB. Sweep jobs rewritten as single-thread processes with row sampling and a vectorized Viterbi: ~12x (trellis) and >100x (GPTQ on big tensors) faster.

### 2026-10-30 ~17:30 CST · #29-01 · E3b trellis tool built and validated
- **Status:** running (full sweep pending) · **Code:** `v3/e3/viterbi.c`, `trellis.py`
- **Result:** tool validated: ×1.29 vs exllamav3 K4 (layer 10 ssm_z, 64 rows); int8 codebook costs +1% at K4.

### 2026-10-30 ~17:30 CST · #— · Calib Hessians on the VM (47 GB, 257 files)
- **Result:** input Hessians on the VM (47 GB, 257 files).

### 2026-10-30 ~17:00 CST · #— · Tracker created
- **Result:** tracker created.

### 2026-10-30 ~17:00 CST · #26-02 · E1b A4 per-family results in; batch 3 queued
- **Status:** running (batch 3 queued) · **Result:** A4 per family (rows in the E1 summary): attn_out 0.0085, ssm_out 0.0141, attn q/k/v 0.0237, gdn_in 0.0241, down 0.0262, gate_up 0.0264, head 0.0333. Batch 3 queued: A4 g128 and clipping.

### 2026-10-30 ~17:00 CST · #— · Calib Hessians downloading to the VM from HF
- **Result:** calib Hessians downloading to the VM from HF.

### 2026-10-30 ~16:40 CST · #26 · E1b A8-all +0.00047 (Large needs A16), A4-all 0.150
- **Status:** done · **Result:** A8 all linears +0.00047 (per-row; Large needs A16); A4 all linears 0.150.

### 2026-10-30 ~16:40 CST · #29 · E3 partial: int codes ×2–3.5 vs trellis
- **Status:** running · **Result:** plain integer codes ×2–3.5 vs trellis (3 tensors so far).

### 2026-10-30 ~16:40 CST · #29 · E3: weight codes, quality per bit (Hessian proxy; topic summary)
- **Status:** running (provisional decision; 3 tensors so far) · **Machine:** CPU · **Code:** `v3/e3/e3.py`
- **Setup:** metric rel = tr(dW H dWᵀ)/tr(W H Wᵀ), the same as hproxy. Shown: median ratio to our EXL3 ours-K at the same bits. Data: 3 tensors so far (layers 10 / 43).
- **Result:**

| bits | GPTQ, per-channel int | GPTQ g128 -> int8 | Lloyd LUT -> int8 | RTN (rotated) |
|---|---|---|---|---|
| 2 | ×2.3 | ×1.7 | ×1.7 | ×12.7 |
| 3 | ×2.1 | ×2.0 | ×2.0 | ×11.3 |
| 4 | ×2.7 | ×2.3 | ×2.2 | ×13.0 |
| 5 | ×3.1 | ×2.5 | ×3.0 | ×13.8 |
| 6 | ×3.5 | ×3.4 | ×4.5 | ×15.3 |
| 8 | ×3.5 | ×19 | ×12 | ×16.3 |

  Plain integer codes cost about 0.6–0.9 bit against the trellis, and GPTQ is mandatory. Folding group scales into int8 has a floor around 1e-4 (≈ K6). On paper, the EXL3 mul1 codebook rounded to int8 adds ~6e-5 relative error: +4% on K4, +17% on K5, +60% on K6; it dominates K7–8.
- **Decision / next:** provisional: a trellis decoding to int8 for ≤ ~K5. Above that, f16 compute (EXL3 as is).

### 2026-10-30 ~16:20 CST · #26 · E1b hook built in a separate tree; no-op run byte-identical
- **Status:** done · **Code:** `/root/e1b` (separate tree)
- **Result:** the hook's no-op run is byte-identical to the normal K6 score.

### 2026-10-30 — · #25 · E0 iu4 / iu8 peaks and scale-group cost measured; fp4 absent
- **Status:** done · **Machine:** GPU (gfx1201)
- **Result:** measured (details in the E0 summary). fp4 absent.

### 2026-10-30 — · #25 · E0: hardware facts (gfx1201, measured; topic summary)
- **Status:** done · **Machine:** GPU (gfx1201; ISA checked for every number)
- **Result:** measured peaks:

| op | peak |
|---|---|
| iu8 16×16×16 | 323 TOPS |
| fp8 16×16×16 | 321 TOPS |
| iu4 16×16×32 | **685 TOPS** |
| iu4 16×16×16 | 328 TOPS |
| f16 (earlier) | 158 TF |
| fp4 / fp6 | none (gfx1250 only) |

Activation scales applied along K (`wmmagrp.hip`, a tight loop, so the worst case):

| scale group | iu4 TOPS | iu8 TOPS |
|---|---|---|
| 64 | 331 | 220 |
| 128 | 444 | 265 |
| 256 | 536 | 295 |
| 512 | 596 | 312 |
| 1024 | 631 | 323 |
| per token / per channel only (epilogue) | ~685 | ~323 |

- **Decision / next:** compute is integer (iu8 + iu4), not fp8; W4A8 is dropped. Group scales along K are expensive, so the scales are per token and per output channel.

### 2026-10-30 — · #1 · Two quantizer processes per card (dead end; V2 era)
- **Status:** negative · **Date:** undated in source; INDEX bounds it at ≤10-30 · **Code:** `ourq.py` on exllamav3 1.6.3, gfx1201
- **Result:** +14% throughput only (GPU already at 100%), and memory fragmentation risk (head quants OOMed before).
- **Decision / next:** one process per card.

### 2026-10-30 — · #2 · Bigger tile batches in the quantizer (dead end; V2 era)
- **Status:** null · **Date:** undated in source; ≤10-30
- **Result:** per-tile trellis cost is flat from 64 to 8,704 tiles per launch.
- **Decision / next:** nothing to gain.

### 2026-10-30 — · #3 · Sharing work across K values in one run (dead end; V2 era)
- **Status:** dropped · **Date:** undated in source; ≤10-30
- **Result:** the K-independent part (Hessian prep, LDL, Hadamard) is ~5 s of ~35 s per layer, so ≤ 15%.
- **Decision / next:** not worth the restructure.

### 2026-10-30 — · #4 · K7's slow trellis kernel (dead end; V2 era)
- **Status:** dropped (not fixed; the INDEX row currently says done, see Rewrite notes) · **Date:** undated in source; ≤10-30
- **Result:** K7 ~40 µs/tile vs ~15 for K4 / K8. Not fixed: needs an exllamav3 extension rebuild. What did work: dropping the diagnostic `block_trace` (79 -> 35 s/layer, bit-identical); `EXL3_QT_OPTIMIZED=1` (identical indices; K1 2.2x, K6 1.26x).

### 2026-10-30 — · #5 · Head quantization OOM at K5 / K6 (dead end; V2 era)
- **Status:** done · **Date:** undated in source; ≤10-30
- **Result:** `refit_scales` allocated full-size temporaries; fixed by an in-place chunked rewrite (`/data/quantize.py.orig` is the original). Head quant still needs ~22 GB host RAM.

### 2026-10-30 — · #6 · Bigger prefill steps in the eval (dead end; V2 era)
- **Status:** null · **Code:** `radiance-kldgold-fast` · **Date:** undated in source; ≤10-30
- **Result:** `--max-num-batched-tokens 4096` is a no-op (chunks are capped at the 2,048 GDN checkpoint interval); `--prefill-span 2` and `--checkpoint-interval 4096` run 4,096-token chunks but take the same 118 s with the same KLD. 92% of GPU time is the EXL3 GEMMs at ~40% of peak.
- **Decision / next:** the fix is the prefill GEMM rewrite.

### 2026-10-30 — · #7 · rocprofv3 on radiance (dead end; V2 era)
- **Status:** dropped · **Date:** undated in source; ≤10-30
- **Result:** unusable: memory-copy tracing breaks the load, and kernel tracing makes the recorded-launch replay fault.
- **Decision / next:** use radiance's own `--profile-ops`.

### 2026-10-30 — · #8 · Card 1 evals ~25% slower (152 vs 119 s) (dead end; V2 era)
- **Status:** done (cause found; fix not made) · **Date:** undated in source; ≤10-30
- **Result:** the full-vocab f32 logits copy over its PCIe3 x4 link.
- **Decision / next:** async copy + background scoring would fix it (~1–1.5 h engine work); not done yet.

### 2026-10-30 — · #9 · TP2 for whole-corpus runs (dead end; V2 era)
- **Status:** negative · **Date:** undated in source; ≤10-30
- **Result:** card 1's x4 link makes it slow.
- **Decision / next:** TP2 only for documents that do not fit one card.

### 2026-10-30 — · #11 · Symmetric "depth compounding" term in kfit (dead end; V2 era)
- **Status:** negative · **Date:** undated in source; ≤10-30 · **Code:** kfit.py (cost_w·(1 + c·cost of earlier layers))
- **Result:** algebraically a global quadratic link on the total, not a depth effect: it improved CV only by fixing the overall level, and its solver step biased against late layers. Replaced by the directional model (error amplified by downstream error, kfit2.py): CV log-RMSE 0.28 (additive) -> 0.19 (link forms) -> 0.11–0.13 (directional).

### 2026-10-30 — · #12 · Lagrangian multiple-choice-knapsack sweep (dead end; V2 era)
- **Status:** superseded (exact solver instead) · **Date:** undated in source; ≤10-30
- **Result:** exact only on hull points; lands well below targets (hence the old "3.99" targets); cannot handle the directional term. Pitfall in the replacement DP's first version: 4 MiB buckets rounded *up* per tensor left ~0.1 bpw unused per mix -> 1 MiB, nearest rounding, exact-size check.
- **Decision / next:** replaced by an exact byte-bucket DP.

### 2026-10-30 — · #13 · Treating all quantizer families alike (dead end; V2 era)
- **Status:** negative · **Date:** undated in source; ≤10-30
- **Result:** turboderp's "plain" exports cost ~1.15–1.23x the KLD of "SC" ones per unit of Hessian proxy error; the proxy is not quantizer-neutral.
- **Decision / next:** fit a factor per family (and per K range for ours).

### 2026-10-30 — · #14 · Measuring single-tensor swaps end to end (dead end; V2 era)
- **Status:** negative · **Date:** undated in source; ≤10-30
- **Result:** paired SE on ~100k positions is ~1.5%; a single tensor's effect is ~0.1–0.5%.
- **Decision / next:** use group swaps, whole mixes, or random multi-tensor perturbation designs.

### 2026-10-30 — · #15 · Noise-injection sensitivity calibration (HIGGS / Q-Palette / exllamav3 style; dead end; V2 era)
- **Status:** negative · **Date:** undated in source; ≤10-30
- **Result:** not cheaper for us: real swaps are 0.2 s reflink mixes, and noise injection would need a bf16 forward that does not fit one card.

### 2026-10-30 — · #16 · Shapley-value interactions (IMPQ, SLQ; dead end; V2 era)
- **Status:** dropped · **Date:** undated in source; ≤10-30
- **Result:** literature puts interactions at ~1% for ~15–18 h per 8B model. Skipped.

### 2026-10-30 — · #17 · RCO: Riemannian constrained optimisation (arXiv 2605.00649; dead end; V2 era)
- **Status:** dropped · **Date:** undated in source; ≤10-30 · **Files:** `docs/research-rco.md`
- **Result:** needs a differentiable torch forward + backward of the full 27B on ROCm, 800–3,200 passes per size, for 0–4% over measured-loss search. Skipped.
- **Decision / next:** the "one backward pass = in-context sensitivity of every tensor" idea is kept as an optional spike.

### 2026-10-30 — · #18 · Gradient boosting / generic black-box surrogates (XGBoost, full GP BO; dead end; V2 era)
- **Status:** negative · **Date:** undated in source; ≤10-30
- **Result:** not usable: the exact solver needs a per-tensor-decomposable model, and black-box BO over ~340 categorical dims needed ~10k evals in AMQ.

### 2026-10-30 — · #21 · Layer 0 as a sensitivity spike (dead end; V2 era)
- **Status:** null (the hypothesis was wrong) · **Date:** undated in source; ≤10-30
- **Result:** K2 on layer 0 ffn_down costs +0.00006 (≈ free). The spike is at the end: L63 ffn_gate_up K2 +0.0065 (80% of the base's KLD); L60–63 ffn_down +0.0006–0.0013.

### 2026-10-30 — · #23 · 2-bit Lloyd embedding (dead end; V2 era)
- **Status:** negative · **Date:** undated in source; ≤10-30
- **Result:** earlier embedding sweep: clearly bad vs 3-bit at normal sizes.
- **Decision / next:** still offered to the optimizer for the research floor (1.2–1.5 bpw) only.

### 2026-10-30 — · #— · Process pitfall: selecting on public
- **Date:** undated in source; placed at the V2-era bound (≤10-30) · **Result:** never. Public is held out (`docs/METHOD.md` §6). Verify-slice reuse also bakes in noise slowly.

### 2026-10-30 — · #— · Process pitfall: `pkill -f` / `pgrep -f` with a pattern in our own ssh command
- **Date:** undated in source; placed at the V2-era bound (≤10-30) · **Result:** killed the shell several times. Anchor patterns.

### 2026-10-30 — · #— · Process pitfall: queue claims by `mkdir` alone on the overlay filesystem
- **Date:** undated in source; placed at the V2-era bound (≤10-30) · **Result:** claims raced: two cards ran one eval. Claims now take a lock.

### 2026-10-30 — · #— · Process pitfall: `golden_pick.py` with `splitlines()`
- **Date:** undated in source; placed at the V2-era bound (≤10-30) · **Result:** split documents on Unicode line separators. Split on `\n` only.

### 2026-10-30 — · #— · Process pitfall: fp8 KV cache
- **Date:** undated in source; placed at the V2-era bound (≤10-30) · **Result:** fp8 KV cache moves KLD. All measurements use bf16 KV.

### 2026-10-30 — · #— · Process pitfall: reference curve including dominated containers
- **Date:** undated in source; placed at the V2-era bound (≤10-30) · **Result:** a reference curve that includes dominated containers (bigger *and* worse, e.g. plain 2.00 / 4.00 / 5.00 / 6.00) made mixes look better than they were. Compare against the Pareto envelope only.

### 2026-10-29 — · #10 · First knapsack (`knap.py`): band sensitivities from big K=2 swaps, linear model, Lagrangian solver (dead end)
- **Status:** superseded (by kfit2 / knap4, #45)
- **Result:** the K=2 swaps overstate slopes ~4x (only relative values usable); a power-law (P) variant blew up and was dropped; a units bug in the head / embedding cost gave every mix a 3-bit head. Fixed version: ~−3.5% vs the reference containers on calib-B on average, and the 2.00 bpw mix was +68% *worse* (non-additivity at low bpw).

### 2026-10-29 — · #35 · Trellis (EXL3-style) token embedding instead of Lloyd (dead end; deferred)
- **Status:** dropped (deferred on 2026-10-29; the trellis embedding was later measured on real KLD and adopted, see #35)
- **Result:** measured 4-bit Lloyd at ≈ +2% KLD at 4.00 bpw vs IQR's trellis embedding ≈ +0.6%: at most ~1.5% total to win, and it needs a row-wise trellis gather kernel.
- **Decision / next:** deferred; only reconsider for vocabulary-tiered embeddings (simulate both first).

### 2026-10-29 — · #24 · Head above K6 (dead end)
- **Status:** null
- **Result:** our K7 / K8 heads have 5x / 16x less proxy error than a 6-bit head, but the KLD gain over a 6-bit head is within noise (~−5% vs the 5-bit head for all of them).
- **Decision / next:** head bits past K6 are wasted except maybe at 8+ bpw.

## Notes on the rewritten entries (before 2026-11-02 ~05:00 CST)

*Applied 2026-11-02 (main session): (a) both radiance-port entries → #74, tdverify → #75 (new INDEX rows); (b) corrections 1, 3, 4, 5, 6, 7, 8 applied to INDEX (#39 set blocked, not running: its GPU test is on gpubox); 2 (#38) and 9 (#37) left as they are: damp3's result set the V3 damping and finalgate closed the speed pass.*

**Conventions applied.** Times: CT and zone-less times are kept as given and labelled CST. Five entries the source gives in CDT were converted (−1 h): 01:55, 01:50, 01:47 and 01:40 CDT on 2026-11-01, and 01:17 CDT (Gate 2e). `~` is kept where the source is approximate, or where the time is inferred from a neighbouring entry. `—` marks entries with no clock time in the source, including every dead-end bullet. Undated V2-era dead ends, and the six process pitfalls, carry the header date 2026-10-30 (INDEX bounds the dead ends at ≤10-30). Two undated 2026-11-01 log bullets (tdverify, damp2) take the date of the entry above them, with `—`. Within a bullet list, a bullet with its own bold label is one entry, and its plain sub-bullets are folded into it; a sub-bullet with its own date takes that date. A dead end whose bullet spans several dates uses the latest one (per-group gain scales 2026-10-31; YAQA sampled positions 2026-11-01). Where a source entry gives no status word, the status was taken from its text (for example, `running` while a gate is still open). Entries are sorted by date, then time, newest first; entries without a clock time follow that day's timed entries. Numbers: every number in the entry bodies appears verbatim in the two archive files, except INDEX cross-references (`#46`, `#33-08`).

**Topic summaries (11).** E0, E1/E1b, E1c, E2, E3, E3b (tool), E3b results, E3c, K3, E4, E4b. E2 has no heading in the source: its block sits inside the E1c section, so it is written as a summary under #28 and dated at its last log result (19:30 CST). E4b has no result; it is placed at E4's date (`~21:00`). E3 is dated 16:40, its last result in the section (INDEX #29 says last 10-31; the 10-31 full set is in the E3b results summary).

#### (a) Entries marked #? (2)

- **2026-10-31 ~04:00 CST · #? · radiance 1.15.0 / ABI 32 port (planned).** Suggested row: `#?` | radiance 1.15.x plugin port (ABI 32, layout fingerprint) and KLD parity | kernels | planned | 2026-10-31 | plugins rebuilt and re-validated (byte-identical KLD) before V2 ships.
- **2026-11-01 ~17:45 CST · #? · radiance 1.15.1 port works (done).** Same row proposed; one row can cover both entries (the 1.15.1 entry has the parity check: 0.027122 vs 0.027122).
- Optional, not numbered: `eval/capstudy.py` (search slices capped at 2k, held-out 8k) sits inside the #58 entry; D6 (GDN a/b merge closed) and B13 (`ourq.py` code stamp) sit inside the sub-1 K entry. tdverify was given #63 by content; no row covers it.

#### (b) INDEX rows whose verdict or status looks wrong (9)

1. **#39 Global gain scale vs K.** INDEX: `null`, 10-31. Source (10-31 ~22:20 CT): "the best factor rises with K in every family … the GPU gaintest (running) shows whether exllamav3's choice already tracks this. If it doesn't, it is a strong candidate for the high-K gap." Suggest `running`, pending the GPU gaintest.
2. **#38 Damping.** INDEX: `done` (11-01), verdict "input 0.005 (0.05 below K1), rest 0.025". Source: 11-02 "**damp3 (running 2026-11-02):** … Grid 0.025 / 0.01 / 0.005 / 0.0025 / 0.001 / 0.0003 / 0.0001" and 10-31 "0.01 elsewhere is harmless". Suggest `running` (damp3), and reconcile "rest 0.025" with "0.01 elsewhere".
3. **#33 YAQA output-side Hessians.** INDEX: `running`. Source: dead end (11-01) "Every backward-based Hessian variant is dropped (2026-11-01)"; log 10-31 ~22:20 CT "two-sided YAQA dropped". The only open arm is #33-08 (Sketch B). Suggest the parent read `negative` (or `dropped`) with 33-08 as the open part; the dead end also says Sketch B was "not tested".
4. **#34 LM head.** INDEX: `done` (11-01, "mean-row −30%"). Source (11-01 17:45 CST): "Floor search fixes the slab head … whole-model check still to do (critique C3.5)"; (11-01 00:40 CST): "Needs hdslab-K5/5.5/6 containers before validation." Suggest `running` for the slab head; the mean-row result stays `done`.
5. **#46 A4 activations.** INDEX: `negative` (11-01, "no, at any size (knap4)"). Source: "The A4 drop above rests on the matmul-core table, not on this slot" (11-01 01:25 CST), and the A4 timing slot is "queued after D2" (11-01 00:45 CST). Suggest keeping `negative` but stating that the timing slot has not been run.
6. **#22 GPTQ damping.** INDEX: `negative` (10-31). Dead-end bullet: "not a dead end, deferred … not worth a V2 requant; use 0.005 for attention / GDN inputs in the V3 requant." Suggest `superseded` by #38.
7. **#4 K7 trellis kernel.** INDEX: `done`. Dead-end bullet: "not fixed: needs an exllamav3 extension rebuild." Suggest `dropped`, or `done` only with a pointer to #37 (qtf2 integer K6 / K7, 11-02).
8. **#19 Per-layer edge offsets.** INDEX last date `≤10-30`. Source: 10-31 ~01:50 CST entry and dead end "(KFIT_EDGE, 2026-10-31)". Suggest last date 10-31.
9. **#37 qtf2 speed pass.** INDEX: `done` (11-02, "finalgate PASS, 51 instances"). Source (11-02 speed pass, lower confidence): "Not done: reusing K-independent work across levels …; refit via Cholesky …; KA 6-7 lane-cooperative search". Check whether the pass is closed.

#### (c) Source entries not placed, or contradictory

1. **Gate 2e MARGINAL rule** (11-01, date only): listed after the ~00:55 CT knap4 entry, but it must precede the 01:17 CDT verdict. Kept as `2026-11-01 —` (sorted to the end of the day).
2. **qtf2 v1** (11-01 17:45): "still ~2x slower than the integer kernel (K3 43k vs 18.6k tiles/s)". The numbers read backwards; copied as written.
3. **Fractional-K heads** (11-01 01:25 CST) say K3.5 / K4.5 OOM and do not fit. The 17:45 CST entry says K3.5 "work[s] alone" (25.6 GB peak). Both kept; the later one supersedes.
4. **Speed pass 2** bullets sit under the 11-01 17:45 CST container but carry 2026-11-02 dates. They are dated 11-02 with `—`.
5. **chain1** (11-01): results are from 10-31 ~23:50 CT; the entry is dated 11-01 with no clock time.
6. Several 10-30 and 10-31 entries have no time zone; written as CST.
7. The 11-01 ~21:20 CT quarter-step entry and some 10-31 entries are out of order in the source; sorted by timestamp.
8. **Trellis token embedding** (dead end, 10-29, "deferred") was measured and adopted later (#35, 10-31 onward). Its status is kept as of its own entry (`dropped`, deferred).
9. **Per-group gain scales**: dead-end dated 10-30, retested 10-31; dated 10-31.
10. **Cross-references inside multi-topic bullets**: the rounded-codebook bullet (#43) also covers #34 and #45; chain1 (#35) covers #63; the knap4 wiring entry (#45) covers #34 and #35; the 11-01 requant bullet (#58) covers #36, #38 and #48.
11. **Dead ends that restate a log entry** (embedding run5/run6, Best-of-N, upstream auto rule, GPTQ) are kept as separate entries, as the brief asks.
12. **Undated process pitfalls** have no INDEX row and no date in the source; placed at 2026-10-30 (≤ V2 era). This is an assumption.
13. **Not converted:** the tracker header, "Plan to the V3 stream (2026-11-01, user)", "Running / queued now", "Backlog" and "Candidate experiments". They are outside the two sources the brief names. The user decisions in the plan appear in the Log entries where the Log records them.
