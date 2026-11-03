# Experiment log (newest first)

Append-only; template and rules in `README.md`. Entries before 2026-11-02 are in `archive/`.

### 2026-11-03 08:40 CST · #33-22 #78-03 #80–#89 · Night queue results (0.8B proxy, 4070; A1 and #80 on CPUs)
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
- **#80 (colo, CD replicate, row-seed 1):** cd3 / ldlq 0.993 (K2.5), 0.993 (K3), 16 / 16; cubic / mul1 0.989 / 0.985, 16 / 16. **PASS**: #77 and #78 replicate on other rows.
- **#81 (A1, Thor + colo):** ug / D mean K2 0.838, K2.5 0.832, K3 0.829 (L30 0.73 / 0.71); up only 0.871 / 0.866 / 0.863 (keeps 80% of the reduction); oracle 0.825. **PASS** (L30 ≤ 0.85, mean ≤ 0.90), up-only preferred. Caveat to check first: the predicted share of H-weighted energy is only 4.5%, yet the coded error drops 17%; that needs explaining (shape effect or a bug) before any GPU work.
- **Files:** `data/night1103/*.txt`, `data/yaqa08/result-fact1{2,3}.txt`, `data/a1/a1-{thor,colo}.jsonl`, `data/e3/cd-colo-rs1.jsonl`

### 2026-11-02 20:05 CST · #79 · Coordinate descent stacks on the cubic codebook (CPU proxy)
- **Status:** running → done · **Machine:** Thor · **Pass rule:** `v3/e3/cd/README.md` (#79)
- **Result** (cd3 / ldlq, both on cubic0.02_i8; mean / param-weighted / tensors ≤ 1): K2.5 0.992 / 0.993 / 16 of 16;
  K3 0.992 / 0.993 / 16. Combined (cubic + cd3) vs ldlq mul1_i8: K2.5 0.979, K3 0.974 (means).
- **Reading:** passes: CD's −0.8% is unchanged on top of the codebook (the two gains add).
- **Decision / next:** #77 stays "real, small"; with #80 (other rows) it decides whether CD gets a GPU version.
- **Files:** `data/e3/cd-thor-stack.jsonl`

### 2026-11-02 20:35 CST · #33-22 #81–#89 · Night queue (no iceman): worker experiments on the 4070, A1 on the CPUs
- **Status:** planned → running · **Machine:** 4070 (`proxy2b/night4070.sh`, one after another), Thor + colo (`v3/e3/a1/a1run.sh`)
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
- **Status:** #77 running (replicate + stacking), #78 passes · **Machine:** Thor + colo · **Pass rule:** `v3/e3/cd/README.md`
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
- **Files:** `data/e3/cd-thor.jsonl`, `data/e3/cd-colo.jsonl`

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
  whether the kernel is), but on the GPU (iceman), not a CPU day.

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

### 2026-11-02 16:52 CST · #77 / #78 · CPU proxy sweep, first read (13 of 16 tensors; ffn_down pending on colo)
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
- **Files:** `data/e3/cd-thor.jsonl`

### 2026-11-02 16:35 CST · #77 / #78 · Tile coordinate descent (C2) and Gaussianized mul1 codebooks (D3): CPU sweep started
- **Status:** running · **Machine:** Thor (20 jobs × 2 threads) · **Code:** `v3/e3/trellis2.py`, `v3/e3/cd/run.sh` @ 48e7b99
- **Pass rules:** `v3/e3/cd/README.md` (written before data).
- **Setup:** 16 real 27B tensors (GDN layers 10/30/50 × qkv, out, gate_up, down; attention layers 19/43 × q, o),
  256 rows, K 2 / 2.5 / 3 / 4, paired proxy error. Smoke test (one small tensor, 64 rows, K2.5): CD 2 sweeps
  −0.5%, gmul1_i8 −0.7% vs mul1_i8 (not a result).
- **Files:** (when done) `data/e3/cd.jsonl`

### 2026-11-02 15:55 CST · #76 · pisess v1: our own pi sessions as an agentic-trace corpus
- **Status:** done · **Machine:** colo (CPU) · **Code:** `xcorp/pisess.py`, `xcorp/pisess_redact.py`,
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
- **Decision / next:** training / statistics only (AGENTS.md); first use #33-19. Lives on colo
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
  `hdiag27.sh` once both iceman cards are free.
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
  when iceman is back and card 1 is given; snapshots at 256 / 512 / 1024.
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
- **Status:** planned · **Machine:** 4070 (check), iceman (use) · **Code:** `v3/yaqa/bf16x.py`, `proxy2b/prectest.py`,
  `collect.py` YAQA_PREC=x2 @ 0ef4e99
- **Question:** can the 27B collector's fp32 GEMMs (~30 s / sequence) run as bf16 hi/lo splits on the matrix cores
  (2 GEMMs per product, 3 for the weight gradient; weights exact in bf16) without losing the fp32 gradient quality?
- **Pass rule:** `proxy2b/YAQA08.md` amendment 7.
- **Setup:** CPU self-test on random data (emulated): x2 vs fp32 relative error 2.5e-6 (products), 6e-7 (diagonal);
  bf16 1.7e-3 / 3.8e-4. The model-level test (`prectest.py`, ~10 min on the 4070) is not run yet.
- **Decision / next:** run prectest + `yaqaFact4.sh` when the 4070 is free; on iceman, the gate run of `hdiag27.sh`
  with YAQA_PREC=x2 also reports one linear's precision against a whole fp32 G and the time per sequence. Then the
  two-stage pipeline (#33-13) if the GEMMs still dominate the profile.

### 2026-11-02 11:50 CST · #33-13 · 27B diagonal H_out: collector and 0.8B dose check prepared
- **Status:** planned · **Machine:** iceman (both cards) + 4070 · **Code:** `v3/yaqa/collect.py` YAQA_DIAG,
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
