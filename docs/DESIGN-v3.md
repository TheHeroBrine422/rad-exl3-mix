# EXL3-V3 design (draft, 2026-10-31)

Consolidates `DESIGN-new-format.md` §8, the V3 experiments (`EXPERIMENTS-new-format.md`) and what V2's run taught.
Open decisions are marked **[decide]**.

## Goal and scope

- **Speed:** the board scores trunk GEMM time per decoder layer at 2048 tokens (`gemm_us`). EXL3 (V1/V2 kernel):
  ~8,030 µs. IQR-V3: 4,935–5,236. FP8: 4,901. Target: **≤ ~5,200 µs at every size**, i.e. ~1.55× faster prefill GEMMs,
  at V2's KLD or better. At ≤ 2.5 bpw, beat IQR-V3's ~4,935 µs: that needs A4 on part of the trunk.
- **Sizes:** ≤ 7.5 bpw (above that V2 is effectively lossless; K7/K8 stay f16 anyway).
- **Hardware:** card 0 only (03:00.0, PCIe x16), TP1. TP2 only at times (when both cards are available).
- **Not in V3:** QAD (V4), vocab-tiered embedding and frequency-tiered head (dropped; a Fisher-driven row-slab
  head is under test, §2).

## 1. Compute format (the speed)

- **W8A8 on int8 WMMA** (iu8 ~323 TOPS dense, E2 ~255 TOPS real at M ≥ 1024 vs our f16 path ~130–137):
  - **Weights:** EXL3 trellis codes as stored; the decoded mul1 value rounded to int8 (fixed global factor). E3:
    ×1.000–1.004 proxy error at K2–6, so **no requant needed for the int8 path** at K ≤ 6. K7/K8 keep the f16 tile.
  - **Activations:** rotated (same Hadamard), int8 with one scale per (token, 128 inputs). E1b on K6: g128 costs
    +0.00005 KLD (~5%), per-token +0.00047.
  - **Fold (IQR-V3's, the key to FP8 speed):** the K loop sees no per-row weight scale. Per 128-input block: one
    FMA per accumulator slot with the token's activation scale (× a block factor shared by all rows); the row scale
    (EXL3's `svh`) is applied once at the store. EXL3 already has exactly this scale structure.
  - Per tensor the solver chooses the compute mode: **f16** (today's kernel) or **A8-g128**; later maybe A8 with
    other group sizes.
- **A4 (iu4; real GEMM ~490 vs ~255 TOPS for iu8, E2) as a per-tensor option at ≤ ~2.5 bpw.** E1b on K6 (base
  0.00101): all linears A4-g128 +0.030; per family (per-row) gate_up +0.025 / down +0.025 / gdn_in +0.023 / attn qkv
  +0.023 / ssm_out +0.013 / attn_out +0.0075; best found: down g64 +0.0038, gate_up g64 +0.008. At 2 bpw (KLD ~0.155)
  A4-g64 on gate_up + down (~42% of the FLOPs) costs ~+8% KLD for an est. ~14% lower `gemm_us`; at 3 bpw the same
  is ~+45%, so A4 only pays at the bottom. Needs int4-grid codes (codebook; E3: ×1.01 at K2, ×1.07 at K3) and an
  iu4 tile with the same fold.
- **Head:** f16. At serving prefill it runs once (last token); it is not in `gemm_us`.
- **GDN a/b projections** (N = 48 and 128, bf16) run as separate `gemm_nt` ops beside the trellis path, each
  re-reading the activation tile. Measure their share of the layer at M = 2048 (`--profile-ops`); if > ~2%, merge
  them into one N = 176 op or fold them into the qkv tile.
- **Decode (M = 1–16):** keep exl3r4d's GEMV; measure it against bytes/bandwidth (4 bpw: 29.4 tok/s ≈ 63% of
  bandwidth). IQR-V3's computed-code GEMV is a reference for later.
- **Open check first:** E2 measured −12…−29% for group scales at 128×128; re-measure with the fold above (activation
  scale per block, weight scale only at the store). If that is ~free, A8-g128 is the default everywhere.

## 1b. Decisions (2026-10-31)

1. **Compute mode lives in the tensor's encoding** (`act=`, group size), written per mix by `radmix` (the same quant
   can run at several modes).
2. **A8/A4 only for prefill steps of M ≥ 512**; below that the f16 path (same speed there, better quality).
3. **Group sizes are solver options**, speed vs quality: per-token, 1024, 512, 256, 128, 64 for A8 and A4
   (E0, iu8 TOPS: 64 220 / 128 265 / 256 295 / 512 312 / 1024 323; E1b, A8 on K6: g128 +0.00005, g256–1024
   +0.00015–0.00018, per-token +0.00047). Kernels instantiated per group size at compile time.
4. **Shared-input linears share one mode** (q/k/v, gate/up, GDN qkv/z). Not one activation tile, though: their input scales (`suh`)
   differ (per-tensor scale refit; signs agree), so each needs its own prep pass unless the requant ties them.
5. **Head activations: f16 only** (2026-10-31). Why, since it is easy to forget:
   - **Prefill** runs every prompt token through the trunk (each hidden state feeds later tokens' attention), but
     the head only on the **last** position: logits for earlier prompt positions would be thrown away, so the server
     does not compute them. One head row of work per request, not per prompt token.
   - **Decode** runs the head once per generated token at M = 1; speculative verify runs it at M ≈ 4–16. Both are
     below the A8/A4 threshold (M ≥ 512), so int8/int4 activations would never be used for it in serving.
   - **Exceptions:** requests for prompt logprobs (`echo` / `prompt_logprobs`) and **our KLD evals**, which need
     logits at every position: there the head runs at full M (~10% of the linear FLOPs in eval profiles). Keeping
     it f16 keeps the evals faithful to serving.
   - So the head's cost is **decode bytes**: every generated token reads all 248k × 5120 weights. Its **K** stays a
     solver option, priced like every tensor by KLD and bytes (decode time is checked at the end, not solved for);
     its activation mode does not.
6. **Embedding (a lookup, no GEMM, so no activation format):** options = V2's Lloyd/fwht128 tables (e1–e8) and a
   **trellis-coded embedding** from the requant, same quantizer as every other tensor but laid out so one row decodes
   alone (no rotation across rows); the solver picks by bytes vs KLD. Needs a row-decode gather kernel.
   Quantize it with **uniform weighting (H = I)**, like the Lloyd tables: its true Hessian is diagonal = token
   frequency, which is the dropped vocab-tiering in disguise (starves rare / multilingual tokens). Est. 2–5% KLD at
   ≤ 3 bpw (trellis-3 ≈ Lloyd-4). Plain int4/int8 tables are dominated by Lloyd at the same bits.
   **Measured (CPU, 2026-10-31, `v3/e3/embtrellis.py`):** at equal bits the trellis has 0.51–0.56× Lloyd's
   reconstruction error (K ≥ 2): ~0.48 bits a value saved = ~0.023 bpw (so trellis-3 ≈ Lloyd-3.5, not Lloyd-4);
   worth ~1% KLD at ~3 bpw by V2's fitted embedding costs. **Menu: quarter steps** (one quarter = 0.012 bpw; the
   generic pattern decode serves them at no extra kernel cost). Scale group size: one scale per row (0.003 bit/value) beats
   G128 by ~15% at equal bits; G1024/G512 lose at every K up to 5.5 (menu sweep): one scale per row only.
7. **Fresh held-out slices** ver3 (~80 docs, monitoring) and ver4 (~240, final) from the unused verify docs, plus
   search-pool selection slices; built once, frozen.
8. **Fractional K: compiled everywhere**. One instance per shift pattern for the GEMV, the narrow tile and the
   prefill pre-decode pass. The prefill GEMM itself only sees decoded int8/f16 fragments, so instances are
   patterns + (mode × group size), not their product.
9. **Speed objective = real-world prefill time** (2026-10-31). Per option, measured on card 0 at M = 2048
   (the recommended chunk), summed over the trunk; the head (last token only) and the embedding (one row a token)
   are ~0 in prefill, so the board's `gemm_us` and real prefill rank options the same (reports show both; `gemm_us`
   calibrated on the V1 entries). **Decode is not in the solve:** at a given size it is ~bytes-bound and nearly fixed
   (the A8/A4 modes never run at decode M, and the K mix barely changes total bytes); decode work (GEMV per K and
   fractional pattern, the spec-decode narrow tile at M 4–16, head K) is done at the end on the chosen mixes.

## 1c. Serving and release notes (2026-10-31)

1. **Mixed batches:** the A8/A4 path triggers on the GEMM's total M, so decode rows batched with a ≥ 512-row prefill
   chunk take it too; outputs can depend slightly on batching. Accepted; documented in the model cards.
2. **Chunked prefill:** ≥ 2048-token chunks are already the recommended setting; time table measured at 2048.
3. **Separate plugin [leaning yes]:** V3 kernels in a new plugin with a new encoding name (e.g. `exl3i`), so
   V1/V2 containers keep loading through the frozen exl3r4d / exl3dense untouched (no format version bump, no risk
   to V2's byte-identity); the qwen35_exl3 architecture plugin dispatches by encoding. Shared code (trellis decode)
   copied or factored into a common header.
4. **Kernel selftest matrix:** every (mode × group) and every K pattern against a torch reference, automated. Cases
   that bit the V1 bring-up and must be in it (unit tests built on synthetic 2-D f16 operands missed all of them):
   - the real row count differs from the band's M (M = 1, 15, 511, 512, 513, 952, 2048): the op must compute
     `rows` of the operand, and the A8/A4 threshold and tile / split-K choice must see the real M;
   - bf16 and f32 output stores (the graph is bf16; logits f32), not only f16;
   - the GDN `ssm_out` input, a 3-D `[T, 48, 128]` buffer read as `[T, 6144]` (row pitch from the flat layout);
   - refusal tests for malformed K / mode / group strings in the encoding, so the gates cannot rot;
   - a per-container **census** of the resolved (kernel, K pattern, mode, group) per tensor, diffed against the
     `.mix`: a mis-parsed `act=` that silently falls back to f16 is correct and slower, invisible to KLD;
   - the loaded plugin's build hash printed (a stale `.so` was benchmarked as new more than once in V1).
5. **Activation-cost model:** CPU proxy (`aqproxy.py` on activation dumps) for the shape, ~30 real probes to
   calibrate (as E1c), the stream refines with whole mixes.
6. **Submission size:** leaderboard scoring is expensive (tens of minutes a checkpoint). Score every variant
   locally and submit only the variants on our own measured frontier, in a modest number per round.
7. **Fractional K in the format (convention, 2026-10-31):** follow exllamav3 1.6.3, which already stores half-K:
   - The rate is implied by the tile width: a 256-weight tile is 16·K uint16 words (K2 32, K2.25 36, K2.5 40,
     K2.75 44), so the trellis plane is u16 `[tiles_k, tiles_n, 16K]` for every K, integer or not. K must be a
     multiple of 1/16; the menu uses quarters.
   - No pattern is stored: each rate has one canonical step mask, the even spread (KA + 0.25: 0x8888, + 0.5: 0xAAAA,
     + 0.75: 0xEEEE; step i shifts KA + bit (i mod 16) bits). exllamav3 export → radiance import → kernels all derive
     (KA, MASK) from the width; a new mask would need a new rate or a stored field (not planned).
   - Sub-1 rates (K0.25–0.75, KA = 0) add the salted 0-bit read (`viterbi_tiles_pat`): same width rule, plus a codebook
     variant flag in the encoding (`mul1s`), since a reader without the salt would decode wrong values silently.
   - Encoding string: today's EXL3 encoding plus the new fields, e.g. `exl3i:k=2.25,cb=mul1,act=a8,g=128`;
     the import derives `k` from the width and refuses a mismatch; the reader's equality check compares all fields.
   - Then grep every K gate in the plugin (V1's K=1 support touched five sites and three tests) and add refusal
     tests for malformed K / mask / act / group values to the selftest matrix.
8. **Engine version:** V3 targets radiance 1.15.x (ABI 32: plugins export a layout fingerprint). Take the
   prefill-softmax fix (PR #362: rescale per row, not per wave; measured on our eval: KLD −0.12%, packed and unpacked
   runs agree with it) with the port, and re-baseline a few V2 picks on the new engine once.
9. **Output store dtype and range:** the int8 epilogue picks its store dtype fresh. Check f16 vs bf16 vs f32 rounding
   of linear outputs with the E1b hook (bf16 storage is ~1.1e-3 relative L2 a GEMM, comparable to A8-g128's cost at
   ≥ 6 bpw), and record max |y| of the residual writers (down, ssm_out, attn_output) over the long verify traces:
   the f16 tile kept for K7/K8 overflows at 65,504, and one inf costs more than any quant choice.

## 1d. turboderp's tensors (2026-10-31)

- **Options at every K**: his published EXL3 tensors join ours in V3's option set, at all bit widths (V2
  allowed only ≥ 5 bits). Cheap: containers and proxy rows exist; the knapsack drops what does not pay.
- **Why they win at high K, first:** per tensor, ours have ~23% lower proxy error at every K (0.74–0.80×; ours
  worse in only 1–4% of tensors), yet V2's picks take 104 of his tensors at 5.5 bpw, 52 at 6.0, 42 at 8.0, and the
  best ours-only 6.0 mix lost clearly on held-out data (0.00112 vs 0.00072). Either the fit's per-family factor
  (KLD per unit of input-side proxy error) differs for real, or it is a fitting artifact plus selection noise.
  **Early V3 experiment:** single-group swaps ours-K5 vs his K5 (and K6) on the same tensors at ~4 depths,
  measured end to end. If his win despite worse proxy error, the input-side proxy misleads at high K: then
  output-side Hessians (YAQA) or calibrating on model-written text (as his self-calibrated "SC" quants do) should
  shape the V3 requant.
- **Free evidence first (CPU):** his repository ships per-tensor tables for this model (`kld_table.json`: a
  sensitivity S and KLD at K1–8 from trace-based noise attribution; `measurement.json`: per-group 2→3-bit dKLD), and
  his recipes give ~22 tensors the top K and ~65 the next one at each size. Rank-correlate S with kfit2's
  sensitivities, and intersect his top-K tensors with the ones V2 takes from him: a large overlap means his trace
  sees something our input-side proxy does not. (Done 2026-10-31: his S vs our fit Spearman 0.71, weak within families: not
  used as a per-tensor prior; sub-2 options are priced from our own measurements.) And read where his SC calibration trace (self-sampled, ~250 × 2048 tokens) enters: the GPTQ
  Hessian, or only the allocation.
- **Fused GEMMs force K ties:** his recipes tie k+v and gate+up to one K because his fast path runs them as one GEMM.
  If V3's time table measures fused launches (q/k/v, gate/up, GDN qkv/z), the solver must tie K and group size
  across them, or price the unfused launches. Decide which before the time table.

## 2. Quality items

Ranked by expected KLD gain at 3–4 bpw (estimates, `EXPERIMENTS-new-format.md`):

| item | est. gain | needs |
|---|---|---|
| fractional K: one generic trellis (shift pattern, any K = a/b); quarter steps K0.25–K5, half steps K5–K8 (26 levels), K0 = drop | −5 to −7% (more below 2 bpw) | quantizer + decode kernels take the pattern as a parameter; one quant run per K |
| row-slab LM head: a K per run of 128-row blocks, vocab rows sorted by the head's Fisher diagonal (~400 rows hold half of it, one block 39%) | under test | a few head GEMMs + a gather in the plugin; heads quantized at half steps as the baseline |
| scale gain per K and family (attn_output / ssm_out larger at ≥ 5 bits) | 0 to −4% | sweep (`quantize/gaintest.py`), then requant |
| damping 0.005 for attention / GDN input projections | ~−0.5 to −1% | requant (`quantize/damptest.py`: −2…−6% on those tensors) |
| K0 / K0.5 | large at < 2 bpw only | as half steps |
| YAQA output-side Hessians (§7) | paper: −15 to −30%; **ours so far: worse** (gates 2, 2b) | full-token gate 2c running; diagonal-only output weighting next |

All of these need a **requant**: K1–K6.5 in half steps is ~12 runs at ~1 h each on one card ≈ 12–14 h (two nights,
or one TP2 night when both cards are available).

## 3. One release, built in overlapping phases (2026-10-31)

All improvements ship together (they compound; the knapsack drops what does not pay). The card is the bottleneck, so
work overlaps it:
- **Phase A:** fold kernel (A8) → time table → speed-aware knap3 → stream on V2's quants with modes {f16, A8}.
  Meanwhile, off-card: half-step trellis, int4-grid codebook, A4 tile (R1 tested and dropped, §7).
- **Phase B:** one requant (one to two nights on card 0, ~16 runs): K1–K8 + half steps + int4-grid K2/K3, damping
  and gain baked in. New options join the stream; the fitted model carries over as the prior.
- **R1 is all-or-nothing:** a residual-stream rotation changes every tensor's basis, so rotated and unrotated
  quants cannot be mixed. Decide R1 before Phase B. If adopted, Phase A's measurements only inform the prior.

## 4. Solver

- `knap3.py`: min KLD s.t. bytes ≤ B, via a Lagrangian on time inside the exact byte DP (cost = KLD-cost + λ·time).
- **Time table:** microbench per (shape, K, mode) at M = 2048 on card 0, the board's own measurement point. Sum over
  the trunk = predicted `gemm_us`.
- **KLD model:** kfit2's structured model as the prior (V2's ~1,250 measurements, activation error 0), plus the
  activation-quant terms from E1b/E1c (`aqcal.py` family factors) as the new, fitted part.
- **Several speed variants per size:** solve at a few λ per size, e.g. *quality* (min KLD), *balanced* (the knee)
  and *fast* (≤ IQR-V3's `gemm_us` at that size). Fewer sizes than V2 so the count stays sane [decide which].

## 5. Search pipeline (lessons from V2)

- Streaming from the start; exploration kinds (tilt, random direction) and auto shares on per-kind beat rates.
- **Data discipline:** fit on fit-pool slices; **select** (local-search accept/reject, best-measured picks) on
  search-pool slices; monitor on a fresh verify slice (**ver3**), final on another fresh one (**ver4**). ver1 / ver2
  are spent. The verify pool has ~520 unused docs; both 250k traces already serve the long check.
- **Validation curve from day one** (held-out score of the best-so-far picks), stop when it goes flat.
- **Budget:** one card, ~30 evals/h (int8 evals ~25% faster), ~600–800 evals ≈ 20–30 h. Fewer targets (release
  sizes only), short high-bpw phases (V2's 6 h one bought ~nothing).
- Long-context check: one card fits the 262k bf16 KV only up to ~4 bpw; above that cap the context or arrange TP2.

## 6. Order of work (Phase A first)

1. Fold kernel: A8-g128 × int8-rounded EXL3 decode, row scale at store; byte-exact torch reference; E2 re-measure.
2. End-to-end on one container (K4, all A8): KLD and `gemm_us` vs V2's kernel. Gate: ≥ 1.4× faster GEMMs at
   ≤ +10% KLD, else stop and rethink.
3. Time table (all shapes × K × mode) on card 0.
4. knap3 with the time constraint; first mixes; stream.
5. Phase B requant once the off-card pieces work; options join the stream; final, long check, release.

Checks added from the session-log mining merge (2026-10-31, `local/mine/merged-2026-10-31.md`):
- **Before step 3 (one card slot):** time gate_up / down / gdn_in at full and half N (the board may time per TP2 rank)
  at stock and 210 W, f16 and A8-g128, plus an M = 16–511 sweep (f16 vs iu8; "same speed below 512" is unmeasured).
  One calibration point (all V1 entries share a kernel) cannot show per-shape ratio changes.
- **Activation prices:** the E1b hook quantizes every prefill step with T > 16, V3 only M ≥ 512 (E1/E1c costs are
  slight upper bounds); and activation costs were measured on a K6 base only. One E1b batch on a K2/K3 base, gated
  at M ≥ 512 vs ungated, checks both.
- **Before step 5:** requant one layer at K5/K6 with the int8-rounded codebook on the card and swap it in (expect ≈ the
  float quant); rerun E3b's float-vs-int8 comparison with the widened scale search. Phase B is 1–2 nights.
- **Selection and release hygiene:** freeze pick rules before held-out reads and log each ver3 read with the decision
  it informed; smoke tests on fit/search slices only; reserve long verify docs for a later slice; record each
  measured container's sha256 (the release must be byte-identical to what was measured); distinct op names for the
  V3 plugin (kernel choice follows plugin file order). radiance #328 (split-KV row sum in libr4d attention) is an
  open PR, not in 1.15.0: our plugins don't implement attention, so check during the port whether our shapes take
  the split-KV path, and if so A/B it like #362.

## 6b. Finer K: what to build (2026-10-31)

- Build the trellis generically once (per-position shift pattern → any rational K). **Menu (2026-10-31): quarter steps
  K0.25–K5, half steps K5–K8**, 26 levels, ~285 GB of trunk quants. Quantize in order of use: 2.25–4.75, then
  1.0–2.0, then 5.5–8, then the sub-1 levels. Heads at half steps (one K step of the head is 0.047 bpw; a half step
  halves the rounding the trunk has to absorb); quarter-step heads are the fallback if the row-slab head (§2) fails
  (a quarter step = 0.012 bpw, near the trunk's own granularity; cost: head quant runs only).
- **The quantizer side already exists for K ≥ 1 (found 2026-10-31):** exllamav3 1.6.3 (the build on the GPU box)
  quantizes half-integer K: `frac_k`, `quantize_tiles_frac_kernel<KA, MASK>`, step width = KA + bit (i mod 16) of a
  16-bit mask (rate KA + popcount/16, in 1/16-bit steps), tile = 16K uint16 words. Only MASK 0xAAAA (KA + 0.5) is
  instantiated, for KA 1–3 in the reference checkout; quarter steps are new instances (0x8888 = KA + 0.25, 0xEEEE =
  KA + 0.75, same even spread as the CPU test) and KA up to 7. **Sub-1 (KA = 0) is not covered:** a 0-bit step
  would repeat the previous state's value; it needs the salted read of the CPU test (`viterbi_tiles_pat`), new in
  the quantizer and every decode kernel. The serving side (our radiance plugin) needs the pattern in GEMV, the
  narrow tile and the prefill pre-decode regardless.
- **Below K2 the damage steepens (measured 2026-10-31):** our proxy error falls 2.11 bits per bit from K1 to K2
  (2.00 from K2 to K3, 1.97–1.88 above), and V2's fitted model (KLD ∝ proxy^1.004, one factor for K1–K3)
  under-predicts the K1-dominated swaps by 20% in-sample, 36% cross-validated (K2: 5%, K3–K8: ±1–4%; only three
  K1 swaps). The real K1→K2 slope is ~2.3–2.5 bits per bit, matching turboderp's note (~2.35 vs ~2.0 on this
  model). So: price every sub-2 option from measurements, never by extrapolating the proxy; give K1 and each sub-2
  level its own family factor in the fit; quantize K0.5 first and K0.25 / K0.75 only if K0.5 earns picks.
- Sub-2 steps matter only for the 1.5–2.0 bpw sizes. Below 1 bit the trellis keeps little (Gaussian
  D(R) = 2^-2R: K0.25 removes ~29% of a weight's error energy, K0.5 50%, K1 75%), so they suit only very insensitive
  tensors. K0 = drop the tensor (zero) or a small low-rank stand-in: no trellis; testable early with zero-donor
  probes on the cheapest tensors (L0–L3 ffn_down, K2 cost there +0.00006). Continuous K beyond
  quarter steps: < 1% more, each K costs a quant run → no.
- Row-slab K (a K per 128-row slab inside a tensor): deferred for the trunk. E3c put its ceiling at ~0.05 bit, but
  E3c saw only the input-side Hessian, which cannot tell rows apart; the output-side Hessian (YAQA's H_out diagonal)
  can. Revisit if H_out shows large per-row sensitivity spread. Needs a K per slab and per-slab scales in the format.
  **For the LM head it is being tested now** (the head's Fisher diagonal is exact and cheap: Σ p(1−p) from the bf16
  golden's top-512, no backward), see §2.

## 7. Outside research

- **R1: tested and dropped (2026-10-31, `EXPERIMENTS.md`):** A8-g128 activation error 6.6× worse, because the norm
  gains that shrink the outlier channel must be folded into the weights. Original note:
  **R1, SpinQuant's residual-stream rotation folded into the weights.** Exact only for R1 (Qwen3.8's conv1d,
  sigmoid/SiLU gates and SwiGLU block head-dim and MLP rotations); norm gains absorbed first; Hessians transformed
  analytically, H' = Qᵀ D⁻¹ H D⁻¹ Q. For weights alone little is expected after fwht128. **For V3 it is an activation
  question:** a massive-activation channel dominates its 128-block's A8/A4 scale; R1 spreads it over 5120. Test on
  CPU with the E1c dumps (inputs of every linear): A8/A4-g128 error with vs without a random orthogonal Q (gains
  divided out) on the residual readers (attn q/k/v, gdn in-proj, gate_up). Pass → fold R1 into the Phase-B requant.
- **REAL-Q (arXiv 2609.00049):** PTQ with a Fisher surrogate at each block's output (F = mean g gᵀ from one end-to-end
  backward pass) and one Adam step on the still-unquantized columns after every 128-column block; up to ~49% lower KL
  than GuidedQuant at W4A16 (scalar grids, small Qwen3 models, QuaRot). Costs: the stage-0 backward (bf16 27B:
  a TP2 night, same as YAQA's Hessians — collect both in one night) and a rewrite of the quantizer loop around
  exllamav3's LDLQ/trellis. Potentially the largest quality item; research track after Phase B, not on V3's path.
- **YAQA (arXiv 2505.22988, Cornell RelaxML, built on QTIP = EXL3's ancestor):** Kronecker-factored Hessian of the
  end-to-end KL, H ≈ H_out ⊗ H_in, and LDLQ with error feedback on both axes. ~30% lower KL than LDLQ on QTIP trellis
  codes; Qwen3-8B 2-bit: KL 0.285 → 0.227 with 2K sequences (Sketch B, ~1 GPU-h at 8B) → 0.205 with 64K.
  **Our exllamav3 already has the quantizer side** (`ldlq_2hess`, `prepare_H_out`); only H_out is missing.
  Another competitor reported no gain (method unknown to us). Cost is all in collecting H_out: an fp32 backward over
  qad windows (1,819 × 2048 = 3.7M tokens), fp32 sums; H_out is m×m for each of the 496 linears (gate and up separate:
  17408² fp32 = 1.2 GB each, ~0.2 TB for the model, ~0.1 TB as triangles), so it runs in passes over layer groups.
  The paper's −15…−30% is **unverified here: two gates so far came out worse than one-sided** (below).
  **Status (2026-10-31):** collection runs (`v3/yaqa/collect.py`: text model streamed onto two cards, sampled-label
  CE, |x|-weighted dy rows at 32 positions per sequence, all 496 trunk linears). Gate 1 passed (5.3 s per 2048-token
  sequence in bf16). Gate 2 with a bf16 backward failed: two-sided LDLQ on blocks 22/23 was +23% (K2) / +28% (K3)
  worse than one-sided. A bf16-vs-fp32 comparison on identical sequences showed why: the bf16 backward's rows are
  wrong for a minority of positions, increasingly toward early layers (H_out diagonal cosine vs fp32: 0.88 at
  layers 48–63, ~0.5 mid-depth, 0.36 at 0–15), and those rows are 3× too large, so they dominate H_out. Collection
  must run with fp32 activations (30 s a sequence; the qad pool ≈ 15 h on two cards). **Gate 2b (fp32, 6,400 sampled
  rows) failed too:** +17%/+19% worse at sigma_reg_out 0.025, +6%/+10% at 0.3 (K2/K3), trending toward one-sided.
  **Gate 2c** (running): full-token sums over the same 200 windows (backward truncated at layer 22, ~22 s/seq). If it
  fails, one more cheap test before dropping: **diagonal-only output weighting** (exllamav3 `--out_scales yaqa`,
  or a row-weighted trellis as in PR #364), since a 17k diagonal is well estimated from rows already on disk while a
  17408² matrix is not. If it passes, add a split-half check (two halves' H_out agree?) before any full collection.
- **Teacher Intervention (arXiv 2302.11812):** a QAT technique → V4 (QAD), not V3.

## 8. Vision tower and DFlash2 drafter (2026-10-31; was hoped for V2)

IQR's releases bundle the bf16 vision tower and z-lab's DFlash2 block-diffusion drafter; DFlash2 is a large decode
gain (published acceptance 4.1–5.5 tokens). Our `qwen35_exl3` architecture plugin declares only the 64 text layers
(no `model.visual.*`, no drafter), so V2 ships text-only. For V3's plugin (new anyway, §1c.3):
- **Vision:** port the media path from radiance's `qwen35_bf16` / `qwen35_fp8` (`rad_block_vit.h`); the tower stays
  bf16; radiance skips it when media is off, so it costs disk only. **The board does not count it either:** it
  scores the language model's bytes (IQR-V3's IQ4R lists 14.99 GB against a 17.57 GB file carrying the tower and
  DFlash2), so bundling costs no bpw.
- **DFlash2:** port the drafter from `qwen35_fp8` (`rad_block_dflash2.h`): the trunk taps at layers 5/19/33/47/61
  into `dflash_src`, the context pass and the query pass; containers via `rad-convert --draft-model`.
- Validate: drafts are verified, so KLD is unchanged; measure acceptance and decode tok/s vs no drafter.
- Decode speed work (§1b.9) then includes the narrow tile at the verify M (4–16).

## Later / backlog

- YAQA output-side Hessians: bf16 forward + backward over the qad pool, a full TP2 night; time it once V3.0 works.
- Row-slab K, continuous K, 2-tier head, residual re-ranker with more data, A4.
- V4 healing: a third-party fixed-code QAD on this model gained much more training only the last 16 layers than all
  of them (on a proxy metric). Start scale-only healing on late layers, judge on verify-pool KLD.
