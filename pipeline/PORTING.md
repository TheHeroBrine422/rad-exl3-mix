# What is ported, and how each port was checked (2026-11-06; level passes, S5b, S7 / S8, tile CD + H_out: 2026-11-07)

Rule (task w52): no new quantizer behaviour; every port bit-identical to the experiment script it replaces, on small
cases (0.8B tensors or single 27B tensors). "Bit-identical" for files = the same tensor names, dtypes and values
(`torch.equal`) and the same metadata dict. Not file bytes: safetensors writes the `__metadata__` keys in hash order,
which differs between processes (found here: the first byte-level comparison failed on every file for that reason
alone, tensors and metadata equal).

Raw outputs of every check: `data/w52-2026-11-06/` (`result.txt` lists the GPU units; its two "C quant rc=127" lines
at 18:58 UTC come from editing the driver while bash was reading it, no command ran: the C unit's real run is in the
BCD block).

| stage | rxq | original (rad-exl3-mix) | check | result |
|---|---|---|---|---|
| S2 tensor quant | `quant.py`, `engine.py` | `quantize/ourq.py` (ladder settings: i8, qtf2, scaled H, σ 0.025 / 0.005) | `tools/run4070.sh A`: 27B layer 10 (6 tensors) + layer 39 q/k/v/o at K2.5, one per-tensor job each, vs hconv `q-none` (ourq, 11-04) | **PASS 10/10** (trellis, suh, svh, marker torch.equal; proxy_err, g_scale, σ, out_scales equal). ffn_down first came out with 17 / 17,408 suh and 1 / 5,120 svh entries 1 f16 ulp off (codes equal): its refit falls back to the CPU on this card and the BLAS thread count differed (4 vs hconv's default); rerun at the default count: identical (unit B, `a2-check.txt`) |
| S2 with H_out | same (`--hout`, ldlq_2hess) | ourq `--hout DIR --sigma-reg-out 0.01` | `run4070.sh D`: 27B L10 in_proj_z / gate, L39 o_proj vs hconv `q-bn1664` | **PASS 2/2 comparable** (in_proj_z as a j > 0 group member, gate_proj: all tensors equal through ldlq_2hess). L39 o_proj: no reference tensors (hconv's bn1664 arm never wrote layer 39, it OOM'd in that layer's MLP); its proxy_err equals the reference log's to all 17 digits |
| S5 writers requant | same (`--hin`, `--wpre`) | ourq `--hin --wpre` (asym27 test08 G1 path) | `run4070.sh C`: rxq mkdenoise (GPU) on sib27test/ref2 → 48 0.8B writers vs d1103 `q-k3asym_r0` | **PASS 48/48** (192 tensors; rxq's mkdenoise ran on the GPU in fp64, as test08 G1 did) |
| S5 G / F | `asym.py` Denoiser | `proxy2b/exp/w07-denoise/mkdenoise.py` | `tests/check_cpu.py`: 4 groups of ref2 (layers 0, 3), CPU fp64, default path and `MKD_LEAN=1` path | PASS / PASS (8 + 8 files, stats equal) |
| S1 H_in blend | `stats.blend_hin` | `proxy2b/exp/d1108/blend.py hin` | `check_cpu.py`: h08 calib + d1111 hinX at w 0.5 vs d1111 `hi-X0.5` | PASS (96 blended + 1 link) |
| S1 H_out bn | `stats.hout_blocks` | `v3/yaqa/hconv/mkhalf.py` (MKHALF_NORM=1) | `check_cpu.py`: hconv 27B snapshots (15) → train-N / test, vs mkhalf.py run on the same snapshots | PASS (6,400 files, mkhalf.json) |
| S4 swj math | `swj.acc / fit / resid` | `proxy2b/exp/w02-fixedscale/fsutil.py` swiglu_* | `check_cpu.py`: 4 random blocks, 81-point grid | PASS (sums, fit, residuals equal) |
| S4 swj container | `swj.apply_level` | `v3/yaqa/swj27/swjapply.py` | `check_cpu.py`: q-k3base + w36 k3lv scales | PASS (28 files) |
| S3 assembly | `job.assemble` | (new: ourq writes per layer itself) | `tests/check_assemble.py`: 0.8B q-k3base split into 150 per-tensor job files over 5 job dirs, reassembled | **PASS** (600 tensors equal, same key → shard map; a level with one changed code stamp is refused) |

### 2026-11-07 (w76): level passes, decoder, C9, S7 / S8, tile CD, i4, hygiene

CPU checks only (w76 had no GPU; the 4070 was off limits): colo (`/data/work/w13/w76`, nice 19, ≤ 8 GB RSS) and the VM.
Raw outputs: `data/w76-2026-11-07/`. The 0.8B (Qwen3.5-0.8B bf16, downloaded to colo) runs on transformers' torch GDN
path (P2B_BLOCK=causal_conv1d,fla) with a synthetic corpus (random token ids: no calib document read, no ledger use).

| stage | rxq | original (rad-exl3-mix) | check | result |
|---|---|---|---|---|
| S4 swj level pass | `swjfit.py` on `forward.py` | `v3/yaqa/swj27/swjfit27.py` | `tests/check_levelpass.py swj / swjc`: 0.8B, :wq source (gate / up + 3% noise), 3 stages, a checkpoint after every document (resume path), --no-cache and the fresh-cache forward | **PASS** both (50 files each, swjfit.json equal; `data/w76-2026-11-07/levelpass.txt`) |
| S5 sibling moments | `sibmom.py` | `v3/yaqa/asym27/sibmom27.py` | `check_levelpass.py sib`: 0.8B, every layer, --wq siblings, 256-token windows | **PASS** (73 files: moments, half/, windows.json, done markers) |
| S1 H_in collection | `hin.py` on `forward.py` | `eval/gold27/mkhin27.py` | `check_levelpass.py hin`: 0.8B, 3 stages, 256-token chunks, a flush every 2 documents; 97 Hessians byte for byte (both sides at 1 thread: at 4 threads on a loaded host the CPU SDPA is not reproducible run to run, mkhin27 against itself included) | **PASS** (97/97, collect.json equal) |
| level decoder | `deq.py` | `v3/yaqa/swj27/exl3deq.py` (swjfit27's) and `v3/yaqa/asym27/deq.py` (sibmom27's, c9e27's); tables from rad-quant-at-home `checker/exl3dec.py` | `tests/check_deq_c9.py lut / deq`: hconv's 27B K2.5 i8 layer 10 (6 tensors) decoded by all three | **PASS** (6/6 torch.equal; 4 codebooks × 65,536 states equal) |
| i4 codebook | `deq.lut("i4")` | w53's C oracle (`v3/infer/ref/exl3i_ref.c` + `exl3i_ref-i4.patch`, built in a scratch copy) | `tests/check_i4lut.py` (VM) | **PASS** (mul1 / i8 / i4: 0 of 65,536 states differ; i4 has 16 values) |
| S5b C9 in the quant job | `post.c9_tensor` (in `job.run_quant`), `job._assemble_c9` | c9e27.py (as above) | `check_deq_c9.py injob`: the writer's tensors as the job holds them + W + its group's H_in → c9-<key> file → assembled LEVEL/c9 | **PASS** (scales and c9.jsonl equal c9e27's); 10-25 s a gdn_out writer on 4 colo threads (f64; load-dependent); on the card: `tools/gpucheck27.sh c9` (prepared; also `tests/check_c9job.py` on a fake job dir: PASS on colo) |
| S5b C9 scales | `post.py c9` | `v3/e3/settle/w75/c9e27.py` (the 27B implementation; w64's `c9w.py` for the 0.8B) | `check_deq_c9.py c9`: the same level's out_proj, α gdn_out 3, DEV=cpu | **PASS** (scales and c9.jsonl equal). attn_o / down not exercised (no attention layer in the shard; down's f64 17,408² metric exceeds the RSS cap) |
| S7 knapsack | `knap.py` | knap5.py run by `v3/levels/reprice/run5.py` with mkjobs2.py's main arm | `tests/check_knap.py`: the 11-07 ladder as the library, the settings file's `[knap]` → env + solve, 2.5 bpw accurate | **PASS** (the output line byte-identical to `data/levels/reprice/swj2/out/main-acc-2.5.jsonl`); hand-off mix: 337 weights, TRUNK order |
| S8 recipe / mix text | `mix.py` | `v3/yaqa/swap27.sh` TRUNK + `swap27k.sh` packmixk | `tests/check_mix.py`: the 13 swap27 arms' `.recipe` / `.mix` (gpubox backup) | **PASS** 13/13 byte for byte |
| tile CD | `tilecd.py` (merged) | `v3/e3/tilecd/tilecd.py` (w50) | `tests/check_tilecd.py` (stub trellis quantizer, VM): same / gate (M = I, fp32 and fp64) / two-sided descent / dense refusal / wrappers | **PASS** 7/7; on the real quantizer: `tools/gpucheck27.sh cd` (prepared) |
| i4 + qtf2 | `engine.setup` (i4 allowed), ourq + exl3frac2 patched | w53's `qtf2-i4.patch` | `tests/check_i4.py` (build flags, ourq's assert, engine installs, sub-1 plan) | **PASS** 4/4; on the card: `tools/gpucheck27.sh i4` (prepared) |
| swap-pair hygiene | `pairs.py` | (new; radiff.py's identity gate generalized) | `tests/check_pairs.py`: synthetic levels and RAD2 containers | **PASS** 22/22 |

Not ported (DESIGN.md "Open items" 5): collect.py (the Σdy² collection: a two-card GPU backward that cannot run on a
CPU; `rxq/hout.py` runs it in place with its production environment, and its bn estimator is ported); the per-mix
asym driver; kfit2's refit; rad-convert / radmix / census (radiance tools, called). Not ported because not in the V3
settings: ourq's --sigma-in-sweep, per-linear --hin files (Sketch B), --hout-diag, --out-scales-off, --head-perm /
--head-mean-row (the head job comes with the slab-head decision), the multi-level memo.

## Notes from the port

- **Per-tensor jobs equal the group run.** exllamav3 draws a Hessian group's input sign vector su once, under the
  seed of the group's first tensor; later tensors of the group reuse it. A job for tensor j > 0 replays that draw
  (seed the RNG with tensor 0's seed, finalize the group Hessian) before quantizing with its own seed. Checked on
  in_proj_z, up_proj, k_proj, v_proj (j = 1, 2) in unit A.
- **CPU refit fallback and thread count:** when the scale refit runs on the CPU (RXQ / OURQ_REFIT_CPU_FALLBACK, small
  cards, 27B ffn_down), its scales depend on the BLAS thread count at the 1-ulp level (17 of 17,408 suh entries at 4 vs
  the default threads); the codes do not. Results from different donors can differ there; verification by score
  (rad-quant-at-home VERIFY.md) absorbs it, bit-identity checks must pin the thread count.
- **ourq's --save-wq rowerr** is computed from the weight after quantize_exl3 has regularized it in place (scaled,
  rotated): the per-row error is not in the original basis. Ported unchanged for parity; flagged in DESIGN.md.
- **Tile CD with H_out is not per-row weights.** H_out is diagonal per output channel in the original basis, but the
  quantizer works after the output sign / scale vector and the 128-blockwise Hadamard: there M = P S H_out S P is
  block diagonal with dense 128 × 128 blocks, and the tiles of one 128-column group are coupled. rxq/tilecd.py uses
  that M (prepare_H_out's own matrix, captured) and decides a group's 8 tile positions in turn with exact cross terms.
- **Level decode is one function.** swjfit27 (exl3deq.py) and sibmom27 / c9e27 (asym27 deq.py) used two decoders;
  they agree bit for bit on a real 27B level, so rxq.deq replaces both and adds i4 (sub-1 / i4 levels were not
  decodable by asym27's deq.py).
- **knap5 must sit in a v3/nf directory** (it reads data relative to its own file and to the cwd): rxq.knap stages a
  symlink tree; abspath keeps the link's directory.
- **The scale refit and the host CPU (GitHub issue 2).** The single-tensor path un-rotated the Hessian for the scale
  refit on the CPU (`unrotate_H(H)`, H on the host): the job's only CPU BLAS use (`tests/check_hostmath.py`: 2 call
  sites, the two `@` of blockwise_preapply_had_l_ / _r_). Today's code, torch 2.14.0+rocm10.1.0 with hipBLASLt off and
  no OpenBLAS setting, 0.8B layers 0-3 (25 tensors): box A (Ryzen 9 9950X, 2 cards) vs box B (EPYC 7443, 4 cards) 20 of
  29 files equal at K2.5 (5 suh + 6 svh values differ), 15 of 29 at K0.5 (9 suh + 11 svh), each one f16 step, trellis
  equal (C9 `c` follows the scales); the pinned stack gave 29 of 29 on both. exl3-patches 0010 un-rotates on the device
  (as original_H() and the batched path do): both stacks, both levels, box A = box B 29 of 29, check 0 call sites;
  against today's code (pinned stack) trellis equal, 5 suh + 6 svh (K2.5) and 6 suh + 2 svh (K0.5) values one f16 step.
  Three paths: card / card (un-rotation and refit fit: no host math); card / CPU refit (the refit runs out of card
  memory and `RXQ_REFIT_CPU_FALLBACK=1` refits on the CPU: host dependent, see "CPU refit fallback and thread count");
  CPU / CPU (even the un-rotation does not fit: 0010's out-of-memory branch, the old result). At the 27B ffn_down size
  (17,408² Hessian, random data, R9700) capped to the declared ffn_down job memory (9 GiB): today's tree un-rotates on
  the CPU and refits on the card, with 0010 both run on the card; neither run fell back to the CPU refit; peak 5.84 GiB
  in both. 0010 changes the code stamp (`rxq_code`), and assembly refuses a level whose tensors disagree on it
  (DESIGN.md, "Stamps"): a level made before 0010 is kept whole or redone whole.
