# #174-02 rules — trellis embedding served from the stored form (w144, 2026-11-13)

Written and committed before any scored run of this experiment. Brief: `rad-local/workers/instr3/w144.md`.

## What is measured

1. **Identity (correctness, not a selection).** On one trunk (`/data/q27/w138/packs/v3-k4.125-r16k4-a8.rad`: K4.125
   i8 ladder trunk, R16 K4 head, A8), two containers per trellis level, at levels K2.5 and K4 (i8, natural rows):
   - **A** = the trunk with `token_embd.weight` taken from the trellis donor (radmix): served by `embed_exl3i` from the
     stored record;
   - **REF** = the trunk with `token_embd.weight` bf16 rows = the oracle decode (`exl3i_decode_rows`, RNE bf16) of the
     same record (`embref.py`), served by `embed_lookup`.

   Scored on golden-ver3-8k (box2 `score.sh`: tp 1, bf16 KV, max-model-len 16384). **Pass:** |mean KLD(A) −
   mean KLD(REF)| ≤ 1e-5 at both levels. Also reported: top-1 agreement, per-doc max |Δ|. Plus the GPU unit test
   (`exl3dense_embed_exl3i_gpu_test`): device row == host row in every bit (every codebook, rates m 2 … 128, both row
   orders), and the host test's agreement with the oracle.
2. **VRAM as served.** Static VRAM (radiance's load report) of A vs the same trunk with V2's Lloyd l3 embedding (V2 3.00's
   token table, radmix) and vs the trunk as built (bf16 embedding, host-pinned). Reported against the trellis entry's
   container bytes (record + suh + svh + codes plane); any extra device buffer is reported.
3. **Time.** `--profile-ops` µs per call of `embed_exl3i` vs `embed_tq` (l3) in the same serve setup, the kernel bench in
   the GPU test (M = 1 … 8192, random ids over the whole table = no shared blocks), and end-to-end 8k prefill and decode
   step (A vs l3 container, same card, interleaved). Expectation: ~0 of a run; reported as is.
4. **KL per option (pricing anchors, A8).** Exactly w135's #174 A8 anchor setup (`search/v3/emb174/a8anchor174.sh`: a
   search incumbent's hostd mix + the embedding donor + `act … a8g128` on every trunk weight, radmix on V2 3.00, rad10
   radiance, s171b whole, bf16 KV, 16k), so each new row pairs per document with w132's A8 score of the same incumbent
   and with w135's Lloyd swap rows:

   | incumbent | trellis arm | paired Lloyd row (w135) |
   |---|---|---|
   | v3s171-2.5-explore-00419 | t2.5 (≈ 398 MB) | base l3 (497 MB, the incumbent) |
   | v3s171-3.5-block-00483 | t4 (≈ 636 MB) | e4 l4 (656 MB) |
   | v3s171-5-pair-00449 | t6 (≈ 954 MB) | e6 l6 (973 MB) |
   | v3s171-6-explore-00372 | t5 (≈ 795 MB) | e5 l5 (814 MB) |

   Each trellis arm has fewer bytes than its paired Lloyd row. Reported per part (calib / xcorp) and pooled: log KL ratio
   vs the incumbent and vs the paired Lloyd row, per-document paired bootstrap SE (`mix/paired.py` convention).

## How the results are used (no selection here)

- No accept / reject decision is made in this experiment. The trellis options go into the menu (menu174 format) with
  `err` from the row-error price model w135 fitted (`embprices174.json` floor model: err = F + a × rel, rel measured on
  the real quantized table), and the anchor rows go to the lead / w143 as fit rows (own labels
  `emb17402-s171b|c|full` / `|x|full`, the incumbents as anchors) for the final re-solve, which picks per size.
- **Price check (reported, decides nothing):** an arm agrees with the price model when its measured pooled log ratio vs
  the paired Lloyd row is within 2 paired SE of the model's prediction.
- If the identity check fails, nothing from this experiment goes into a menu.

## Data note

s171b is the slice the 11-13 overfit audit proposed to retire (241 search decisions, #174 / #172 / ref134 reads). These
four reads are pricing rows paired per document with w135's Lloyd rows on the same slice (a fresh slice would need every
Lloyd arm re-scored); they make no accept / reject decision. golden-ver3-8k is read only for the identity check (A vs REF,
same trunk), which ver3's status ("spent for any selection") allows. Verify (ver4) and public are not touched.

## Amendment 1 (before any anchor run; identity / VRAM / time runs had started, no anchor data exists)

- **Build control.** The anchors run on this branch's build (rad10-perf install 5c9956c + W/home), not rad10/home
  (a3799b5), which w132 / w135 scored with. First eval: the 2.5 incumbent (v3s171-2.5-explore-00419) **unswapped**, A8,
  same command. If its per-position rows equal w132's A8 rows of that incumbent (byte-identical `.rows`), the trellis arms
  pair with w132's incumbent rows and w135's Lloyd rows as planned. If they differ, each incumbent is also scored unswapped
  on this build (3 more evals) and every trellis arm pairs with its same-build incumbent; the comparison with w135's Lloyd
  rows is then reported as cross-build (labelled so).

## Amendment 2 (2026-11-13 4:40 PM CST, before any run below; lead notes 11-13: host-RAM embedding, K8 mul1 one kernel)

The lead's notes ask whether a host-RAM embedding read by the same `embed_exl3i` (mul1 K8 trellis, codes in pinned host
memory) is ≈ bf16 and what reading it over PCIe costs. Added, nothing above changes:

5. **Host gather cost (no KL).** `exl3dense_embed_exl3i_gpu_test --bench-only [--host] [--tokens golden-ver3-8k ids]`
   on one box2 card: the same kernel with record / suh / svh in device memory vs in mapped pinned host memory
   (`--host`), K2.5 / K4 / K8, M = 1 (decode) … 8192 (8k prefill), uniform ids (worst case) and real text ids. Reported
   as is, beside the served per-op µs of bf16 `embed_lookup` (host-pinned, radiance's default) from perf144.
6. **KL of the host candidates (pricing, no selection).** Three more A8 arms in exactly the setup of item 4
   (a8anchor144.sh, ds8, s171b whole, this branch's build), all on v3s171-2.5-explore-00419:
   - **bf16**: the checkpoint's bf16 table (the K0.125 base pack's token_embd);
   - **t8m**: mul1 K8 trellis (`/data/q27/emb/trellis/emb-t-m1-K8.rad`, quant144.sh + donor144.sh as the other levels);
   - **i8row**: int8 per-row RTN (bf16 scale = absmax / 127), served as its bf16 dequantized rows (`embrtn144.py`): the
     KL an int8 row gather would give, without building that gather.
   Reported: pooled per-document paired log KL ratio (calib / xcorp parts, `calobs.py`) of t8m vs bf16 and i8row vs bf16
   (and each vs the ctl row). **"≈ bf16"** for an arm means its paired 2-SE interval vs bf16 contains 0 and lies within
   ±1%. The row errors `rel` (int8rel144.py: int8-row / int8-g128 / int8-g128h; K8 from embsrc.py) and their floor-model
   `err` are reported beside. No accept / reject here: the lead decides the host format and the loader default.
