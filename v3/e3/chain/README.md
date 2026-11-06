# #102 · Chained tail-biting: one trellis ring per n row tiles (review 2026-11-05 idea 1), CPU proxy

Worker w38, colo. Source: `local/mine/review-2026-11-05-ideas.md` idea 1. Changes the codes, so it must be decided
before the 27B ladder. **Runtime cost:** decode / prefill only if adopted (one extra neighbour-word read per tile,
see "Format and kernels"); the proxy itself is quantizer-side.

## Question

EXL3 codes each 16×16 tile as a 256-step tail-biting ring: the first ~16/K weights read their 16-bit window from the
tile's own last bits. #57 measured that constraint at +5.5% (K2) / +4.0% (K4) proxy error against a free start
(i.i.d. Gaussian tiles; free/TB 0.948 / 0.962). Idea 1: let tile r take its start window from the last word of tile
r−1 in the same 16-column block (the tiles LDLQ quantizes together), closing the ring once per chain of n tiles; if the
constraint's cost scales as 1/run length, n = 4 recovers ~3–4% proxy error.

## Setup

- `viterbi.c` = a copy of `v3/e3/viterbi.c` plus `viterbi_chain_tb`: a chain of n consecutive tiles is one ring of
  L = 256 n positions (step i shifts pat[i mod plen] bits; integer K: one-element pattern, K2.5 = [2, 3]); ring
  constraint (s₀ ≫ k₀) = low (16 − k₀) bits of s_{L−1}; exllamav3's search (NTRY 1): a free run rolled by L/2 gives the
  wrap group at position 0, then one constrained run. At n = 1 and integer K it is `viterbi_tiles_tb` exactly.
- `trellis_chain.py`: `gauss` (i.i.d. tiles, #57's check, plus the n curve to 32) and `tensor` (trellis.py's block LDLQ
  on 128 sampled rows, rotation seed 0, damp 0.01, factors {1, 1.15, 1.3} stacked; the chains never cross factors;
  U = chol(H⁻¹) computed once per job and shared by the arms, so every arm sees the same rows and metric). Arms per
  (tensor, K): free start (n = 0, the old CPU reference), n = 1 (EXL3 today), 2, 4, 8 (8 = all 8 row tiles of a factor
  in one ring). Proxy error rel = tr(ΔW H ΔWᵀ) / tr(W H Wᵀ) under the calib H_in, best factor per arm.
- Tensors: #57's set (`v3/e3/f16gap.sh`: 16 tensors, layers 0 / 10 / 11 / 43 / 63, all families), codebook mul1
  (as #57; the trellis constraint is codebook-agnostic), K 2 / 2.5 / 3 / 4. Gaussian: 2,048 tiles at K1 / 2 / 2.5 / 3 / 4.
- `run.sh` (colo, single-threaded jobs, nice 10, ≤ 6 at once beside #103's 4), `report.py data/`.

## Gate (passed 2026-11-05 ~11:00 PM CST)

Reproduce #57 at n = 1 within 0.2%: Gaussian tiles (128, seed 7, mul1) give free 6.4661e-2 / TB 6.8175e-2 at K2
(0.948) and 4.3723e-3 / 4.5438e-3 at K4 (0.962), identical to #57's numbers to every printed digit, and the chained
search at n = 1 returns arrays bit-identical to `viterbi_tiles_tb` (K2, K4). Fractional ring sanity: K2.5 free/TB
0.953 (between K2 0.948 and K3 0.954).

## Rule (from the task, written before data; statistic fixed here 2026-11-05 ~11:05 PM CST)

**Adopt for the ladder if n = 4 cuts proxy error ≥ 2% at K2, K2.5 and K3** — the median over the 16 tensors of
rel(n=4) / rel(n=1) ≤ 0.98 at each of the three K — **and every tensor family ≤ 1.00** (geometric mean of the family's
ratios at n = 4, at every K run). Otherwise not adopted; the n curve, the kernel / format implications and the quantizer
memory / time are reported either way.

**Prior (written before the tensor data):** probably fails. A ring of 256 n positions and n rings of 256 have the same
bit budget and the same local structure (every weight reads a 16-bit window of a circular bit string); only the global
topology differs, which should not matter for i.i.d.-like targets. What #57's free start had is ~16 − K *free* bits per
tile that no ring can encode, and chaining does not give those back: the free-vs-ring gap per tile is the free start's
advantage, not a cost of short rings. What chaining can amortize is the search heuristic's loss (a guessed wrap group,
once per ring instead of once per tile) — #57 put that at ≤ ~0.4% (K2) with 16 candidates. First Gaussian check
(128 tiles, n = 2): −0.4% at K2, +0.07% at K4.

## Results (2026-11-06 3:11 AM CST) — **fail: not adopted**

Full tables: `report.md` (`python3 report.py data/`), raw lines `data/` (64 tensor jobs × 5 arms, 5 Gaussian jobs).

**Rule:** n = 4 / n = 1, median over the 16 tensors: **0.9961 (K2), 0.9963 (K2.5), 0.9976 (K3)** vs ≤ 0.98 needed;
families above 1.00 at some K: attn_k, attn_qg, attn_v, ssm_z (single small tensors, noise). **Not adopted.**

The n curve (real tensors, median of rel(n) / rel(n=1); free / TB = the free start's advantage, geometric mean):

| K | free / TB | n = 2 | n = 4 | n = 8 | share of the free-vs-TB gap recovered at n = 4 |
|---|---|---|---|---|---|
| 2 | 0.9440 | 0.9960 | 0.9961 | 0.9958 | 5% |
| 2.5 | 0.9464 | 0.9957 | 0.9963 | 0.9968 | 10% |
| 3 | 0.9505 | 0.9970 | 0.9976 | 0.9966 | 4% |
| 4 | 0.9585 | 0.9976 | 0.9979 | 0.9969 | 1% |

Gaussian tiles (2,048 i.i.d. tiles, mse(n) / mse(n=1)) — the clean version of the same curve:

| K | free | 16 wrap tries, n = 1 | n = 2 | n = 4 | n = 8 | n = 16 | n = 32 |
|---|---|---|---|---|---|---|---|
| 1 | 0.9376 | 0.9927 | 0.9923 | 0.9912 | 0.9914 | 0.9915 | 0.9915 |
| 2 | 0.9485 | 0.9962 | 0.9962 | 0.9961 | 0.9960 | 0.9960 | 0.9960 |
| 2.5 | 0.9521 | – | 0.9977 | 0.9972 | 0.9974 | 0.9976 | 0.9974 |
| 3 | 0.9544 | 0.9980 | 0.9980 | 0.9976 | 0.9975 | 0.9976 | 0.9976 |
| 4 | 0.9605 | 0.9990 | 0.9990 | 0.9991 | 0.9990 | 0.9990 | 0.9989 |

Reading:
- **No 1/n scaling.** The gain is flat from n = 2 to n = 32 and equals what 16 wrap candidates per tile already give
  (#57's NTRY 16): −0.4% at K2, −0.2% at K3, −0.1% at K4. A chained ring only amortizes the wrap *guess*; a ring of
  256 n positions is not a better code than n rings of 256 (same bits, same local window structure). The 5–6% free
  start advantage is the ~16 − K unpaid start bits per tile, which no ring can encode — the prior above, confirmed.
- **New number:** #57's tail-biting cost holds on real tensors under block LDLQ: free / TB 0.944 / 0.946 / 0.951 / 0.959
  at K2 / 2.5 / 3 / 4 (Gaussian 0.949 / 0.952 / 0.954 / 0.961). So earlier free-start CPU sweeps are 4–6% optimistic
  on real tensors too (ratios between arms unaffected, as #57 said).
- Per-tensor noise: small tensors (attn_k / attn_v, 1,024 rows; attn_qg L11) swing ±1–2.5% between arms because
  LDLQ feeds a different code choice forward; ffn / ssm / attn_output tensors sit at 0.993–0.998 at every n.
- If the residual ~0.2–0.4% at K2–3 were ever wanted, the format-preserving route is more wrap candidates in the
  quantizer (NTRY 16 gives the same number with no format change; ~16× search time) — not worth it either (#57).

## Format and kernels (not needed: not adopted)

What adoption would have cost, for the record:
- **Format:** a per-tensor `chain = n` flag; codes otherwise the same size. Chains must be aligned to n tiles along N
  (EXL3 stores `encoded[tiles_k][tiles_n][256 K / 16]`, so the row tiles of one 16-row k-slice — the proxy's
  16-column block — are contiguous; tile r−1 is the previous 256 K / 32 words).
- **Decoders:** every decoder indexes a tile's words modulo W = 256 K / 32 for the wrap (`exl3_dq.cuh`:
  `ptr[i % (bits * 256 / 32)]`, `dq`, `dq2`, `dq4`, `dq2x2`, ...). Chained: the wrapped word (i = W − 1) comes from
  the previous tile (`ptr[−1]`), or at a chain start from the chain's last tile (`ptr[(n − 1) W + W − 1]`): one
  select per load, and the GEMM's shared-memory tile set must include the neighbour (true when its N extent is a
  multiple of n tiles and chain-aligned; GEMV / narrow-tile / MoE / int8-plane / reconstruct kernels and the CPU
  decoders all need the same change). Fractional K (qtf2): same, with the ring length L = 256 n.
- **Quantizer:** the GPU search (qtf2 / exllamav3) runs one ring per thread block: history L × edges bytes per ring
  (K2: 4 MB per tile → 4 n MB per chain; same total per tile), but a 16-row k-slice then has tiles_n / n independent
  blocks instead of tiles_n — e.g. N = 1,024 (attn_k / v) gives 16 blocks at n = 4, too few for 64 CUs, so the
  quantize time of narrow tensors grows up to ~n× (LDLQ is sequential over k-slices; nothing else to batch).
- **CPU proxy cost:** same work per tile: arm time n ≥ 2 vs n = 1 +0…+8% (ffn_down median 564 s → 581–610 s; others
  163 s → 161–172 s), traceback memory n × 256 × 2^(16−k_min) bytes per ring (n = 8 at K2: 32 MB / thread).

**Runtime cost:** none (not adopted; the format and kernels stay as they are).

## Files

`viterbi.c` (copy of `v3/e3/viterbi.c` + `viterbi_chain_tb`), `trellis_chain.py`, `run.sh`, `report.py`, `report.md`,
`data/` (one jsonl per job). Ran on colo `/data/work/w13/w38/chain` (built there, `-march=native`), 2026-11-05 11:00 PM –
2026-11-06 3:11 AM CST, ≤ 6 single-threaded jobs (9 for the last 2 h). A runner bug (a filter that let the second pool
start one ffn_down job twice) was caught at 1:15 AM CST: the two copies were killed and the job rerun; every data file
was checked to hold exactly the arms n = 0, 1, 2, 4, 8.
