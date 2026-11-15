# Speed of the KL-IQR entries, IQR-V4 neighbours (#194-02, w177)

Card: box2 card 0 (rocm-smi GPU[3]); one radiance serve per row; bench.py over the chat API: batch 1, greedy, 512 generated tokens, ignore_eos, 2 warm-ups + 5 runs per context, medians. Prefill tok/s = prompt tokens / TTFT. Power cap from rocm-smi.

| row | decode ms/tok ctx0 | ctx4k | ctx16k | decode tok/s ctx0 | prefill tok/s 4k | prefill tok/s 16k | TTFT 4k (ms) | TTFT 16k (ms) | power cap (W) | load (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| iq3r-s | 27.00 | 27.53 | 28.81 | 37.0 | 2582 | 2433 | 1607 | 6755 | 210 | 17.3 |
| iq3r | 27.53 | 28.09 | 29.37 | 36.3 | 2577 | 2431 | 1610 | 6760 | 210 | 17.5 |
| iq4r-s | 28.99 | 29.56 | 30.86 | 34.5 | 2557 | 2416 | 1621 | 6802 | 210 | 16.4 |
| iq5r | 34.88 | 35.40 | 36.75 | 28.7 | 2535 | 2396 | 1635 | 6860 | 210 | 20.6 |
| nano | 34.79 | 35.52 | 36.80 | 28.7 | 1237 | 1209 | 3350 | 13593 | 210 | 19.6 |
| small | 39.74 | 40.32 | 41.59 | 25.2 | 1382 | 1354 | 3001 | 12141 | 210 | 23.6 |
| tiny | 36.13 | 36.72 | 37.95 | 27.7 | 1389 | 1359 | 2984 | 12098 | 210 | 20.6 |
| v3-k4-uniform-proxy | 29.21 | 30.13 | 31.87 | 34.2 | 2453 | 2328 | 1690 | 7060 | 210 | 9.1 |

Notes:
- Entries (Nano, Tiny, Small) and IQR-V4 rows run on THEIR build (Perire qc/qwen38-kl-guided-iqr 3eec7eb6); the IQR-V4 rows are the same engine, so the comparison is the containers'. The V3 proxy row runs on our build (rad10-e157b).
- The V3 row is NOT an incumbent and NOT size-matched: `q27v3-635bac02-k4-mad-a8.rad` (w150 pack, uniform K4, A8 act) is 18.25 GB: 12.23 GB trunk, 2.54 GB bf16 token embedding, 2.54 GB bf16 head, 0.92 GB bf16 vision tower (rad-info tensor sums); 5.15 text-plane bpw, 5.43 file bpw. Its decode and prefill are confounded by the bf16 head and embedding. Our 3.0 / 4.0 / 5.0 / 6.0 incumbents could not be built on box2 (levels K4.625-K7.5 exist only as finals). Read it as a V3 kernel reference only.
- All rows: card 0 (rocm-smi GPU[3]), power cap 210 W (rocm-smi Max Graphics Package Power), spec decoding off (IQR-V4's drafter off), --max-num-seqs 8 (the decode176 harness; the KLD reads used 16), --max-model-len 40960, bf16 KV.
- gemm_us (the board metric) is not measured: rad-kbench on their build declines every gemm_nt_q case (no operand recipe for that op), so there is no bench for their GEMM. End-to-end only, as the lead allowed.
- Prefill tok/s = prompt tokens / client TTFT (includes the first decode step); 4k prompts are about 4,146 tokens, 16k about 16,435.
