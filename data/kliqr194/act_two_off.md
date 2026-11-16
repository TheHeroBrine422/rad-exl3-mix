# KL-IQR-V1 entries with two-pass activations OFF (#194-02, w178)

Each container is read as shipped, with one load-time change: `RADIANCE_ACT_TWO=0` overrides the container's `quantization_config.act_two=1` (the override is printed at load: `radiance: RADIANCE_ACT_TWO=0 overrides ...`). Nothing else changes: same files (w177's), build, golden (`golden-t183`), bf16 KV, `--tp 1`, `--max-num-seqs 16`. `act_group` stays as the container says (Nano 32, Tiny and Small 128). Speed: card 0, 210 W, batch 1, 512 tokens (bench.py).

## Headline

- **KL-IQR-V1-3.00 (Nano)**: act_two off changes t183 mean KL by -0.4 % (paired 95 % CI [-1.2, +0.3] %) against act_two on.
- **KL-IQR-V1-4.00 (Tiny)**: act_two off changes t183 mean KL by +1.2 % (paired 95 % CI [-0.0, +2.4] %) against act_two on.
- **KL-IQR-V1-5.50 (Small)**: act_two off changes t183 mean KL by +6.5 % (paired 95 % CI [+4.8, +8.4] %) against act_two on.

## Quality (t183, gen rows; text-plane bpw)

| entry | text-plane bpw | act_two on (w177) KL [95 % doc CI] | act_two off KL [95 % doc CI] | off vs on, paired (95 % CI) | on vs IQR-V4 / off vs IQR-V4 (same bpw) | on vs ours / off vs ours (same bpw) |
|---|---:|---:|---:|---:|---:|---:|
| KL-IQR-V1-3.00 (Nano) | 2.9980 | 0.02957 [0.02341, 0.03747] | 0.02946 [0.02341, 0.03721] | -0.4 % [-1.2, +0.3] | -6.8 % / -7.2 % (IQR-V4 0.03173) | +15.5 % / +15.1 % (ours 0.02560) |
| KL-IQR-V1-4.00 (Tiny) | 3.9982 | 0.00718 [0.00597, 0.00854] | 0.00727 [0.00605, 0.00868] | +1.2 % [-0.0, +2.4] | -12.3 % / -11.2 % (IQR-V4 0.00819) | +1.0 % / +2.2 % (ours 0.00711) |
| KL-IQR-V1-5.50 (Small) | 5.4989 | 0.00116 [0.00098, 0.00133] | 0.00123 [0.00106, 0.00140] | +6.5 % [+4.8, +8.4] | -23.6 % / -18.7 % (IQR-V4 0.00151) | -11.2 % / -5.4 % (ours 0.00130) |

Guards (README rule: tails and top-1 next to the mean; gen rows, 500 document resamples, 95 % intervals on off / on):

| entry | KL99 off / on (ratio [CI]) | KL99.9 off / on (ratio [CI]) | top-1 on → off (diff, pp [CI]) |
|---|---:|---:|---:|
| KL-IQR-V1-3.00 (Nano) | 0.4233 / 0.4281 (0.989 [0.965, 1.011]) | 1.869 / 1.876 (0.996 [0.957, 1.035]) | 94.03 → 94.02 (-0.003 [-0.044, +0.046]) |
| KL-IQR-V1-4.00 (Tiny) | 0.0952 / 0.0936 (1.017 [0.988, 1.050]) | 0.427 / 0.427 (0.998 [0.941, 1.044]) | 96.84 → 96.82 (-0.020 [-0.065, +0.021]) |
| KL-IQR-V1-5.50 (Small) | 0.0154 / 0.0146 (1.060 [1.032, 1.106]) | 0.065 / 0.064 (1.009 [0.944, 1.111]) | 98.51 → 98.47 (-0.039 [-0.082, +0.016]) |

Same-day control: Nano with the container's own setting, re-read on 11-15 with the same harness: KL 0.029568546 vs w177's 0.029568546 (identical to the printed digits: the harness reproduces w177's reads, so the off-vs-on difference is the setting alone).

Sanity gate (lead): no entry's act_two-off KL is above 3× its act_two-on value (ratios: nano 0.996, tiny 1.012, small 1.065). The load did not fail.

## Speed (card 0, 210 W, batch 1, 512 tokens; medians over 5 runs; prefill = prompt tokens / TTFT)

| row | decode ms/tok ctx0 | decode ms/tok ctx16k | prefill tok/s 4k | prefill tok/s 16k | TTFT 4k (ms) | power cap (W) | load (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| KL-IQR-V1-3.00 (Nano), act_two on (w177) | 34.8 | 36.8 | 1237 | 1209 | 3350 | 210 | 19.6 |
| KL-IQR-V1-3.00 (Nano), act_two on (same-day control) | 35.0 | 36.7 | 1234 | 1209 | 3360 | 210 | 21.6 |
| KL-IQR-V1-3.00 (Nano), act_two OFF | 108.4 | 110.5 | 2120 | 2061 | 1957 | 210 | 20.5 |
| IQR-V4 iq3r-s (w177, nearest size) | 27.0 | 28.8 | 2582 | 2433 | 1607 | 210 | 17.3 |
| KL-IQR-V1-4.00 (Tiny), act_two on (w177) | 36.1 | 37.9 | 1389 | 1359 | 2984 | 210 | 20.6 |
| KL-IQR-V1-4.00 (Tiny), act_two on (same-day control) | 36.1 | 37.9 | 1389 | 1361 | 2984 | 210 | 20.6 |
| KL-IQR-V1-4.00 (Tiny), act_two OFF | 32.1 | 34.0 | 2478 | 2353 | 1674 | 210 | 20.6 |
| IQR-V4 iq4r-s (w177, nearest size) | 29.0 | 30.9 | 2557 | 2416 | 1621 | 210 | 16.4 |
| KL-IQR-V1-5.50 (Small), act_two on (w177) | 39.7 | 41.6 | 1382 | 1354 | 3001 | 210 | 23.6 |
| KL-IQR-V1-5.50 (Small), act_two on (same-day control) | 39.7 | 41.6 | 1382 | 1352 | 3002 | 210 | 23.5 |
| KL-IQR-V1-5.50 (Small), act_two OFF | 37.5 | 39.4 | 2463 | 2340 | 1684 | 210 | 23.7 |
| IQR-V4 iq5r (w177, nearest size) | 34.9 | 36.7 | 2535 | 2396 | 1635 | 210 | 20.6 |
| Nano repeat, act_two on (control, run first) | 34.9 | 36.8 | 1234 | 1209 | 3361 | 210 | 19.6 |
| Nano repeat, act_two OFF (run second) | 108.3 | 110.4 | 2118 | 2060 | 1959 | 210 | 19.6 |

Share of the gap to IQR-V4 at the nearest size that act_two accounts for (same-day control as 'on'; a share is the part of the gap that switching act_two off would remove):

| entry | decode ms ctx0: gap to IQR-V4 | act_two share | prefill tok/s 4k: gap to IQR-V4 | act_two share |
|---|---:|---:|---:|---:|
| KL-IQR-V1-3.00 (Nano) | 7.96 ms | off is slower (act_two helps decode) | 1348 tok/s | 66 % |
| KL-IQR-V1-4.00 (Tiny) | 7.14 ms | 57 % | 1168 tok/s | 93 % |
| KL-IQR-V1-5.50 (Small) | 4.87 ms | 47 % | 1153 tok/s | 94 % |


Off vs same-day control, per entry (negative = act_two off is faster):

| entry | decode ms ctx0 | decode ms ctx16k | prefill tok/s 4k | prefill tok/s 16k |
|---|---:|---:|---:|---:|
| KL-IQR-V1-3.00 (Nano) | +210.1 % | +201.0 % | +71.8 % | +70.5 % |
| KL-IQR-V1-4.00 (Tiny) | -11.2 % | -10.4 % | +78.4 % | +72.9 % |
| KL-IQR-V1-5.50 (Small) | -5.7 % | -5.3 % | +78.2 % | +73.0 % |

## Method and caveats

- Quality: `kl_mean.py` (gen rows), `kl_boot.py` (per-document 95 % CI, B = 2000, seed 194), `paired_boot.py` (off vs on, the same resampled documents for both). Rows files on box2 `/data/q27/w178/rows/`, summaries in `raw/act2off/`.
- Speed: `speed178.sh` (w177's `speed.sh` harness with an env column). Each row loads its own serve; the same-day control row is the container as shipped. The IQR-V4 rows are w177's (11-15, 2:47–3:59 PM CST), not re-run.
- Not done: a second speed day, the t186n split (not asked), gemm_us (no bench for their GEMM).
- Other per-container switches looked at: see the report (`out/w178.md`).
