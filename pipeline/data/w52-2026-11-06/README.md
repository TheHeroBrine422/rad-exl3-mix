# w52 check and profile runs, 2026-11-06 (RTX 4070, VRAM cap 7.2 GiB, ladder settings)

GPU units (tools/run4070.sh), times UTC (CST = UTC-6):
- A   18:55:52-18:58:32 (160 s held): 27B L10 (6) + L39 attention (4) at K2.5 vs hconv/q-none → a-K2.5/, a-check.txt
- BCD 19:27:32-19:35:50 (498 s held): a2 (L10 ffn_down at K2.5, default threads) → a2-check.txt; profile K2 (10
  tensors), K4 / K6 (9, no ffn_down) → b-K*/; C = 0.8B asym writers (rxq mkdenoise on the GPU + 48 writer jobs) vs
  d1103/q-k3asym_r0 → c-asym/, c-check.txt, c-dn-stats.jsonl; D = H_out (bn1664) on 3 tensors vs hconv/q-bn1664 →
  d-K2.5/, d-check.txt
- total GPU lock time: 658 s.
result.txt's lines "18:58:34 C quant rc=127" / "C lock released" are an artifact: the driver was edited while bash was
reading it (no command ran, nothing touched the GPU after A's release).
CPU checks (tests/check_cpu.py, 4070 CPU, 4 threads): cpu-check.txt.
log.jsonl per job dir: ourq's proxy record + rxq.stages (exclusive seconds per stage), peak VRAM, max RSS.
profile-4070*.md: tools/profreport.py over these dirs.
