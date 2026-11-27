# w235 — HEPH (grid-aligned trellis) vs our mad + Hadamard trellis on DS4.1-F routed experts

Report (local): `rad-local/workers/out/w235.md`. Raw results: `data/heph235/out/{heph,ours}-<tag>/` (one jsonl per matrix),
tables `data/heph235/tables-*.md` (agg.py output). Same 18 experts / rows as w211 (`data/ds41-quant/w211/`).

| file | what |
|---|---|
| `heph.c` | HEPH Viterbi: generic L / T / shift pattern (fractional K, salted 0-bit steps), per-element branch weights, exact DP, tail-biting two-pass (free pass rolled by T/2 -> seam group, constrained pass) |
| `heph.py` | ctypes wrapper, MagStrat codebook (`codebook`), shift pattern, MXFP4 exponent Huffman cost (`exponent_cost`) |
| `test_heph.py` | DP == brute force (free + constrained, tiny L, incl. fractional / 0-bit patterns); tail-biting + on-grid validity of full encodes |
| `heph_check.py` | one matrix: exact atoms + e8m0 bytes from the checkpoint, HEPH encodes for `HEPH_SPECS` (variant:K:L[:seed]), metrics in the original domain |
| `ours_metrics.py` | w211's `base` baseline re-run (F3 factors) keeping the best reconstruction: rel / rel_diag / rel_mse in the original domain (rel equals w211's number exactly) |
| `run_grid.sh`, `env.sh`, `pull.sh` | restartable xargs driver (box2), job env, copy results to the repo |
| `agg.py` | markdown tables (rate-matched ratios, bootstrap over matrices, crossover) |

## Rerun (box2, CPU only)
```
# inputs: /data/ds41ck-full; calib experts shards Hero422/deepseek-v4.1-flash-1017-quant-calib experts/{index.json,L05-s0,L20-s0,L35-s0}.safetensors
#   -> /data/q27/w235/calib (hf_hub_download, dataset is public); code: this dir -> /data/q27/w235/code/heph235, w211 dscommon.py + experts.json ->
#   code/w211, v3/e3 (built .so) -> code/e3; w211's hf/scripts (hesstri.py) at /data/q27/w211/hf/scripts (env.sh)
cd /data/q27/w235/code/heph235 && gcc -O3 -march=native -fopenmp -shared -fPIC heph.c -o libheph.so && python3 -I test_heph.py
export HEPH_SPECS=std:1:12,std:2:12,std:3:12,std:2:16,std:2:12:2,std:3:12:2,sw:2:12,sw:3:12,h1:3:12
TAG=main ./run_grid.sh heph 4 "5 20 35"                  # HEPH, 36 matrices, 288 / 320 rows (w211 slice 0)
TAG=main W235_KS=1,2,3,4,1.5,2.5 ./run_grid.sh ours 12 "5 20 35"  # baseline re-run (K = 4 for the rate matching; 1.5 / 2.5 = ours' fractional pattern search, non-tail-biting)
HEPH_SPECS=std:1.5:12,std:2.5:12,sw:1.5:12,sw:2.5:12 TAG=main ./run_grid.sh heph 3 "5 20 35"    # fractional K
HEPH_SPECS=std:1:12,std:2:12,std:3:12,sw:2:12,sw:3:12 TAG=allrows W235_ROWS_GU=0 W235_ROWS_DN=0 ./run_grid.sh heph 8 "5 20 35"   # all rows
./pull.sh   # on the VM: results -> data/heph235/out
python3 -I agg.py ../../data/heph235/out > tables.md
```

Notes: `agg.py` interpolates ours between its integer-K points for the integer-K tables (ours' fractional points only in the fractional table);
`ours_metrics.py` also stores `rel_unrot` (full H, unrotated, float64, the evaluation `heph_check.py` uses): equals `rel` to 2e-7 (down) / 3e-6 (gate_up)
on the two matrices re-run for the check (`out/ours-chk/`). The jobs sourced `env.sh` with an early bug (row counts forced to 288 / 320 even for the
all-rows tag); that run was killed and redone after the fix, nothing from it is in the data.
Results are deterministic: the codebook seed is fixed (default 1), the Viterbi has no randomness.
