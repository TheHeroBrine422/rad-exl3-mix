# w211 — DS4.1-F routed-expert quantizer harness + first settings sweep (mad, CPU, weight-error proxy only)

Report: `rad-local/workers/out/w211.md` (local). Raw results: `res/` (one jsonl per (matrix, setting) job, slices 0 / 1), `experts.json`
(chosen experts), `verify_dequant.log`, `tables.md` (agg.py output).

## Files
| file | what |
|---|---|
| `dscommon.py` | MXFP4 dequantizer (exact), expert / pooled Hessian loaders, chunked LDLQ (`ldlq_prepared`, bit-identical to `trellis.ldlq_trellis`) |
| `verify_dequant.py` | dequantizer vs the reference model code (`ds41pipe/ds41pipe/ref/kernel.py`, `model.py`): weights bit-equal, Expert output equal |
| `pick_experts.py` | the expert set (3 layers x hot / medium / cold x 2, shard s0 only) from `counts.json` + the shards' stats |
| `ds_check.py` | one (matrix, setting) job: rotation, unit-RMS rows, factor search, LDLQ + mad trellis for K = 1..4, rel under own / pooled H |
| `run_grid.sh` | restartable xargs driver (P parallel single-thread jobs) |
| `agg.py` | tables for the report |

## Rerun (box2; paths in `dscommon.py` are env-overridable)
```
# inputs: /data/ds41ck-full (weights); calib/experts/L{05,20,35}-s0.safetensors + index.json, calib/dense/L{05,20,35}.ffn_in.safetensors
#   from HF dataset Hero422/deepseek-v4.1-flash-1017-quant-calib (huggingface_hub.hf_hub_download, local_dir=/data/q27/w211/calib);
#   hf/scripts = the dataset's scripts/ (hesstri.py); ref/ = ds41pipe/ds41pipe/ref; code/e3 = v3/e3 (e3.py trellis.py viterbi.c viterbi3.c)
cd code/e3 && gcc -O3 -march=native -fopenmp -shared -fPIC viterbi.c -o libviterbi.so && gcc -O3 -march=native -fopenmp -shared -fPIC viterbi3.c -o libviterbi3.so
python3 -I pick_experts.py counts.json calib experts.json
W211_SLICE=0 ./run_grid.sh 30      # slice 0 (576 / 640 rows); W211_SLICE=1 for the disjoint replicate; SETTINGS="base d0.1" to restrict
python3 -I agg.py out experts.json 0,1 > tables.md
```
Env of the jobs: `E3_V2=3 E3_TB=1 OMP_NUM_THREADS=1 OMP_WAIT_POLICY=passive`. Rows: `W211_ROWS_GU` (576 of 4608) / `W211_ROWS_DN` (640 of 5120) per
slice, the same rows for every setting of a matrix (seeded permutation, disjoint slices); `W211_ROWS=0` = all rows (~10 h for the grid on 32 threads).
