## #123 integrity scan: verdict

selftest: PASS (clean: True, span: False, nan: False, zerodiag: False, indefinite: False)

**H_in: 257 files scanned (exhaustive), 256 pass every check, 1 fail.**
- FAIL model.language_model.layers.7.post_attention.input.safetensors: diagonal: 1 entries not > 0; notes: longest exact-zero run 1126 entries (review)
- Cauchy-Schwarz: max ratio over all files 0.986778 (model.language_model.layers.0.mlp.down_proj.input.safetensors, [7214, 13323]); entries > 1 + 1e-5: 0; > 1 + 1e-3: 0; non-finite: 0
- damped Cholesky (1e-3 x mean diag): 256 / 256 succeed (files that failed an earlier check are not factored); min L_ii^2 / lambda 1
- exact zeros: 1 files (model.language_model.layers.7.post_attention.input.safetensors 5120)
- metadata rows: ['10259331']; k: [5120, 6144, 17408]; sizes ok (header check)
- sha256: HF LFS oid (dataset 2dcf0061, 257 files) mismatches 0; download metadata mismatches 0; Thor copy mismatches 0 (257 hashed); w86 dry-run pins 8 compared, mismatches 0; HF files not on colo: none

**H_out (bn1920): 400 keys, 400 files, 400 pass; extra files [].**
- min entry 4.23e-23, max / min within a tensor up to 3.28e+23
- sha256: Thor copy mismatches 0 (400 hashed); gpubox source (VM backup, 400 files) mismatches 0; w86 dry-run pins 13 compared, mismatches 0

**Verdict (rule): FAIL** (1 failing items)
