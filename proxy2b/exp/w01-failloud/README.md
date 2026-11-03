# w01-failloud: side-Hessian files must exist (ranked.md A8)

**Question / mechanism.** Silent fallbacks can waste a night:
- `ourq.py --hout / --hout-diag / --hin` quantize any linear without a file one-sided, on the calib H_in, without a
  word (`os.path.exists` in `quantize_group` / `hessian`).
- A collector that misses a linear (a hook not firing, truncated backward) writes no file for it.
- `collect.py` indexes `wins[s]` unbounded: past the qad pool's 1,350 windows (2048 tokens; `../w01-windows`) it
  dies with an IndexError hours into a run.

Each of these turns a whole 27B night into a mixed or failed result that looks normal.

## Patches (no GPU run; merge after review)
1. **`ourq-sidecheck.patch`** (`quantize/ourq.py`).
   - Before any work: for every trunk linear to be quantized (per `--layers`, `--only`; lm_head not checked, it has
     its own path), a file must exist in each given `--hout` / `--hout-diag` dir.
   - For `--hin`: either the per-key file or the group file (both layouts ourq reads).
   - Otherwise: exit, with the count and the first missing key.
   - `--side-partial` restores the old behaviour, with a warning.
   - The new flag is left out of the stamp's argument dict. As with every ourq.py edit, the file hash changes the
     stamp: resuming an unfinished OUTDIR across the merge needs `OURQ_MIXED=1`.
2. **`collect-check.patch`** (`v3/yaqa/collect.py`).
   - `START + NSEQ > windows` → exit at start-up (not in YAQA_GATE).
   - Each save (`save_diag`, `_save_hout`) first checks that every linear that should have sums has them, and exits
     naming the missing ones before writing anything. The expected set: YAQA_DIAG, all trunk linears but GDN
     a / b; YAQA_FULL, the FULL layers' linears. With SAVE_EVERY 25, a missing linear stops the run within the first
     25 sequences.
   - Applies with or without `../w01-genmask/genmask.patch`, in either order, with the same result.

**Draft pass rule (merge gate, no data):** both self-tests ALL PASS; and no existing 0.8B side dir would be refused
(checked: see below).

## Self-tests (CPU; outputs saved)
- `selftest/test_sidecheck.py` → `test_sidecheck.out`: **ALL PASS, 13 cases.** It runs the real top of ourq.py,
  original and patched, with stub modules (argparse, layer_plan, the new check, the stamp's argument dict).
  - Refuses: one missing --hout / --hout-diag file, a missing --hin group, an empty dir.
  - Accepts: complete dirs, --hin group or per-key layouts, --only subsets, `--only-head`, keys outside `--layers`.
  - `--side-partial` warns and runs.
  - The stamp's argument dict is identical to the original's in every running case.
- `selftest/real-dirs-4070.out`: the same check on the 4070's existing dirs (0.8B plan, layers 0–23):
  - dv8192, dv1024, ident, prec1k/x2, snap1024, snap8192, snap8192diag, snap8192mix, c18 snapshot 512 (B / xw / dy),
    snap8192/hin;
  - **all 150 / 150**, none refused.
  - 27B dirs on iceman were not checked. Old 27B gate scripts that pass a partial `--hout` dir with `--layers 0-63`
    now need `--layers` narrowed or `--side-partial`.
- `../w01-genmask/selftest/cpu_e2e.out` (colo CPU, tiny Qwen3.5): **ALL PASS**.
  - The collector exits naming `0.mlp.down_proj` when that linear's hook is removed, and writes nothing.
  - NSEQ past the window list exits before the first sequence. Control: the original dies with IndexError after the
    in-range windows.
  - Mask-off outputs are bit-exact with the original collector.

**GPU time:** none. **Changes:** no KL effect. It protects the 27B diagonal night (#33-13) and every later `--hout`
quant. ranked.md A8.
