# radiance KLD recorder patches

Applied to radiance 1.13.0 to build the evaluation server (`RXM_KLDBIN`, `radiance-kldgold-fast`):
- `kld.orig.cpp`: radiance's KLD recorder, unmodified (the base for the diff).
- `kld.cpp`: our version (per-row scores {kl, top1, nll_q, nll_p} written next to the run; scored inline against a
  bf16 golden).
- `engine_kld-async.patch` / `engine_kld.async.cpp`: asynchronous copies of the scored rows (validated at TP1).
Build radiance with these files in place of the originals; `eval/gold2.sh` runs the result.

All files here are radiance source (https://codeberg.org/StillDeadcode/radiance), Apache-2.0, Copyright 2026 Deadcode
and the radiance contributors; all but `kld.orig.cpp` are modified (see NOTICE).
