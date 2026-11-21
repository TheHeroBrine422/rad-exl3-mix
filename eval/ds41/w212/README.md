# w212 — DS4.1-F baseline harness (official release on bakon's `deepseek41-wip` tree, box2)

Status: slice + harness + conversion done; **the engine cannot serve the release on box2** (no number). Report: `rad-local/workers/out/w212.md`.

| file | what |
|---|---|
| `../mkslice41.py` | builds the verify slice `ds41-verify-v1` (150,008 scored positions) from the HF golden; `plan` then `build` (colo, HF token) |
| `../slice-ds41-verify-v1.manifest.json` | the 184 documents (name, category, part, row0, tokens, scored) |
| `../ledger.jsonl` | the 184 documents, role `check`, use `ds41-verify-v1` |
| `../recipes/ds41-release.recipe` | the conversion recipe: MXFP4 experts + the Engram table (fp8 × e8m0[1x32]) kept as released, other trunk weights fp8 → bf16 (`cast`, exact), norms / hc / router f32 |
| `convert.sh` | `rad-convert` of the release with that recipe (`--arch deepseek41 --max-spec 0 --add engram-tables.safetensors`); `convert-first.sh` = the first try (Engram bf16, 651 GiB, unservable) |
| `score.sh TAG` | writes the YAML and runs `radiance --config` (`kld-ref` / `kld-out` are YAML-only), TP4, locks card0–3; run twice for run-to-run noise |
| `score41.py` | masked (gen.u8) mean KLD + per-document bootstrap CI, per category, top-1, ppl ratio; two reports → paired run-to-run difference (tested on synthetic rows only) |
| `join-mik.sh`, `smoke-mik.sh`, `score-mik.sh` | w212b: join mikoskinen's container parts + sha check; serve + one greedy request; score the slice with it (same locks / outputs as `score.sh`). Results `data/ds41-baseline/mik225/` |
| `mksmoke.py` | cuts a few documents out of a golden for a smoke run |
| `engine-patches/0001-…patch` | the one-line engine experiment (`in.file_relayout = true`), **not sufficient** (see report); commit on local branch `ds41-baseline` of `forks/radiance-ds41` |

Rebuild: `python3 mkslice41.py plan POOLS MAIN100B WORKDIR MANIFEST` / `build MANIFEST OUT`; box2 `/data/q27/w212/` holds gold-v1, the container
(`ds41-official.rad`, 473.47 GiB) and `runs/`.
