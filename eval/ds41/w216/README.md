# w216 · cleaning `exl3-v3` of bakon's EXL3 / GGUF / sidecar paths, KV size

Scripts (box2, `/data/q27/w216/`):
- `kvsize.sh LEN...` — deepseek41 KV allocation a card at each `--max-model-len` (bakon's reference build
  `/data/radiance-ds41` + mikoskinen's EXL3 2.25 container, TP4, bf16 KV; starts the server, reads the plan, kills
  it; takes the four card locks, `flock -n`). Logs: `data/ds41-kv/w216/<LEN>/serve.log`.
- `build.sh` — configure + build the synced `exl3-v3` tree in `/data/q27/w216/src` (restartable).
- `tests.sh` — in-tree `ctest -L fast`, install, symbol gate, host SIMT-emulator tests (linked by hand with
  `-lstdc++ -lm`), libquant exl3v3 selftest + tiles, then under card3's lock: `r4d_selftest --mxcheck` (MXFP4
  grouped GEMM: drafter_mx forms vs stock, byte for byte; MXREF lines = stock vs a host reference), exl3i /
  exl3quant / qwen35_exl3 out of tree.
- `tests2.sh` — exl3v3 tiles and `engine_e2e_test` under card3's lock, the exl3i plugin's gate.

Sync the VM worktree first with `rad-local/workers/w216/sync.sh` (machine-specific, not in the repo).
Results: `data/ds41-port/w216/` (test logs), `data/ds41-kv/w216/`. Report: `rad-local/workers/out/w216.md`.
