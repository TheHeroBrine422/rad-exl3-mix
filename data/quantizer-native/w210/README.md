# w210: native EXL3-V3 quantizer — host steps on the card, batching, release extras (2026-11-20 / 21, box2 cards 2 + 3, R9700)

Report: `rad-local/workers/out/w210.md`. Code: radiance fork branch `exl3-v3` (`libquant/lq_exl3v3_pipe.hip.h`,
`lq_exl3v3_extras.hip.h`, `tests/exl3v3/`). box2's CPU was shared with w211 (≤ 32 threads) during every run: host-side
seconds are noisy, card-side ones are not.

- `raw/base/`: w209's host pipeline (radiance 97eb3dab) on layer 3, K2 / K4 (`scripts/prof.sh base build-base 3 2,4`; no warm-up call,
  so its K2 line carries the first call's start-up).
- `raw/plain/`: the card pipeline, same settings (`prof.sh plain build 3 2,4 --warm`). `raw/ident/ident.txt`: both builds'
  records / suh / svh compared byte for byte (`scripts/ident.sh`).
- `raw/syn/`: `exl3v3_tool batch` on synthetic experts (`scripts/synall.sh`, `syn.sh`; `-r` = a repeat at the end of the session,
  `-ss4` = `--scale-sample 4`).
- `raw/ladder/`, `raw/release/`: layer 3 with the ladder's / the release's settings (`scripts/lay.sh`); `raw/ref/*.jsonl`: the
  release pipeline (rad-exl3-mix `pipeline/rxq` over the exllamav3 fork b0052f7c) on the same tensors (`scripts/refs.sh`, `ref.sh`;
  block order and C9 off on both sides), `*.tilecd.jsonl` its tile CD records.
- `raw/extras/`: each extra through `exl3v3_tool` (`scripts/extras.sh`), incl. `swj.jsonl` (the tool's fit, then
  `tests/exl3v3/ref/swj_compare.py` = rxq/swj.py on the same inputs).
- `agg.py`: every table of the report (`python3 agg.py`). `pull.sh`: copy from box2 (needs `W210_RSH`).

H_out: `Hero422/qwen3.8-27b-hout-diag` `block-normalized.safetensors`, split per key with box2 `/data/q27/tools/houtsplit.py`.
No KLD data was read (proxy errors on the calib Hessians and synthetic inputs only): no overfit audit needed.
