# search/v3/ref134 — reference scores for the V3 convergence plot (#171 plot, w134, 11-13)

The V3 search's plot (`search/convplot4.py`) puts every number on "KLD ÷ V2 at the same bpw, same slice". That needs V2
measured on the held-out monitor (golden-ver3-8k) at every searched size, and a real competitor line: deadcode's
IQR-V4 containers (`StillDeadcode/qwen3.8-27b-iqr-v4`, public) scored on the same golden instead of a guessed ratio.

| file | what |
|---|---|
| `ref134.py` | ref171.py's procedure (host's own score_cmd, one free card lock = one eval slot, ram.lock shared) for several goldens per container, per-part rows (c / x) on two-part search slices, `@BPW` = the bpw to plot at |
| `run134.sh` | ds8 driver: downloads (background, ≤ 2 IQR files on disk, HOLD < 160 GB free) and scores one container at a time on ver3-8k + the search slice (`SL`, default s171b); IQR files deleted once scored, V2 files kept in `/data/q27/mixes`. Restartable |

- V2: 1.50–7.00 (3.00–5.00 were on disk and on ver3 from ref171.py; here on s171b only). Both reference families are
  A8-neutral: their own shipped containers (V2 f16 activations, IQR-V4 its int8 activations).
- IQR-V4 containers carry the DFlash2 drafter and the vision tower: plotted at the published LM-only bpw (the model
  card's table: 2.470 … 7.420), `bpw_file` keeps the file's. The KL is the language model's (the drafter does not
  change outputs).
- Rows: ds8 `/data/q27/w134/refs/results.jsonl` (rxrun.multi format, tags `ref-V2-<size>`, `ref-IQR4-<name>`),
  copied to `data/searchv3/w134/refs134.jsonl`; the VM loop pulls the live file.
- Ledger: uses `ref134-ver3-8k` (104 verify docs) and `ref134-s171b-c` (43 calib docs), role check, no decisions.

Run (ds8): `cd /data/q27/w134 && setsid -f bash code/run134.sh > run134.out 2>&1 < /dev/null` (log `run.log`).
Plot (VM): `/workspace/rad-local/plots/convplot4-loop.sh [once]` → `/workspace/v3-convergence.png` every 30 min.
