# v3/qad221 — w221: QAD wave on the w220 trainer (late layers only, tied front half, clip, 3.00)

Runs `../qad220/qad220.py` (unchanged) with w214's data / order, budgets 0.25 M / 0.5 M, compiled, reads on j217. Scripts = w220's with W = `/data/q27/w221` (cards 2 and 3 only).
- `jobs221.txt` — the six runs; `card221.sh CARD jobs221.txt` — per-card dispatcher (train, then served reads; restartable; remove `jobs221.txt.cursor` to re-walk).
- `run221.sh`, `serve221.sh`, `score221.sh`, `env221.sh` — w220's with paths changed.
- `readout221.py` — the table (box2: `python3 code/v3/qad221/readout221.py /data/q27/w221 --json readout221.json`).
Rerun on box2: copy `/data/q27/w220/code` to `/data/q27/w221/code`, add this directory, `setsid nohup code/v3/qad221/card221.sh 2 code/v3/qad221/jobs221.txt < /dev/null > logs/card221-2.log &` (and 3).
Raw results: `../../data/qad221/`.
