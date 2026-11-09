# w94 — head inputs at scored rows (#124, merge 11-08 A2b) and head quantizer settings (#125, A6)

Rules: `RULES.md` (650436c before data; amendment 1 1cc3f12, amendment 2 c1ee37a, each before its arm). Results: LOG
2026-11-08 5:55 PM (#124), 6:30 PM (#125), 7:05 PM (#125-01), 7:13 PM CST (#125-02); tables in `data/`.

## Outcome

- **#124:** gen masking raises head-only KL 3–7% and changes no floor, plan or hull. Pricing on qad rows (the Fisher's
  own pool) overstates slab heads about 2×. Price on held-out gen rows: `data/sim94-search-gen.json:plans94.json`
  for knap5 `KN_HEADGRID`, with `KN_HG_DONORS=3,4,5,6,7` until the new donors exist. Grid vs today's menu: −1.9 / −4.4 /
  −3.2 / −1.9% at 2.0 / 2.5 / 3.0 / 4.0 (`data/knap-headgrid.md`).
- **#125 (CPU proxy):** proxy unverified (G2 FAIL at K3), so nothing was adopted from it. τ was negative: the optimum is
  ≈ 1.005, under 1%.
- **#125-01 / -02 (real quants, card 0):** use two-sided softmax-Fisher LDLQ **without** mean-row for the head donors.
  Against today's donors it gives 0.908 / 0.899 head KL at K3 / K4 (search-gen) and 0.874 / 0.860 (qad-even). Mean-row
  makes it worse (+7–12%).

**Donor flags** (today's plus the H_out part; no `--head-mean-row`):

    ourq.py -K <k> -o <dir> --only-head --device 0 --save-wq --head-perm perm.npy --qtf2 --frac-ext <SRC> \
      --scaled-hessian --sigma-reg 0.025 --codebook i8 --hout <HD> --hout-blocks group --sigma-reg-out 0.1

- `<HD>/lm_head.safetensors` holds `hout_diag = fisher_qad_gen[perm]` (float32), the same perm as `--head-perm`.
  `gpu94.sh` writes it from `headsens.npz` and `perm.npy`.
- The fork needs patch 0008 and `exl3-two-sided-cpu-weight.patch`. Without the patch the head crashes in the proxy
  metric.
- Cost: +5–6% quant time per donor (100 vs 94.5 s at K3). Same bytes, format and kernels.

## Files

| file | what | where it ran |
|---|---|---|
| `mkdocs94.py` → `docs94.json` | documents / row windows of the dumps (fixed before data) | VM |
| `headx94.py` | bf16 forward on ONE card, layer-streamed (HF modules), final-norm output at the window rows (`--cpu` test mode) | gpubox card 0 |
| `gate94.py` | G1: dumped rows vs the golden's top-512 | gpubox |
| `headsim94.py` | headsim.py's exact head-only KL, every row kept, τ grid, lse / argmax for the CPU proxy | gpubox card 0 |
| `run94.sh` | #124 driver (takes card0.lock, then ram.lock shared, per step) | gpubox |
| `report94.py`, `simjson94.py`, `knaprep94.py` | #124 / #125 readouts; headsim-format JSON per row set; knap5 grid table | VM / colo |
| `mkblocks94.py` → `blocks94.json`, `hq94.py`, `q94.sh`, `s94.sh` | #125 CPU proxy (64 stratified head blocks) | colo |
| `s94b.sh`, `strata94.py`, `posthoc94.py` | post hoc, report-only (after G2 failed) | colo / VM |
| `gpu94.sh`, `gpu94b.sh`, `report94b.py`, `report94c.py` | #125-01 / -02 real head quants + headsim + readouts | gpubox card 0 / VM |
| `exl3-two-sided-cpu-weight.patch` | 1-line fork fix (two-sided path + a > 5e8-weight tensor) | for the lead |

Rerun #124: on gpubox, `/data/w94/code/run94.sh` (code = this folder + `v3/yaqa/headslab.py`). It is idempotent.
Machine data: gpubox `/data/w94` (dumps, sims, T0 donors `q/two0/K3,K4`, patched fork `exl3/`); colo
`/data/work/w13/w94` (dumps, per-row proxy KL); per-row npz copies in `/workspace/rad-local/workers/w94/data/`.

**Runtime:** none (statistics, quantizer settings).
