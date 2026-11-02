# Experiment log (newest first)

Append-only; template and rules in `README.md`. Entries before 2026-11-02 are in `archive/`.

### 2026-11-02 06:20 CST · #— · Tracker reorganised
- **Decision:** the two old trackers (`EXPERIMENTS-new-format.md`, `EXPERIMENTS.md`) are archived verbatim in
  `archive/`; every experiment so far was numbered #1–#62 in `INDEX.md`; open plans and the backlog moved to
  `BACKLOG.md`. Old paths keep a one-line pointer. From now on: one numbered row per experiment, one templated log
  entry per event.

### 2026-11-02 05:50 CST · #33-08 · YAQA Sketch B on 0.8B: two-sided objective optimised but does not track KL
- **Status:** running (unchanged) · **Machine:** 4070 · **Code:** `proxy2b/objcheck.py` (CPU)
- **Question:** is the 1K loss a quantizer bug (two-sided objective not optimised) or the objective itself?
- **Setup:** 14 linears of layers 0/5/11/19/23; per linear, one-sided error tr(E Hc Eᵀ)/tr(W Hc Wᵀ) with the calib
  input Hessian, and two-sided tr(E Hi Eᵀ Ho)/tr(W Hi Wᵀ Ho) with the 1K Sketch B factors; K3 arms base, ldlq, yb1024.
- **Result:** yb1024 has a 1.4–8× lower two-sided objective than base on every linear, and a 1.4–3.5× higher
  one-sided error (e.g. layer 5 in_proj_qkv: two-sided 5.9e-3 → 7.6e-4, one-sided 3.0e-3 → 8.8e-3).
- **Decision / next:** the quantizer path works; at 1K sequences and damping 0.01 the two-sided objective is the
  wrong target. Wait for 2K–16K before diagnosing; keep every snapshot for H_out research (`local/TODO.md`).
- **Files:** 4070 `/data/logs/objcheck.log`

### 2026-11-02 05:25 CST · #33-08 · YAQA Sketch B on 0.8B: 1K read negative
- **Status:** running (unchanged) · **Machine:** 4070 · **Code:** `proxy2b/yaqaB.py`, `yaqaBrun.sh`
- **Pass rule:** `proxy2b/YAQA08.md` (amendment 2 written before any data); `ldlq` is the one-sided reference,
  `base` (our tuned quant) reported beside it.
- **Setup:** 0.8B, i8 codebook, K3 and K2; arms base / ldlq (σ 0.01, no scaled Hessian) / yb1024 (per-linear
  H_in, H_out from 1,024 sequences, σ 0.01 both sides); golden g08-search1.
- **Result:**

  | K | base | ldlq | yb1024 | yb vs ldlq |
  |---|---|---|---|---|
  | 3 | 0.0989 | 0.1049 | 0.1184 | +12.8% (z 9.4) |
  | 2 | 0.535 | 0.556 | 0.690 | +24.1% (z 19.6) |

- **Decision / next:** collection continues; 2K/4K/8K scored ~09:00–09:15 CST, 16K ~13:20 CST. The 2K–8K reads
  decide whether to stop before 16K.
- **Files:** 4070 `/data/yb/result.txt`
