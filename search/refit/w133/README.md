# search/refit/w133 — #173: a better sensitivity model on the measurements we already have (w133, 2026-11-13)

Question (user 11-13): would a better model fitted on the search's own rows + w129's calibration rows give better picks
than kfit3 (search/v3/fit.py) + search? Protocol `RULES-173.md` (5363227, committed before any fit; amendment 1 =
frozen candidates `candidates173.json`, 50554a8, before any T / P / S run). Results: LOG #173.

| file | what |
|---|---|
| `ds133.py` | replay of the driver's state (on_result / add_obs) into one dataset: fit rows with the result index they became available at, decided events (candidate vs the incumbent it was judged against, per part log r + paired SE, size offset, call), s129 swaps, assignments. Per-part log r == the driver's (max diff 0) |
| `m133.py` | `XModel` = fit.py's SModel + optional terms: `prof` (label offsets profiled out) with ev133's `relabel: size` (one offset per (label, size)), `lam2` (part curvature in log P), `qg` (per-group q), `headk`, `sib`, `adj`, `re` (per-tensor shrinkage), `mode` (A8 hook: per-mode P(1 + a) + c, fitted only when A8 rows are in the fit). ext {} == SModel (selftest 1e-15); `tables` / `cost` honour re / qg / headk so moves.Pred / mckp solve with it |
| `ev133.py` | the protocol: `fit` / `pred` units (restartable), `s` (slice split), `report` (weighted ratio MSE per part, calls, AUC, clear-cut sign, accepted gains, paired bootstrap vs M0) |
| `sol133.py` | step 3: moves.mckp re-solve per size from the current incumbent with each model; bytes moved / tensors changed vs incumbent and vs w129's init pick, cross-predictions |
| `run133.sh` | colo runner (xargs, nice 19, single-threaded) |
| `dev173.json` / `candidates173.json` | development variants (D only) / the frozen candidates |

Data: `data/refit/w133/` (ds-S0 / ds-P datasets, fits and predictions per split, reports, picks/).

```
R=local/workers/w131/run
python3 ds133.py OUT.json.gz --state $R/drv/state.jsonl --assign $R/drv/assign --registry $R/registry.json --menu $R/menu129h.json \
    --prior $R/fit129h2.obs.jsonl --w129 data/refit/w129 [--until 'MM-DD HH:MM:SS'] [--a8-slices s171c,..]
python3 ev133.py fit DS candidates173.json M1 CUT OUTDIR                 # (colo; MENU=... for the menu path there)
python3 ev133.py pred DS candidates173.json M1 CUT OUTDIR EVDS LO HI
BOOT=4000 python3 ev133.py report OUTDIR "T=pred-*-ds-S0-*.json"
python3 sol133.py OUT.json DS candidates173.json FITDIR CUT M0,M1 --state .. --assign .. --init $R/init --save-dir picks
```
Using M1 in the search = fit.py with two changes (not made here: a patch for the lead if adopted): the search rows'
labels carry their size (`S@size|p|...`) and the label offsets are profiled out, plus the λ1 / μ1 curvature terms
(XModel's `prof`, `lam2`; `relabel`).
