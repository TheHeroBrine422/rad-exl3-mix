# search/refit/w129 — pass-1 calibration swaps and the search's starting refit (#169, 11-12 / 11-13)

Rule: `RULES-prior.md` (committed before data). Results and decisions: LOG 2026-11-13 1:15 AM CST. Data: `data/refit/w129/` (menus and obs files gzipped: gunzip before use).

| file | what |
|---|---|
| `draw129.py` | fit129 / s129-c document lists (merged pool, least-used first) |
| `mkslices129.sh` | the goldens on ds8 (fit129, s129-c, s129-x, goldmix s129) |
| `mkspecs129.py` | the 84 mixes as rxrun.multi eval lines (`data/refit/w129/specs129.jsonl`) |
| `feed129.py`, `pack129.sh` | VM feeder: queue a mix once its levels are final + packed, pack only the needed levels, disk guard (HOLD below 60 GB), re-queue refused evals as TAG.rN, delete finished mix containers |
| `fit129.py` | rows → prior test (β 0 / fit / 1), held-out H, ladder flag, the rule's final refit, slopes |
| `floor129.py` | report-only: per-part labels / additive floor on the held-out xcorp miss; the fit129-only refit |
| `final129.py` | refit for the driver on a given menu (fit.py output format + the rows fitted + slopes b) |
| `mkmenu129.py` | mkmenu.py with the K2.5 diag files replaced by their diagsum (ds8's ladder out/ dirs are gone) |
| `th129.py`, `fit-headinit.patch` | trunk × head anchors (K3 / K4 × R16 u3–u5): held-out check, refit fit129h2; the fit.py head-θ start fix |
| `hd129.sh`, `hdcheck129.py` | the 17 R16 head donor containers on ds8 (`/data/q27/heads/donors/hd-<plan>.rad`) + spot check |

## Load into the search driver
Recommended model for the R16-head search: **`fit129h2`** (β 0, the 84 fit129 rows + 6 trunk × R16 head rows; needs
`fit-headinit.patch` merged into search/v3/fit.py, else the driver's refits leave the head unpriced). Menus:
- V2 3.00 head (`menu129.json`): CONF `menu` = it, `prior_obs` = `fit129c.obs.jsonl`, and before the first start
  append to `ROOT/state.jsonl` a line `{"ev": "fit_done", "group": "all", "n_results": 0, "out": ".../fit129c.out.json"}`:
  replay reloads it (every obs tag is in prior_obs), so the first proposals use it; the driver refits after
  `refit_every` results.
- R16 heads (`menu129h.json`, donors on ds8): the same with `fit129h2.out.json` / `fit129h2.obs.jsonl` (V2's head priced
  as R16 u2.75, its measured equivalent; `fit129h` = the older 84-row variant with it as u4 and the head unpriced).
A different menu (other levels, other heads) changes the model's `eb` (the middle option's error): refit on that menu
with `final129.py MENU fit129c.obs.jsonl 0 OUT.out.json [--head-map base=u4]` (colo, ~1 min) and seed with that.
Slopes b above ~4.5 are artefacts of the calibration rows' fixed V2 head (LOG): keep the driver's b_default there.
