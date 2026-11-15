# #185 w162 · identity and timing runs, 2026-11-15 (ds8; VM clock UTC, CST = UTC−6)

Raw results of `search/v3/run183/vload/README.md` (see there for the commands). The 3.5 MB rows files stay on ds8
(`/data/q27/w162/idb/`, `/data/q27/w162/timing/`, `/data/q27/w162/ident/`); their sha256 is in `subset/rows.sha256` and
`timing/` (the rows of the three timing paths hash to the same value).

- `subset/` : four #183 mixes with the trunk lines of levels k1, k3.75, k7 served from final/ (`*.final.mix`), their run
  logs and JSON reports, the control (pair158 through the new build), `run.log` (the runner's log).
- `timing/` : one full k2.5 candidate three ways (mix / packs / final): open, loader and score logs, JSON, times.
- `planes/` : per-weight identity (`cmp_dumps.py`) for levels k0.125, k2.5, k7.5 (336 of 336 each); the k2.5 manifests;
  `format_test.log` (the fork's host test on the build).
- `tests/` : the final installed build's pairact run (rows SAME) and its k0.125 plane check.
