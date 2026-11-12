# Ideas runs (research-idea passes, 2026-11-02 → 2026-11-08)

Reports from our idea-generation passes: several Claude Opus agents ("lenses"), each reading the same snapshot of the
plan docs, experiment log and a paper corpus, then a critic agent ranking the union (`ranked.md`). They are raw
proposals with the agents' own estimates, not results: what was tested and how it came out is in
`docs/experiments/INDEX.md` / `LOG.md` (experiment numbers are cited there).

| run | lenses | notes |
|---|---|---|
| `2026-11-02/` | codes, composable, crossarea, heal, hout, kernels, lowbpw, measure, open, redteam + `ranked.md` | round 0 |
| `2026-11-02-faraway/` | faraway | exploration lens, run on its own |
| `2026-11-07/` | the same ten + faraway, `ranked.md` | round 1 |
| `2026-11-08r2/` | codes, composable, faraway, lowbpw, measure, redteam, `ranked.md` | round 2, scoped to pre-freeze (ladder) changes; its §1 shortlist became #138–#148 |

Paths inside the reports point at our working machines and local folders; they are kept as written.
