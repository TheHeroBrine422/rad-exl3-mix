# search/v3/run171 — the V3 search's first real run (#171, w131, 11-13)

What changed before it (search/v3, rules in `../RULES-parts.md`, committed before any run):
- **fit.py: per-part labels.** A two-part slice gives one row per part (`S|x|full`, `S|c|pre<k>`; driver `add_obs`), each
  part with its own offset and anchor (the incumbent on the same documents) and its own paired SE. Optional
  `fit_opts.xscale`: x-part rows predicted as exp(soff) · P^λ (one fitted λ, prior log λ ~ N(0, 0.5)); moves keep
  predicting P (calib scale). **Adopted** by test A (λ 0.765; held-out x log-RMSE 0.146 → 0.085, calib 0.028 → 0.027).
- **fit.py: two starts.** w129's head-θ start (s_head 0.5) is right for menus whose head err is the head-only KL and
  diverged on the stand-in's synthetic menu (log-RMSE 2–8 at HEAD, nfev cap): the fit now runs from both starts and keeps
  the lower cost (fit129h2 reproduced exactly: the head start wins there).
- **seqrule.py: `stat` "pmean" (default)** — the accept / early-stop statistic is the equal-weight mean of the parts' log
  ratios, se = √(Σ se_p²) / n_parts (no stop until every part has 3 documents); "both parts better" stays; `stat:
  "pooled"` = the old one. Hosts: `../earlystop-parts.patch` (pipeline/rxrun/earlystop.py = seqrule.py again; supersedes
  `earlystop-sizeadj.patch`, which was never staged: ds8 stopped on raw KLD until this run).
- **boundsim.py `--stat pmean`**: test B — the pick procedure still picks **B p2 w5** (×1.94 gain/h vs F; S1 tied
  within 2 %, B p2 w5 wins on good-move loss 1.3 % vs 7.5 %). Outputs `data/searchv3/w131/`.
- **driver.py**: per-part fit rows; frontier `kld_norm` per part (offset, λ) then the parts' geometric mean; decisions
  record `logr_pooled` beside the statistic.
- **slices.py poolcheck** fixed: calib docs carry the dataset name in `source`, so the old test skipped every doc and
  always passed; now a doc is calib by `part == "c"` or the golden's calib corpus hash.

| file | what |
|---|---|
| `draw171.py` | c parts: merged fit + search pool, least-used first (headx collections count 0), ≥ 45,000 scored positions at cap 4,096 each |
| `mkslices171.sh` | ds8: golden_pick (c) + pickx 130 xcorp pieces (x, earlier x parts excluded) + goldmix → `/data/q27/w131/slices/golden-s171?` |
| `mkinit171.py` | first incumbents: the starting fit's knapsack optimum per size (as the `solve` move), 0.005 bpw margin |
| `ref171.py` | the monitor's V2 references scored on ver3-8k with ds8's own score command (a hostd eval cannot score a bare container); lines appended to the coordinator's results.jsonl for convplot3 |
| `guard171.py` | VM: deletes scored mix containers, HOLD below 100 GB free on ds8 (off at 130), (re)starts the coordinator |

## Run (VM; state in `/workspace/rad-local/workers/w131/run/`)
```
W=/workspace/rad-local/workers/w131/run
# once, before the first start: the starting model (search/refit/w129/README.md)
echo '{"ev": "fit_done", "group": "all", "n_results": 0, "out": "'$W'/fit129h2.out.json"}' >> $W/drv/state.jsonl
setsid nohup python3 search/v3/run171/guard171.py $W/coord.json v3s171 > /dev/null 2>&1 < /dev/null &   # starts the coordinator
setsid nohup python3 search/v3/driver.py $W/driver.json > $W/driver.out 2>&1 < /dev/null &
python3 search/v3/driver.py $W/driver.json --status ; cat $W/drv/status.txt ; tail $W/drv/driver.log
# stop: touch $W/drv/STOP (driver); kill the guard; touch /workspace/rad-local/multi/v3/HOLD (coordinator drains and exits)
```
More slices: `draw171.py` (registry = `$W/registry.json`) → `mkslices171.sh` on ds8 → register (`slices.py add` with the
copied kld.json, or as in the LOG entry) → ledger (`golden-s171*-c` role exists; new names need their role first).
