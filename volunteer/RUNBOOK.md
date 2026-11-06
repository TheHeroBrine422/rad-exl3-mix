# rad-quant@home: operator runbook

Start the server, load a job batch, watch it, verify the results, pay points. Everything here runs on the queue host
(the server container) or on our own boxes; nothing goes over HTTP that changes the queue except the search's batch
endpoint (its own token, eval_mix on public goldens only). Commands use `A` for admin.py:

```bash
# in the container (data and results read-write):  A="docker compose run --rm admin"
# beside it, inside the running server:            A="docker compose exec server radquant admin"
# plain process:                                   A="python3 -I volunteer/server/admin.py --db Q.sqlite --data DATA --results RESULTS"
```

## 1. Start the server

1. Build the image from a commit: `volunteer/docker/server/build.sh <commit>` (tags `radquant-server:0.2.0`, `:0.2.0-<commit>`).
2. `cd volunteer/docker/server && cp .env.example .env`; set the volume directories (`RADQUANT_DB_DIR`, `_DATA_DIR`,
   `_OUT_DIR`, `_SOURCES_DIR`: local disk; the DB not on NFS; spool and results share `OUT`), `RADQUANT_PUBLISH`
   (loopback or the tunnel's address) and, on SELinux hosts, `RADQUANT_SELINUX_RELABEL=1`. `chown 10001:10001` the
   DB and OUT directories.
3. Secrets in `RADQUANT_SECRETS_DIR` (≥ 32 random characters each, `chmod 0400`, owned by 10001):
   `admin-token` (read-only summary) and `search-token` (the search's batch API), and set
   `RADQUANT_ADMIN_TOKEN_FILE=/run/secrets/radquant/admin-token`, `RADQUANT_SEARCH_TOKEN_FILE=/run/secrets/radquant/search-token`.
   `python3 -c "import secrets; print(secrets.token_urlsafe(32))" > secrets/admin-token`.
4. `docker compose up -d && docker compose ps` (healthy within seconds).
5. TLS: a terminator in front (Caddy: `queue.<domain> { reverse_proxy 127.0.0.1:8740 }`) or a tunnel. Nothing is
   public until this step; it needs the user's go-ahead (outward-facing).
6. Our own workers: a verifier for spot checks on one of our cards (`run-local.sh` on the GPU box under its card lock,
   `--gpu-lock '/root/card{card}.lock'`), registered with `$A add-worker verifier-gpubox` and flagged
   `$A verifier ID`. Without a verifier running, drawn results wait (their points stay pending).
7. Donors: `$A invite --uses 1 --note "<who>"` per donor; send the code and the queue URL privately. One person with
   several boxes: `$A invite --uses 3 --personal --note "<who>"` (every registration with it is one donor). Big-box
   donors we know personally: `$A trust ID` after they registered.
8. Status page (public, read-only; on by default every 60 s at `/`): write `campaign.json` into the DB directory
   (`{"title": ..., "ladder": {"levels": 73}, "search": {"evals": N}}`; README "Public status page"). For a static
   host as well: `RADQUANT_STATUS_DIR=/srv/radquant/out/status` and publish that directory (e.g. `rsync` from cron).
   Look at it before anything is public: `curl -s http://127.0.0.1:8740/ | less`.

## 2. Load data (once per model / level / golden)

```bash
$A datapack qwen38-27b <rev> /srv/radquant/sources/qwen38-27b-bf16 /srv/radquant/sources/hessians/calib \
   [--hout <snap> <dir>] [--hf] [--bf16-rad /srv/radquant/sources/qwen38-27b-bf16.rad]   # also writes model.json
$A ladder-add qwen38-27b <rev> k3-i8 /srv/radquant/sources/ladder/k3-i8.rad           # each level the search mixes from
$A golden-add xcorp-s07 /srv/radquant/sources/goldens/xcorp-s07 --public --corpus xcorp --pool search
$A corpus-add xcorp-m2 /srv/radquant/sources/corpora/xcorp-m2.jsonl --public --chunk-docs 64  # for golden_record
$A manifest --jobs 8                                                                         # after every change
```

- Data rules (AGENTS.md): **only public data.** `golden-add` / `corpus-add` refuse anything without `--public` and any
  document whose source is one of ours (agent traces, session logs); `manifest` never lists a golden or corpus whose
  marker is not public; the server re-checks the marker on every file it serves. Never put a trace-derived golden in
  the data root at all. A golden built from calib must be in the document-use ledger (`eval/pools/ledger.jsonl`) and
  follow the pools: search accept / reject only on search-pool or xcorp goldens, never verify.
- Disk: `ladder-add` writes each level once more as pieces (reflinked from the `.rad` on XFS / btrfs, copied
  otherwise: ~10-17 GB a 27B level). `df -h` first.

## 3. Load a job batch

- **A ladder level** (quant jobs): `$A enqueue-level --model qwen38-27b --rev <rev> --K 3 --codebook i8 --level k3-i8
  <frozen settings> --head` (64 layer jobs + the head; requirements from model.json: 27B layers 32 GB / 24 GB RAM, the
  head 48 GB RAM).
- **Search candidates** (eval_mix): from the search box, `client/searchq.py --server https://<queue> --token-file
  search-token submit SPEC.json`, then `wait ID` and `collect ID OUT` (rows per candidate in radiance's FILE.rows
  layout). Or on the queue host: `$A batch-submit SPEC.json`, `batch-status ID`, `batch-collect ID OUT`. The spec is in
  `server/batch.py`. Repeated candidates reuse their earlier result.
- **Reference goldens** (trusted big boxes): `$A enqueue-golden --model qwen38-27b --rev <rev> --corpus xcorp-m2
  --method tp2|staged --top 512 --max-model-len 32768`.
- Priorities: `--priority` on `enqueue-noop` / `enqueue-golden` / the batch spec's `priority`; higher first.

## 4. Watch

- `$A list [--level L] [--state S]`, `$A events [JOB]`, `$A workers` (flags, declared caps, last seen).
- `curl -H "Authorization: Bearer $(cat secrets/admin-token)" http://127.0.0.1:8740/api/v0/admin/summary`: counts per
  level and state, workers, lumens, spot-check counts, gang reservations, spool bytes, the last events.
- `docker compose logs -f server`.
- Donors: `$A donors` (display name, opt-in, workers, lumens, rank). A name that should not be on the page:
  `$A donor ID --hide` (`--unhide`; `--display-name` / `--leaderboard on|off` to correct an entry). Two donors who
  are one person: `$A link WORKER DONOR`. Donors see their own side with `radquant status | jobs | points | logs`.
- A job that keeps failing: `$A events JOB` (worker errors are in the `fail` events), fix, `$A requeue JOB`. A level
  to redo: `$A requeue --level L --force` (accepted results are moved aside). Stop work: `$A cancel --level L`.
- A gang job waits for a multi-card worker to drain (`reserve` events); a drain that has not finished after
  `RADQUANT_RESERVE_MAX_S` is given up and may be retried by another worker.

## 5. Verify

- **Quant levels:** `$A check --level L --rows-frac 0.25` (CPU checker: ~15 s per 27B layer sampled, ~1 min full;
  `RADQUANT_ADMIN_CPUS` / `_MEM_LIMIT` bound it). A FAIL quarantines the donor and requeues its unconfirmed results;
  `--no-quarantine` to look first.
- **Spot checks** run by themselves: 4% of every donor's accepted results plus its first of each type are re-run on a
  verifier and compared (`spot_draw`, `spot_pass`, `spot_fail` events; `verify_state` in `$A list`). `$A spot JOB`
  draws one by hand.
- **On a `spot_fail`:** the donor is already quarantined and its unconfirmed results requeued. Read the comparison
  (`verify_detail` of the job: worst tensor / rows out of tolerance / container hashes). A hardware fault or a
  tolerance that is too tight (eval rows: calibrate on the first R9700 pairs) is lifted with `$A unquarantine ID`; a
  cheat stays, and `$A disable-worker ID` revokes the token.
- **Export** only verified work: `$A export --level L OUT --checked` (ourq layout: knap4 and the search read it).

## 6. Pay points

Lumens are granted by the server as results are verified and revoked on quarantine; nothing to do per job.
`$A points` (table) / `$A points --json` (per worker: id, name, lumens, verified jobs, flags); `$A donors --json`
per donor (what the leaderboard aggregates). Worker names are what donors registered with and are never published;
the status page shows only opted-in display names. Credits in the model cards: the donors with lumens
on that release's levels (`$A list --level L` has the `by` worker of each job).

## 7. Upgrades and backups

- A job type's meaning changes → bump its `VERSION`; a release → bump `volunteer/VERSION`. Deploy the server first,
  then the new worker image; old workers stop getting the changed type.
- The v0 names (`RXMQ_*`, `RXMW_*`, `rxmq`) are read in 0.2 with a deprecation note and removed in 0.3.
- Backups: `sqlite3 queue.sqlite ".backup q.bak"` (consistent under WAL) and the results directory; the data root can
  be rebuilt from the sources (`datapack`, `ladder-add`, `manifest`).
