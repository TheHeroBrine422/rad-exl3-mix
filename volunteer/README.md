# volunteer/: donated compute for the quantization ladder (v0)

A Folding@home-style job system. Donors run a worker (a container, or the script on our own boxes), it pulls jobs from
our queue, runs them and uploads the results; we verify every result before it counts. **All code is here and open
(Apache-2.0); every job type is built into the worker.** The server only names a job type and sends parameters
that must match that type's schema; it never sends code.

v0 runs the 27B quantization ladder (64 `quant_layer` jobs + 1 `quant_head` job per level) through the queue, first
on our own machines, then on donors' machines. It has been built and tested locally (TESTS.md) but **not deployed**:
nothing listens beyond localhost yet. The server runs either as a plain Python process or as a container
(`docker/server/`, built and tested); the donor worker image (`docker/Dockerfile.rocm`) is ready to publish but not
built yet.

## Architecture

```
            admin.py (on the server host: enqueue levels, requeue / cancel, check, export)
                 |
                 v
   data root ---------- server.py (stdlib HTTP + SQLite WAL) -------- results/<job>/   -> export -> ourq layout
   models/<m>@<rev>/     lease / heartbeat / upload / complete / fail     spool/<job>/<lease>/ (uploads in flight)
   manifest.json         GET /data/<file> (HTTP ranges)
                 ^                     ^
                 |  sha256-checked     |  bearer token per worker
                 |  downloads          |
            worker.py  (donor: lease -> fetch into a capped cache -> run built-in job -> upload -> complete)
```

- **Server** (`server/server.py`, Python stdlib only, so it runs on the queue host as is): workers register with an invite code
  and get a token (only its sha256 is stored). The job table holds type, params, inputs, requirements, state, lease
  owner / expiry, attempts, result. A reaper thread puts jobs whose leases missed their heartbeat back in the queue;
  a job fails for good after `max_attempts` (default 5). Completion is idempotent: once a job is accepted, any
  later completion or upload for it is acknowledged and ignored, whichever lease it comes from. A late result from
  an expired lease is still accepted if nobody has finished the job yet, and the newer lease is then told to stop.
  Files are served one at a time with HTTP Range support (resumable downloads), and only to a worker holding an
  active lease on a job that lists that file. No HTTP endpoint changes anything as admin. With `--admin-token-file`
  there is one read-only overview (`GET /api/v0/admin/summary`); `GET /healthz` is unauthenticated and returns only ok.
- **Admin** (`server/admin.py`): `datapack` cuts the bf16 checkpoint into one file per layer
  (`weights/LNN.safetensors`, byte-identical tensors) and links the input Hessians and H_out snapshots. `manifest`
  hashes everything once (incrementally). `enqueue-level` creates one level's jobs (validated with the same
  schemas the worker uses, deduplicated by settings stamp + layer). `list`, `events`, `requeue`, `cancel` manage
  jobs. `check` runs the checker and stores the verdict. `export` writes the accepted shards in exactly
  `quantize/ourq.py`'s layout (`q-LNN.safetensors`, `q-head.safetensors`, `proxy.jsonl`, `stamp.json`,
  `model.safetensors.index.json`, `config.json`), so knap4 and the search read them unchanged.
- **Worker** (`worker/worker.py`): a loop of lease → fetch → run → upload → complete, with a heartbeat thread
  throughout. Configured by CLI flags only. Every step can be resumed:
  - The current job and its phase are in `STATE/current.json`, written atomically.
  - Downloads go to `<sha256>.part` and resume with a Range request. They are renamed only once the sha256 matches.
  - Uploads resume from the server's offset.
  - ourq itself skips a finished shard.

  A killed worker restarted on the same state dir picks up its job if the lease is still alive, and drops it
  otherwise. The input cache is content-addressed with a byte cap (`--cache-gb`). Least-recently-used blobs are
  evicted, except the ones the current job needs, and the server never leases a job whose inputs exceed the
  declared cap.

  Stopping: SIGTERM/SIGINT or `--stop-file` finishes the current job and then exits. A second signal, or
  `--on-stop abandon`, hands the job back immediately (the abandoned attempt is not counted).

  Other controls:
  - `--window HH:MM-HH:MM` (repeatable, local time): no new leases outside the windows.
  - `--gpu-lock FILE`: flock around GPU jobs, for shared boxes.
  - `--vram-cap-gb`: caps torch's allocator in job processes.
- **Checker** (`checker/`): verifies returned shards without trusting the worker (below).
- **Server container** (`docker/server/`): a slim Python image (stdlib server + numpy for the checker), non-root
  (uid 10001), read-only root filesystem, configured by `RXMQ_*` environment variables, with a `/healthz`
  healthcheck and a compose file (`restart: unless-stopped`; SQLite DB, data store and results as volumes). Built
  and tested on the VM (TESTS.md section 7). See "Server container" below.
- **Worker container** (`docker/`): `Dockerfile.rocm` (gfx1201) pins Python, ROCm/torch wheels, exllamav3 `a3bfbfd0`
  plus our patch series `quantize/exl3-patches/`, and the repo commit (`build.sh` builds from a `git archive` of
  one commit, without our dev shell scripts). It precompiles ourq's side extensions. Versioned tags, OCI labels
  pointing at this repo, and nothing secret inside: the queue URL and the token are given at run time.
  `Dockerfile.cuda` is the same for NVIDIA (quantization only). `run-local.sh` runs the worker without docker.
  Both worker Dockerfiles pass `docker build --check`. Neither image has been built: our build hosts can't handle
  several GB of wheels plus the exllamav3 compile, and building one is the first step of publishing.

## Job types (`jobs/`)

| type | does | inputs (per job) | outputs | needs |
|---|---|---|---|---|
| `quant_layer` | `ourq.py --layers L-L --save-wq` + per-row error claim | `weights/LNN.safetensors`, the layer's 4 input-Hessian files, `config.json`, optional H_out files of the snapshot | `q-LNN.safetensors`, `proxy.jsonl`, `stamp.json`, `job.json`, `rowerr.npz`, `ourq.log` | GPU; declares ≥ 12 GB RAM, ≥ 7 GB VRAM (32 GB card at the 27B in practice) |
| `quant_head` | `ourq.py --only-head --save-wq` | `weights/head.safetensors`, `lm_head.input` Hessian, `config.json` | `q-head.safetensors` + the same small files | GPU, **≥ 24 GB host RAM declared** (peak ~22 GB), ≥ 24 GB VRAM |
| `eval_mix` | interface only (v0 workers refuse it) | mix recipe, cached ladder shards, golden slice | `kl.npy` (one float32 per scored position), `summary.json` | GPU ≥ 24 GB |
| `noop` | sleeps, reads its inputs, writes a deterministic file | `test/*` | `out.bin` | nothing |

Parameters (`quant_layer`): `model`, `revision`, `arch` (`qwen3_5`), `layer`, `K` (multiple of 1/8, 0.125–8),
`codebook` (`mul1|i8|i4|i8c`), `level` (a name), `settings` (`sigma_reg`, `sigma_reg_in`, `sigma_reg_out`,
`scaled_hessian`, `qtf2`, `hout` snapshot id, `seed_offset`) and `stamp`, the settings stamp: a sha256 prefix of
everything except the layer. The worker recomputes the stamp and refuses a mismatch, so every job of a level
provably carries the same settings. `job.json` records the settings stamp, ourq's own code stamp (taken from the
shard), the image id, the GPU and the timings. In the image all paths are fixed, so ourq's code stamp is the same
for every donor running the same image.

**Versions.** The server and the worker each derive a job's inputs and outputs from their own copy of `jobs/`. So
every type has a `VERSION`, bumped on any change of parameters, inputs, outputs or meaning. A worker declares the
versions it runs, the server leases only matching jobs, and the dedupe key of `enqueue-level` includes the version.
(Learned in the 4070 test: a server started before `rowerr.npz` was added refused that upload five times.)
Deploy order: server first, then the new image; old workers simply stop getting jobs of the changed type.

The per-row claim exists because ourq's own `wq/<key>.rowerr.npz` is computed on the weight `quantize_exl3` has
regularized in place (err ≈ ref; the fix `proxy2b/exp/w04-blockgain/rowerr.patch` is not merged), so the job
computes it itself (`jobs/rowerr_tool.py`) from ourq's dequantized weights. Those stay local and are deleted
before upload.

## Security model

**What the server can make a worker do:**
- Run one of the job types built into the worker's image, with parameters that pass that type's strict schema
  (unknown keys, wrong types and out-of-range values are rejected) and the type's extra checks (settings stamp).
- Download files from the configured server URL only. File names must be relative manifest paths of a fixed form
  (no `..`), and they must be exactly the files the job type requires for those parameters (plus the optional
  files it allows, e.g. the layer's H_out files of the named snapshot). Each file is checked against its sha256
  before use.
- Upload only the output files the job type defines, from the job's own output directory, each under a size cap.
- Use GPU, RAM and disk within the caps the donor declared, plus the cache cap.

**What it cannot do:**
- Send code, shell commands or argv. ourq.py's command line is built by `jobs/quant_common.py` from validated
  values and run without a shell.
- Make the worker read or upload any other local file.
- Point the worker at another host.
- Make it run outside its time windows.
- Make it register, log in or open any port: the worker only makes outgoing requests.

Tested against a fake malicious server (TESTS.md): unknown types, extra parameters, path traversal, inputs the
type doesn't allow, an unimplemented type, a bad settings stamp. All were refused, with no download attempted.

**What a worker can do to the server:**
- Lease and finish jobs (one active lease at a time, by default).
- Read only the files its own active leases list.
- Upload only the declared outputs of its own leases, within the caps. Chunks ≤ 64 MB, JSON bodies ≤ 1 MB.

A result is accepted only if its sha256 and size match what was uploaded and the job type's structural check
passes: for quant jobs, exact tensor names, dtypes and shapes against our own bf16 weights, plus the log and the
stamp. Points (v1) come only after the checker passes. Tokens are per worker, stored hashed, and can be disabled
(`admin.py disable-worker ID`).

**Not covered in v0:**
- TLS. The server speaks plain HTTP, so it must sit behind a TLS terminator (e.g. Caddy) or a tunnel before
  leaving localhost or the LAN. Tokens are bearer secrets.
- Rate limiting.
- Sandboxing beyond the container. The worker image runs as uid 10001; the donor run line below adds
  `--cap-drop ALL --security-opt no-new-privileges`, only the GPU devices and one volume for the state. Two
  ROCm-driven exceptions:
  - `seccomp=unconfined`, which AMD's container setup uses: the ROCm runtime makes NUMA memory-policy calls that the
    default profile allows only with `CAP_SYS_NICE`. `--cap-add SYS_NICE` with the default profile may be enough;
    that is untested.
  - No `--read-only`: torch's extension loader takes a lock file in the extension directory on every job.
- The server container runs as uid 10001 with a read-only root filesystem, no capabilities and `no-new-privileges`.
  Its port is published on the host's loopback only by default.
- Donors only ever get public data: the published model and calib Hessians, our H_out diagonals, and later goldens
  built from calib slices. `datapack` links only what it is given.

## Verification (`checker/`)

For each returned shard:
1. **Structure:** exactly `{key}.trellis` (int16 `[in/16, out/16, 16K]`), `.suh` (f16 `[in]`), `.svh`
   (f16 `[out]`) and `.mul1` (int32 marker) for every linear of the layer plan, with the shapes our bf16 weights
   imply. `proxy.jsonl` covers each tensor once with the job's K, codebook and scaled-Hessian setting. `job.json`
   carries the job's settings stamp.
2. **Proxy recompute:** `exl3dec.py` decodes each tensor from its codes on the CPU. It is a numpy transcription of
   exllamav3's decode: integer and fractional K, qtf2's sub-1 salt, the mul1/i8/i8c/i4 codebooks. The checker
   then computes rel = tr(ΔW H ΔWᵀ) / tr(W H Wᵀ) against our own Hessian and bf16 weight.
   - **Full mode:** compares rel with the logged `proxy_err`. exllamav3 logs its post-refit error against the
     undamped Hessian, which is the same quantity.
   - **Sampled mode (`--rows-frac 0.25`):** scores random 128-wide output blocks. The checker picks the seed, so
     the worker can't know which blocks. It compares them with the worker's per-row claim on exactly those
     blocks, and also checks the claim's total against the log.

   Tolerance: `|ratio − 1| ≤ 2e-3`. Measured spread for honest results is ≤ 1.7e-4, which comes from fp16
   rounding of the stored scales. Results are not bit-identical across GPUs, but the check compares the codes a
   worker returned with that worker's own numbers, so cross-GPU nondeterminism doesn't enter it.
3. **Not yet:** a quality check that catches valid-but-bad codes with an honest log. v1 has a per-tensor bound
   from neighbouring levels and our own reference runs (seed spread ~0.5%, TESTS.md).

Cost on the queue host (cpu1) (8 threads, niced), per 27B layer:

| mode | wall | peak RSS |
|---|---|---|
| full | 50–75 s | 2.6 GB |
| sampled 25% | 13–18 s | — |

Batch checks should go layer by layer: the Hessian load and tr(W H Wᵀ) are the same for every level.

## Data per job

Measured numbers are in TESTS.md.

| | down | up |
|---|---|---|
| 27B layer | ~0.8 GB weights + ~0.7 GB Hessians (≈ 1.5 GB) | 50–300 MB shard + ~1 MB small files |
| 0.8B layer (test) | 43 MB weights + 38 MB Hessians | see TESTS.md |

The Hessians and weights of a layer are the same for every level, so a donor working down the ladder layer by
layer hits its cache. Cache-aware leasing is a v1 item.

## How to run

### Server without Docker (tests, our own boxes)

```bash
A="python3 -I volunteer/server/admin.py --db Q.sqlite --data DATA --results RESULTS"
$A datapack qwen38-27b <rev> <bf16 ckpt dir> <calib hessians dir> [--hout <snap> <dir>]
$A manifest --jobs 8
$A enqueue-level --model qwen38-27b --rev <rev> --K 3 --codebook i8 --level k3-i8 \
   --sigma-reg 0.025 --sigma-reg-in 0.005 --scaled-hessian --qtf2 --head
python3 -I volunteer/server/server.py --db Q.sqlite --data DATA --spool SPOOL --results RESULTS --bind 127.0.0.1 --port 8740
#   (expose only through a TLS terminator / tunnel)
$A invite --uses 1                       # one per donor
$A list --level k3-i8; $A check --level k3-i8 --rows-frac 0.25; $A export --level k3-i8 OUT --checked
```

Every `server.py` flag can also be set as `RXMQ_<FLAG>` in the environment (`RXMQ_DB`, `RXMQ_PORT`,
`RXMQ_SPOOL_MAX_GB`, ...); a flag on the command line wins. `admin.py` reads `RXMQ_DB`, `RXMQ_DATA` and
`RXMQ_RESULTS` the same way.

### Server container (a plain Linux host)

Files: `docker/server/Dockerfile`, `entrypoint.sh` (`rxmq serve | admin ... | healthcheck | version`),
`compose.yaml`, `.env.example`. Needs Docker with the compose plugin (tested: Docker 29.8, compose 5.6; Podman untested).

```bash
# 1. build (from a commit; --worktree for uncommitted code). Tags rxm-queue:0.1.0, :0.1.0-<commit>, :latest
volunteer/docker/server/build.sh
# 2. configure: copy .env.example to .env next to compose.yaml and set at least the directories
cd volunteer/docker/server && cp .env.example .env
#    RXMQ_DB_DIR=/srv/queue/db  RXMQ_DATA_DIR=/srv/queue/data  RXMQ_OUT_DIR=/srv/queue/out  RXMQ_SOURCES_DIR=/srv/models
sudo mkdir -p /srv/queue/{db,data,out} && sudo chown 10001:10001 /srv/queue/{db,data,out}   # or RXMQ_UID/GID = yours
#    SELinux enforcing (getenforce)? also RXMQ_SELINUX_RELABEL=1 (bind mounts get :z)
# 3. start; it restarts by itself (unless-stopped) and is healthy once /healthz answers
docker compose up -d && docker compose ps
# 4. data, levels, invites (admin = admin.py inside the image, same volumes)
docker compose run --rm admin datapack qwen38-27b <rev> /srv/rxmq/sources/qwen38-27b-bf16 /srv/rxmq/sources/hessians/calib
docker compose run --rm admin manifest --jobs 8
docker compose run --rm admin enqueue-level --model qwen38-27b --rev <rev> --K 3 --codebook i8 --level k3-i8 ...
docker compose exec server rxmq admin invite --uses 1
docker compose exec server rxmq admin list --level k3-i8
docker compose run --rm admin check --level k3-i8 --rows-frac 0.25          # CPU-heavy: 8 CPUs / 8 GB by default
docker compose logs -f server; docker compose restart server; docker compose down   # down keeps the volumes
```

What lives where (each `*_DIR` is a named volume by default, or a host path):

| volume | in the container | contents | server access |
|---|---|---|---|
| `RXMQ_DB_DIR` | `/srv/rxmq/db` | `queue.sqlite` (+ `-wal`, `-shm`): jobs, leases, workers, events | read-write; local disk only, not NFS |
| `RXMQ_DATA_DIR` | `/srv/rxmq/data` | per-file inputs + `manifest.json` | read-only (`run --rm admin` writes it) |
| `RXMQ_OUT_DIR` | `/srv/rxmq/out` | `results/<job>/` and `spool/` (uploads in flight) | read-write; one filesystem, because completion renames spool to results |
| `RXMQ_SOURCES_DIR` | `/srv/rxmq/sources` | bf16 checkpoint and Hessians that `datapack` symlinks from | read-only |
| `RXMQ_SECRETS_DIR` | `/run/secrets/rxmq` | `admin-token` (optional) | read-only |

`datapack` writes symlinks to the Hessians. Run it through `docker compose run --rm admin` with source paths under
`/srv/rxmq/sources`, so that the links resolve inside the server container too. Running admin.py on the host instead
also works, against the same bind-mounted files: the WAL database handles two processes, and the tests do exactly
this. SQLite gives its journal files the database owner when run as root. Files the host side writes must be
readable by uid 10001 (admin.py writes them 0644).

Settings (`.env`; defaults in brackets):

| variable | meaning |
|---|---|
| `RXMQ_PUBLISH` [`127.0.0.1:8740`] | host address:port to publish on; keep it on loopback (or LAN / tunnel), with TLS in front |
| `RXMQ_LEASE_S` [120], `RXMQ_HEARTBEAT_S` [30], `RXMQ_RETRY_S` [30], `RXMQ_MAX_LEASES` [1] | leases and polling |
| `RXMQ_CHUNK_MAX_MB` [64] | largest upload PUT |
| `RXMQ_UPLOAD_MAX_MB` [0 = off] | ceiling per uploaded file, on top of each job type's own caps |
| `RXMQ_SPOOL_MAX_GB` [0 = off] | uploads get 507 (the worker backs off and resumes) while the spool holds this much |
| `RXMQ_ADMIN_TOKEN_FILE` [off] | enables the read-only `GET /api/v0/admin/summary` (counts per level, workers, last events) for that bearer token (≥ 32 characters, a file in `RXMQ_SECRETS_DIR`, readable by uid 10001) |
| `RXMQ_UID` / `RXMQ_GID` [10001] | user the container runs as |
| `RXMQ_MEM_LIMIT` [2g], `RXMQ_ADMIN_MEM_LIMIT` [8g], `RXMQ_ADMIN_CPUS` [8] | resource caps |
| `RXMQ_SELINUX_RELABEL` [off] | `1`: bind mounts get `:z` (needed on SELinux hosts) |

Nothing that changes the queue goes over HTTP. Admin commands run on the host, via `docker compose exec` or
`run --rm admin`; the admin token gives a read-only view only.

### Donor worker (ROCm, Docker)

The donor needs a Linux host with the amdgpu kernel driver (no ROCm install needed) and Docker. The token is given at
run time only, by one of:
- `RXMW_INVITE`: a one-time invite. The worker registers on its first start and keeps the token in the state
  volume (`/var/lib/rxm/token`, mode 0600). The invite stays visible in `docker inspect`, which is harmless because
  it is used up.
- `RXMW_TOKEN_FILE`: a file mounted read-only.
- `RXMW_TOKEN`: the token in an env variable. This shows in `docker inspect`, so prefer the other two.

```bash
IMG=docker.io/<hub-user>/rxm-worker:0.1.0-rocm7.14-gfx1201
docker run -d --name rxm-worker --restart unless-stopped \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" \
  --security-opt seccomp=unconfined --security-opt no-new-privileges --cap-drop ALL \
  --memory 32g --stop-timeout 900 \
  -v rxm-state:/var/lib/rxm \
  -e RXMW_SERVER=https://<queue-url> -e RXMW_INVITE=<invite-code> -e RXMW_NAME=<your-name> \
  -v /etc/localtime:/etc/localtime:ro \
  "$IMG" --gpu-name r9700 --vram-gb 32 --ram-gb 30 --cache-gb 200 --window 22:00-08:00
# a token you already have, instead of the invite:
#   -v "$PWD/rxm-token:/run/secrets/rxm_token:ro" -e RXMW_TOKEN_FILE=/run/secrets/rxm_token
docker logs -f rxm-worker
docker stop rxm-worker        # finishes the current job (up to --stop-timeout), then exits
docker kill -s TERM rxm-worker  # (while that stop waits) hand the job back now instead
```

- **Disk:** `--cache-gb` caps the input cache in the state volume. Allow that plus one job's staging, ~2 GB for a
  27B layer. The server never leases a job whose inputs exceed the cap.
- **Time windows:** `--window` (repeatable) or `RXMW_WINDOWS=a-b,c-d`, in the container's local time (the
  host's, via the `/etc/localtime` mount, or `-e TZ=...`). Outside them the worker takes no new jobs.
- **Caps:** `--ram-gb`, `--vram-gb` and `--gpu-name` declare what jobs may use. `--vram-cap-gb` caps torch's
  allocator on a shared card. Every flag also has an `RXMW_*` environment variable.

`docker/worker/compose.yaml` (with `.env.example`) holds the same settings plus `restart: unless-stopped`, for donors
who prefer compose:

```bash
cd volunteer/docker/worker && cp .env.example .env    # set RXMW_SERVER, RXMW_INVITE, VIDEO_GID, RENDER_GID, caps
docker compose up -d && docker compose logs -f
```

### Publishing the worker image (you, on a host with Docker and ~40 GB free)

```bash
IMAGE_REPO=docker.io/<hub-user>/rxm-worker volunteer/docker/build.sh rocm <commit>   # never from --worktree
docker image inspect docker.io/<hub-user>/rxm-worker:0.1.0-rocm7.14-gfx1201 --format '{{json .Config.Labels}}'
docker push docker.io/<hub-user>/rxm-worker --all-tags
```

Tags:
- `0.1.0-rocm7.14-gfx1201`: what donors pin.
- `0.1.0-rocm` and `rocm`: these move.
- `rocm-<commit>`: the exact source commit.

Bump `volunteer/VERSION` with every release, and a job type's `VERSION` with any change of its meaning (above).
Deploy the server first, then the new image. Before the first push, set the hub namespace in the default of
`docker/worker/compose.yaml` (`RXMW_IMAGE_REF`) and in the run line above. Run one real job with the built image on
our own R9700 first (`--max-jobs 1`). That checks the GPU flags and the read-only code paths, which no build host has
exercised yet.

### Worker on our own boxes (no Docker)

```bash
STATE=... EXL3_SRC=... EXL3_PATH=... TORCH_EXT=... GPU_LOCK=... volunteer/docker/run-local.sh --server ... --token-file ...
```

### Tests

- `tests/e2e_noop.py`: no GPU, ~2.5 min. Runs the server as a subprocess, or `E2E_SERVER=docker` for the
  container. The workers always run on the host.
- `tests/quant_e2e.sh`: one real layer vs a direct ourq run.
- `tests/checker_tamper.py`.

## v1

- **Web UI on the donor side:** start / stop button, time windows, storage and bandwidth limits (the worker
  already has the flags; the UI writes them and the stop file). An upload bandwidth limit and a preference for
  small-upload jobs on slow uplinks.
- **Points and leaderboard** (notes, user 2026-11-04):
  - Points only for checker-accepted results.
  - Base points per job type scaled by its reference cost (R9700 card-hours).
  - A small quick-return bonus; nothing for timed-out, reassigned jobs.
  - A share of the points for bytes served.
  - Opt-in names and teams, per-donor stats, "your card quantized K2.375 of EXL3-V3", credits in the model cards.

  The `events` table already logs every lease, expiry, accept and check.
- **Quality check:** a per-tensor error bound from neighbouring levels (the error must fall with K, smooth in log)
  and from our own reference runs. Random full re-checks on our hardware.
- **Cache-aware scheduling:** prefer jobs whose inputs a worker already holds (same layer, next level). Bootstrap
  seeding of the ~105 GB.
- **Healing jobs:** one per ladder level per layer group, hours each, checkpointed state of tens of MB.
- **Layer-streamed statistics collection** for new models: Σdy² partial sums (~10 MB up), input-Hessian partials
  by (token chunk, layer group).
- `eval_mix` implementation with spot re-scoring.
- TLS / a reverse proxy in front, rate limits, token rotation, signed images
  (cosign), reproducible builds.
