# volunteer/ v0: tests (2026-11-04)

Times are US Central (CST). Machines: the VM (noop end-to-end, the server container), the RTX 4070 sandbox (the real
`quant_layer`, CPU checks), the queue host (cpu1) (server throughput, checker timing on the 27B). Nothing was exposed beyond localhost.
Section 7 (the server container, w13b) was added the same evening.

## 1. End-to-end with noop jobs (`tests/e2e_noop.py`, VM, localhost)

A server with `--lease-s 6 --heartbeat-s 1.5`, workers as subprocesses. Each run takes ~50 s. Three runs
(3:44–4:10 PM CST): **39/39 checks passed every time**. w13b grew the suite to 54 checks (local) / 61 (container):
see section 7.

| scenario | what is checked | result |
|---|---|---|
| tokens | no token / bad token → 401; bad invite → 403; an invite with 3 uses registers 3 workers, the 4th → 403 | pass |
| data rules | a file without an active lease listing it → 403; not in the manifest → 404; `../` → 404 | pass |
| 3 workers, one killed (SIGKILL) mid-job | 9 jobs × 3 s; the victim's lease expires, the job is re-leased and accepted **once**, by another worker (events: `lease, expire, lease, accept`); every result byte-exact | pass |
| killed worker restarted after its lease died | finds `current.json`, heartbeat says the lease is gone, drops the job, no leftovers | pass |
| killed worker restarted within its lease | resumes the same lease and completes it (events: `lease, accept`, no expiry) | pass |
| corrupted upload | bytes with one bit flipped, true sha256 declared → 422 `bad: [out.bin]`, staged file deleted; correct re-upload accepted | pass |
| chunked / resumable upload | wrong offset → 409 with the resume point; two chunks at offsets 0 / 3000 | pass |
| duplicate completion | a second `complete` (even with a different sha) → 200 `duplicate`, ignored; upload to an accepted job → 409; stored result unchanged, one `accept` event | pass |
| caps | upload of a name that is not an output → 403; over the per-file cap → 413 | pass |
| another worker's lease | heartbeat on someone else's lease → 403 | pass |
| abandon | `fail` with `abandon` → job back to `queued`, attempt not counted | pass |
| disk cap eviction | cache cap 700 kB, four 300 kB inputs a, b, c, a → 2 evictions, cache ≤ cap; a job needing 900 kB is not leased to this worker | pass |
| Range resume | 120 kB of a 300 kB input already in `<sha>.part` → 180 kB downloaded, sha256 verified | pass |
| malformed jobs from our server | `noop` with `seconds: "3"` and with an extra key `cmd` (injected into the DB) → every worker rejects, the job fails after `max_attempts` (2) with 2 `reject` events; workers keep running | pass |
| fake malicious server | 6 crafted leases (unknown type `shell`; extra key `argv`; input `test/../../etc/passwd`; an input the type does not allow; `eval_mix` (unimplemented); `quant_layer` with a wrong settings stamp) → all 6 rejected via `fail` with `reject`, **no download attempted** | pass |
| graceful stop | SIGTERM → finishes the current job, exits 0; `--on-stop abandon` → job handed back within 5 s, attempt not counted | pass |
| time windows | window arithmetic incl. across midnight; a worker outside its window leases nothing | pass |

Request overhead: a 1 s noop job takes 1.01 s lease-to-complete on localhost.

## 2. Server throughput (the queue host (cpu1), loopback, niced)

One noop job with a 1.5 GB input and a 64 MB output: download + sha256 check **4.7 s** (~320 MB/s, limited by
hashing), upload of 64 MB **0.48 s** (both sides hash it). The stdlib server (sendfile for data) is far from the
bottleneck for home connections; the queue host's 10 Gbps link is the limit for the bootstrap.

## 3. Checker (CPU decode + proxy recompute)

**Decoder validation** against ourq's own dequantized weights (`--save-wq`) on 27B layer 10 at K2.5 i8 (fractional,
the lead's hconv `q-none` run): every tensor matches to a relative Frobenius error of 4.0e-4 (fp16 rounding of the
stored scales and of `wq`). The recomputed proxy matches the logged `proxy_err` to ≤ 1.6e-4 (in_proj_qkv 1.3e-5,
in_proj_z 3.0e-5, out_proj 8.7e-6, gate 4.9e-6, up 4.3e-6, down 1.6e-4).

**ourq's `rowerr.npz` is wrong** (known; `proxy2b/exp/w04-blockgain/rowerr.patch` not merged): err/ref ≈ 1.0008
instead of 0.0032 for in_proj_z. The jobs therefore compute the per-row claim themselves (`jobs/rowerr_tool.py`);
its totals equal the logged proxy to ≤ 1e-5.

**Sampled mode** (25% of output blocks, checker's seed) needs the per-row claim: without it, a block sample can't
be compared with a whole-tensor number (out_proj's sample came out 1.66× its mean). With the claim, two seeds give
≤ 1.7e-4 on the sampled blocks.

**Timing per 27B layer on the queue host (cpu1)** (8 OpenBLAS threads, `nice -n 19`, the lead's sweeps running):

| mode | wall | notes |
|---|---|---|
| full | 50.6–75.3 s | peak RSS 2.6 GB; down_proj is ~40% (17408² Hessian: 2.4 s load, 2 GEMMs) |
| sampled 25% | 13.4 s | |

At 25%: ~14 min per level (64 layers) in one process; several in parallel. Cheapest correct decode path: this
numpy decoder on the CPU (0.8–5.4 s per tensor). exllamav3's reconstruct needs a GPU. `mix/hproxy.cpp` decodes
radiance `.rad` containers (HIP), not raw ourq shards, so it would need a conversion first.

**Tolerance:** `|ratio − 1| ≤ 2e-3`, ~12× the largest honest deviation measured (1.7e-4 on the 27B, 2–3e-5 on the
0.8B). Cross-GPU nondeterminism does not enter: the check compares the codes a worker returned with that worker's
own logged numbers.

**Tamper test** (`tests/checker_tamper.py`, on the 0.8B layer-5 result, 4070 CPU): **8/8 as expected**.

| case | checker |
|---|---|
| untouched, full / sampled | pass (2.0e-5 / 3.0e-5) |
| 1% of one tensor's trellis words changed, full / sampled | fail (ratio off by 4.0 / 3.9) |
| per-row claim 10% low | fail (0.11) |
| logged proxy 5% low | fail (0.053) |
| a tensor missing from the shard | fail (structure) |
| wrong settings stamp in job.json | fail (structure) |

## 4. One real `quant_layer` on the RTX 4070 (`tests/quant_e2e.sh`)

0.8B, layer 5, K3 i8, settings as `proxy2b/q2b.sh`: qtf2, scaled Hessian, σ 0.025, input group 0.005. Everything
ran on the box: server on localhost, worker with `--vram-cap-gb 7.2`, and every GPU step inside one `flock` on the
shared GPU lock.

**Session 1** (lock held 4:25:57–4:26:41 PM CST, **43.5 s of GPU**):

| step | result |
|---|---|
| direct `ourq.py` (as q2b.sh, `--layers 5-5 --save-wq`) | 4.9 s wall, of which the layer 3 s: **~1.9 s per-process overhead** (imports, CUDA init, cached extension loads); peak VRAM 976 MiB |
| worker | lease → stage → ourq → rowerr_tool → upload took ~7 s per attempt; ourq and the rowerr step ran fine every time, but **every upload was refused**: the server had been started before `rowerr.npz` became an output (version skew). 5 attempts, then the job failed as designed. Fix: VERSION 2 for the quant types, version in the dedupe key, README rule |
| direct, `--seed-offset 1` | 4.4 s; per-tensor proxy vs seed 0: −0.37% to +0.21% (in_proj_z −0.37, qkv −0.18, out −0.13, up −0.03, gate +0.21, down +0.20): the seed spread a quality bound has to allow |

**Server-side path on the real output (CPU, no GPU):** the direct run's shard + proxy log + a job.json + the
per-row claim from `rowerr_tool.py` (CPU fallback) → `quant_layer.accept()` OK (exact keys / dtypes / shapes
against the datapack's bf16 weights, log, stamp) → checker PASS, full 2.0e-5, sampled 2.2e-5 → tamper test 8/8
(section 3).

**Session 2** (byte-for-byte comparison, plus `tests/dec_rates.sh`: the decoder at K0.5 i8 (sub-1, salt),
K2.125 i8 (1/8 step), K3.5 mul1 and K2.25 i8c): **NOT RUN YET.** It is queued on the GPU lock since 4:27 PM CST.
At 5:27 PM CST the lock was still held by a deadlock between two of the lead's scripts (`hconv.sh` holds the lock
and waits for no `quantize/ourq.py` process to exist; `hest.sh`'s `flock … python3 …/quantize/ourq.py` matches that
pattern while it waits for the lock). The session needs ~3 min of GPU. It writes `quant_e2e.out` /
`dec_rates.out` in its work dir and leaves the GPU lock as soon as it finishes.

## 5. Data per job

| job | down | up |
|---|---|---|
| 0.8B layer (measured) | 81.3 MB: weights 43.0 MB + 4 Hessians 38.3 MB + config 3 kB | shard 8.1 MB + rowerr ~0.3 MB + logs / stamps ~35 kB |
| 27B layer (from the files) | **1.55 GB**: weights 0.74–0.77 GB + Hessians 0.79 GB (once per layer: the same for every level) | shard ≈ params × K / 8: 116–120 MB at K2.5, ~190 MB at K4, ~380 MB at K8; rowerr ~1 MB |
| 27B head | 2.54 GB weights + 52 MB Hessian | ~0.4 GB at K2.5 (1.27 G params) + rowerr 4 MB |
| bootstrap (whole trunk + head) | 48.7 GB weights + 50.3 GB Hessians + 2.6 GB head ≈ 102 GB | — |

Worker overhead per quant job, measured on the 4070: ~1.9 s for ourq's process start, plus a second short process
for the per-row claim. Staging, hashing and HTTP are well under a second for the 0.8B; the 27B's 1.55 GB of inputs
hash in ~5 s. Compared with ~2–3 min of quantization per 27B layer on an R9700, the overhead is a few percent.

## 6. Not tested in v0

- The worker images: not built. Both Dockerfiles pass `docker build --check`. A ROCm build (several GB of wheels
  plus an exllamav3 compile) is too heavy for our build hosts; it is the first step of publishing (README).
  Untested, therefore:
  - the GPU run flags (`/dev/kfd`, `/dev/dri`, the video / render groups, `seccomp=unconfined`);
  - whether torch's extension loader runs from the image's precompiled `torch_ext` without writing into the
    read-only source tree.
- The server container on Podman, and on the queue host itself.
- `quant_head` end to end.
- Any 27B job through the queue.
- Cross-GPU (R9700 vs 4070) differences.
- TLS.
- More than one machine.

## 7. The server container (`docker/server/`, VM, 2026-11-04, 5:36–6:14 PM CST)

Docker 29.8.2 and compose 5.6.0 on the VM (Fedora, SELinux enforcing).

**Image:** `build.sh --worktree` builds `rxm-queue:0.1.0` in 19 s, 284 MB, on `python:3.14-slim-trixie`
(digest `sha256:930e1abba7c4…`) + numpy 2.5.4. Contents checked: only LICENSE, NOTICE and
`volunteer/{common,jobs,server,checker}` + the entrypoint and healthcheck. User 10001, OCI labels (source, revision,
version, created, licences), `HEALTHCHECK` on `/healthz`.

**The noop suite against the container** (`E2E_SERVER=docker tests/e2e_noop.py`). The server runs from
`compose.yaml` (project `rxmq-e2e`), with the test dir's `db/`, `data/`, `out/` and `secrets/` bind-mounted. The
workers, admin.py and the test's own SQLite reads all run on the host against the same files. Final code: **61/61
twice** (6:10 and 6:14 PM CST). The plain-process path (`E2E_SERVER=local`) on the same code: **54/54**. Earlier
iterations passed 59/59 and 60/60 as checks were added. Every check from section 1 passes in both modes, plus:

| check | result |
|---|---|
| `/healthz` without a token → 200; the container turns `healthy` within ~4 s (`start_interval` 2 s) | pass |
| server process (PID 1) runs as uid 10001; read-only rootfs, `cap_drop: ALL`, restart policy `unless-stopped` | pass |
| `docker compose exec server rxmq admin ...` and `docker compose run --rm admin ...` beside the running server | pass |
| port published on `127.0.0.1` only | pass |
| server-wide upload ceiling (`RXMQ_UPLOAD_MAX_MB=48`) under a job type's 64 MiB cap → 413 at byte 48 MiB + 1 | pass |
| read-only admin summary: no token / a worker's token / a wrong token → 401; admin token → 200 (counts per level, workers without tokens) | pass |
| worker configured by environment only: `RXMW_INVITE` registers once, token file written 0600, the next start uses it; `RXMW_IMAGE` reported in caps | pass |
| **restart mid-run** (2 workers, 6 jobs × 3 s, 200 kB outputs; at the time 2 accepted, 2 leased, 2 queued): `docker compose restart` (0.7 s) | back in 0.9 s; all 6 accepted once, byte-exact, `PRAGMA integrity_check` ok, no lease expired; workers exit 0 |
| **crash mid-run:** SIGKILL of the container's PID 1 from the host | the restart policy brings it back (RestartCount 1), healthy in 0.5 s; same result |
| **recreate mid-run:** `docker compose down` + `up` (a new container on the same volumes) | back in 1.2 s; same result |
| **crash mid-upload:** a 40 MiB output in 1 MiB chunks, SIGKILL when 8–9 MiB are in the spool | the partial upload survives in the spool (once with a half-written chunk: 8.5 → 8.8 MiB); the worker resumes from the server's offset (sent 39.2–40.0 MiB in total, not 48+), accepted once, byte-exact |

The local mode runs the same restart scenarios with SIGTERM (graceful exit, code 0), SIGKILL and a respawn.

**Defaults as a user would run them:** named volumes, no `.env`. `docker compose up -d` gives a healthy server in
4 s, with or without `RXMQ_SELINUX_RELABEL=1`. Admin works via `run --rm admin invite` and `exec server rxmq admin
workers`. `down -v` leaves nothing behind.

**Found and fixed on the way:**
- **SELinux:** a bind-mounted host dir is not writable by the container (`unable to open database file`), so
  `RXMQ_SELINUX_RELABEL=1` adds `:z` (opt-in, because it relabels the host dir).
- **manifest.json was written 0600** (`mkstemp`). When admin.py ran as another user (root on the host), the
  container's server could not read it: every data request got 404 and nothing was leased. `write_json_atomic` now
  gives the mode `open()` would (0644 under umask 022). The server also logs an unreadable manifest and keeps the
  last good one.
- The spool and the results must share a filesystem (completion renames). The compose file puts both under one
  volume, and the server falls back to copy + delete on `EXDEV`.
- Docker's port proxy resets connections while a container starts: the test client now treats that as "not up".

**Worker image (not built):** `docker build --check` passes for `Dockerfile.rocm` and `Dockerfile.cuda` (after
re-declaring the base-image ARG after `FROM`). The donor `docker run` flags parse (`docker create` with stand-in
devices: `seccomp=unconfined`, `no-new-privileges`, numeric `--group-add`, `--memory`, `--stop-timeout`).
`docker/worker/compose.yaml` renders with `docker compose config` and refuses to start without `RXMW_SERVER`,
`VIDEO_GID` or `RENDER_GID`.
