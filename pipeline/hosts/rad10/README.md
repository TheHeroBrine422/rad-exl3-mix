# rad10 — radiance with ROCm 10.1 on box2 / ds8 (#170, w130, 2026-11-13)

radiance supports ROCm 10.1 or 7.2 only; box2 and ds8 had built it with the pip ROCm SDK 7.14.1 (`/data/q27/rad/`). This
directory installs ROCm 10.1.0 beside the pip SDK and builds radiance the way radiance's `Dockerfile` does, into
`/data/q27/rad10/` (never `/data/q27/rad/`). The torch venv keeps pip 7.14.

| step | script | what |
|---|---|---|
| 1 | `rocm10-get.sh` | `/opt/rocm-10.1` = `/opt/rocm/core-10.1` of `rocm/dev-ubuntu-24.04:10.1.0-full` (sha256:5ed1362e…; AMD clang 24.0.0git, ROCm/llvm-project d6f6cb69; HIP 7.16). No `/opt/rocm` is created. Needs amdgpu interface ≥ 3.64 (GUIDE: "this kernel's amdgpu interface is 3.N"); box2 / ds8 (7.0.14-pve): 3.64.0 |
| 2 | `llvm10-vopd.sh SRC` | the Dockerfile's `llvm` stage natively: ROCm's libLLVM at the commit its clang names + `docker/llvm-vopd-pairing.patch`, configured as the Dockerfile; then the shadow tree `/data/q27/rad10/rocm-repeatable` (links into `/opt/rocm-10.1`, own `lib/llvm/bin` + the patched libLLVM). ~3.5 min on ds8 (28 jobs) |
| 3 | `rad10build.sh` | core + our five plugins (exl3quant, qwen35_exl3, exl3dense, exl3r4d, exl3i) from `/data/q27/rad10/src`, device code by the shadow clang with `--rocm-path=/opt/rocm-10.1` and `-DRAD_HIP_VOPD_REPEATABLE=ON`; gfx1201, FFmpeg off, host code by the system g++ 15 (as the 7.14 build; the image uses g++-14). `VOPD=0`: plain 10.1, pairing off, `*-novopd` dirs. ~10 min |
| — | `llvmcheck10.sh` | radiance's `docker/llvm-check.sh` with host paths: pairing off, ROCm's clang and the patched one emit the same device code; pairing on, three compiles agree |
| 4 | `packhome.sh` | pack homes `rad10/home-pack`, `rad10/ref714/home-pack`: a3799b5's `qwen35_exl3` cannot pack from the HF checkpoint (below), so these carry 1aa18de's |
| — | `plug714.sh` | the 7.14 comparison: the same plugin source with pip 7.14 against the frozen 7.14 core → `rad10/ref714/home` |
| 5 | `val10.sh`, `kldcmp.py`, `valbox.sh` | validation (one card by the locks): tests, packs, KLD rows 7.14 vs 10.1, K4H text + decode step, FUSE re-test; `valbox.sh` on the second host |
| 6 | `hostjson10.py HOST.json [--back]` | hostd `score_cmd*` → `rad10/install` + `rad10/home`, `pack_cmd` → `rad10/install` + `rad10/home-pack`; old file `HOST.json.bak-714` |

Source in `/data/q27/rad10/src`: `git archive` of radiance-hero `qc-exl3-v3-perf` a3799b5 + `core-42960d5..ca5491d.patch`
(= `git diff 42960d5 ca5491d`: rad-convert `--no-error` and radiance `--kld-progress`, which the frozen 7.14 install
carries and the perf branch does not; core only, applies cleanly since the perf branch changes only `plugin/`).

Second host: copy `/opt/rocm-10.1` and `/data/q27/rad10` (without `build-llvm`, `llvm-project`) over ds8 ↔ box2's 10 GbE
link (same paths), then `valbox.sh CARD`.

**Known source issue (not the toolchain):** a3799b5 `qwen35_exl3` fails every rad-convert from the HF checkpoint
(`name map for 'v.patch.weight': declared twice`: rad-convert declares at max_enc_patches 16384 when the checkpoint has
`vision_config`; the plugin's probe maps `v.patch.weight`, then `VisionTower::declare(gate_weights=true)` maps it again).
8afc1fc would instead pack the bf16 vision tower into every container. Hence the pack homes. Serving text-only packs
with a3799b5 is fine.
