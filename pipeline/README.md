# rxq — the V3 quantizer pipeline (`pipeline/` of rad-exl3-mix)

The V3 quantizer pipeline for EXL3 trellis codes (Qwen3.8-27B first; model-agnostic in time): input statistics,
per-tensor trellis quantization with LDLQ, post-quantization scale fits, and the level assembly that the knapsack and
packer consume. One Python package (`rxq`), one settings file (`config/v3.toml`), one job unit (a tensor at a level)
that a compute-donation queue (rad-quant-at-home) can hand out.

- **DESIGN.md** — stages, the job unit, the settings and their evidence, open items.
- **PORTING.md** — what was ported from the research scripts (rad-exl3-mix) and how each port was checked.
- **PROFILE.md** — where the quantizer's time goes, per stage and family, and the ranked optimization targets.

## Run

Needs: python ≥ 3.11, torch with CUDA or ROCm, the exllamav3 fork (rad-exl3-mix `quantize/exl3-patches/`) on
`PYTHONPATH`, a rad-exl3-mix checkout for the run-time kernels.

```
export PYTHONPATH=/path/to/rad-exl3-mix/pipeline:/path/to/exllamav3-fork
export RXQ_RXM=/path/to/rad-exl3-mix RXQ_EXL3_SRC=/path/to/exllamav3-fork/exllamav3/exllamav3_ext
export TORCH_EXTENSIONS_DIR=/path/to/torch_ext     # kernel build cache
# one layer at K2.5, the settings file's defaults (= the ladder's quantizer)
python -m rxq quant --layers 10-10 -K 2.5 --ckpt MODEL --hess HESSIANS -o out/K2.5/L10
# several levels in one process; a settings override
python -m rxq quant --keys model.language_model.layers.39.self_attn.q_proj -K 2,4,6 --ckpt MODEL --hess HESSIANS \
    -o 'out/K{K}' --set quant.tile_cd.enable=true
python -m rxq job spec.json -o out/job17        # a queue job (DESIGN.md "Job unit")
python -m rxq assemble level/K2.5 out/K2.5/*    # job outputs -> an ourq-layout level (rad-convert exl3i src=...)
python -m rxq stamp                             # the settings stamp of config/v3.toml
python -m rxq levelargs LEVEL                   # an ourq level's settings as rxq quant arguments (refuses what rxq can't reproduce)
python -m rxq.stats blend-hin CALIB XCORP 0.5 OUT       # S1: xcorp H_in blend
python -m rxq.hin MODEL META OUT --scratch DIR          # S1 level pass: input Hessians (published layout)
python -m rxq.hout collect OUT NSEQ / estimate OUT DST  # S1: Σdy² collection (collect.py in place, both cards) / bn estimator
python -m rxq.stats hout SNAPROOT OUT dy                # S1: block-normalized H_out (OUT/all)
python -m rxq.swjfit MODEL META OUT --src NAME=LEVEL --scratch DIR --fused --no-cache   # S4 level pass: swj scales
python -m rxq.swj apply LEVEL SCALES OUT                # S4: swj scales into a level
python -m rxq.post apply LEVEL SWJ,LEVEL/c9 OUT         # S4 + S5b: swj and the jobs' C9 scales (assembled into LEVEL/c9) in one pass
python -m rxq.sibmom MODEL META OUT --level LEVEL      # S5 level pass: writers' sibling moments
python -m rxq.asym denoise CALIB MOMENTS OUT --lam 1 --rho 0 --ckpt MODEL   # S5: writers' G / F
python -m rxq.post c9 LEVEL OUT --model M --hess H      # S5b as a level / mix pass (asym writers: --dn DN); plain writers get
                                                        # C9 in their quant jobs (post.c9 on), nothing to run
python -m rxq.post plan -K 4                            # the post passes a level gets under the settings
python -m rxq.knap library LEVELS... -o lib.json --packs DIR   # S7: the library of levels
python -m rxq.knap solve lib.json out.jsonl -t 2.5,3 -l 0,3e-7  # S7: knap5 over the library
python -m rxq.knap handoff out.jsonl lib.json MIXDIR --tag T --frontier F   # S7 -> search stream (mix files + frontier)
python -m rxq.mix recipe LEVEL                          # S8: rad-convert recipe of a level (levelmix PACK: radmix mix)
python -m rxq.ladder levels | estimate [--ldlq blocks] | req   # the 55 levels in build order, R9700 card-hours per pass,
                                                        # the per-tensor VRAM / seconds table (w81's 11-08 measurements)
python -m rxq.ladder post [--swj-batch N]               # per level, the steps after its jobs; up to N levels of a ladder
                                                        # pass that have an swj step, in build order, share one swj level pass
python -m rxq.ladder jobs MANIFEST --model q27 --rev R --pass 1 -o jobs.jsonl   # rad-quant-at-home rxq_quant jobs (inputs
                                                        # pinned by the datapack manifest's sha256): admin.py enqueue-jobs
python -m rxq.pairs mixes A.rad B.rad --arm attn_qg     # swap-pair hygiene (also: levels, same, splice)
```

`VRAM_CAP_GB` caps the CUDA allocator (shared cards); `RXQ_REFIT_CPU_FALLBACK=1` lets an out-of-memory scale refit
(27B ffn_down on ≤ 12 GB cards) run on the CPU. **Test only:** `--device cpu` with `RXQ_CPU_STANDIN=1` runs the fork's real
quantize.py on the CPU with `rxq/cpukern.py`'s stand-in kernels (a greedy trellis encoder: valid codes, worse than the
real search; stamped `cpu-standin`, refused by assemble unless `--allow-standin`): dry runs of the job / queue wiring.

## Tests

CPU (no GPU, seconds to minutes; results of the 11-07 runs in `data/w76-2026-11-07/`):
- `tests/check_tilecd.py` — tile CD: rxq == the research module, M = I gate, two-sided descent, wrappers (stub quantizer).
- `tests/check_i4.py` — the i4 + qtf2 path (build flags, ourq's assert, engine, sub-1 plan); `tests/check_i4lut.py OUT` —
  the i4 codebook vs w53's C oracle.
- `tests/check_pairs.py OUT` — swap-pair hygiene, splice, no-op gate on synthetic levels and RAD2 containers.
- `tests/check_mix.py SWAP27` — S8 recipe / mix text vs the swap27 arms' files; `tests/check_knap.py OUT` — S7 driver vs
  w63's knapsack solve (~1 min).
- `tests/check_levelpass.py WORK RXM MODEL08` — swjfit / sibmom / hin vs the research tools on the 0.8B (colo, ~1 h), and
  `swjb`: a source fitted in a two-source swj level pass equals the source fitted alone (`swjb` alone: ~4 min on 4 CPUs);
  `tests/check_deq_c9.py WORK RXM LEVEL MODEL HESS` — decoder, C9 and the quant job's C9 path vs the research tools on
  a 27B level; `tests/check_c9job.py JOB_OUT C9E27 MODEL HESS WORK` — a quant job's C9 files vs c9e27.py.
- `tests/check_cpu.py OUT` — the 11-06 CPU ports (blend, H_out bn, swj math / apply, mkdenoise both paths; 4070 data).
- `tests/check_assemble.py LEVEL OUT N` — level assembly round trip from an ourq level, mixed-stamp refusal.
- `tests/check_grouphead.py [QUANTIZE_PY]` — group-head su draw (11-08 gpucheck27 i4 FAIL): rxq per-tensor == ourq's
  group run, `ourq --only` without the group head differs, σ does not move su, the sub-1 damping rule per family, the
  check_quant BADREF guard (seconds).

- `tests/check_cpukern.py [RQAH]` — the CPU stand-in kernels: every encoded tile == the decode of its packed trellis
  (rxq.deq and rqah's exl3dec), sub-1 / fractional / integer rates, i4 / i8 / mul1 (seconds).
- `tests/check_cpujob.py FORK FORK_0008 [WORK]` — the quant job end to end on the stand-in (tiny two-layer model; K0.125 /
  0.5 i4, K2.5 i8, K6.5 mul1; H_out dense and blocks): decode == the quantizer's own weight, the logged settings, tile CD
  and C9 where they belong, assemble's stand-in / mixed-stamp refusals (~2 min).
- `tests/check_blkorder.py FORK FORK_0008 [WORK]` — #117 route B (quant.block_order) with H_out (dense / blocks) and tile CD:
  identity order == plain bit for bit, the consensus order changes readers only, decode == wq (~15 min on 2 CPUs).
- `tests/check_blkorder84.py FORK [WORK]` — rxq's block order == w84's blkorder84.py (installed as w84q.py did) bit for
  bit under w84's p0 settings, tile CD off and on (~8 min). GPU twin: `tools/blkorder27.sh` (p0perm's 9 readers).
- `tests/check_ladder.py [RQAH]` — the 55 levels and passes, every job of every level through rqah's rxq_quant checks,
  manifest pins, estimates, the post scripts at `--swj-batch` 1 and 4 (one swjfit call per level that has an swj step /
  up to 4 levels of a ladder pass that have an swj step, in build order, share one swj level pass), the swj level pass
  count and hours (seconds).

GPU (prepared scripts, the lead runs them):
- `tools/gpucheck27.sh cd|i4|c9` — gpubox card 0: tile CD on the real quantizer (rxq vs research module, M = I gate,
  two-sided ratios with H_out), the qtf2 i4 kernel check and ourq-vs-rxq i4 bit-identity, C9 in the quant job vs
  c9e27.py on the card (and its per-writer time). Stage the code tree as in
  `tools/CDHOUT27.md` (into `/data/w76chk/code`, plus `quantize/ourq.py` and `v3/e3/tilecd/`).
- `tools/cdhout27.sh` / `tools/insert-cdhout27.sh` — the 27B K4 no-harm pair of tile CD with H_out (rule
  `tools/CDHOUT27.md`), with the swap-pair hygiene check.
- `tests/check_quant.py REF_LEVEL JOB_OUT` — per-tensor jobs vs an ourq level (GPU run first; the ourq reference must
  hold each key's group head, i.e. q_proj / in_proj_qkv / gate_proj in its `--only`, or the key is BADREF); `tools/run4070.sh A|B|C|D`
  — the 11-06 4070 checks (they turn the 11-07 settings off to match their references).

Licence: Apache-2.0 (LICENSE, NOTICE); exllamav3 is MIT (LICENSES/).
