# #178 — GDN in_proj_a / in_proj_b quantized (A2) + container census before scoring (B2) — w142, 2026-11-13

Source: `local/mine/merged-q-20261113-0740.md` items A2 and B2; lead notes 11-13 (IQ8R form first, radiance 2.0's stock
INT8 G128 FWHT128 tensor, trellis arm required). Rule: `RULES-178.md` (committed 9d59b5a before any arm ran).
Results: `data/ab178/` (README there).

## A2 files

| file | what |
|---|---|
| `RULES-178.md` | the pre-registration: arms, containers, slice, statistic, reading |
| `abq.cpp` | one weight through a radiance libquant quantiser outside rad-convert: dlopen's `libquant.so`, builds the `RadQuantWeight` and the recipe's options (kinds from the quantiser's own option table), calls `encoding` + `quantize` exactly as rad-convert does. Used with `gptqsim` (GPTQ on any grid, decoded, stored bf16): the arms' values are libquant's own, not a re-implementation |
| `abgptq.py` | arms G (GPTQ, `calib=` the published layer-input Hessian written as RADGRAM2) and R (RTN: no calib) in the IQ8R recipe's form (`codes=i8 group=128 scale=bf16 transform=fwht128 clamp=sym rule=search`, damp 0.01), a \| b fused per layer (GPTQ rows are independent: one factorization a layer), 8 layers at a time on the CPU; proxy error per weight |
| `abtrellis.py` | arms T4 / T5 / T6: a \| b fused and padded to 128 rows (exllamav3's LDLQ asserts N % 128 = 0), rxq quant (config/v3.toml, codebook i8, `inputs.hout.enable=false` and `quant.refit=true`: no H_out exists for a/b) in the layer-input group (`qwen35.layer_plan` patched in-process), `--save-wq` decode |
| `abpatch.py` | writes an arm's a/b bf16 values over a container's own a/b planes, in place, after checking every target is plain bf16 of the same shape and bytes |
| `run178.sh` | ds8 runner: trellis quant on a queued card, per size the incumbent's .mix + `act a8g128` lines → radmix once, per arm a copy + abpatch + score on golden-s171d (card lock + ram.lock shared), copy deleted |
| `docsums178.py`, `ab178.py` | per-document KL sums (hostd.doc_sums) → the RULES-178 table (pmean over parts, paired stratified bootstrap, trunk price, reading) |
| `tprof.sh`, `tprof-run.sh`, `tprof.py` | timing: arm B served on 127.0.0.1 with `--profile-ops`, one 9k-token prompt + 256 greedy tokens; the a/b `gemm_nt` share of prefill and decode next to the trunk's activation quantizer |

Run (ds8): `abgptq.py CKPT HESS code/abq LIBQUANT gq --jobs 8` (CPU, ~70 min under load), then `run178.sh` (trellis
quant ~4.5 card-min, 10 scores ~6 card-min each), `docsums178.py > docsums178.json`, `ab178.py docsums178.json
REGISTRY ab178.json`; `tprof-run.sh 2.5 CARD`.

## B2 files (`census/`: patches for the lead; nothing shared was edited in place)

| file | what |
|---|---|
| `census.py`, `census-178.patch` | `store/census.py` extended: R16 row order and the exl3i encoding version (1 / 2 mul1i4 / 3 R16) per weight; a form string per weight (`exl3i mul1i8 K4 r16`, `plain - bf16`, `affine fwht128 u3`); the head and embedding forms printed; `--fast` (encoding + plane sizes instead of bytes: 0.2 s vs ~15 s); donor-name check (a level pack named `-k<K>-<cb>` must hold that K and codebook); with `--base`, a weight with no act line must keep the base's act mode (radmix's rule: catches a mix that silently drops its act lines); expectations `--expect-act`, `--expect-codebook`, `--expect-head PATTERN`, `--expect-emb PATTERN`, `--max-version N` |
| `hostd-census.patch` | `rxrun/hostd.py` run_eval: HOSTCONF `eval.census_cmd` ({mix} {mixfile} {base}) runs on every mix before scoring; a non-zero exit fails the unit unscored; a pass is remembered in `<mix>.census` |
| `test_hostd_census.py` | offline check of the hostd patch on test_hostd_es.py's stubs (8/8 PASS; test_hostd_es.py still 11/11 on the patched tree) |
| `finalpack.sh` | the final-pack path: radmix → full census (byte check vs donors and base) with the release expectations (A8 trunk, mul1i8, R16 exl3i head, version ≤ 3); the container appears only on PASS (`.bad` otherwise) |

Suggested host config line (ds8 / box2; `store` is in hostd's TREE_DIRS, so the patch changes the hosts' tree stamp:
restage both together):
`"census_cmd": ["/data/q27/venv/bin/python", "-I", "/data/q27/stage/code/store/census.py", "{mix}", "--mix", "{mixfile}", "--base", "{base}", "--fast", "--expect-codebook", "mul1i8", "--max-version", "3"]`
(+ `"--expect-act", "a8g128"` once every search eval is A8: it refuses f16 evals such as w135's f16 anchors).
