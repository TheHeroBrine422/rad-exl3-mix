# mad17702 — the mad codebook in exl3i and its byte-sum decode GEMV (#177-02, w150, 2026-11-14)

Radiance fork branch `qc-exl3-v3-mad` (off `qc-exl3-v3` 179220f); what changed and every number: the fork's
`plugin/exl3i/PERF.md`, section "Codebook mad and its byte-sum decode GEMV". This directory holds the box2 scripts
and the summed data. All runs on box2 under `/data/q27/w150/` (HIP card index, `/data/q27/locks/card<N>.lock`, ram.lock
shared for model loads), ROCm 10.1 with the VOPD-repeatable clang, plugins built standalone against
`/data/q27/rad10-perf/install` (the rad10-perf core, as w137's #175 builds).

| script | what |
|---|---|
| `plugb150.sh SRC OUT [XFLAGS] [PLUGINS]` | our plugins (exl3quant qwen35_exl3 exl3dense exl3r4d exl3i) from a radiance tree, host tests, OUT/home (w137's plugb.sh TC 10 + extra HIP flags) |
| `gt.sh CARD BUILD TEST [ARGS]`, `ctall.sh CARD BUILD` | one / every device test of a build's exl3i under the card lock (`res/BUILD-ctall.txt`) |
| `dkm.sh CARD ROUNDS "RATES" "ROWS"`, `dktab150.py RESDIR [M]` | decode bench (bench_twopass, every trunk shape + the head, M 1 or `MS`, `--cold`: each launch on its own copy of the weight, `BURST`: N back-to-back launches / N); rows `BUILD:CODEBOOK:MODE[:ENV]` (mode `a8q`: mad's GEMV on the producer's xb rows, as served; `a8`: the act a8g128 row by itself); per-rate family table, 48 GDN : 16 attention layer, trellis GB/s |
| `isa150.sh BUILD OUT.s [UNIT]` | device asm of an exl3i unit with the build's flags (w137's isa.sh); `kloop.py` (k175) counts its loops |
| `radretag.cpp` | a `.rad` copy with its exl3i encodings' codebook renamed (`mul1i8/...` → `mad/...`): w139's stage-3 hash levels were packed in mul1i8's slot; the codes are mad's |
| `kld150.sh CARD TAG=HOME=PACK...` | whole-model KLD rows (`tools/score.sh`, golden-search27c-c, bf16 KV); env passes through (`EXL3I_MAD_SPLIT=1 EXL3I_A8_MIN_ROWS=1000000`: every prefill launch as 8-row GEMV pieces) |
| `e2b.sh CARD ROUND TAG=HOME=PACK[=ENV]...`, `e2bsum.py` | e2e (w138's #176 server settings, bench.py: 8k prefill + ctx-0 decode, 3 runs a round), rows interleaved per round |
| `dprof150.sh`, (`../iqrcmp/dprof.py`) | decode per-op profile (`--profile-ops`, 320 steps; the lead's dprof.sh with the plugin home a parameter) |

`data/`: `dk*/` (bench CSVs: `dkf` the final rate table, `dkN` mad on nt_pass vs the GEMV, `dkT` / `dkT2` T 2–8, `dk1`
the first pass), `e2/` (e2e jsonl + `e2bsum` tables: `e2c` mad b5 vs i8 twins, `e2f` decomposition, `e2g` the final
build, `e2b` with IQR iq2r), `kld/` (KLD reports, `paired.txt`: cmp115.py paired differences; the per-position rows
stay on box2 `res/kld/*.rows`), `prof/` (dprof tables), `tests/` (the final build's device test outputs).

Packs (box2 `/data/q27/w150/packs/`): `q27v3-144dd6e1-k2.625-mad-a8.rad`, `-k4.125-mad-a8.rad` = w139's stage-3
`mix-a8.rad` (`/data/q27/w139/s3/q27v3-144dd6e1-k<K>-hash/`) relabelled by radretag (112 / 119 bytes differ). Twins:
`/data/q27/w139/s3/q27v3-144dd6e1-k<K>-i8/mix-a8.rad`. The KLD checks are equivalence checks on the search pool's
golden-search27c-c (w137 / w139's), no selection.
