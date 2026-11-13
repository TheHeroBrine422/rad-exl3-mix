# k175 — kernel profiling pass on box2 (#175, w137, 2026-11-13)

Step 0: Code000's RDNA4 ISA suite (`radisa check`, `run.sh` audit) over our plugins on both of box2's toolchains; the
FUSE hang under ROCm 7.14 found (ISA) and fixed; FUSE on by default. Step 1: the 8k prefill and batch-1 decode of a V3
K4.125 A8 pack against IQR-V4 S on box2, per op and with clocks. Fork commits on `qc-exl3-v3-perf` (= `qc-exl3-v3`):
5273e01 (asm literals), e776f32 (FUSE loop + default on + tests), 3055e75 (PERF.md: the results, read that first).

All scripts ran on box2 under `/data/q27/w137/` (card locks `/data/q27/locks/card<N>.lock` + `ram.lock` shared; HIP card
index; restartable: finished units are skipped).

| script | what |
|---|---|
| `plugb.sh TC SRC OUT` | our five plugins from a radiance tree, standalone, with ROCm 10.1's VOPD clang (TC 10, against `rad10-perf/install`) or the pip ROCm 7.14.1 (TC 714, against `rad/install`); compile_commands.json for radisa; host tests |
| `lint.sh TC PB OUT [FILTER]`, `lintsum.py` | `radisa check` of every exl3i / exl3dense / exl3r4d unit, source mode (each unit with its build's flags) and object mode (the built `.hip.o`); summary by rule and source line |
| `audit.sh`, `auditsum.py` | `run.sh` (radisa collect + audit) over the final artifacts of the staged perf build and of our plugins with the fix |
| `asmlit.hip` | every changed inline-asm form, `"s"` constant vs the constant in the asm text, 2^26 inputs, bit for bit |
| `tp.sh`, `ctgpu.sh` | `test_twopass` (FUSE on / off, a timeout for a hang) and every gpu ctest of a build |
| `kld.sh`, `q2.sh` | whole-model KLD rows (golden-search27c-c, K4.125 A8 pack) per build × FUSE setting; q2 = the final tree end to end |
| `pf.sh`, `pq.py`, `mk8k.py`, `e2esum.py` | e2e: radiance server (w138's #176 settings), w138's `bench.py` client, ctx 8k (prefill) and ctx 0 (decode) rounds; `prof1` / `prof5` = `--profile-ops` servers after 1 / 5 prefill-only 8k requests; `clk` = 6 prefills with rocm-smi sampled |
| `profsum8k.py` | per-op breakdown of one warm 8k prefill (prof5 − prof1, / 4) by category, V3 vs IQR |
| `smisum.py`, `dclk.sh` | clocks / power / temperature during prefill (`clk`) and during 512-token decodes |
| `kb.sh` | `bench_twopass` of today's device code vs the asm-literal fix (decode M 1 / 4, A8 phases M 512 / 2048) |
| `rp.sh`, `kt.py` | an unprofiled per-kernel trace: **did not work** (rocprofv3 breaks radiance's AQL backend; `RADIANCE_AQL_STAMP` doubles prefill time and wrote nothing on SIGINT) |
| `q1.sh` | the card-1 queue (e2e rounds, kb, dclk, rp) |

`data/`: `kld.txt` (every whole-model run: mean, rows vs today's, sha), `test_twopass.txt`, `lintsum-{src,obj}.txt`,
`audit.txt`, `profsum8k.txt` (+ `prof/` server logs), `e2e-card{0,1}.txt` (+ `e2e-*.jsonl`), `clk.txt` (+ smi /
bench raw), `dclk-*.jsonl` / `smi-dclk-*.txt`, `kb/` (bench_twopass CSVs). The 8k prompt is w138's ctx-16k text (public
radiance docs) cut to 8,192 tokens (`mk8k.py`): timing only, no corpus data.
