# data/ab178/17803 — #178-03 (w149, 2026-11-13): GDN a/b B / T4 / T5 / T6 on the 5.0 / 6.0 / 6.5 incumbents, s173f

Rule `search/v3/ab17803/RULES-17803.md` (264c3f4). Containers: #171 incumbents `v3s171-5-pair-00449`,
`v3s171-6-explore-00372`, `v3s171-6.5-pair-00588`, A8 trunk, bf16 embedding (`emb-bf16.rad`), head as in the incumbent's
mix (hd-u5 / u6 / u7). Slice golden-s173f (35 calib + 130 xcorp documents, 203,251 positions). T arms are simulated:
w142's decoded a/b values written over the bf16 a/b planes.

## Result (`ab17803.out`, the pre-registered table)

| size | arm | bpw saved | d [95% CI] | c / x | price g | N = d − g | RULES-178 reading |
|---|---|---|---|---|---|---|---|
| 5.0 | T4 | 0.00918 | −2.10% [−4.70, +1.46] | −0.57 / −3.64% | 0.72% | −2.82% | null |
| 5.0 | T5 | 0.00801 | +7.77% [−0.29, +12.52] | +12.22 / +3.31% | 0.63% | +7.14% | null (unverified) |
| 5.0 | T6 | 0.00684 | +1.07% [−2.46, +4.23] | +4.02 / −1.88% | 0.53% | +0.54% | null |
| 6.0 | T4 | 0.00918 | +17.38% [+0.18, +22.74] | +26.83 / +7.94% | 0.32% | +17.06% | null (unverified) |
| 6.0 | T5 | 0.00801 | −4.81% [−9.24, +1.65] | −6.68 / −2.93% | 0.28% | −5.09% | null (unverified) |
| 6.0 | T6 | 0.00684 | −0.19% [−7.39, +7.83] | −3.07 / +2.69% | 0.24% | −0.43% | null |
| 6.5 | T4 | 0.00918 | +19.53% [+1.41, +28.02] | +28.92 / +10.14% | 0.15% | +19.39% | **rejected** (unverified) |
| 6.5 | T5 | 0.00801 | +3.00% [−11.34, +10.19] | +20.25 / −14.26% | 0.13% | +2.87% | null |
| 6.5 | T6 | 0.00684 | +6.57% [−12.73, +15.03] | +22.84 / −9.70% | 0.11% | +6.46% | null (unverified) |

KLD of B: 5.0 0.01227, 6.0 0.00690, 6.5 0.00519. bpw saved with the embedding excluded (trunk + head denominator
25,624,600,064): T4 0.00964, T5 0.00841, T6 0.00718.

**Per-size choice by the rule:** 5.0 **T4**; 6.0 **T5** (noise-level: T4's reading is null); 6.5 **B** (bf16 a/b).

## Why the numbers are noisy (`posthoc17803.out`, descriptive, written after the data)

- One calib document, `c:3181`, holds 55 / 71 / 67% of the calib part's KL at 5.0 / 6.0 / 6.5 and moves −2…+46% under
  any a/b change. Without it, every arm's calib change is within ±2.3%.
- The xcorp part does the same with 3–4 long chat / agentic documents (wildchat 404 / 76 / 497, smoltalk2/longalign 243,
  hermes-fc 519), each ±20–75 nats.
- So the document bootstrap understates the uncertainty, and the arm-to-arm pattern isn't monotone in K (5.0: T5 worst;
  6.5: T6 worse than T5).
- **Trimmed view** (the 2 largest-change documents per part removed) and median per-document log ratio:

| size | T4 | T5 | T6 |
|---|---|---|---|
| 5.0 | −2.1% / −0.9% | −0.1% / +0.5% | +1.7% / +0.2% |
| 6.0 | **+4.4% / +4.8%** | +0.9% / −0.1% | −1.0% / −0.1% |
| 6.5 | **+2.5% / +2.3%** | −3.4% / −1.9% | −1.7% / −0.1% |

  Read this way, T4 costs a few % at 6.0 / 6.5, about 10× the price of its bytes there (0.32 / 0.15%), while T5 and T6
  are ~0. That fits the hypothesis behind the test, but it is post-hoc and from one slice.

## Census (`census/`)

- B mixes: `CENSUS PASS [checked: mix base act codebook emb version]` at all three sizes (full byte check).
- Each arm copy, checked with B's mix + base: `CENSUS FAIL (96) [checked: mix base]`, as expected. The 96 mismatches are
  exactly `blk.N.ssm_{a,b}.weight: not named by the mix and differs from base` for all 48 GDN layers (`ARMCENSUS PASS 96`).
- `census/try1/`: the first attempt checked the copies with `--base B.rad` and no `--mix`. `store/census.py` then compares
  nothing (0 mismatches, `[checked: base]`), so the guard refused all 9 arms and none was scored (patch in
  `search/v3/ab17803/census-base-needs-mix.patch`). All T scores were written after the fixed rerun (17:38 CST).

Files: `docsums17803.json` (per-document KL sums), `ab17803.json/.out`, `posthoc17803.json/.out`, `run17803.log`,
`mix/` (mix specs + radmix logs), `scores/` (radiance kld summaries), `census/`.
