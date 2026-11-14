# emb17403 — trellis embedding back in the budget (w157, #174-03)

Deadcode (11-14) counts the embedding as VRAM, so the user put it back in the budget: the V3 menu offers trellis token
tables (w144's `embed_exl3i`, gathered from the stored record), and size = container bytes (trunk + head + embedding).
Pre-registration: `RULES.md` (committed 5b6e57d before any run). Results: LOG #174-03.

**Serving branch:** radiance-hero `qc-exl3-v3-emb157` head **120132a**: w144's three commits rebased onto `qc-exl3-v3`
plus one drafter commit. Rebased twice: onto 3bc8ad2 (7163603, dcadb35, 9dcd0cb + 67cd2ce; one comment conflict in
exl3quant merged), then (lead note 11-14 6:10 AM CST) onto 8a00735 without conflicts (d032bbb, 7585524, 4e727a4 +
120132a). The drafter commit: the DFlash2 drafter embeds through `embed_exl3i`: block hook `op_embed_ext` in `arch/common/rad_block_dflash2.h`, off by default; qwen35_exl3 declares
it over `df_x` with the trunk's own record / suh / svh, so the mask token's row and the anchor rows are the trellis
rows decoded exactly; host test). Worktree `/workspace/wt/w157-radiance`.

| file | what |
|---|---|
| `RULES.md` | pre-registration: build pass, serving identity on a mad pack, time, DFlash2, eighth donors + determinism, eighth prices spot check, menu rule |
| `mk157.sh` | VM → box2: `git archive` of the branch, all five plugins for ROCm 10.1 and 7.14 (`plugb.sh`, OVL=1 for the tree's `rad_block_dflash2.h`), host tests |
| `plugb.sh`, `gt.sh`, `ctall.sh` | w152's (`v3/bench/step179/`), work dir `/data/q27/w157` |
| `quant157.sh`, `q-eighths.txt` | box2 card 1: the eighth-step token tables with w144's tool / settings from the code tree w144 used (`stage/code-old-c015001`, code stamp checked), K2.5 rerun first (determinism) |
| `slim157.sh`, `embcmp157.py` | donors: w144's recipe (src paths at box2's `ladder-src`), `--reuse` of the K2.5 donor, then radmix `drop` of every weight but `token_embd.weight` (0.2–0.8 GB instead of 4.3 GB); token_embd bytes / sha256 compare |
| `mix157.sh` | the arms: B (mad K2.5 pack, A8, bf16 embedding), T (+ t2.5), R (B with the oracle rows of t2.5), t<K> (eighths), P3t4 (w153's drafter pack + t4); `check` = confounder check |
| `gpu157.sh` | box2 card 1 stages: `gtest` (embed_exl3i GPU test dev + `--host`, ctall, both toolchains), `kld`, `perf` (w144's protocol), `df2` (drafter acceptance / greedy) |
| `docsums157.py`, `spot157.py` | per-document KL sums; identity + paired ratios vs B + floor-model prediction check |
| `menu17403.py` | the menu's embedding block (`token_embd.weight` entry for `fin133.py menu --emb-menu`) |
| `patches/census-expect-emb.patch` | `store/census.py`: `--expect-emb trellis`, `--expect-codebook` no longer applied to the token table, embedding-donor name check |

Data: `data/emb17403/`. Menu block: `data/emb17403/menu-emb17403.json`.
