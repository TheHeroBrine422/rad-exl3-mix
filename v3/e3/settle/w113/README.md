# w113 · #148 P16-TP: the P16 output-row interleave made tensor-parallel safe (TP 1 / 2 / 4)

Rule: `../w113.md` (ed8c0fd, committed alone before any data). Data: `../data/w113/` (`a/` real-quant f2, `b/` KL pair,
`tp/` functional check). Follows #130 (w101) and #146 (w111).

**Runtime:** prefill + decode if adopted (a store-side remap in every kernel writing these tensors; for ssm_qkv the TP
map has a few more integer ops than full P16's at TP < 4); the experiment itself changes nothing.

## 1. How radiance shards these tensors today (`forks/radiance-hero` qc-exl3-v3 42960d5)

| what | where |
|---|---|
| world ∈ {1, 2, 4}: attention refuses world > n_head_kv (4); heads divided per rank (24 q, 4 kv; 16 k / 48 v GDN heads) | `plugin/qwen35_exl3/rad_block_attn_gated_exl3.h:74-79`; `arch/common/rad_arch.h:1027` (n_head), `:1044` (n_head_kv; `geom_from` is called without `attn_world_ok`, `qwen35_exl3.cpp:73`, so no replication); `qwen35_exl3.cpp:141-142` (GDN heads) |
| per-rank dims: q_dim = n_head·head_dim, kv_dim = n_head_kv·head_dim; GDN k_dim = n_head_k·128, v_dim = n_head_v·128, conv_dim = 2 k_dim + v_dim | `arch/common/rad_arch.h:744-745`, `rad_block_gdn_exl3.h:29-31` |
| ssm_qkv: RAD_SHARD_ROW with 3 row parts {k_dim, k_dim, v_dim} (per rank): rank r owns its slice of q, of k and of v | `rad_block_gdn_exl3.h:121-125` |
| ssm_z: RAD_SHARD_ROW, one part (v_dim) | `rad_block_gdn_exl3.h:132-134` |
| attn_qg [n_head, q 256 | gate 256]: RAD_SHARD_ROW, one part; attn_k / attn_v: RAD_SHARD_ROW, one part | `rad_block_attn_gated_exl3.h:94-99` |
| the row cut: no parts → rows [r·N/t, (r+1)·N/t); parts → per part [O_p + r·P_p/t, O_p + (r+1)·P_p/t), concatenated in part order (no gather: rectangles of the stored entry) | `core/format/share.cpp:104-153` (`:138-140` one part, `:141-153` parts) |
| exl3 planes: codes / exl3i record cut with the same parts (record rows = 16 weight rows, parts / 16), svh rows follow the parts, suh whole (row-sharded weight) | `plugin/qwen35_exl3/exl3_dense_op.h:150-171` (codes / record), `:185-186` (suh), `:200-206` (svh) |
| **exl3i (V3) weights refuse TP > 1 today** ("a split of the record is not written yet") | `exl3_dense_op.h:114-119` |
| conv1d / a_log / dt_bias follow the same head split (qkv parts) | `rad_block_gdn_exl3.h:137-144` |

Rows each rank holds (natural rows; 27B):

| tensor | N | TP 2 rank ρ | TP 4 rank r | TP 4 blocks a rank |
|---|---|---|---|---|
| ssm_qkv | 10240 = q 2048 \| k 2048 \| v 6144 | q [1024ρ, +1024), k 2048 + [1024ρ, +1024), v 4096 + [3072ρ, +3072) | q [512r, +512), k 2048 + [512r, +512), v 4096 + [1536r, +1536) | 4 + 4 + 12 = 20 |
| ssm_z | 6144 | [3072ρ, +3072) | [1536r, +1536) | 12 |
| attn_qg | 12288 (24 heads × 512) | [6144ρ, +6144) = 12 heads | [3072r, +3072) = 6 heads | 24 |
| attn_k / attn_v | 1024 (4 kv heads × 256) | [512ρ, +512) | [256r, +256) = 1 head | 2 |

Every boundary is a multiple of 128, so **every natural 128-row Hadamard block lies inside one rank at TP 2 and TP 4**,
and the TP 4 sets nest in the TP 2 sets (TP 2 rank ρ = TP 4 ranks 2ρ, 2ρ+1, part by part).

**Full P16 (gperm(N, 16)) under TP** (`perm113.py`, index arithmetic): a stored block holds one 16-row tile of 8
consecutive natural blocks (1024 rows). At TP 4, 48 / 80 ssm_qkv blocks and 16 / 48 ssm_z blocks hold rows of two
ranks (not servable without cross-rank exchange); attn_k / attn_v blocks span ranks at TP 2 already (8 / 8). Where the
row sets do fit (TP 2 for qkv / z / qg; TP 4 for qg) the blocks sit at other ranks' stored positions (qkv TP 2: 38 / 80,
qg TP 4: 72 / 96), so the loader would need a block gather.

## 2. P16-TP

perm (stored row j holds natural row perm[j]) = `perm113.tpperm(parts, 4)`: inside every TP 4 rank set S_r (its rows
in increasing order, n = N / 4), stored position S_r[i] holds natural row S_r[gperm(n, 16)[i]] (gperm = w101's).

- Widths (natural blocks a stored block): ssm_qkv 8 (the rank's q, k and v heads mix), ssm_z 8, attn_qg 8 (exactly
  full P16's block row sets, at other positions), **attn_v / attn_k 2** (the rank's one KV head: a stored block =
  tiles 0–3 or 4–7 of both its 128-row halves). Report-only V arm for v_proj: nest 2, width 4, serves TP 1 / 2 only.
- **One stored order for TP 1 / 2 / 4.** The loader's cut is unchanged: rank r of t reads the same stored rectangles
  as for a natural tensor, and those hold exactly its natural rows (as a set), so no gather and no cross-rank data. svh
  is stored in the same order and cut the same way; suh is unchanged.
- **Kernel map** (`p16tp_map.c`, table-free; equals the table for every family, TP and rank — checked exhaustively in
  `perm113.py __main__` and `tpcheck113.py`): stored local column j → natural local column 16·p16tp_tile(j >> 4) +
  (j & 15), parameters per tensor and serving TP: parts' tiles per TP 4 rank c_p (qkv 32 / 32 / 96, z 96, qg 192,
  k / v 16), prefix sums C_p, nb = C_n / 8, m = 4 / t. One part reduces to natural tile = s·8nb + 8 (k' mod nb) +
  ⌊k' / nb⌋, s = ⌊k / 8nb⌋, k' = k mod 8nb. The format needs one field per tensor (nest: 0 natural, 4 P16-TP).

## Files

| file | where | what |
|---|---|---|
| `perm113.py` | any | PARTS, rank_rows (the loader's cut), tpperm, arm_perm (P / T / V), local_map (closed form); `__main__` = properties |
| `p16tp_map.c` | any | the kernel map in C (what the kernels implement) |
| `mkperm113.py` | gpubox | arm checkpoint (rows permuted) + permuted bn H_out + perm.json |
| `a113.sh` | gpubox card 0 | #148-01: units N / P / T (8 tensors) and V (4 v_proj) × K2.5 / K4 × seeds 0 / 1 → f2il113.py → report148a.py |
| `f2il113.py`, `report148a.py` | gpubox CPU | f2 per unit (natural basis), model ratios; table + per-family gate |
| `slice113.sh` | gpubox CPU | golden-search27e (xc8k-b2-c1 + fresh calib search docs) |
| `b113.sh`, `mksplice113.py`, `bytes113.py`, `verdict113.py` | gpubox | #148-02: T units → dense splice → pack / radmix into w111's N mixes → gates → 12 scores → verdict |
| `tpcheck113.py` | gpubox CPU | #148-03: per-rank TP 2 / TP 4 compute with the closed-form remap vs TP 1 (bit for bit) on real T-quantized tensors |

Run (gpubox; code staged at `/data/w113/code` = git archive ed8c0fd + this dir):
```
setsid nohup bash /data/w113/code/v3/e3/settle/w113/a113.sh > /data/w113/a/run.out 2>&1 < /dev/null &
bash /data/w113/code/v3/e3/settle/w113/slice113.sh            # after (a)'s gate
echo ssm_qkv,ssm_z,attn_qg > /data/w113/b/scope               # = report148a's gate line
setsid nohup bash /data/w113/code/v3/e3/settle/w113/b113.sh prep 0 1 score verdict > /data/w113/b/run.out 2>&1 < /dev/null &
python3 tpcheck113.py /data/w113/a/out/T-K2.5-s0 /data/w113/a/perm/T/perm.json OUT.json
```

## Results

**#148-01 (a), real rxq f2 (gpubox card 0, 11-12 00:28–01:27 UTC, stamp 87a7e4ab in all 16 units):** R = geo over
K2.5 / K4 × seeds 0 / 1 of f2_X / f2_N; keep = ln R_T / ln R_P (`../data/w113/a/report148a.txt`).

| family (tensors) | R_P (full P16) | R_T (P16-TP) | keep | gate |
|---|---|---|---|---|
| ssm_qkv (L10) | 0.6769 | 0.6879 | 0.959 | PASS |
| ssm_z (L10) | 0.9481 | 0.9409 | 1.144 | PASS |
| attn_qg (L11, L39) | 0.8729 | 0.8734 | 0.995 | PASS |
| attn_v (L3, L11, L39, L63) | 0.8647 | 1.0017 | −0.012 | **FAIL** |

- attn_v per tensor: P 0.905 / 0.976 / 0.971 / 0.652, T 1.005 / 0.998 / 0.999 / 1.004; report-only V (TP 2-only,
  width 4) 0.923 / 0.983 / 0.997 / 0.999 (keep 0.18). The composable model predicts the same (L63: P 0.66, T 1.00,
  V 0.98): v_proj's gain needs rows of different KV heads in one stored block, which no TP-safe order has.
- attn_qg T ≡ P up to the stored block order (identical block row sets): 0.995 as expected. ssm_z's T is a little better
  than P (its stored blocks group a different 8 natural blocks each; the model agrees: T 0.923 vs P 0.927). Seeds agree to ±0.003; w101's ed97adb7 numbers
  (qkv 0.678, z 0.949, q_proj 0.825 / 0.926) match P here.
- (b) scope: ssm_qkv + ssm_z + attn_qg (112 tensors).

**#148-03 TP functional check (CPU, `../data/w113/tp/`): PASS.** C map == table for all five families at TP 1 / 2 / 4
(0 tiles differ); on the 8 real T-quantized tensors (T-K2.5-s0), every rank at TP 2 and TP 4, computing only from its
loader-cut stored rows (each local 128-row block = one whole stored block, output Hadamard inside the rank) and writing
through the closed-form map, equals the TP 1 result un-permuted by the table **bit for bit** (0 outputs differ); vs
x·W_naturalᵀ (float64) rel. error ≤ 1.9e-15.

**Slice golden-search27e** (built 11-12 01:32 UTC by `slice113.sh`, first use): xcorp 130 pieces of `xc8k-b2-c1`
(never scored; 1,408 docs of 27b-x / 27c-x / 27d-x / mX excluded), 187,589 positions; calib 72 search-pool docs never
in the ledger and not in 27d-c (74,195 tokens, 65,334 scored). Ledger: +72 rows `golden-search27e-c` (role select).

**#148-02 (b), 27B KL (gpubox card 0, units 01:32–02:04 UTC, packs / gates to 02:11, 12 scores 02:11–02:59 UTC).**
Gates: G1 PAIR OK (T_S vs N_S-v: 112 arm weights changed, 675 identical, codes plane only), G2 BYTES113 PASS 112 / 112,
G3 stamp, both seeds.

| pair | seed 0 | seed 1 | geo | xcorp | calib | SE comb | z |
|---|---|---|---|---|---|---|---|
| **N → T (P16-TP, 112; decides)** | 0.9933 | 1.0079 | **1.0005** | 0.9991 | 1.0284 | 0.0151 | +0.04 |
| N → P (full P16, 128; reported) | 0.9954 | 1.0122 | 1.0038 | 1.0016 | 1.0476 | 0.0072 | +0.52 |
| P → T (reported) | 0.9978 | 0.9957 | 0.9968 | 0.9976 | 0.9817 | 0.0139 | −0.23 |

- **VERDICT FAIL** (geo > 0.995, calib part > 1): no adoption proposal on this evidence.
- Per document (docstat, N → T): both parts per-doc geo **0.992 / 0.994**, median 0.996 / 0.994, 56% / 55% of docs
  better; the xcorp part is 59% one swe-smith document (r 0.984 both seeds); without it xcorp 1.004 / 1.038. Calib
  pooled 1.012 / 1.045 but per-doc geo 1.002 / 0.995 (median 0.995 / 0.975): a few heavy docs.
- Full P16 on this third slice: no gain either (per-doc geo 0.994 / 1.004). Joint full P16 over 27c (112) + 27d (128)
  + 27e (128): geo **0.9965, z −0.79**; the 128-tensor P16 on 27d + 27e 0.9985 (z −0.26). This slice's T pair with
  w101's and w111's P16 pairs: 0.9932, z −1.31 (arms differ).
- Cleanup: w111's four mixes (66 GB) and all w113 mixes, patches, wq, codes and permuted checkpoints deleted; the
  slice golden-search27e stays on gpubox `/data/w113/b/`.

## Proposal

Rule: (b) FAIL → no adoption. What the data say: P16-TP keeps the whole real-quant gain of P16 on ssm_qkv / ssm_z /
attn_qg (and is TP 1 / 2 / 4 safe with no loader change), and loses v_proj's (no TP-safe form keeps it; a TP 2-only
form keeps 18%). But the KL effect of P16 itself has not held up: 0.9933 → 0.9903 → 1.0038 over three fresh slices (joint
z −0.79 for full P16); per-document geo suggests ≈ −0.5%, below what our slices can resolve. **Recommendation: do not
adopt P16 in either form before the freeze**, as the cost (a per-tensor format field, the remap in every kernel that
writes these tensors, +0.05…0.25% prefill gemm for full P16's map, untimed for P16-TP's) is certain and the gain is
unresolved. If the user wants it anyway, P16-TP on ssm_qkv / ssm_z / attn_qg is the form to ship (below).

**What the kernel rewrite must implement (if adopted):** a per-tensor `nest` field (0 natural, 4 P16-TP) and the parts
(c_p) in the exl3i record header / plan; in every kernel that writes these tensors (A8 / A4 prefill, nt / rows / rate /
T2 decode epilogues) the y column `16·p16tp_tile(j >> 4) + (j & 15)` (`p16tp_map.c`, svh stays at the stored column),
with multiply-shift magic numbers for ⌊·/c_p⌋ and ⌊·/nb⌋ per launch; the rank's m = 4 / t from the plan's world size.
The loader needs nothing new beyond the exl3i record split itself (not yet written: exl3i weights refuse TP > 1 today):
the stored rectangles per rank are the natural ones. Timing as w111 (A8 microbench on the four shapes, e2e prefill,
decode at batch 1 / 8); w111's raster variant (run the stored blocks of one natural group back to back) carries over
within a rank.
