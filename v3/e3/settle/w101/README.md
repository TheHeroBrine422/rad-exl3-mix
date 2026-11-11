# w101 · #130 output-row 16-row tile interleave (P16) · #131 per-section K in fused q_proj / in_proj_qkv

Rule: `../w101.md` (acf54f2, before data; amendment 1 = the (c) recipe 0446e9b, amendment 2 = v_proj f2 80016b2, both
before their data). Data: `../data/w101/` (`a/` #130-01, `b/` #130-02, `131/` #131, `modelx/`, `c/` and `v/` when done).
Kernel / timing note: `kernel-note130.md`.

**Runtime:** #130 = prefill + decode if adopted (store-side un-permute in every kernel writing ssm_qkv / ssm_z /
attn_qg; ~0 expected, to be timed, `kernel-note130.md`); the experiments themselves change nothing. #131 = none
(per-128-row-block K, the slab head's feature; failed anyway).

| file | where | what |
|---|---|---|
| `perm101.py` | — | `gperm(R, 16)` (composable perm2.py verbatim), `inverse`, `balanced` (B16, report only) |
| `mkperm101.py` | gpubox | P16 checkpoint (rows permuted) + permuted bn H_out for a key list; read-back check |
| `il101.sh` | gpubox card 0 | #130-01: 4 tensors × K2.5 / K4 × seeds 0 / 1 × N / P rxq quants (`--save-wq`), f2, report |
| `f2il101.py` | gpubox CPU | w81's two-sided f2 of every unit (P un-permuted; perm-basis f2 asserted equal), model ratio |
| `report130a.py` | — | #130-01 table + verdict |
| `stab101.py`, `report130b.py` | colo | #130-02: h_even / h_odd from the 15 Σdy² blocks, high-rate model in / out of sample |
| `model101x.py` | colo | lead note 11-09: model P16 ratios for k / v_proj, mlp gate / up, down's input side |
| `ilv101.sh` | gpubox card 0 | amendment 2: il101.sh on L3 / L11 / L39 / L63 v_proj (report only) |
| `il101c.sh` | gpubox card 0 | #130-03: 112 tensors × N / P × seeds 0 / 1 at K2.5 → `mksplice101.py` (dense bf16 patch ckpt) → rad-convert `--reuse` + radmix into mix-base / mix-s1 → gates (rxq.pairs, `bytes101.py`, stamp) → 8 scores on golden-search27c → f2 → `verdict130c.py` |
| `sec101.py`, `q101.sh`, `report131.py` | colo | #131: prep + per-(tensor, K) two-sided block proxy rows, queue, verdict |

Run (all restartable, finished units skipped; stop = `touch <dir>/STOP`):
```
# gpubox (code staged at /data/w101/c/code = git archive 0446e9b + this dir)
setsid nohup bash /data/w101/c/code/v3/e3/settle/w101/il101c.sh > /data/w101/c/run.out 2>&1 < /dev/null &
setsid nohup bash /data/w101/c/code/v3/e3/settle/w101/ilv101.sh > /data/w101/v/run.out 2>&1 < /dev/null &
# colo
setsid nohup bash /data/work/w13/w101/e3/settle/w101/q101.sh > /data/work/w13/w101/sec/q.log 2>&1 < /dev/null &
python3 model101x.py /data/work/hout27/hout-bn /data/models/qwen38-27b-bf16 /data/challenge/hessians/calib OUT.jsonl 3,10,31,63
```

## Results

**#130-01 (a), real rxq f2 on card 0 (stamp ed97adb7, 11-09):** R = geo over K2.5 / K4 × 2 seeds of f2_P / f2_N:
L10 in_proj_qkv **0.678** (model 0.658), L10 in_proj_z 0.949 (0.927), L11 q_proj 0.825 (0.802), L39 q_proj 0.926
(0.920). Seeds agree to ±0.002, K2.5 vs K4 within 0.015. The unweighted f1 goes *up* (×1.17 qkv): P16 trades
plain error for error where h is small, as intended. **PASS** (≤ 0.90); scope of (c) = qkv + q_proj (geo 0.874) + z.

**#130-02 (b), out-of-sample stability (colo, 112 tensors, G-b 6e-8):** P16 cost ratio in-sample → held out: qkv
0.779 → 0.784 (**kept 0.977**), z 0.918 → 0.922 (0.959), q_proj 0.851 → 0.857 (0.960). Per-head mean-h Spearman
between halves 0.97 / 0.95 / 0.97 (min 0.77). B16 (balanced, h-fitted) is ~8% lower again on qkv (0.72) and also holds
(kept 0.98). **PASS.**

**Model, more families (lead note; in-sample full bn1920, P16 / N):** in_proj_qkv 0.787, q_proj 0.857, in_proj_z
0.922, v_proj 0.923 (min 0.66), k_proj 0.952 (min 0.72), mlp gate 1.000, up 1.000, down's input side 1.007 (L3 / L10 /
L31 / L63: 1.007 / 1.005 / 1.002 / 1.015). MLP: nothing to gain (its h is flat across 128-row blocks), and the input
side of down gets slightly worse. **v_proj real f2** (amendment 2, report only, (a)'s code): L3 0.906, L11 0.976, L39
0.971, L63 0.657 (geo 0.87), matching the model tensor by tensor (realized share 0.93–1.08).

**#130-03 (c), 27B dense-splice KL pair (112 tensors at K2.5, stamp 87a7e4ab, golden-search27c):** N → P16 pooled
r **0.9930** [0.982, 1.005] (seed 0) / **0.9937** [0.983, 1.003] (seed 1), **geo 0.9933**; xcorp 0.9931 / 0.9941
(geo 0.9936), calib 0.9904 / 0.9856 (geo 0.9880); combined SE 0.0070, z −0.95. **PASS** (geo ≤ 0.995, both parts
< 1). Gates: PAIR OK (only the 112 arm weights differ, codes plane), BYTES101 PASS 112 / 112, stamp on all 4 units.
Per document: 62–69% of documents better, per-doc geo 0.979 / 0.979, leave-one-out pooled ≤ 0.996, without the two
xcorp docs > 20% of the part 0.989 / 0.998. Realized f2 over all 112 tensors: qkv 0.806, z 0.936, q_proj 0.881
(model 0.787 / 0.922 / 0.857; 81–90% of the predicted log gain). Wall: 4 quant units 11–12 min each on the card,
packs ~10 s (reuse) + radmix ≤ 7 min, scores 57–84 s. Mixes kept on gpubox `/data/w101/c/mix/` (4 × 17 GB) until the
lead decides; wq / patch / P16 checkpoints deleted.

**#131, per-section K (colo, 8 tensors × 16 cells, held out over block halves):** mean gain **−0.018 bit** vs ≥ 0.10
→ **FAIL**. attn_qg B2.5 / B4 −0.003 / +0.003, ssm_qkv −0.037 / −0.034. In-sample gains are +0.05–0.13 bit and vanish
or reverse held out (4 blocks per section per half). L19 attn_qg is the only cell family > +0.1 (+0.10 / +0.13). L3
attn_qg (q holds 5% of the energy, high-rate ceiling 0.60 bit) picks splits *against* the energy share and loses
−0.11 held out. The high-rate ceiling is not realized; not investigated further. Section boundaries fall on 128-row
blocks for both tensors (asserted: qkv 2048 / 4096; q_proj blocks 4j, 4j+1 q and 4j+2, 4j+3 gate). No 27B pair.
