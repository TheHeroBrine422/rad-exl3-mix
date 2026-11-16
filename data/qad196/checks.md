# #196 checks (w182, 11-15 / 11-16), raw files next to this note

| check | what was run | result |
|---|---|---|
| switches off = #191 arm A (trainer) | `runs/A/train.jsonl` vs `w170/runs/L0B0T0Q0D0/train.jsonl` (#191) | the first 18 step records identical (rows, lr, train_kl to 17 digits); steps 19–49 drift (GPU non-determinism on a different card): final train_kl 0.0309907 vs 0.0309926; fixed-eval KL 0.054984 vs 0.055164 (0.3 %) |
| switches off = #191 arm A (served) | `served/A` vs #191's `L0B0T0Q0D0` (served) | x181g 0.045296 vs 0.045287 (+0.02 %); fe38 0.054708 vs 0.054798 (−0.16 %): run-to-run noise on the served KL is about 0.2 % |
| un-shrink with c forced to 1 = identity | `runs/ONE` (export-only with `data/one25.safetensors`, c = 1 on all 192 linears) | 400 scale files / 800 tensors, all exactly 1.0; params all 0.0 |
| un-shrink row reconstruction | `unshrink196.py --selftest` on layers.3.self_attn.o_proj | served-row reconstruction vs `QLinear.forward` (random x): rel. err 0.0038 (bf16 rounding) |
| un-shrink range | `data/U25.stats.json`, `data/U40.stats.json` | raw c_r means: 2.5 o_proj 1.010, v 1.005, down 1.014, out_proj 0.987, qkv 1.011; 4.0 all 0.998–1.001; clamp hits only on a few in_proj_qkv rows (0.002–0.004 % of rows) |
| norm base patch (identity) | `patchnorms196.py` on the base with the checkpoint's own norms (209 planes), then `cmp` with the source base | read-back ok; `cmp` rc 0 (byte-identical); container norm values = HF checkpoint norms (max |diff| 0, 5 sampled) |
| norm arms served = torch | `data/qad196/torch/RN-sub-fe38.npz` vs `served/RN/score-fe38.json.rows` (`eval196.py cmp`) | torch 0.056006 vs served 0.055578 (+0.77 %); median |row diff| 0.0011, mean 0.0045 (A: +0.78 %, median 0.0011: the same gap) |
| 4.0 torch student (bf16 lm_head, t3.375 table) | `torch/ctl40-sub-fe38.npz` vs `served/ctl40` | torch 0.007221 vs served 0.007389 (−2.3 %); the served u3.75 head is not in the torch student |
| 2.5 torch student | `torch/A-sub-fe38.npz` vs `served/A` | torch 0.055135 vs served 0.054708 (+0.78 %); #191 reported +0.6 to +1.0 % |
| 4.0 lr | `run196.sh` sets `--lr 1e-4` for any 4.0 level (A4, R4, UR4 = final) | the first UR4 (lr 1e-3, off-rule) is kept as `runs/UR4-lr1e3-offrule` and `served/UR4-lr1e3-offrule`: a supplement, not an arm |
