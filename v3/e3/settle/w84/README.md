# w84 · #117 at 27B: residual-stream 128-block permutation (rule and results: `../w84.md`)

**Runtime cost: none.** The permutation only changes the order in which LDLQ visits the residual readers' 128-blocks.
Done as a quantizer option (route B), it changes neither the checkpoint nor the format nor the kernels. The converter
(route A) is an exact transform of the stored tensors.

| file | what |
|---|---|
| `perm117.py` | the converter: `plan CKPT` (classify every tensor; refuses an unclassified 5120-dim tensor), `ckpt ORDER CKPT OUT`, `hess ORDER HESS OUT` (packed upper triangles of the residual-input Hessians re-indexed, the rest linked), `hout ORDER HOUT OUT` (writers' H_out re-indexed). `permute_tensor()` is the per-tensor rule everything else calls |
| `order82.json` | w82's consensus order (`data/w82/order82.json`): `stored_order[p]` = natural block at stored position p, heaviest last |
| `blkorder84.py` | route B: `install(Q, order)` wraps exllamav3's `quantize_exl3` / `scaled_ldl` / `ldlq` so a residual reader's LDLQ runs over the blocks in the stored order (natural block `stored_order[39]` first) and the result is stored back in natural order; `"ident"` = identity (no-op gate) |
| `w84q.py` | `rxq quant` with blkorder84 installed (`w84q.py ORDER RXQ-ARGS`) |
| `signchk84.py` | gate G2 (card): route A (converted tensor + permuted sign draws + natural-order `regularize`) vs route B, bit for bit, plus the two traps |
| `gate84.py` | gate G3 (CPU): layer-streamed bf16 forward of the original vs the converted model (f32 / f64), MTP head, vision merger fc2; `gate84.py verdict DIR` |
| `test117.py` | converter CLI round trips (Hessians, one checkpoint shard) |
| `pickx84.py`, `mk27b.sh` | golden-search27b: xcorp part (65 stratified pieces of night1105's xc8k-b1 golden) + calib part (36 search-pool docs, `calib_slice.py` + `golden_pick.py`) |
| `w84.sh`, `run84.sh` | gpubox driver (card 0, locks as cdhout27.sh): `chk` (G2), `p0` / `p1` (noop gate, arm, splice, pack, mix hygiene, scores, pairs), `verdict`, `vcheck` (the lead's verify check of the K2.5 H_out arms) |
| `pair84.py`, `verdict84.py`, `vcheck84.py`, `proxy84.py` | paired statistics (= tools/paired.py on one golden; pooled over parts with a stratified bootstrap), the rule, the verify check, the readers' proxy ratio |

## Run

- gpubox: stage the repo's `pipeline/`, `store/`, `v3/e3/{exl3frac2,exl3frac,exl3cb}` and this folder under
  `/data/w84/code` (same layout), then `bash mk27b.sh` (CPU), then `nohup setsid bash run84.sh &` (restartable; log
  `/data/w84/result.txt`) and `bash w84.sh vcheck`.
- colo: `gate84.py ORDER CKPT META OUT f32|f64 2 384` (META: the calib `tokens.i32` / `kld.json` / `pools-v3.json`),
  then `gate84.py verdict OUT`; `test117.py CKPT HESS TMP`.
- Shipping the order in the ladder: install `blkorder84` in the quantizer (rxq `engine.setup`) for the readers. A
  converted checkpoint would also work, but then the converter run must permute the natural sign draws and map
  exllamav3's g_scale tile sample (G2's two traps), and every consumer of the residual (MTP, DFlash2 taps, vision
  merger, embedding / head) must take the permutation.
