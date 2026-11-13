# emb174 — embedding options in the V3 search menu (w135, #174)

The live search (#171) had one embedding option, `base` = V2 3.00's own token table (Lloyd l3) at every size. This
folder gives it the embedding ladder the plugin serves from the stored form, prices it, anchors the prices with real
swaps through the running coordinator, and hands a menu to the lead / w132 for the next rotation.

| file | what |
|---|---|
| `radinspect.py` | one weight's directory entry (quantizer, options, encoding, bytes, head / tail hash) of `.rad` containers, local or HTTP range reads |
| `emb174.sh` | ds8 donors from V2 containers: `/data/q27/emb/emb-{e4,e5,e6,bf16}.rad` = the K0.125 pack + V2's token_embd (radmix), checked against the source |
| `radfetch.py`, `relay174.sh` | the V2 5.50 / 6.50 / 8.00 sources as sparse files holding only tables + token_embd: HTTP range reads through the VM (ds8's own HF reads ran at 0.4 MiB/s; xet download stalled) |
| `embll.sh` | Lloyd widths V2 never shipped: `emb-e2.rad`, `emb-e7.rad` = K0.125 pack re-converted (`--reuse`) with V2's rule `rtn table=<fitlevels.py codebook> group=128 scale=bf16 transform=fwht128 rule=search` |
| `embmenu.py` | IN menu → menu with options e2, base (l3), e4, e5, e6, e7, bf16 (bytes from the donors, err = V2's fitted floor); `--prices` adds the trellis price table (not servable) |
| `fit-embprior.patch` | fit.py: menu emb `err` → prior P_0 ratio per embedding label (fitted levels: log-ratio residual, sd 0.5; unseen levels priced by it instead of "no gain") |
| `calsubmit.py` | the calibration evals: an incumbent's eval line + an embedding donor, whole slice, one at a time through the coordinator |
| `calobs.py` | calibration results → fit rows (own labels `emb174-<slice>|<part>|full` with the incumbents as anchors) + per-part log ratios |
| `mkfitin.py`, `embgain.py`, `predpick.py` | offline refit input (live obs + menu + cal rows); gain at fixed trunk; the driver's own solver per size and embedding (predicted pick) |
| `a8anchor174.sh`, `a8docs.py`, `a8pull.sh` | after the 10:11 pause: anchor swaps on A8 containers on ds8 (card locks, one at a time), paired with w132's A8 scores of the same incumbents; per-doc sums → calobs.py |
| `loo174.py`, `loo174.sh`, `loosum.py` | leave-one-calibration-eval-out fits on colo: embedding offset at P_0 vs after the compounding × V2 prior off / on |
| `prof174.sh` | stored-form check: `--profile-ops` on emb-e4 vs emb-bf16 (placement lines, static VRAM, embed_tq / embed_lookup µs) |
| `embcb174.py` | embtrellis.py with another trellis codebook (mul1_i8 vs mul1: KL per byte of a trellis embedding code) |
| `embpick.py` | the per-size embedding pick from the measured swaps (upgrade curve × option factor, l2 cost curve, b × extra bpw) |

Encodings (V2 HF containers, radinspect 11-13): all `rtn` Lloyd-Max codebooks under fwht128, a bf16 scale per 128
values, except bf16 (plain). l3 = V2 1.50–3.50, l4 = 4.00–5.00, l5 = 5.50–6.00, l6 = 6.50–7.50, bf16 = 8.00 / 9.00.

Serving (qwen35_exl3, qc-exl3-v3): `embed_tq` for a Lloyd table at B = 1..8 (any width is served from the stored
form, codes resident on the card), `embed_lookup` for bf16 (by default pinned in host memory and read over the link in
place: 0 VRAM; `--embedding-placement`). **No trellis embedding gather exists**: trellis levels (`v3/e3/embgpu.py`) can only be
evaluated as bf16 fake-quant donors and would count bf16 bytes under the VRAM-size rule until a kernel exists; they
are priced in `embprices174.json` only. Runtime of any option: a gather of one 5120-value row per token (≤ 10 KB in
bf16): ~0 for prefill (`gemm_us` unchanged) and decode; VRAM = the option's bytes.

Result (LOG #174, 11-13 11:25 AM CST): the l3 floor's share of the KLD grows from −0.5 % (2.0) to −44 % (6.0); per-size
pick l2 ≤ 2.5, l3 3.0–4.0, l4 / l5 4.5–5.0, l5 / l6 ≥ 5.5; a fitted embedding term does not extrapolate across sizes
(leave-one-out), so pick from the measured swaps.
