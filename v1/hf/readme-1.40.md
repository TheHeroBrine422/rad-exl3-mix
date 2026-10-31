# Qwen3.8-27B EXL3 1.40 — radiance container

One size point of [turboderp's published EXL3 conversion](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/SC_1.40bpw_H3),
read in **bit for bit** — the trellis, codebooks, scales and zero points as published, nothing
re-quantised — served by the fast `exl3r4d` kernel for the radiance quantization challenge.

**Needs radiance 1.13.0 plus the [`qc-exl3-ladder` plugin](https://codeberg.org/Hero422/radiance/src/branch/qc-exl3-ladder)**
(the `qwen35_exl3` architecture with the K=1 fast path, `exl3_dense`/`exl3r4d` kernels and the
`exl3quant` import-only quantiser, ABI 31). With the plugin in `$RADIANCE_HOME`, the default
load picks `exl3_r4d` ahead of `exl3dense`.

File `qwen3.8-27b-exl3-1.40bpw.rad` — 6.84 GiB, **2.19 real bpw** (container bytes × 8 /
26,895,998,464 params). K=1 on 210 tensors.

## Serve

```
radiance --model Hero422/qwen3.8-27b-exl3-1.40bpw --kv-cache-dtype bf16 \
    --max-model-len 262144 --max-num-seqs 1 --host 0.0.0.0 --port 8000
```

## How this file was made

The recipe (in this repository) is the one that serves all nine points: `bits=` is deliberately
absent, so K is read from each weight's own trellis in the export (last dimension 16*K), and
`codebook=mul1` is checked against each projection's marker. The delta net's a|b projections keep
their bf16 values.

```
EXL3_SRC=turboderp/Qwen3.8-27B-exl3@SC_1.40bpw_H3
rad-convert Qwen/Qwen3.8-27B --recipe q38-27b-exl3-ladder.recipe \
    --set quantization_config.quant_method=exl3 -o qwen3.8-27b-exl3-1.40bpw.rad
```
