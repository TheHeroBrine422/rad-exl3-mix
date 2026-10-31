#!/usr/bin/env python3
"""Create the nine challenge repos and upload files.

PHASE=meta: create repos, upload README.md / recipe / LICENSE  (fast)
PHASE=rad:  upload the .rad containers, one repo at a time, in size order
"""
import os
import sys
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN") or open(os.environ.get("HF_TOKEN_FILE", "/workspace/.hf_token")).read().strip()
api = HfApi(token=TOKEN)
USER = "Hero422"
MODELS = os.environ.get("RXM_HF_MODELS", "/workspace/models")
STAGE = os.environ.get("RXM_HF_STAGE", os.path.dirname(os.path.abspath(__file__)))   # readme-*.md, recipe

# size, source branch, GiB, real bpw, K note
SIZES = [
    ("1.40", "SC_1.40bpw_H3", "6.84 GiB", "2.19", "K=1 on 210 tensors"),
    ("1.60", "SC_1.60bpw_H3", "7.40 GiB", "2.37", "K=1 on 157 tensors"),
    ("1.80", "SC_1.80bpw_H3", "7.97 GiB", "2.55", "K=1 on 90 tensors"),
    ("2.00", "SC_2.00bpw_H3", "8.54 GiB", "2.73", "K=1 on 14 tensors"),
    ("2.20", "SC_2.20bpw_H3", "9.10 GiB", "2.91", "K=1 on 3 tensors"),
    ("3.00", "SC_3.00bpw_H4", "11.52 GiB", "3.68", "K 2–6 per tensor"),
    ("4.00", "SC_4.00bpw_H5", "14.65 GiB", "4.68", "K per weight from the published trellis"),
    ("5.00", "SC_5.00bpw_H6", "17.48 GiB", "5.58", "K 4–6 per tensor"),
    ("6.00", "SC_6.00bpw_H6", "20.31 GiB", "6.49", "K 4–8 (466 K=6 / 134 K=7 / 46 K=8 ops)"),
]


def readme(size, branch, gib, bpw, knote) -> str:
    return f"""# Qwen3.8-27B EXL3 {size} — radiance container

One size point of [turboderp's published EXL3 conversion](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/{branch}),
read in **bit for bit** — the trellis, codebooks, scales and zero points as published, nothing
re-quantised — served by the fast `exl3r4d` kernel for the radiance quantization challenge.

**Needs radiance 1.13.0 plus the [`qc-exl3-ladder` plugin](https://codeberg.org/Hero422/radiance/src/branch/qc-exl3-ladder)**
(the `qwen35_exl3` architecture with the K=1 fast path, `exl3_dense`/`exl3r4d` kernels and the
`exl3quant` import-only quantiser, ABI 31). With the plugin in `$RADIANCE_HOME`, the default
load picks `exl3_r4d` ahead of `exl3dense`.

File `qwen3.8-27b-exl3-{size}bpw.rad` — {gib}, **{bpw} real bpw** (container bytes × 8 /
26,895,998,464 params). {knote}.

## Serve

```
radiance --model Hero422/qwen3.8-27b-exl3-{size}bpw --kv-cache-dtype bf16 \\
    --max-model-len 262144 --max-num-seqs 1 --host 0.0.0.0 --port 8000
```

## How this file was made

The recipe (in this repository) is the one that serves all nine points: `bits=` is deliberately
absent, so K is read from each weight's own trellis in the export (last dimension 16*K), and
`codebook=mul1` is checked against each projection's marker. The delta net's a|b projections keep
their bf16 values.

```
EXL3_SRC=turboderp/Qwen3.8-27B-exl3@{branch}
rad-convert Qwen/Qwen3.8-27B --recipe q38-27b-exl3-ladder.recipe \\
    --set quantization_config.quant_method=exl3 -o qwen3.8-27b-exl3-{size}bpw.rad
```
"""


def main():
    phase = os.environ.get("PHASE", "meta")
    only = os.environ.get("ONLY")
    for size, branch, gib, bpw, knote in SIZES:
        if only and size != only:
            continue
        rid = f"{USER}/qwen3.8-27b-exl3-{size}bpw"
        if phase == "meta":
            try:
                api.create_repo(rid, exist_ok=True, private=False)
                print(f"[{size}] repo exists: {rid}", flush=True)
            except Exception as e:
                print(f"[{size}] create_repo failed: {e}", flush=True)
                continue
            text = readme(size, branch, gib, bpw, knote)
            open(f"{STAGE}/readme-{size}.md", "w").write(text)
            api.upload_file(
                repo_id=rid,
                path_or_fileobj=f"{STAGE}/readme-{size}.md",
                path_in_repo="README.md",
                repo_type="model",
            )
            api.upload_file(
                repo_id=rid,
                path_or_fileobj=f"{STAGE}/q38-27b-exl3-ladder.recipe",
                path_in_repo="q38-27b-exl3-ladder.recipe",
                repo_type="model",
            )
            api.upload_file(
                repo_id=rid,
                path_or_fileobj=f"{STAGE}/LICENSE-model",
                path_in_repo="LICENSE",
                repo_type="model",
            )
            print(f"[{size}] meta uploaded", flush=True)
        elif phase == "rad":
            src = f"{MODELS}/q38-27b-exl3-{size}bpw.rad"
            api.upload_file(
                repo_id=rid,
                path_or_fileobj=src,
                path_in_repo=f"qwen3.8-27b-exl3-{size}bpw.rad",
                repo_type="model",
            )
            print(f"[{size}] rad uploaded", flush=True)


if __name__ == "__main__":
    main()
