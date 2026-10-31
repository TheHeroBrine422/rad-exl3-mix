#!/bin/bash
# hfup.sh TAG REPO BRANCH EMBBITS -- overwrite Hero422/REPO with /root/models/final/q38-27b-TAG.rad (same file name the
# repo already uses), its recipe (src as $EXL3_SRC) and a README; one commit. Log: /root/hfup.log
T=$1 R=$2 BR=$3 EB=$4
export HF_TOKEN=$(cat /root/.hf_token)
M=/root/models/final/q38-27b-$T.rad
[ -f $M ] || { echo "$T: no container" >> /root/hfup.log; exit 1; }
D=/root/hfstage/$R; rm -rf $D; mkdir -p $D
ln $M $D/$R.rad 2>/dev/null || cp $M $D/$R.rad
sed "s#src=/workspace/references/td-[^ ]*#src=\$EXL3_SRC#" /root/models/final/$T.recipe > $D/q38-27b-exl3-ladder.recipe
cp /root/LICENSE.qwen $D/LICENSE
GIB=$(python3 -c "import os;print(round(os.path.getsize('$M')/2**30,2))")
BPW=$(python3 -c "import os;print(round(os.path.getsize('$M')*8/26895998464,3))")
SIZE=${R#qwen3.8-27b-exl3-}; SIZE=${SIZE%bpw}
cat > $D/README.md <<MD
---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags: [radiance, exl3]
---
# Qwen3.8-27B EXL3 $SIZE — radiance container

One size point of [turboderp's published EXL3 conversion](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/tree/$BR),
read in **bit for bit** (the trellis, codebooks and scales as published, nothing re-quantised), with the token
embedding quantised to a **${EB}-bit Lloyd codebook** under a 128-point Hadamard rotation. Served by the fast
\`exl3r4d\` kernel for the radiance quantization challenge.

**Needs radiance 1.13.0 plus the [\`qc-exl3-ladder\` plugins](https://codeberg.org/Hero422/radiance/src/branch/qc-exl3-ladder)**
(commit 14b7fe0 or later: the \`qwen35_exl3\` architecture, the \`exl3_dense\` / \`exl3r4d\` kernels and the
\`embed_tq\` gather for the quantised embedding; ABI 31).

File \`$R.rad\` — $GIB GiB, **$BPW real bpw** (container bytes × 8 / 26,895,998,464 params, embedding and head included).

## Serve

\`\`\`
radiance --model Hero422/$R --kv-cache-dtype bf16 \\
    --max-model-len 262144 --max-num-seqs 1 --host 0.0.0.0 --port 8000
\`\`\`

## How this file was made

The trunk and head are imported from the export by the \`exl3\` import-only quantiser (K read per tensor from
the export's trellis). The token table is quantised by radiance's \`rtn\` with a ${EB}-bit codebook (Lloyd-Max,
fitted to the table's rotated, group-normalised values), a bf16 scale per 128 values and \`transform=fwht128\`.
The recipe is in this repository.

\`\`\`
EXL3_SRC=turboderp/Qwen3.8-27B-exl3@$BR
rad-convert Qwen/Qwen3.8-27B --recipe q38-27b-exl3-ladder.recipe \\
    --set quantization_config.quant_method=exl3 -o $R.rad
\`\`\`

## Quality

Scored with the challenge's golden files (bf16 reference, bf16 KV); full public-corpus numbers follow.
MD
s=$(date +%s)
nice hf upload Hero422/$R $D . --commit-message "$T: quantised token embedding (${EB}-bit Lloyd, fwht128), $BPW real bpw" >> /root/hfup-$T.log 2>&1
echo "$T -> $R rc=$? $(( $(date +%s) - s ))s $BPW bpw" >> /root/hfup.log
rm -rf $D
