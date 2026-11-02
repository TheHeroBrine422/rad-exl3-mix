#!/bin/bash
# setup.sh -- the 4070 box's inputs for the 2B proxy, into /data (public data only, downloaded here):
#   /data/models/qwen35-2b                       Qwen/Qwen3.5-2B (bf16, tied embedding; same tokenizer as Qwen3.8-27B)
#   /data/challenge/calib-meta/{kld.json,tokens.i32,gen.u8}   the published calib golden minus top.bin / rows.f32
#   /data/challenge/corpus/calib.jsonl            the corpus (categories / token counts for calib_slice.py)
#   /data/challenge/tools/score.py                the challenge's scorer (check_score.py's reference)
set -e
python3 - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("Qwen/Qwen3.5-2B", local_dir="/data/models/qwen35-2b")
R = "StillDeadcode/qwen3.8-27b-quant-challenge"
for f in ["golden/calib/kld.json", "golden/calib/tokens.i32", "golden/calib/gen.u8", "corpus/calib.jsonl", "tools/score.py"]:
    hf_hub_download(R, f, repo_type="dataset", local_dir="/data/challenge/dl")
PY
mkdir -p /data/challenge/calib-meta /data/challenge/corpus /data/challenge/tools
ln -sf /data/challenge/dl/golden/calib/{kld.json,tokens.i32,gen.u8} /data/challenge/calib-meta/
ln -sf /data/challenge/dl/corpus/calib.jsonl /data/challenge/corpus/
ln -sf /data/challenge/dl/tools/score.py /data/challenge/tools/
python3 -c "import json; m=json.load(open('/data/challenge/calib-meta/kld.json')); print(len(m['docs']), 'docs', m['rows'], 'rows')"
echo SETUP DONE
