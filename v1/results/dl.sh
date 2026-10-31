#!/bin/bash
# dl.sh REPO REV DEST -- every file of a Hugging Face repo at REV into DEST (curl, resumable)
R=$1 V=$2 D=$3; mkdir -p $D; cd $D
B=https://huggingface.co/$R/resolve/$V
for f in $(curl -s https://huggingface.co/api/models/$R/tree/$V | python3 -c "import json,sys;[print(e['path']) for e in json.load(sys.stdin) if e['type']=='file']"); do
  curl -sfL --retry 5 -C - -o "$f" "$B/$f" && echo "ok $f $(stat -c %s "$f")" || echo "FAIL $f"
done
echo DLDONE
