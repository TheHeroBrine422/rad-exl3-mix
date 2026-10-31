#!/bin/bash
# dl4.sh -- turboderp/Qwen3.8-27B-exl3 @ SC_4.00bpw_H5 into /workspace/references/td-SC_4.00bpw_H5
D=/workspace/references/td-SC_4.00bpw_H5; mkdir -p $D; cd $D
B=https://huggingface.co/turboderp/Qwen3.8-27B-exl3/resolve/SC_4.00bpw_H5
for f in $(curl -s https://huggingface.co/api/models/turboderp/Qwen3.8-27B-exl3/tree/SC_4.00bpw_H5 | python3 -c "import json,sys;[print(e['path']) for e in json.load(sys.stdin) if e['type']=='file']"); do
  curl -sfL --retry 5 -C - -o "$f" "$B/$f" && echo "ok $f $(stat -c %s "$f")" || echo "FAIL $f"
done
echo DLDONE
