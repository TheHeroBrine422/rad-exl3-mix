#!/bin/bash
# ladder.sh BPW... : per point, wait for the container, 2k KLD fast + scalar, score, append /root/ladder.jsonl
cd /root
for b in "$@"; do
  m=/root/models/q38-27b-exl3-${b}bpw.rad
  while [ ! -f $m ]; do sleep 20; done
  for arm in ${ARMS:-fast scalar}; do
    if [ $arm = fast ]; then K=exl3r4d,exl3dense,libavx,libr4d,spechub,libref; else K=exl3dense,libavx,libr4d,spechub,libref; fi
    w=$(./rec.sh $m /root/rec2k-$b-$arm /root/corpus2k.jsonl /root/rec2k-$b-$arm.log --kernels $K)
    err=$(grep -cE "^E |fatal" /root/rec2k-$b-$arm.log)
    python3 -I scorerec.py golden2k /root/rec2k-$b-$arm > /root/rep2k-$b-$arm.json 2>/root/rep2k-$b-$arm.err
    python3 -c "
import json,sys
r=json.load(open(\"/root/rep2k-$b-$arm.json\"))
print(json.dumps({\"bpw\":\"$b\",\"arm\":\"$arm\",\"wall\":\"$w\",\"errs\":$err,\"kld\":r[\"kld\"][\"mean\"],\"top1\":r[\"top1_agreement\"],\"ppl_ratio\":r[\"ppl\"][\"ratio\"]}))" >> /root/ladder.jsonl || echo "{\"bpw\":\"$b\",\"arm\":\"$arm\",\"wall\":\"$w\",\"errs\":$err,\"fail\":1}" >> /root/ladder.jsonl
    rm -rf /root/rec2k-$b-$arm
  done
done
echo LADDERDONE >> /root/ladder.jsonl
