#!/bin/bash
# devprof.sh TAG -- card 1, dev home: 8k-doc record (one 4096-token step pair... see flags) with --profile-ops;
# per-shape us/call, exl3_dense total, and the 8k KLD (must stay 0.006406194530860617 for a bit-exact change)
T=$1
ssh-gpu.sh "cd /root; CARD=1 RHOME=/root/dev-home ./gold.sh /root/models/q38-27b-exl3-4.00bpw.rad golden8k corpus8k.jsonl /root/dp-$T bf16 16384 --max-num-batched-tokens 8192 --profile-ops >/dev/null; grep -E '^E ' dp-$T.log | head -3; python3 - $T <<'PY'
import collections,sys
L=open('/root/dp-%s.log'%sys.argv[1]).read().split('per-op timing, rank 0')[1].splitlines()
agg=collections.defaultdict(list); tot=0; allop=0
for l in L:
    f=l.split()
    if len(f)>=9 and f[3]=='dev':
        try: ms=float(f[1])
        except: continue
        allop+=ms
        if f[6]=='exl3_dense': agg[f[7]+' '+f[8]].append(float(f[2])); tot+=ms
for k,v in sorted(agg.items()): v.sort(); print('%-22s %10.1f us/call' % (k, v[len(v)//2]))
print('exl3_dense total ms %.1f of all-op %.1f' % (tot, allop))
PY
sed -n 4p dp-$T.json; rm -rf dp-$T"
