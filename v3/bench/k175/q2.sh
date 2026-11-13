#!/bin/bash
# q2.sh -- #175 (w137) the final tree (srcd): plugins on both toolchains, every gpu ctest (test_twopass at the default =
# FUSE on, _fuse, _nofuse), whole-model KLD rows at the default on both. Card 1.
cd /data/q27/w137
J=8 bash plugb.sh 714 /data/q27/w137/srcd /data/q27/w137/b714d && J=8 bash plugb.sh 10 /data/q27/w137/srcd /data/q27/w137/b10d || exit 1
bash ctgpu.sh b714d 1; bash ctgpu.sh b10d 1
bash kld.sh 1 d714 d10
for t in d714 d10; do cmp -s kld/base10.json.rows kld/$t.json.rows && echo "$t rows == base10" || echo "$t rows DIFFER"; done
echo "q2 done $(date -u +%T)"
