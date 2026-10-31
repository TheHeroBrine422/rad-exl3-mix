#!/bin/bash
# pmc.sh PLUGIN.so [M N K BITS STACKED] -- counters of the gemm_nt kernel (card 1) via rocprofv3
SO=$1; shift; A="${*:-2048 34816 5120 4}"
ssh-gpu.sh "cd /root && export ROCR_VISIBLE_DEVICES=1; rm -rf /tmp/pmc; for set in 'SQ_WAVES SQ_BUSY_CYCLES SQ_WAVE_CYCLES SQ_WAIT_INST_ANY' 'SQ_INSTS_VALU SQ_INST_CYCLES_VALU SQ_INSTS_LDS SQ_INSTS_SALU' 'SQC_LDS_BANK_CONFLICT SQC_LDS_IDX_ACTIVE SQ_INSTS_TEX_LOAD GRBM_GUI_ACTIVE'; do rocprofv3 --pmc \$set -d /tmp/pmc -o run --output-format csv -- ./ntbench $SO $A 3 ${STK:-1} > /dev/null 2>&1; done; f=\$(find /tmp/pmc -name '*counter_collection.csv' | head -5); python3 - \$f <<'PY'
import csv,sys,collections
tot=collections.defaultdict(float); n=collections.Counter()
for f in sys.argv[1:]:
    for r in csv.DictReader(open(f)):
        if 'gemm_nt_kernel' not in r.get('Kernel_Name',''): continue
        tot[r['Counter_Name']]+=float(r['Counter_Value']); n[r['Counter_Name']]+=1
for k in sorted(tot): print('%-24s %16.0f  (per dispatch %.3g)'%(k,tot[k],tot[k]/max(1,n[k]/ (1))))
PY"
