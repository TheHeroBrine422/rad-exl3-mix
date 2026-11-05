#!/bin/bash
# phase1.sh -- d1108 amendment 3: the decisive H_out arms first (runs a snapshot of run.sh)
export D1108_STEPS="gen build gold coll hout" D1108_GEN_LIMIT=1152 D1108_COLL=C D1108_NOHIN=1 \
  D1108_HOB="C:0.25 C:0.5 C:1" D1108_BUILD="selfgen-train:C selfgen-eval:E"
exec bash /data/yb/d1108/snap/run.sh
