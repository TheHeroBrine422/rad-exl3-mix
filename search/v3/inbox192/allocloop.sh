#!/bin/bash
# RULES-sizes199 amendment 3 (11-16): every 3 h, consensus refresh (s195s1 whole results) + xavg (own193) for every
# active size -> lead inbox files (decide), skipping assignments already submitted (ledger of sha256s).
# Stop: touch /workspace/rad-local/workers/lead-srch/run/drv/STOP-allocloop
R=/workspace/rad-exl3-mix; W=/workspace/rad-local/workers/lead-srch/run; L=/workspace/rad-local/workers/lead-srch/allocloop
mkdir -p $L; echo $$ > $L/pid
SIZES="1.5,1.75,2,2.5,2.75,3,3.25,3.5,3.75,4,4.5,5,5.5,6,6.5,7"
while [ ! -e $W/drv/STOP-allocloop ]; do
  n=$(date -u +%m%d%H%M); out=$L/r$n; mkdir -p $out/cons $out/xavg
  cd $R
  nice -n 10 timeout 1800 python3 search/v3/inbox192/consensus192.py $W/driver.json $out/cons --whole --slice s195s1 \
      --decide $SIZES --probe "" --kmax 7.5 > $out/cons.log 2>&1
  # (11-16 4:05 AM CST) xavg over each size's neighbours only (|dz| <= 0.75): the all-sizes mean from 1.5 to 7 lost by 3-50 %
  for z in ${SIZES//,/ }; do
    nb=$(python3 -c "S=[float(x) for x in '$SIZES'.split(',')]; z=$z; print(','.join(f'{x:g}' for x in S if abs(x-z)<=0.75))")
    mkdir -p $out/xw; nice -n 10 timeout 600 python3 search/v3/alloc193/own193.py $out/xw --sizes $nb --allocs xavg >> $out/xavg.log 2>&1
    for f in $out/xw/xavg-$z.json $out/xw/xavg-$z-*.json; do [ -e "$f" ] && mv "$f" $out/xavg/; done; rm -rf $out/xw
  done
  python3 - "$out" "$W/drv/inbox" "$L/ledger.txt" "$n" <<'EOF'
import json, sys, os, glob, hashlib
out, inbox, ledger, n = sys.argv[1:5]
seen = set(open(ledger).read().split()) if os.path.exists(ledger) else set()
sent = 0
for f in sorted(glob.glob(f"{out}/cons/*.json")) + sorted(glob.glob(f"{out}/xavg/*.json")):
    b = os.path.basename(f)[:-5]
    if b.endswith("-h"): continue                                 # 'h' variants barely move
    d = json.load(open(f))
    if d.get("mode") != "decide" or "assign" not in d: continue
    h = hashlib.sha256(json.dumps(d["assign"], sort_keys=True).encode()).hexdigest()[:16]
    if h in seen: continue
    kind = "cons195" if "/cons/" in f else "xavg"
    size = d["size"]; var = b.split("-", 2)[2] if kind == "xavg" and b.count("-") >= 2 else ""
    name = f"lead-{kind}-{size:g}{('-' + var) if var else ''}-{n}.json"
    d["note"] = f"allocloop {n} ({kind}{(' ' + var) if var else ''}); RULES-sizes199 am. 3"; d["extra_slices"] = []
    tmp = f"{inbox}/.{name}.tmp"; json.dump(d, open(tmp, "w")); os.replace(tmp, f"{inbox}/{name}")
    seen.add(h); sent += 1
    open(ledger, "a").write(h + "\n")
print(f"{n}: submitted {sent}")
EOF
  sleep 10800
done
