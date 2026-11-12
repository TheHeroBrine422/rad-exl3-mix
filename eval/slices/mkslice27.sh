#!/bin/bash
# mkslice27.sh NAME:XBATCH:SEED ... -- fresh two-part 27B search slices (w120, 11-12; docs/design/eval-data.md "two-part,
# calib-sparing"; sizes and evidence in eval/slices/README.md). CPU only, on the host that holds the calib golden and
# night1105's xcorp goldens (gpubox; ram.lock shared). Per NAME (e.g. 27g:xc8k-b2-c0:11201):
#   calib part  $OUT/golden-search$NAME-c: calib_slice CTOK tokens, cap CCAP, search pool, excluding every ledger document
#               (excl.py MODE: never | no27b) and the calib parts built before it in $OUT -> golden_pick
#   xcorp part  $OUT/golden-search$NAME-x: pickx.py XN pieces of XBATCH, heavy categories cut to HEAVY scored positions,
#               excluding the xcorp documents of every earlier x part (XPREV + $OUT/golden-search*-x) and hin27/mX
#   $OUT/ledger/golden-search$NAME-c-kld.json  (copy back; on the VM: ledger_add.py, role in ledger-roles.json first)
#   $OUT/registry.jsonl  one line per slice: name, parts, docs / rows, settings, built (UTC); "decisions": 0 until used
# Restartable: a finished part (kld.json) is skipped; .part dirs are rebuilt. Exits non-zero on the first failure.
set -u
: ${RXM_WORK:=/root}
: ${CODE:?staged rad-exl3-mix tree (eval/pools/ledger.jsonl at the commit the slices are built from)}
: ${OUT:=/data/slices}
: ${PY:=python3}
: ${CORPUS:=/data/challenge/corpus/calib.jsonl} ${GOLD:=/data/challenge/golden/calib} ${N:=/data/night1105}
: ${CTOK:=45000} ${CCAP:=4096} ${MODE:=never} ${XN:=130} ${HEAVY:=agentic,tools,long,reasoning:2048}
: ${MX:=/data/hin27/mX/kld.json}
: ${XPREV:=/data/w91/golden-search27b-x /data/w92/golden-search27c-x /data/w110/golden-search27d-x /data/w113/b/golden-search27e-x /data/w114/golden-search27f-x}
POOLS=$CODE/eval/pools/pools-v3.json LED=$CODE/eval/pools/ledger.jsonl
mkdir -p $OUT/excl $OUT/ledger
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $OUT/build.log; }
die() { log "MKSLICE STOP: $*"; exit 1; }
shrun() { ( exec 7>$RXM_WORK/ram.lock; flock -s 7; nice -n 10 "$@" ); }
[ $# -gt 0 ] || die "usage: mkslice27.sh NAME:XBATCH:SEED ..."
for g in $XPREV; do [ -f $g/kld.json ] || die "XPREV $g/kld.json missing"; done
for spec in "$@"; do
  IFS=: read NAME XB SEED <<< "$spec"; [ -n "$SEED" ] || die "bad spec $spec"
  GC=$OUT/golden-search$NAME-c GX=$OUT/golden-search$NAME-x
  if [ ! -f $GC/kld.json ]; then
    prev=$(ls -d $OUT/golden-search*-c 2>/dev/null | grep -v -- "-c\.part$" | grep -vx "$GC" | while read d; do [ -f $d/kld.json ] && echo $d; done)
    $PY -I $CODE/eval/slices/excl.py $LED $POOLS $OUT/excl/$NAME-c $MODE $prev > $OUT/excl/$NAME-c.log || die "excl $NAME"
    docs=$(EXCLUDE_GOLDEN=$OUT/excl/$NAME-c POOLS=$POOLS $PY -I $CODE/mix/calib_slice.py $CORPUS $CTOK $CCAP $SEED search 2> $GC.slice.log) \
      || die "calib_slice $NAME"
    [ -n "$docs" ] || die "calib_slice $NAME: no documents left ($(cat $OUT/excl/$NAME-c.log))"
    echo "$docs" > $GC.docs; rm -rf $GC.part
    shrun $PY -I $CODE/eval/golden_pick.py $GOLD $CORPUS $GC.part $GC.jsonl $docs > $GC.pick.log 2>&1 && mv $GC.part $GC || die "golden_pick $NAME"
    cp $GC/kld.json $OUT/ledger/golden-search$NAME-c-kld.json
    log "$NAME calib part: $(cat $OUT/excl/$NAME-c.log) | $(cat $GC.slice.log)"
  fi
  if [ ! -f $GX/kld.json ]; then
    ex=(); for g in $XPREV $(ls -d $OUT/golden-search*-x 2>/dev/null); do [ -f $g/kld.json ] && [ "$g" != "$GX" ] && ex+=(--exclude-orig $g/kld.json); done
    [ -f $MX ] && ex+=(--exclude-orig $MX)
    rm -rf $GX.part
    shrun $PY -I $CODE/eval/slices/pickx.py $N/gold/$XB $N/corpora/chunks/$XB.jsonl $N/gold/$XB/corpus.meta.jsonl $XN $SEED \
        $GX.part $GX.jsonl "${ex[@]}" --cap-cat $HEAVY 2> $GX.pick.log && mv $GX.part $GX && mv $GX.part.docs $GX.docs || die "pickx $NAME"
    log "$NAME xcorp part: $(head -1 $GX.pick.log) | $(tail -1 $GX.pick.log)"
  fi
  if ! grep -q "\"name\": \"search$NAME\"" $OUT/registry.jsonl 2>/dev/null; then
    $PY - "$NAME" $GC $GX "$XB" "$SEED" "$CTOK" "$CCAP" "$MODE" "$XN" "$HEAVY" >> $OUT/registry.jsonl <<'PY' || die "registry $NAME"
import json, sys, time
import numpy as np
name, gc, gx, xb, seed, ctok, ccap, mode, xn, heavy = sys.argv[1:]
def part(g):
    m = json.load(open(f"{g}/kld.json")); gen = np.fromfile(f"{g}/gen.u8", np.uint8)
    return {"dir": g, "docs": len(m["docs"]), "tokens": sum(d["tokens"] for d in m["docs"]), "rows": m["rows"], "scored": int(gen.sum())}
print(json.dumps({"name": f"search{name}", "c": part(gc), "x": part(gx), "xbatch": xb, "seed": int(seed),
                  "calib": {"tokens": int(ctok), "cap": int(ccap), "fresh": mode}, "xcorp": {"pieces": int(xn), "heavy_cap": heavy},
                  "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "decisions": 0, "status": "unused"}))
PY
    log "$NAME registered: $(tail -1 $OUT/registry.jsonl | cut -c1-300)"
  fi
done
log "MKSLICE DONE $*"
