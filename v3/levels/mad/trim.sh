#!/bin/bash
# trim.sh [ROOT] -- (ds8) free the per-level copies the mad ladder no longer needs (as the 144dd6e1 ladder's ladder-src
# keeps): for every level whose DONE is current (K <= 4.5: DONE.scales set, i.e. its swj pass is in; K > 4.5: DONE),
# delete out/*/t-*.safetensors (the job codes; level/ and final/ hold them) and level/q-L*.safetensors (final/ holds the
# bits; level/'s stamp.json, proxy.jsonl, c9/, index and config stay), then write TRIMMED. Never touches final/, dup/,
# specs/, logs, diag / c9 / log files in out/. Idempotent. (#177-03, w151)
R=${1:-/data/q27/ladder-mad}; PY=/data/q27/venv/bin/python
for D in $R/q27v3-*-mad; do
  [ -f $D/DONE ] && [ ! -f $D/TRIMMED ] && [ -f $D/final.recipe ] && [ -d $D/final ] || continue
  K=${D##*-k}; K=${K%-mad}
  $PY -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if float(sys.argv[2]) > 4.5 or d.get('scales') else 1)" $D/DONE $K || continue
  [ $(ls $D/final/q-L*.safetensors 2>/dev/null | wc -l) -eq 64 ] || { echo "$(date -u +%T) $D: final/ has not 64 shards: not trimmed"; continue; }
  b=$(du -sm $D | cut -f1)
  rm -f $D/out/*/t-*.safetensors $D/level/q-L*.safetensors
  echo "{\"t\": \"$(date -u +%FT%TZ)\", \"mb_before\": $b, \"mb_after\": $(du -sm $D | cut -f1)}" > $D/TRIMMED
  echo "$(date -u +%T) trimmed $(basename $D): $(cat $D/TRIMMED)"
done
