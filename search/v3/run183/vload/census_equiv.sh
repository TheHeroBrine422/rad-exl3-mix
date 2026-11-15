#!/bin/bash
# w162 (#185 step 3): census equivalence, on ds8. For each mix: radmix writes NAME.rad (the search's own tool); census.py
# (the production copy, w158) runs on the .rad; census_manifest.py runs on the manifest (base line + the same mixfile).
# The per-weight rows and the mismatch lists must be identical (the file size is not compared: the manifest's is
# rad-info --bytes, checked separately). A negative expectation (--expect-act a8t on a8g128 mixes) must give the same
# mismatch list on both. Scratch: OUT. Usage: bash census_equiv.sh OUT CENSUS_PY CENSUS_MANIFEST_PY
set -u
OUT=${1:-/data/q27/w162/census-equiv}; CEN=${2:-/data/q27/w162/census.py}; CM=${3:-/data/q27/w162/census_manifest.py}
RADMIX=${RADMIX:-/data/q27/w158/tools/radmix}; RADINFO=${RADINFO:-/data/q27/w162/rad10/install/bin/rad-info}
PY=/data/q27/venv/bin/python; BASE=/data/q27/srch183/base/base-T4.rad; M=/data/q27/srch183/mixes
mkdir -p "$OUT"
export RADIANCE_HOME=/data/q27/w162/rad10/home
run() {  # run NAME MIXFILE EXPECT...
  n=$1; mf=$2; shift 2
  "$RADMIX" "$OUT/$n.rad" "$BASE" "$mf" > "$OUT/$n.radmix.log" 2>&1 || { echo "$n: radmix FAILED"; return; }
  { echo "base $BASE"; cat "$mf"; } > "$OUT/$n.mix"
  $PY -I "$CEN" "$OUT/$n.rad" --mix "$mf" --base "$BASE" --fast "$@" --json "$OUT/$n.rad.json" > "$OUT/$n.census.log" 2>&1
  $PY -I "$CM" --census "$CEN" "$OUT/$n.mix" --mix "$mf" --base "$BASE" --fast "$@" --json "$OUT/$n.mix.json" > "$OUT/$n.census-virt.log" 2>&1
  $PY -I - "$OUT/$n.rad.json" "$OUT/$n.mix.json" "$n" <<'EOF'
import json, sys
a, b, n = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
same_w, same_m = a["weights"] == b["weights"], a["mismatches"] == b["mismatches"]
print(f"{n}: weights {len(a['weights'])} / {len(b['weights'])} same={same_w}; mismatches {len(a['mismatches'])} / {len(b['mismatches'])} same={same_m}; "
      f"bytes {a['bytes']} (rad) vs {b['bytes']} (manifest: 0, see rad-info --bytes)")
EOF
  echo "$n: census .rad: $(tail -1 "$OUT/$n.census.log")  |  manifest: $(tail -1 "$OUT/$n.census-virt.log")"
  "$RADINFO" --bytes "$OUT/$n.mix" | sed "s/^/$n: rad-info --bytes of the manifest = /"
  stat -c "$n: radmix output %s bytes" "$OUT/$n.rad"
  rm -f "$OUT/$n.rad"
}
EXP="--expect-act a8g128 --expect-emb trellis --max-version 4"
for spec in "pair158:$M/v3s183-3.5-pair-00158.mix" "thompson86:$M/v3s183-7-thompson-00086.mix" "block13:$M/v3s183-1.25-block-00013.mix"; do
  run "${spec%%:*}" "${spec#*:}" $EXP
done
# drop: the search mix with blk.0.ssm_ab dropped (its act key goes with it); cmp-only mix in identity.sh (pairdrop)
{ cat "$M/v3s183-3.5-pair-00158.mix"; echo "drop blk.0.ssm_ab.weight"; } > "$OUT/pairdrop.mixfile"
run pairdrop "$OUT/pairdrop.mixfile" $EXP
# negative: the same mix against a wrong expectation must fail the same way on both
run neg-pair158 "$M/v3s183-3.5-pair-00158.mix" --expect-act a8t --expect-emb trellis --max-version 4
echo done
