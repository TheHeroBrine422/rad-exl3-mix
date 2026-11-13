#!/bin/bash
# emb174.sh -- the search's embedding donor containers on ds8 (w135, #174): one per distinct embedding encoding of the
# 16 EXL3-V2 containers (radinspect.py on HF: Lloyd-Max l3 / l4 / l5 / l6 g128 fwht128 + bf16 plain). l3 is the V2 3.00
# base's own (menu option "base", no donor). Each donor = radmix of the smallest level pack (K0.125, as hd129.sh) with
# token_embd.weight taken from a V2 container that carries the encoding (radmix takes only the named weight from a
# donor, so the rest of the base does not matter). V2 4.00 is already on ds8; for 5.50 / 6.50 / 8.00 radfetch.py
# range-reads from HF (Hero422/qwen3.8-27b-exl3-v2-rad) only what radmix needs (tables + token_embd, a sparse file;
# a whole-file hf_hub_download stalled on ds8, 11-13), deleted once the donor is built and checked (token_embd bytes
# + head / tail hash = the source's: radinspect.py). Out: /data/q27/emb/emb-<enc>.rad. Restartable (.part, skip finished).
set -eu
Q=/data/q27; D=$Q/emb; PY=$Q/venv/bin/python; RM=$Q/tools/radmix; HERE=$(cd "$(dirname "$0")" && pwd)
U=https://huggingface.co/Hero422/qwen3.8-27b-exl3-v2-rad/resolve/main
LV=q27v3-144dd6e1-k0.125-i8; B=$(ls $Q/packs/$LV-s*.rad | head -n 1)
mkdir -p $D/dl
log() { echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S CST') $*" | tee -a $D/emb174.log; }
for spec in e4:4.00 e5:5.50 e6:6.50 bf16:8.00; do
  e=${spec%%:*}; s=${spec#*:}; O=$D/emb-$e.rad; V=qwen3.8-27b-exl3-v2-${s}bpw.rad
  [ -f $O ] && continue
  src=$Q/mixes/$V
  if [ ! -f $src ]; then
    src=$D/dl/$V
    [ -f $src ] || [ "${NOFETCH:-0}" = 0 ] || { log "$V: no source yet (NOFETCH: relay174.sh brings it)"; continue; }
    [ -f $src ] || { log "fetch $V (token_embd + tables, sparse)"; nice -n 10 $PY -I $HERE/radfetch.py $U/$V $src 2>> $D/dl.log || exit 1; }
  fi
  echo "token_embd.weight $src" > $D/emb-$e.mix
  rm -f $O.part
  flock $Q/locks/conv.lock flock -s $Q/locks/ram.lock nice -n 10 $RM $O.part $B $D/emb-$e.mix > $D/emb-$e.mixlog 2>&1
  a=$($PY -I $HERE/radinspect.py $O.part | $PY -I -c "import json,sys; r=json.loads(sys.stdin.read()); print(r['bytes'], r['hash'], r['scheme'])")
  b=$($PY -I $HERE/radinspect.py $src | $PY -I -c "import json,sys; r=json.loads(sys.stdin.read()); print(r['bytes'], r['hash'], r['scheme'])")
  [ "$a" = "$b" ] || { log "FAIL $e: donor '$a' vs source '$b'"; exit 1; }
  mv $O.part $O; log "$O: token_embd $a (= $V)"
  case $src in $D/dl/*) rm -f $src ;; esac
done
rmdir $D/dl 2>/dev/null || true
ls $D/emb-e4.rad $D/emb-e5.rad $D/emb-e6.rad $D/emb-bf16.rad > /dev/null 2>&1 && log "EMB174 DONE" || log "emb174: some donors still missing"
