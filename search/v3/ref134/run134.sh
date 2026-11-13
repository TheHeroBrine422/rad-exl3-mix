#!/bin/bash
# run134.sh -- ds8 (#171, w134): reference containers for the V3 convergence plot, scored one at a time (one eval
# slot: ref134.py takes the first free card lock like a hostd eval, so the search's evals keep the other cards).
# V2 at every searched size (1.50-7.00; 3.00-5.00 already on disk and on ver3 from ref171) and deadcode's IQR-V4
# (public HF repos), each on golden-ver3-8k (the monitor) and on the search slice SL (default s171b). A background
# loop downloads the next file while the current one scores (<= 2 IQR files on disk; HOLD below 160 GB free);
# IQR files are deleted once scored on both goldens, V2 files stay in /data/q27/mixes. Restartable.
set -u
Q=/data/q27; W=$Q/w134; C=$W/code; mkdir -p $W/refs $W/iqr $W/dl
SL=${SL:-s171b}; GS="ver3-8k=$Q/goldens/golden-ver3-8k,$SL=$Q/w131/slices/golden-$SL"
PY="env PYTHONPATH=$Q/stage/code/pipeline $Q/venv/bin/python"
log() { echo "$(TZ=America/Chicago date '+%m-%d %H:%M:%S CST') $*" | tee -a $W/run.log; }
# tag|repo|file|dest dir|LM bpw (empty: file bpw)
LIST="V2-1.50|Hero422/qwen3.8-27b-exl3-v2-rad|qwen3.8-27b-exl3-v2-1.50bpw.rad|$Q/mixes|
V2-2.00|Hero422/qwen3.8-27b-exl3-v2-rad|qwen3.8-27b-exl3-v2-2.00bpw.rad|$Q/mixes|
V2-2.50|Hero422/qwen3.8-27b-exl3-v2-rad|qwen3.8-27b-exl3-v2-2.50bpw.rad|$Q/mixes|
IQR4-iq2r|StillDeadcode/qwen3.8-27b-iqr-v4|qwen3.8-27b-iq2r-v4.rad|$W/iqr|2.470
IQR4-iq3r-s|StillDeadcode/qwen3.8-27b-iqr-v4|qwen3.8-27b-iq3r-s-v4.rad|$W/iqr|3.157
IQR4-iq4r-s|StillDeadcode/qwen3.8-27b-iqr-v4|qwen3.8-27b-iq4r-s-v4.rad|$W/iqr|4.142
IQR4-iq5r-s|StillDeadcode/qwen3.8-27b-iqr-v4|qwen3.8-27b-iq5r-s-v4.rad|$W/iqr|5.071
V2-6.00|Hero422/qwen3.8-27b-exl3-v2-rad|qwen3.8-27b-exl3-v2-6.00bpw.rad|$Q/mixes|
IQR4-iq6r-s|StillDeadcode/qwen3.8-27b-iqr-v4|qwen3.8-27b-iq6r-s-v4.rad|$W/iqr|6.063
V2-7.00|Hero422/qwen3.8-27b-exl3-v2-rad|qwen3.8-27b-exl3-v2-7.00bpw.rad|$Q/mixes|
V2-1.75|Hero422/qwen3.8-27b-exl3-v2-rad|qwen3.8-27b-exl3-v2-1.75bpw.rad|$Q/mixes|
IQR4-iq2r-l|StillDeadcode/qwen3.8-27b-iqr-v4|qwen3.8-27b-iq2r-l-v4.rad|$W/iqr|2.797
IQR4-iq3r|StillDeadcode/qwen3.8-27b-iqr-v4|qwen3.8-27b-iq3r-v4.rad|$W/iqr|3.472
IQR4-iq3r-l|StillDeadcode/qwen3.8-27b-iqr-v4|qwen3.8-27b-iq3r-l-v4.rad|$W/iqr|3.804
IQR4-iq4r|StillDeadcode/qwen3.8-27b-iqr-v4|qwen3.8-27b-iq4r-v4.rad|$W/iqr|4.473
IQR4-iq4r-l|StillDeadcode/qwen3.8-27b-iqr-v4|qwen3.8-27b-iq4r-l-v4.rad|$W/iqr|4.793
IQR4-iq5r|StillDeadcode/qwen3.8-27b-iqr-v4|qwen3.8-27b-iq5r-v4.rad|$W/iqr|5.433
V2-5.50|Hero422/qwen3.8-27b-exl3-v2-rad|qwen3.8-27b-exl3-v2-5.50bpw.rad|$Q/mixes|
IQR4-iq6r|StillDeadcode/qwen3.8-27b-iqr-v4|qwen3.8-27b-iq6r-v4.rad|$W/iqr|6.439
V2-6.50|Hero422/qwen3.8-27b-exl3-v2-rad|qwen3.8-27b-exl3-v2-6.50bpw.rad|$Q/mixes|
IQR4-iq6r-l|StillDeadcode/qwen3.8-27b-iqr-v4|qwen3.8-27b-iq6r-l-v4.rad|$W/iqr|6.939
IQR4-iq7r|StillDeadcode/qwen3.8-27b-iqr-v4|qwen3.8-27b-iq7r-v4.rad|$W/iqr|7.420"

scored() {  # tag -> 0 if scored on every golden (or given up on)
  local n; n=$(grep -c "\"tag\": \"ref-$1\"" $W/refs/results.jsonl 2>/dev/null); [ "${n:-0}" -ge 2 ] || [ -f $W/refs/ref-$1.giveup ]
}

download() {
  echo "$LIST" | while IFS='|' read -r t repo f d b; do
    scored $t && continue; [ -f $d/$f ] && continue
    while [ $(ls $W/iqr/*.rad 2>/dev/null | wc -l) -ge 2 ] && [ $d = $W/iqr ]; do sleep 30; done
    while [ $(df --output=avail -BG /data | tail -1 | tr -dc 0-9) -lt 160 ]; do log "dl HOLD: < 160 GB free"; sleep 300; done
    log "dl $f"
    nice -n 10 $Q/venv/bin/python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('$repo', '$f', local_dir='$W/dl')" >> $W/dl.log 2>&1 \
      && mv $W/dl/$f $d/$f.part && mv $d/$f.part $d/$f && log "dl $f done $(stat -c %s $d/$f)" \
      || { log "dl $f FAILED"; sleep 60; }
  done
}

download & DL=$!
trap 'kill $DL 2>/dev/null' EXIT
# the V2 references already on disk (ver3 rows from ref171): this slice only
for s in 3.00 3.50 4.00 4.50 5.00; do
  $PY $C/ref134.py $Q/w131/host-ref.json $W/refs "$SL=$Q/w131/slices/golden-$SL" ref-V2-$s=$Q/mixes/qwen3.8-27b-exl3-v2-${s}bpw.rad >> $W/run.log 2>&1
done
echo "$LIST" | while IFS='|' read -r t repo f d b; do
  scored $t && { [ $d = $W/iqr ] && rm -f $d/$f; continue; }
  while [ ! -f $d/$f ]; do sleep 20; kill -0 $DL 2>/dev/null || { log "downloader gone, $f missing"; break; }; done
  [ -f $d/$f ] || continue
  log "score $t"
  $PY $C/ref134.py $Q/w131/host-ref.json $W/refs "$GS" "ref-$t=$d/$f${b:+@$b}" >> $W/run.log 2>&1 || {
    n=$(cat $W/refs/ref-$t.fails 2>/dev/null || echo 0); echo $((n + 1)) > $W/refs/ref-$t.fails
    [ $((n + 1)) -ge 2 ] && { touch $W/refs/ref-$t.giveup; log "$t: gave up after 2 failed scores"; }; }
  scored $t && [ $d = $W/iqr ] && rm -f $d/$f
done
wait $DL
log "RUN134 DONE"
