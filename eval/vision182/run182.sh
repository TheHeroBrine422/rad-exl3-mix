#!/bin/bash
# run182.sh STAGE... -- (ds8) #182 vision KLD eval (w156). Restartable: every unit skips when its output exists.
# Build: W/rad/{install,home} = radiance qc-exl3-v3-kldmm (5fa0b46 = b83698d + KL-mode media), rad10build.sh with FFmpeg.
# Log W/logs/run182.log. Stages:
#   corpus  mkcorpus.py: set + answers (gen182.sh) -> W/corpus/{corpus,meta}.jsonl
#   reg     (card CARD) text regression of the build: V2 3.00 on golden-search27c-c-cut13, this build vs rad10-perf (rows sha)
#   gold    (cards CARD + CARD2, TP2, as gold3.sh) bf16 checkpoint --kld-record --kld-top 512 of the corpus, bf16 KV, the tower
#           on (--mm-max-patches 4096, as gen182.sh) -> W/gold/vis182
#   pack    (conv.lock excl. + ram.lock shared) mad K2.5 / K4 / K6 (ladder-mad final.recipe, codebook=mad) with this build's home
#           (qwen35_exl3 packs the bf16 tower) + act a8g128 on the 336 trunk linears (= a8.mix) -> W/packs/K*.rad
#   v2      V2 2.50 (HF Hero422/qwen3.8-27b-exl3-v2-rad, W/ref) / 4.00 (mixes/) + the tower: radmix `add v.*` from K2.5's pack
#   score   (card CARD) every candidate in W/cands.txt (TAG PATH) on gold/vis182 (once it exists) and on golden-s173e-x (the
#           xcorp part of s173e, no calib docs): kld json + rows -> W/res/TAG-{vis,s173x}.json
#   swap    (card CARD) swap182.sh's K2.5 + K6-group mixes (W/cands-swap.txt) on gold/vis182 only, 8 seqs
#   qad     (card CARD) qad182.sh's #181 QAD containers + tower (W/cands-qad.txt) on gold/vis182 only, 8 seqs
#   a8      (card CARD) K6 with A8 activations scored against K6 with f16 activations (gold/k6f): a calibration-free trunk
#           perturbation, for the vision / text ratio of a perturbation that owes nothing to the text-only Hessians
W=/data/q27/w156; Q=/data/q27; I=$W/rad/install; H=$W/rad/home; RAD=$I/bin/radiance; CONV=$I/bin/rad-convert; INFO=$I/bin/rad-info
L=$Q/locks; CARD=${CARD:-7}; CARD2=${CARD2:-}; G=$W/gold/vis182; GX=$Q/w131/slices/golden-s173e-x
P=$W/packs; R=$W/res; PY=$Q/venv/bin/python
mkdir -p $P $R $W/gold $W/logs; ulimit -n 65536
exec >> $W/logs/run182.log 2>&1
say() { echo "$(date -u +%T) $*"; }
say "start $* (pid $$)"
ACT=()
for l in $(seq 0 63); do
  if [ $((l % 4)) = 3 ]; then ws="attn_qg attn_k attn_v attn_output"; else ws="ssm_qkv ssm_z ssm_out"; fi
  for w in $ws ffn_gate_up ffn_down; do ACT+=(--set "exl3.act.blk.$l.$w.weight=a8g128"); done
done
MM=(--mm-max-patches 4096)

st_corpus() {
  [ -f $W/corpus/corpus.jsonl ] && return 0
  $PY -I $W/tools/mkcorpus.py $W/set $W/gen $Q/models/qwen38-27b-bf16 $W/corpus.part > $W/logs/mkcorpus.log 2>&1 \
    && mv $W/corpus.part $W/corpus; say "corpus: $(wc -l < $W/corpus/corpus.jsonl) docs"
}

kld() {   # kld OUT HOME BIN GOLDEN MODEL [args]: one --kld-ref run on CARD
  local o=$1 h=$2 b=$3 g=$4 m=$5; shift 5
  [ -f $o.json ] && return 0
  ROCR_VISIBLE_DEVICES=$CARD RADIANCE_HOME=$h timeout 3600 $b --model $m --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 \
    --max-num-seqs ${SEQS:-1} --host-headroom-mib 8192 --kld-ref $g --kld-out $o.json.part "$@" > $o.log 2>&1 \
    && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json
  say "kld $(basename $o) rc $? $($PY -c "import json; j=json.load(open('$o.json'))['all']; print(j['kld']['mean'], j['top1_agreement'])" 2>/dev/null) rows $(sha256sum < $o.json.rows 2>/dev/null | cut -c1-16)"
}

gpu() {   # gpu FUNCTION: card CARD (exclusive) + ram.lock shared
  ( flock 5; flock -s 4; say "card $CARD held"; "$@"; say "card $CARD released" ) 5>$L/card$CARD.lock 4>$L/ram.lock
}

st_reg() {
  local g=$Q/w124/gold/golden-search27c-c-cut13 m=$Q/mixes/qwen3.8-27b-exl3-v2-3.00bpw.rad
  kld $R/reg-perf $Q/rad10-perf/home $Q/rad10-perf/install/bin/radiance $g $m
  kld $R/reg-kldmm $H $RAD $g $m
}

st_gold() {   # the second card: CARD2 (waited for), else the first of 6..0 whose lock is free (polled every 30 s)
  [ -f $G/kld.json ] && return 0
  if [ -n "$CARD2" ]; then exec 3>$L/card$CARD2.lock; flock 3
  else
    while [ -z "$CARD2" ]; do
      for c in 6 5 4 3 2 1 0; do exec 3>$L/card$c.lock; flock -n 3 && { CARD2=$c; break; }; exec 3>&-; done
      [ -n "$CARD2" ] || sleep 30
    done
  fi
  say "gold: cards $CARD + $CARD2"
  ROCR_VISIBLE_DEVICES=$CARD,$CARD2 RADIANCE_HOME=$H timeout 7200 $RAD --model $Q/models/qwen38-27b-bf16 --tp 2 \
    --kv-cache-dtype bf16 --max-model-len 16384 --max-num-seqs 4 --host-headroom-mib 6144 "${MM[@]}" \
    --kld-record $G.part --kld-corpus $W/corpus/corpus.jsonl --kld-top 512 > $W/gold/vis182.log 2>&1 \
    && [ -f $G.part/kld.json ] && mv $G.part $G
  say "gold rc $? $(ls $G 2>/dev/null | tr '\n' ' ')"
  exec 3>&-
}

pack() {   # pack NAME RECIPE
  local n=$1 rc=$2
  [ -f $P/$n.rad ] && return 0
  [ "$(df --output=avail -BG /data | tail -n 1 | tr -dc 0-9)" -gt 170 ] || { say "pack $n: under 170 GB free"; return 1; }
  rm -f $P/$n.rad.part
  ( flock -x 7; flock -s 6
    while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt 40 ]; do sleep 20; done
    say "pack $n: start"
    cd $Q && RADIANCE_HOME=$H RADIANCE_HOST_DEVICES=1 nice -n 10 $CONV --no-error models/qwen38-27b-bf16 --recipe $rc \
      --set quantization_config.quant_method=exl3 "${ACT[@]}" -o $P/$n.rad.part > $P/$n.log 2>&1 && mv $P/$n.rad.part $P/$n.rad
    say "pack $n: rc $? $(stat -c %s $P/$n.rad 2>/dev/null) B" ) 7>$L/conv.lock 6>$L/ram.lock
  [ -f $P/$n.rad ] && $INFO -v $P/$n.rad > $P/$n.info 2>&1
}
st_pack() {
  for K in 2.5 4 6; do pack K$K $Q/ladder-mad/q27v3-635bac02-k$K-mad/final.recipe; done
  say "packs: $(for K in 2.5 4 6; do echo -n "K$K v.=$(grep -cP '^\s+v\.\S+\s' $P/K$K.info 2>/dev/null) "; done)"
}

st_v2() {   # V2 + the tower (radmix add v.* from K2.5's pack)
  [ -f $P/K2.5.rad ] || { say "v2: no K2.5 pack (tower donor)"; return 1; }
  grep -oP '^\s+\Kv\.\S+(?=\s.*@\d+)' $P/K2.5.info | sort -u | awk -v d=$P/K2.5.rad '{print "add", $1, d}' > $P/tower.mix
  for x in 2.50:$W/ref/qwen3.8-27b-exl3-v2-2.50bpw.rad 4.00:$Q/mixes/qwen3.8-27b-exl3-v2-4.00bpw.rad; do
    local n=V2-${x%%:*} b=${x#*:}
    [ -f $P/$n.rad ] && continue
    [ -f $b ] || { say "v2: no $b"; continue; }
    ( flock -s 6; $W/tools/radmix $P/$n.rad.part $b $P/tower.mix > $P/$n.mix.log 2>&1 && mv $P/$n.rad.part $P/$n.rad
      say "v2 $n: $(tail -n 1 $P/$n.mix.log) ($(wc -l < $P/tower.mix) tower weights)" ) 6>$L/ram.lock
    [ -f $P/$n.rad ] && $INFO -v $P/$n.rad > $P/$n.info 2>&1
  done
}

st_score() {   # the text runs need no golden of ours; the vision runs wait for it
  [ -f $G/kld.json ] || say "score: no golden yet: s173e-x only"
  while read -r t m; do
    [ -z "$t" ] || [ "${t:0:1}" = "#" ] && continue
    [ -f $m ] || { say "score: no $m"; continue; }
    [ -f $G/kld.json ] && kld $R/$t-vis $H $RAD $G $m "${MM[@]}"
    kld $R/$t-s173x $H $RAD $GX $m
  done < $W/cands.txt
}

st_list() {   # st_list FILE: its candidates (TAG PATH) on the vision golden only (it holds the text control), 8 documents
  [ -f $G/kld.json ] || { say "list: no golden"; return 1; }   # in flight (KL mode at 1 runs ~170 positions / s on a K pack)
  local SEQS=8
  while read -r t m; do
    [ -f $m ] && kld $R/$t-vis $H $RAD $G $m "${MM[@]}"
  done < $1
}

st_a8() {   # a calibration-free perturbation of the same trunk: K6 with A8 (a8g128 RTN activations) against K6 with f16
  # activations as the reference (gold/k6f, one card): is the image rows' extra KL specific to the text-calibrated weights?
  local f=$P/K6f.rad
  if [ ! -f $f ]; then
    awk '{print "act", $2, "f16"}' $Q/mad/tools/a8.mix > $P/f16act.mix
    ( flock -s 6; $W/tools/radmix $f.part $P/K6.rad $P/f16act.mix > $P/K6f.mix.log 2>&1 && mv $f.part $f ) 6>$L/ram.lock
    say "a8: K6f $(tail -n 1 $P/K6f.mix.log)"
  fi
  if [ ! -f $W/gold/k6f/kld.json ]; then
    ROCR_VISIBLE_DEVICES=$CARD RADIANCE_HOME=$H timeout 3600 $RAD --model $f --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 \
      --max-num-seqs 8 --host-headroom-mib 8192 "${MM[@]}" --kld-record $W/gold/k6f.part --kld-corpus $W/corpus/corpus.jsonl \
      --kld-top 512 > $W/gold/k6f.log 2>&1 && [ -f $W/gold/k6f.part/kld.json ] && mv $W/gold/k6f.part $W/gold/k6f
    say "a8: gold k6f $(ls $W/gold/k6f 2>/dev/null | tr '\n' ' ')"
  fi
  [ -f $W/gold/k6f/kld.json ] && SEQS=8 kld $R/A8-vis $H $RAD $W/gold/k6f $P/K6.rad "${MM[@]}"
}

for s in "$@"; do
  case $s in
    corpus) st_corpus ;;
    reg) gpu st_reg ;;
    gold) gpu st_gold ;;
    pack) st_pack ;;
    v2) st_v2 ;;
    score) gpu st_score ;;
    swap) gpu st_list $W/cands-swap.txt ;;
    qad) gpu st_list $W/cands-qad.txt ;;
    a8) gpu st_a8 ;;
  esac
done
say "END $*"
