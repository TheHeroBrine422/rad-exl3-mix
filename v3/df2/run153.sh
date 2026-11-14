#!/bin/bash
# run153.sh STAGE... -- (w153, #180, ds8 HIP 7) DFlash2 + vision in qwen35_exl3 on a V3 pack: K4.125 trunk (act
# a8g128 on the 336 trunk weights, as w138's iqrcmp packs) + the R16 head (R16s003 K4) + bf16 embedding. Restartable:
# every unit skips when its output exists. Log W/logs/run153.log. Stages:
#   pack   (conv.lock exclusive + ram.lock shared, as ds8's hostd packs) P0 text-only (rad10-perf/home-pack, as hostd packs today), and with this
#          branch's home (W/home): P1 = + tower + z-lab DFlash2 (IQR-V4 S's drafter lines: MXFP4 linears, u2 head) +
#          the rescoring head (int8 copy of the original lm_head), P2 = P1 with block-fp8 drafter linears, P3 = P1
#          without the rescoring head; (P4, the drafter linears left bf16, is refused at declare: the stock DFlash2
#          block serves fp8a8 / mxfp4a8 linears only, so P2's block fp8 is the closest to the bf16 drafter); rad-info + whash (trunk P0 vs P1, tower / draft head P1 vs IQR-V4 S)
#   kld    (card + ram shared) golden-search27c-c-cut13: today's home (w147 = qc-exl3-v3 179220f) on P0, W/home on P0,
#          W/home on P1 spec 0, W/home on P1 at its default depth
#   serve  (card + ram shared) servers on the media binary, greedy: P1 off / on, IQR-V4 S off / on, P2 / P3 on:
#          image prompts (genimg), text prompts (gen, top-20 logprobs), bench.py ctx 0 gen 256 + ctx 8k gen 32
#          (warm 1 + 3, as v3/bench/iqrcmp/run.sh), /metrics, VRAM; a second bench round for P1 on and IQR on
#   reb    (card + ram shared) after the rebase onto e054e71 (W/home-r, build153r.sh): kld on P0 / P1 spec 0 / P1, and
#          P1 off / on servers (text gen + bench), as above
#   cmp    dcmp of the pairs (spec on vs off = exactness; ours vs IQR on images)
W=/data/q27/w153; Q=/data/q27; I=$Q/rad10-perf/install; CONV=$I/bin/rad-convert; RAD=$I/bin/radiance; INFO=$I/bin/rad-info
HP=$Q/rad10-perf/home-pack; HT=$Q/w147/home; HD=$W/home; HR=$W/home-r; DF=$W/models/z-lab-Qwen3.8-27B-DFlash2
IQR=$W/ref/qwen3.8-27b-iq4r-s-v4.rad; G=$Q/w124/gold/golden-search27c-c-cut13; L=$Q/locks; CARD=${CARD:-7}
P=$W/packs; R=$W/res; RC=$W/recipes; PR=$W/in/prompts; PB=$W/in/p137/prompts
mkdir -p $P $R $W/logs; ulimit -n 65536
exec >> $W/logs/run153.log 2>&1
say() { echo "$(date -u +%T) $*"; }
say "start $* (pid $$)"
ACTSET=()
for l in $(seq 0 63); do
  if [ $((l % 4)) = 3 ]; then ws="attn_qg attn_k attn_v attn_output"; else ws="ssm_qkv ssm_z ssm_out"; fi
  for w in $ws ffn_gate_up ffn_down; do ACTSET+=(--set "exl3.act.blk.$l.$w.weight=a8g128"); done
done

pack() {   # pack NAME HOME [rad-convert args]
  local n=$1 h=$2; shift 2
  [ -f $P/$n.rad ] && return 0
  [ "$(df --output=avail -BG /data | tail -n 1 | tr -dc 0-9)" -gt 150 ] || { say "pack $n: under 150 GB free"; return 1; }
  rm -f $P/$n.rad.part
  # ds8 packs hold ram.lock shared after a RAM admission (hostd's ram_mode "shared", pack_ram_gib 16)
  ( flock -x 7; flock -s 6
    [ "$(free -g | awk '/^Mem:/ {print $7}')" -gt 40 ] || { say "pack $n: under 40 GB available"; exit 1; }
    say "pack $n: start"
    cd $Q && RADIANCE_HOME=$h RADIANCE_HOST_DEVICES=1 nice -n 10 $CONV models/qwen38-27b-bf16 "$@" --recipe $RC/$n.recipe \
      --set quantization_config.quant_method=exl3 "${ACTSET[@]}" --no-error -o $P/$n.rad.part > $P/$n.log 2>&1 \
      && mv $P/$n.rad.part $P/$n.rad
    say "pack $n: rc $? $(stat -c %s $P/$n.rad 2>/dev/null) B"
  ) 7>$L/conv.lock 6>$L/ram.lock
  [ -f $P/$n.rad ] && $INFO $P/$n.rad > $P/$n.info 2>&1
}

kld() {   # kld TAG HOME PACK [radiance args]
  local t=$1 h=$2 m=$3; shift 3
  [ -f $R/kld-$t.json ] && return 0
  say "kld $t"
  ROCR_VISIBLE_DEVICES=$CARD RADIANCE_HOME=$h timeout 1800 $RAD --model $m --tp 1 --kv-cache-dtype bf16 \
    --max-model-len 16384 --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $R/kld-$t.json.part "$@" \
    > $R/kld-$t.log 2>&1 && mv $R/kld-$t.json.part.rows $R/kld-$t.json.rows && mv $R/kld-$t.json.part $R/kld-$t.json
  say "kld $t rc $? $(python3 -c "import json; j=json.load(open('$R/kld-$t.json'))['all']; print(j['kld']['mean'], j['top1_agreement'])" 2>/dev/null) rows $(sha256sum < $R/kld-$t.json.rows 2>/dev/null | cut -c1-16)"
}

PORT=$((8153 + CARD)); SP=
up() {   # up TAG HOME MODEL [radiance args]
  local t=$1 h=$2 m=$3 i=0; shift 3
  ROCR_VISIBLE_DEVICES=$CARD RADIANCE_HOME=$h $RAD --model $m --tp 1 --kv-cache-dtype bf16 --max-model-len 40960 \
    --max-num-seqs 8 --mm-max-patches 4096 --host 127.0.0.1 --port $PORT "$@" > $R/serve-$t.log 2>&1 &
  SP=$!
  while [ $i -lt 1800 ] && ! curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null; do sleep 3; i=$((i + 3)); kill -0 $SP 2>/dev/null || break; done
  curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null
}
down() { kill -INT $SP 2>/dev/null; for j in $(seq 60); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill $SP 2>/dev/null; wait $SP 2>/dev/null; SP=; }
srv() {   # srv TAG HOME MODEL WHAT [radiance args]; WHAT: i (images) t (text gen) b (bench) 2 (bench round 2 only)
  local t=$1 h=$2 m=$3 wh=$4; shift 4
  [ -f $R/$t.done ] && return 0
  say "serve $t"
  if up $t $h $m "$@"; then
    rocm-smi --showmeminfo vram 2>/dev/null > $R/vram-$t.txt
    case $wh in *i*) python3 -I $W/tools/genimg.py $PORT $PR/img-prompts.jsonl $W/in/img $R/img-$t.jsonl 64 > $R/genimg-$t.txt 2>&1 ;; esac
    case $wh in *t*) python3 -I $W/tools/gen.py $PORT $PR/gen8.jsonl $R/txt-$t.jsonl 1 128 > /dev/null 2>&1 ;; esac
    case $wh in *b*) rm -f $R/bench-$t.jsonl
      python3 -I $W/tools/bench.py --port $PORT --prompts $PB --out $R/bench-$t.jsonl --tag $t --ctx 0 --warm 1 --runs 3 --gen 256 --server-log $R/serve-$t.log > $R/bench-$t.txt 2>&1
      python3 -I $W/tools/bench.py --port $PORT --prompts $PB --out $R/bench-$t.jsonl --tag $t --ctx 8k --warm 1 --runs 3 --gen 32 --server-log $R/serve-$t.log >> $R/bench-$t.txt 2>&1 ;; esac
    curl -s -m 10 http://127.0.0.1:$PORT/metrics > $R/metrics-$t.txt
    touch $R/$t.done
    say "serve $t: $(grep -c '"error"' $R/img-$t.jsonl 2>/dev/null) image errors, $(grep -c '"error"' $R/txt-$t.jsonl 2>/dev/null) text errors; $(tail -n 1 $R/bench-$t.txt 2>/dev/null)"
  else say "serve $t: no health ($(grep -m1 '^E ' $R/serve-$t.log | cut -c1-200))"; fi
  down
}

gpu() {   # gpu FUNCTION...: under the card lock (exclusive) and ram.lock (shared)
  ( flock 5; flock -s 4; say "card $CARD + ram (shared) held"; "$@"; say "card $CARD released" ) 5>$L/card$CARD.lock 4>$L/ram.lock
}

st_pack() {
  pack P0-text $HP
  for n in P1-mx-rr P2-fp8-rr P3-mx; do pack $n $HD --draft-model $DF; done
  [ -f $R/whash-trunk-P0-P1.txt ] || python3 -I $W/tools/whash.py $INFO $P/P0-text.rad $P/P1-mx-rr.rad --prefix blk. --prefix output --prefix token_embd > $R/whash-trunk-P0-P1.txt 2>&1
  [ -f $R/whash-P1-IQR.txt ] || python3 -I $W/tools/whash.py $INFO $P/P1-mx-rr.rad $IQR --prefix v. --prefix dflash. > $R/whash-P1-IQR.txt 2>&1
  for n in P1-mx-rr P2-fp8-rr P3-mx; do
    [ -f $R/whash-$n.txt ] || python3 -I $W/tools/whash.py $INFO $P/$n.rad --prefix v. --prefix dflash. --prefix blk. --prefix output --prefix token_embd > $R/whash-$n.txt 2>&1
  done
  say "whash: $(head -n 2 $R/whash-trunk-P0-P1.txt | tail -n 1) | $(head -n 3 $R/whash-P1-IQR.txt | tail -n 1)"
}
st_kld() {
  kld T-P0 $HT $P/P0-text.rad
  kld D-P0 $HD $P/P0-text.rad
  kld D-P1-s0 $HD $P/P1-mx-rr.rad --num-speculative-tokens 0
  kld D-P1 $HD $P/P1-mx-rr.rad
}
st_serve() {
  srv P1off $HD $P/P1-mx-rr.rad itb --num-speculative-tokens 0
  srv P1on $HD $P/P1-mx-rr.rad itb
  srv IQRoff $HD $IQR ib --num-speculative-tokens 0
  srv IQRon $HD $IQR itb
  srv P2on $HD $P/P2-fp8-rr.rad tb
  srv P3on $HD $P/P3-mx.rad tb
  srv P1on-r2 $HD $P/P1-mx-rr.rad b
  srv IQRon-r2 $HD $IQR b
}
st_reb() {
  kld R-P0 $HR $P/P0-text.rad
  kld R-P1-s0 $HR $P/P1-mx-rr.rad --num-speculative-tokens 0
  kld R-P1 $HR $P/P1-mx-rr.rad
  srv R-P1off $HR $P/P1-mx-rr.rad tb --num-speculative-tokens 0
  srv R-P1on $HR $P/P1-mx-rr.rad itb
}
st_cmp() {
  for x in P1off:P1on R-P1off:R-P1on P1on:R-P1on IQRoff:IQRon P1off:IQRoff P1on:P2on P1on:P3on; do
    a=${x%%:*}; b=${x#*:}
    for k in img txt; do
      [ -f $R/$k-$a.jsonl ] && [ -f $R/$k-$b.jsonl ] && python3 -I $W/tools/dcmp.py $R/$k-$a.jsonl $R/$k-$b.jsonl > $R/dcmp-$k-$a-vs-$b.txt 2>&1 \
        && python3 -I $W/tools/tiemargin.py $R/$k-$a.jsonl $R/$k-$b.jsonl > $R/tie-$k-$a-vs-$k-$b.txt 2>&1
    done
  done
  say "cmp done"
}
for s in "$@"; do
  case $s in
    pack) st_pack ;;
    kld) gpu st_kld ;;
    serve) gpu st_serve ;;
    reb) gpu st_reb ;;
    cmp) st_cmp ;;
  esac
done
say "END $*"
