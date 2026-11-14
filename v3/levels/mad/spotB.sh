#!/bin/bash
# spotB.sh [CARDS] -- (ds8) RULES-17703 stage B, per level K in 2.5 4 6, restartable (every step skips when its output exists):
#   wait   until ladder-mad's level is DONE at the current scales (K <= 4.5: DONE.scales set; K6: DONE)
#   arms   M = a hard-link copy of the mad level's final/ with stamp.json args.codebook "i8" (mad codes in mul1i8's slot for
#          the importer; the original kept as stamp-mad.json; the real final/ untouched) + a recipe with codebook=mul1i8;
#          I = the i8 twin's final/ + recipe from box2 ladder-src (10 GbE link), recipe src rewritten
#   pack   hostd's pack_cmd (rad10 home-pack, rad-convert --no-error): conv.lock excl. + ram.lock shared after MemAvailable
#          >= 16 GiB; mix = radmix pack + a8.mix (act a8g128 on the 336 trunk linears); the pack is kept (w150 hand-off)
#   check  radents.py: non-trunk entries byte-identical, trunk byte counts equal, sizes equal -> check-K.json (fail: no score)
#   score  rad10-perf radiance on golden-s173e, home-base3 (I) / home-hq (M, EXL3I_BENCH_IQRH 1), bf16 KV, max-model-len
#          16384, max-num-seqs 1; first free card of CARDS (default "6 5 4 3 2 1 0"; never 7 = w153's); card lock + ram.lock shared
#   sums   per-document KL sums of every score (rxrun.hostd.doc_sums) -> docsums.json; both arms' level proxy.jsonl -> proxy/
# Out /data/q27/mad/spotB. Process name: `bash /data/q27/mad/tools/spotB.sh`. (#177-03, w151)
set -u; Q=/data/q27; S=$Q/mad/spotB; T=$Q/mad/tools; L=$Q/locks; RP=$Q/rad10-perf/install/bin/radiance; W=$Q/w139s3
G=$Q/w131/slices/golden-s173e; PY=$Q/venv/bin/python; CARDS=${1:-6 5 4 3 2 1 0}
BOX="ssh -i /root/.ssh/dbox -o StrictHostKeyChecking=no -o Compression=no"
mkdir -p $S/proxy; exec >> $S/spotB.log 2>&1
say() { echo "$(date -u +%T) $*"; }
current() {   # DONE at the current scales stamp (rxrun.run.level_current semantics for this level)
  $PY -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if float(sys.argv[2]) > 4.5 or d.get('scales') else 1)" $1/DONE $2 2>/dev/null
}
pack() {   # pack DIR: DIR/final.recipe -> DIR/pack.rad -> DIR/mix-a8.rad
  local D=$1
  if [ ! -f $D/pack.rad ]; then
    ( flock -x 7; flock -s 6
      while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt 16 ]; do sleep 20; done; say "pack $D"
      cd $Q && RADIANCE_HOME=$Q/rad10/home-pack RADIANCE_HOST_DEVICES=1 nice -n 10 $Q/rad10/install/bin/rad-convert --no-error \
        models/qwen38-27b-bf16 --recipe $D/final.recipe --set quantization_config.quant_method=exl3 -o $D/pack.rad.part > $D/pack.log 2>&1 \
        && mv $D/pack.rad.part $D/pack.rad; say "pack $D rc $?" ) 7>$L/conv.lock 6>$L/ram.lock
  fi
  [ -f $D/pack.rad ] || { say "PACK FAIL $D"; rm -f $D/pack.rad.part; return 1; }
  if [ ! -f $D/mix-a8.rad ]; then
    ( flock -s 6; $Q/tools/radmix $D/mix-a8.rad.part $D/pack.rad $T/a8.mix > $D/mix.log 2>&1 && mv $D/mix-a8.rad.part $D/mix-a8.rad
      say "mix $D: $(tail -n 1 $D/mix.log)" ) 6>$L/ram.lock
  fi
  [ -f $D/mix-a8.rad ]
}
for K in 2.5 4 6; do
  ML=$Q/ladder-mad/q27v3-635bac02-k$K-mad; IN=q27v3-144dd6e1-k$K-i8; M=$S/M-$K; I=$S/I-$K
  until current $ML $K; do [ -f $Q/mad/STOP ] && exit 2; sleep 60; done
  if [ ! -f $M/final.recipe ]; then
    rm -rf $M; mkdir -p $M; cp -al $ML/final $M/final
    cp $M/final/stamp.json $M/stamp-mad.json
    $PY -c "import json,sys; p=sys.argv[1]; d=json.load(open(p)); d[-1]['args']['codebook']='i8'; d[-1]['w151_shim']='mad codes (a 0xd0474b b 0x2545f491) in the mul1i8 slot for EXL3I_BENCH_IQRH 1 scoring (RULES-17703); original: ../stamp-mad.json'; json.dump(d,open(p+'.new','w'),indent=1)" $M/final/stamp.json
    rm $M/final/stamp.json && mv $M/final/stamp.json.new $M/final/stamp.json        # unlink first: the real final/ keeps its stamp
    sed "s|src=$ML/final|src=$M/final|; s|codebook=mad |codebook=mul1i8 |" $ML/final.recipe > $M/final.recipe.part && mv $M/final.recipe.part $M/final.recipe
    cp $ML/level/proxy.jsonl $S/proxy/M-$K.jsonl; say "arm M $K ready"
  fi
  if [ ! -f $I/final.recipe ]; then
    mkdir -p $I/stage; rsync -a -e "$BOX" root@10.29.6.6:/data/q27/ladder-src/$IN/final root@10.29.6.6:/data/q27/ladder-src/$IN/final.recipe $I/stage/ \
      && rsync -a -e "$BOX" root@10.29.6.6:/data/q27/ladder-src/$IN/level/proxy.jsonl $S/proxy/I-$K.jsonl || { say "pull $IN FAIL"; exit 1; }
    rm -rf $I/final; mv $I/stage/final $I/final
    sed -E "s|src=[^ ]*/$IN/final|src=$I/final|" $I/stage/final.recipe > $I/final.recipe.part && mv $I/final.recipe.part $I/final.recipe; say "arm I $K ready"
  fi
  grep -q "codebook=mul1i8" $M/final.recipe && ! grep -q "codebook=mad" $M/final.recipe && grep -q "src=$I/final" $I/final.recipe || { say "RECIPE CHECK FAIL $K"; exit 1; }
  pack $I || exit 1; pack $M || exit 1
  if [ ! -f $S/check-$K.json ]; then
    python3 -I - $T $I/mix-a8.rad $M/mix-a8.rad $S/check-$K.json $T/a8.mix <<'EOF2' && say "check $K: $(cut -c1-200 $S/check-$K.json)" || { say "CHECK FAIL $K"; exit 1; }
import json, os, sys
sys.path.insert(0, sys.argv[1]); import radents as R
a, b, out = sys.argv[2:5]; A, B = R.entries(a), R.entries(b)
T = {l.split()[1] for l in open(sys.argv[5]) if l.startswith("act ")}          # the 336 exl3i trunk linears
names = sorted(set(A) | set(B)); tr = [n for n in names if n in T]; nt = [n for n in names if n not in T]
missing = [n for n in names if n not in A or n not in B]
nt_diff = [n for n in nt if n in A and n in B and (A[n][1] != B[n][1] or R.digest(a, A[n][0], A[n][1]) != R.digest(b, B[n][0], B[n][1]))]
tr_bytes = [n for n in tr if n in A and n in B and A[n][1] != B[n][1]]
r = {"entries": len(names), "trunk": len(tr), "non_trunk": len(nt), "missing": missing, "non_trunk_differ": nt_diff,
     "trunk_bytes_differ": tr_bytes, "size_a": os.path.getsize(a), "size_b": os.path.getsize(b)}
r["pass"] = not (missing or nt_diff or tr_bytes) and r["size_a"] == r["size_b"]
json.dump(r, open(out + ".part", "w")); os.replace(out + ".part", out); sys.exit(0 if r["pass"] else 1)
EOF2
  fi
  grep -q '"pass": true' $S/check-$K.json || { say "check $K not passed: no scores"; exit 1; }
  for a in I M; do
    D=$S/$a-$K; hm=base3; [ $a = M ] && hm=hq; o=$D/score-s173e
    [ -f $o.json ] && continue
    while :; do for c in $CARDS; do [ $c = 7 ] && continue; exec 5>$L/card$c.lock; flock -n 5 && break 2; exec 5>&-; done; sleep 20; done
    ( flock -s 6; say "score $a-$K card $c"
      ROCR_VISIBLE_DEVICES=$c RADIANCE_HOME=$W/home-$hm $RP --model $D/mix-a8.rad --tp 1 --kv-cache-dtype bf16 --max-model-len 16384 \
        --max-num-seqs 1 --host-headroom-mib 8192 --kld-ref $G --kld-out $o.json.part > $o.log 2>&1 \
        && mv $o.json.part.rows $o.json.rows && mv $o.json.part $o.json; say "score $a-$K rc $?" ) 6>$L/ram.lock
    exec 5>&-
    [ -f $o.json ] || { say "SCORE FAIL $a-$K ($o.log)"; exit 1; }
  done
done
env PYTHONPATH=$Q/stage/code/pipeline $PY - $G $S > $S/docsums.json.part <<'EOF3' && mv $S/docsums.json.part $S/docsums.json && say "docsums done"
import glob, json, os, sys
from rxrun.hostd import doc_sums
G, S = sys.argv[1:3]; out = {}
for r in sorted(glob.glob(f"{S}/[IM]-*/score-s173e.json.rows")):
    arm = os.path.basename(os.path.dirname(r)); d, kl = doc_sums(G, r); sc = json.load(open(r[:-5]))
    out[arm] = {"docs": d, "kld": float(kl.mean()), "positions": int(len(kl)), "score": {k: v for k, v in sc.items() if not isinstance(v, (list, dict))}}
json.dump(out, sys.stdout)
EOF3
say "SPOTB DONE"
