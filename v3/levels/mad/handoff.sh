#!/bin/bash
# handoff.sh K... -- (ds8) the stage-B mad levels for #177-02 (w150; lead note 11-13 7:30 PM CST) to box2 /data/q27/mad-packs/
# over the 10 GbE link, per K: <level>/final/ (the real level: stamp.json codebook mad + "mad": {a, b}), <level>.recipe (the
# real recipe, codebook=mad, src rewritten to the box2 copy), <level>-shim.rad (spotB's pack: the same codes in mul1i8's slot,
# importer stamp shimmed to i8; serves only through EXL3I_BENCH_IQRH 1). A README says which is which. Restartable (rsync).
set -u; Q=/data/q27; S=$Q/mad/spotB; D=/data/q27/mad-packs
BOX="ssh -i /root/.ssh/dbox -o StrictHostKeyChecking=no -o Compression=no -c aes128-gcm@openssh.com"
$BOX root@10.29.6.6 "mkdir -p $D"
for K in "$@"; do
  n=q27v3-635bac02-k$K-mad; L=$Q/ladder-mad/$n
  [ -f $S/M-$K/pack.rad ] || { echo "no shim pack for K$K yet"; exit 1; }
  $BOX root@10.29.6.6 "mkdir -p $D/$n"; rsync -a --whole-file -e "$BOX" $L/final/ root@10.29.6.6:$D/$n/final/ || exit 1
  sed "s|src=$L/final|src=$D/$n/final|" $L/final.recipe | $BOX root@10.29.6.6 "cat > $D/$n.recipe"
  rsync -a --whole-file -e "$BOX" $S/M-$K/pack.rad root@10.29.6.6:$D/$n-shim.rad || exit 1
  echo "$(date -u +%T) K$K handed off"
done
$BOX root@10.29.6.6 "cat > $D/README.txt" <<EOF
#177-03 (w151) stage-B mad ladder levels for #177-02 (w150), $(date -u +%FT%TZ). rad-exl3-mix 0ce451c, codes stamp 635bac02,
code stamp c3ee8275, mad a 0xd0474b b 0x2545f491 mask 0x3F3F3F3F L16 bias 126 (v3.toml [quant.codebook.mad]).
Per level q27v3-635bac02-k<K>-mad:
  <level>/final/       the level (64 q-L shards; swj at K <= 4.5; C9). stamp.json: codebook "mad", args.codebook "mad", "mad": {a, b}
  <level>.recipe       its rad-convert recipe: codebook=mad (the exl3i importer name #177-02 adds), src = the final/ here
  <level>-shim.rad     plain pack (bf16 embedding / head, no act keys) of the same codes with the importer stamp shimmed to i8
                       (mul1i8 slot): serves only through the EXL3I_BENCH_IQRH 1 bench build (w139 home-hq); RULES-17703 scored
                       its A8 mix (+ act a8g128 on the 336 trunk linears) on golden-s173e
Re-pack with the mad importer: rad-convert --no-error <bf16 ckpt> --recipe <level>.recipe --set quantization_config.quant_method=exl3 -o X.rad
EOF
echo "README written"
