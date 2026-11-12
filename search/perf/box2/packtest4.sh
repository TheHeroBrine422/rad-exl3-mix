#!/bin/bash
# packtest4.sh -- box2 (w123, lead note 11-12 3:30 PM): the k3 level packed by stock rad-convert, then by the patched one
# with --no-error, under the PROPOSED rule (conv.lock exclusive + ram.lock SHARED: never stalls level work), niced 19.
# A sampler logs VmRSS / RssAnon / RssFile / MemAvailable every 10 s (logs/pack4-<tag>.mem) and kills the pack if host
# MemAvailable < 25 GiB or RssAnon > 30 GiB. Then cmp (ram.lock shared), lines -> logs/packs.jsonl, both copies deleted.
cd /data/q27/w123; L=q27v3-144dd6e1-k3-i8; mkdir -p packs
pk() {  # pk TAG BIN [args]
  local t=$1 b=$2; shift 2; local O=/data/q27/w123/packs/$L-$t.rad; rm -f $O $O.part logs/pack4-$t.mem
  local t0=$(date +%s.%N)
  flock /data/q27/locks/conv.lock flock -s /data/q27/locks/ram.lock bash -c "
    cd /data/q27; RADIANCE_HOME=/data/q27/rad/home RADIANCE_HOST_DEVICES=1 nice -n 19 $b models/qwen38-27b-bf16 \
      --recipe /data/q27/ladder/$L/final.recipe --set quantization_config.quant_method=exl3 $* -o $O > /data/q27/w123/logs/pack4-$t.log 2>&1 &
    P=\$!; echo \$P > /data/q27/w123/logs/pack4-$t.pid; peak=0; kill=''
    while kill -0 \$P 2>/dev/null; do
      a=\$(awk '/^RssAnon/{print \$2}' /proc/\$P/status 2>/dev/null); f=\$(awk '/^RssFile/{print \$2}' /proc/\$P/status 2>/dev/null)
      r=\$(awk '/^VmRSS/{print \$2}' /proc/\$P/status 2>/dev/null); m=\$(awk '/^MemAvailable/{print \$2}' /proc/meminfo)
      echo \"\$(date +%s) \$r \$a \$f \$m\" >> /data/q27/w123/logs/pack4-$t.mem
      if [ -n \"\$a\" ] && { [ \$m -lt 26214400 ] || [ \$a -gt 31457280 ]; }; then kill \$P; kill=1; echo KILLED >> /data/q27/w123/logs/pack4-$t.mem; fi
      sleep 10
    done; wait \$P; echo \$? > /data/q27/w123/logs/pack4-$t.rc"
  local t1=$(date +%s.%N)
  python3 - "$t" "$t0" "$t1" "$O" <<'P' >> logs/packs.jsonl
import json, os, sys
t, t0, t1, O = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
M = [l.split() for l in open(f"logs/pack4-{t}.mem") if l[0].isdigit()]
M = [[int(x) for x in m] for m in M if len(m) == 5]
rc = open(f"logs/pack4-{t}.rc").read().strip()
print(json.dumps({"pack": f"k3-{t}", "rule": "conv excl + ram shared", "rc": rc, "wall": round(t1 - t0, 1),
                  "bytes": os.path.getsize(O) if os.path.exists(O) else 0, "peak_rss_gib": round(max(m[1] for m in M) / 2**20, 2),
                  "peak_anon_gib": round(max(m[2] for m in M) / 2**20, 2), "peak_file_gib": round(max(m[3] for m in M) / 2**20, 2),
                  "min_memavail_gib": round(min(m[4] for m in M) / 2**20, 1), "samples": len(M), "killed": any("KILLED" in l for l in open(f"logs/pack4-{t}.mem"))}))
P
}
pk stock /data/q27/rad/install/bin/rad-convert
pk noerr /data/q27/w123/radbuild/bin/rad-convert --no-error
S=packs/$L-stock.rad N=packs/$L-noerr.rad
if [ -s $S ] && [ -s $N ]; then r=$(flock -s /data/q27/locks/ram.lock cmp -s $S $N && echo IDENTICAL || echo DIFFER); else r=MISSING; fi
echo "{\"cmp\": \"k3 stock vs noerr\", \"result\": \"$r\"}" >> logs/packs.jsonl
rm -f $S $N; echo PACKTEST4 DONE >> logs/packs.jsonl
