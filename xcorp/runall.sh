#!/bin/bash
# runall.sh OUTDIR PAR [build.py args...] -- every manifest source (not the agent traces) through build.py, PAR at a
# time; logs OUTDIR/logs/<slug>.log. Sources already done (OUTDIR/<slug>.jsonl) are skipped.
O=$1 P=$2; shift 2
cd /data/xcorp; mkdir -p $O/logs
python3 -I -c "import json; print('\n'.join(json.load(open('ref/corpus/manifest.json'))['datasets']['sources']))" > $O/sources.txt
one() { s=$1; shift; slug=${s//\//__}; [ -f $O/$slug.jsonl ] && return
  OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false python3 -I build.py "$s" $O "$@" > $O/logs/$slug.log 2>&1; }
export -f one; export O
xargs -a $O/sources.txt -P $P -I{} bash -c 'one "$@"' _ {} "$@"
echo "RUNALL DONE $(ls $O/*.jsonl | wc -l) of $(wc -l < $O/sources.txt)"
