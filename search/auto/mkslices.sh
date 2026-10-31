#!/bin/bash
# fit2, fit3 (fit pool, ~150k tokens, disjoint from calA except calA's agentic/long/long-agentic documents, which
# the fit pool has too few of -- shared so every slice keeps the corpus's category mix) and ver2 (verify pool,
# ~340k tokens, disjoint from ver1 except ver1's long/long-agentic documents, same reason)
: ${RXM_WORK:=/root}; export RXM_WORK   # paths (README: Paths)
cd ${RXM_WORK}
shared() { # GOLDEN CATEGORIES -> "idx:8192" args of that golden's documents in those categories
  python3 -I - $1 "$2" <<'PY'
import json, os, sys
g, cats = sys.argv[1], sys.argv[2].split(",")
docs = [json.loads(l) for l in open(os.environ["RXM_WORK"] + "/challenge/corpus/calib.jsonl")]
print(" ".join(f"{d['orig']}:8192" if int(docs[d['orig']]['tokens']) > 8192 else str(d['orig'])
               for d in json.load(open(f"{g}/kld.json"))["docs"] if docs[d["orig"]]["category"] in cats))
PY
}
mk() { # NAME POOL TOKENS SEED EXCLUDES SHAREFROM CATS
  [ -d golden-$1 ] && return
  D=$(EXCLUDE_GOLDEN=$5 python3 -I hproxy/calib_slice.py challenge/corpus/calib.jsonl $3 8192 $4 $2 2>/tmp/$1.err); cat /tmp/$1.err
  S=$(shared $6 $7)
  ALL=$(echo $D $S | tr " " "\n" | sort -t: -k1,1n -u | tr "\n" " ")
  nice python3 -I golden_pick.py challenge/golden/calib challenge/corpus/calib.jsonl golden-$1 corpus/$1.jsonl $ALL > /tmp/$1.pick 2>&1 || cat /tmp/$1.pick
}
mk fit2 fit 150000 21 ${RXM_WORK}/golden-calA ${RXM_WORK}/golden-calA agentic,long,long-agentic
mk fit3 fit 150000 31 ${RXM_WORK}/golden-calA,${RXM_WORK}/golden-fit2 ${RXM_WORK}/golden-calA agentic,long,long-agentic
mk ver2 verify 340000 41 ${RXM_WORK}/golden-ver1 ${RXM_WORK}/golden-ver1 long,long-agentic
for g in fit2 fit3 ver2; do python3 -c "
import json,numpy as np,collections
docs=[json.loads(l) for l in open('${RXM_WORK}/challenge/corpus/calib.jsonl')]
m=json.load(open('${RXM_WORK}/golden-$g/kld.json')); g=np.fromfile('${RXM_WORK}/golden-$g/gen.u8',np.uint8)
c=collections.Counter(docs[d['orig']]['category'] for d in m['docs'])
print('$g', len(m['docs']), 'docs', m['rows'], 'rows', int(g.sum()), 'scored;', dict(c))"; done
