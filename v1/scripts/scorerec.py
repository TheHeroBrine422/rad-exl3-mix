#!/usr/bin/env python3
"""scorerec.py GOLDEN RECDIR -- score a radiance --kld-record dir (logp.f16 [rows, n_vocab]) against a
radiance-kld-top-1 golden with the challenge score.py arithmetic."""
import json, os, sys
import numpy as np
sys.path.insert(0, "/root")
from score import Reference, Accumulator
ref = Reference(sys.argv[1])
rec = sys.argv[2]
meta = json.load(open(os.path.join(rec, "kld.json")))
lp = np.memmap(os.path.join(rec, "logp.f16"), np.float16, "r", shape=(meta["rows"], meta["n_vocab"]))
acc = Accumulator()
r0 = 0
for d in range(ref.n_docs):
    doc = ref.docs[d]
    n = doc["tokens"] - 1 - doc["score_from"]
    acc.add(ref.score_doc(d, np.asarray(lp[r0:r0 + n], np.float32)), doc["source"])
    r0 += n
print(json.dumps(acc.report(), indent=1))
