#!/usr/bin/env python3
"""mkmeta.py GOLDEN OUT DOCARG... -- a tokens-only META (kld.json "docs" + tokens.i32, the layout swjfit27.py /
mkhin27.py read) of GOLDEN's documents: DOCARG = idx or idx:cap (calib_slice.py's output; first cap tokens). No
top.bin / rows (nothing is scored). docs[] keep source, tokens (after the cap), score_from, orig = idx (the ledger's
document id); GOLDEN's header (corpus, corpus_fnv1a64, model, ...) is kept, so eval/pools/ledger_add.py accepts OUT
(use = OUT's directory name). Refuses to overwrite a META of other documents."""
import hashlib, json, os, sys, time
import numpy as np

g, out, args = sys.argv[1], sys.argv[2], sys.argv[3:]
if len(args) == 1 and " " in args[0]: args = args[0].split()
m = json.load(open(f"{g}/kld.json")); toks = np.memmap(f"{g}/tokens.i32", np.int32, "r")
toff = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
docs, T = [], []
for x in args:
    i, _, c = x.partition(":"); i = int(i); d = m["docs"][i]
    n = min(d["tokens"], int(c)) if c else d["tokens"]
    T.append(np.asarray(toks[toff[i]:toff[i] + n])); docs.append({"source": d["source"], "tokens": int(n), "score_from": 0, "orig": i})
hd = {k: v for k, v in m.items() if k not in ("docs", "rows", "top")}
hd.update({"format": "tokens-only", "golden": os.path.abspath(g), "tokens": int(sum(len(t) for t in T)),
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "docs": docs})
kj = json.dumps(hd).encode()
if os.path.exists(f"{out}/kld.json"):
    if json.load(open(f"{out}/kld.json"))["docs"] != docs: sys.exit(f"{out} holds another META")
    print(f"{out}: same documents, nothing to do"); sys.exit(0)
os.makedirs(out, exist_ok=True)
np.concatenate(T).astype(np.int32).tofile(f"{out}/tokens.i32.part"); os.replace(f"{out}/tokens.i32.part", f"{out}/tokens.i32")
open(f"{out}/kld.json.part", "wb").write(kj); os.replace(f"{out}/kld.json.part", f"{out}/kld.json")
print(f"META {len(docs)} docs, {sum(len(t) for t in T)} tokens -> {out} (kld.json sha256 {hashlib.sha256(kj).hexdigest()[:16]})")
