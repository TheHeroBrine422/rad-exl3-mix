#!/usr/bin/env python3
"""keffcmp.py K [BLOCKS] -- i.i.d. N(0,1) Keff of the generated CPTC table (tables/table-K.npy) vs the global hash
codebooks mad / mul1 on the SAME samples (paper Table 2 protocol: T=256, tail-biting, seed disjoint from generation)."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("CPTC_E3DIR", "."))
import cptc, trellis
K = int(sys.argv[1]); nb = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
x = np.random.default_rng(777).standard_normal((nb, 256)).astype(np.float32)
out = {"K": K, "blocks": nb, "target": cptc.TARGET_KEFF[K]}
for name in ("cptc", "cptc_presnap", "mad", "mul1"):
    if name.startswith("cptc"):
        T = np.load(os.path.join(cptc.TBL, f"table-K{K}{'-presnap' if name == 'cptc_presnap' else ''}.npy"))
        cb = cptc.decode_table(*cptc.COEF[K], T).astype(np.float32)
    else:
        cb = np.ascontiguousarray(trellis.codebook(name), dtype=np.float32)
    q, _ = cptc.encode_rev(x, cb, K, 16)
    out[name] = round(cptc.keff_of(x, q), 4); print(name, out[name], flush=True)
print(json.dumps(out), flush=True)
