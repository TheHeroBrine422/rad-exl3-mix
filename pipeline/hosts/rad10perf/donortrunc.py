"""donortrunc.py DONOR.rad QDIR PERM OUT.rad [trunc|stoch] -- a second bf16 donor of the same R16 head (#164-02, w136): OUT
= a copy of DONOR with output.weight = QDIR/wq/lm_head.safetensors (f16, R16 rows) un-permuted and narrowed to bf16 by
truncation (trunc, biased) or stochastic rounding (stoch, unbiased, seed 0); headslab.py's donors round to nearest. The
KLD gap between two donors is the size of the bf16 rounding alone, the yardstick for the quantized head vs the
round-to-nearest donor. Prints JSON (values that differ from DONOR)."""
import json, os, shutil, sys
import numpy as np
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radents import entries  # noqa: E402

D, QD, PERM, O = sys.argv[1:5]; MODE = sys.argv[5] if len(sys.argv) > 5 else "trunc"; rng = np.random.default_rng(0)
if not os.path.exists(O):
    shutil.copyfile(D, O + ".part")
    off, nb, shape = entries(O + ".part")["output.weight"]
    perm = np.load(PERM)
    with safe_open(f"{QD}/wq/lm_head.safetensors", "np") as f:
        W = f.get_tensor("weight")
    out = np.memmap(O + ".part", np.uint16, "r+", offset=off, shape=tuple(shape))
    ref = np.memmap(D, np.uint16, "r", offset=off, shape=tuple(shape))
    diff = 0
    for c in range(0, W.shape[0], 16384):
        u = W[c:c + 16384].astype(np.float32).view(np.uint32)
        if MODE == "stoch":   # add a uniform 16-bit dither below the kept bits, then truncate: P(up) = the dropped fraction
            u = u + rng.integers(0, 1 << 16, u.shape, dtype=np.uint32)
        t = (u >> 16).astype(np.uint16)
        out[perm[c:c + 16384]] = t
        diff += int((t != ref[perm[c:c + 16384]]).sum())
    out.flush(); del out
    os.rename(O + ".part", O)
    print(json.dumps({"out": O, "mode": MODE, "values_differing_from_donor": diff, "of": int(W.size)}))
