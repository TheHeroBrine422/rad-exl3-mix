#!/usr/bin/env python3
"""vtguard.py COUNTS.npy OUT.npy -- the language-neutral vocab ranking for the tiered embedding: rank each id by
min(calib-frequency rank, token id). Qwen's byte-level BPE ids follow merge order, i.e. frequency in Qwen's own
multilingual training data, so common tokens of every language stay in the top tiers even when calib (~94%
English/code) never shows them. Output: a pseudo-count array (higher = earlier), embtier.py's COUNTS format."""
import sys
import numpy as np
cnt = np.load(sys.argv[1]); V = len(cnt)
crank = np.empty(V, np.int64); crank[np.argsort(-cnt, kind="stable")] = np.arange(V)
r = np.minimum(crank, np.arange(V))
np.save(sys.argv[2], (V - r).astype(np.float64) + np.arange(V)[::-1] / V)   # ties: smaller id first
