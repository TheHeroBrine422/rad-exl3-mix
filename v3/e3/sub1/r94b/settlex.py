#!/usr/bin/env python3
"""settlex.py -- w43's settle.py with cbx.py's exact exllamav3 codebooks added to SETTLE_CB (mul1x, mul1x_i8); same
arguments and output as settle.py (w45, #94 rerun). settle.py itself is unchanged."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settle, trellis2, cbx
trellis2.codebook = cbx.codebook                            # settle.main reads trellis2.codebook(SETTLE_CB)
if __name__ == "__main__":
    settle.main()
