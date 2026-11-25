#!/usr/bin/env python3
"""list_sources.py [CELL] -- source ids of the registry (one per line), optionally one cell."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sources
for s in sources.S:
    if len(sys.argv) < 2 or s["cell"] == sys.argv[1] or s["id"] == sys.argv[1]: print(s["id"])
