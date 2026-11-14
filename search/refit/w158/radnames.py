#!/usr/bin/env python3
"""radnames.py X.rad -- every weight entry name of a .rad container (directory read only), one a line (w158)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mkmenu as MK  # noqa: E402
for n in MK.rad_entries(sys.argv[1])[1]: print(n)
