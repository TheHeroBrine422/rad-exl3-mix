#!/usr/bin/env python3
"""mkgenh.py stub"""
import glob, os, shutil, sys
CAL, NG, OUT = sys.argv[1:4]
for w0 in [float(x) for x in sys.argv[4].split(",")]:
    d = f"{OUT}/w{w0:g}"; os.makedirs(d, exist_ok=True)
    for f in glob.glob(f"{NG}/*.input.safetensors"): shutil.copy(f, f"{d}/{os.path.basename(f)}")
    open(f"{d}/stats.jsonl", "w").close()
print("MKGENH DONE stub")
