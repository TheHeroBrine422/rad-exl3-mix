import glob, os, sys
sys.path.insert(0, sys.argv[1]); import radents as R
bad = 0
for p in sorted(glob.glob(sys.argv[2] + "/*.rad")):
    E = R.entries(p); end = max(x[0] + x[1] for x in E.values()); sz = os.path.getsize(p)
    if end > sz or len(E) != 1120: bad += 1; print("BAD", os.path.basename(p), len(E), end, sz)
print(f"{len(glob.glob(sys.argv[2] + '/*.rad'))} packs, {bad} bad")
