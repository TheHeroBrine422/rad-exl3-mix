import json, sys
sys.path.insert(0, sys.argv[1]); import radents as R
p = sys.argv[2]; E = R.entries(p)
json.dump({n: [E[n][1], R.digest(p, E[n][0], E[n][1])] for n in sorted(E)}, sys.stdout)
