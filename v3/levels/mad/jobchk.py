# jobchk.py LEVELDIR... -- after ds8's crash (11-14): job dirs whose job.json / log.jsonl do not parse (NUL / torn lines)
import glob, json, os, sys
for L in sys.argv[1:]:
    bad = []
    for d in sorted(glob.glob(f"{L}/out/*")):
        try:
            if os.path.exists(f"{d}/job.json"): json.load(open(f"{d}/job.json"))
            if os.path.exists(f"{d}/log.jsonl"): [json.loads(l) for l in open(f"{d}/log.jsonl") if l.strip()]
        except Exception as e: bad.append(os.path.basename(d))
    print(os.path.basename(L), len(glob.glob(f"{L}/out/*")), "jobs,", len(bad), "unparseable:", bad)
