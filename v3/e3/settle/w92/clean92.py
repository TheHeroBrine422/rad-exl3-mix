"""clean92.py OUT -- before a rerun of an rxq job dir: keep log.jsonl's whole lines whose tensor file exists, remove tensor /
C9 files that log.jsonl does not record (killed between the save and the log line), as rad-quant-at-home rxq_quant run()."""
import json, os, sys
o = sys.argv[1]; lp = f"{o}/log.jsonl"; ok = set()
if os.path.exists(lp):
    rows = []
    for l in open(lp):
        try: r = json.loads(l)
        except ValueError: break                     # a torn last line
        if os.path.exists(f"{o}/t-{r['key']}.safetensors"): rows.append(l if l.endswith("\n") else l + "\n"); ok.add(r["key"])
    with open(lp + ".part", "w") as f: f.writelines(rows)
    os.replace(lp + ".part", lp)
for f in os.listdir(o):
    for p in ("t-", "c9-"):
        if f.startswith(p) and f.endswith(".safetensors") and f[len(p):-len(".safetensors")] not in ok: os.remove(f"{o}/{f}")
