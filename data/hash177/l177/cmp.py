import json,glob,collections
V=collections.defaultdict(dict)
for f in sorted(glob.glob("*.jsonl")):
    tag=f.rsplit("-",1)[1].replace(".jsonl","")
    for l in open(f):
        d=json.loads(l); key=(d["layer"],d["w"],int(d["bits"]))
        for a,v in d["rel"].items(): V[key][(tag,a)]=v
for k in sorted(V):
    v=V[k]
    if ("md","V16") in v:
        b=v[("md","V16")]
        print(k," ".join("%s/%s=%.4f"%(t,a,x/b) for (t,a),x in sorted(v.items())))
