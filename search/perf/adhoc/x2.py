import json, numpy as np, os
R=[json.loads(l) for l in open("se2/pairs.jsonl")]
def nm(m): return os.path.basename(m).replace(".rad","").replace("mix-","")
print("== a4cost / swap27 pairs on golden-search27 (calib, 103 docs, 123k pos)")
for r in R:
    if r["part"]=="golden-search27" and ("a4cost" in r["model_a"]+r["model_b"] or "swap27" in r["model_a"] and "swap27" in r["model_b"]):
        a,b=nm(r["model_a"]),nm(r["model_b"])
        if "a4cost" in r["model_a"]+r["model_b"] and a.split("-")[0]!=b.split("-")[0]: continue
        print(f'{a:22s} {b:22s} dlog {100*r["logr"]:+6.2f}  se {100*r["se_boot"]:5.2f}  z {r["logr"]/r["se_boot"]:+5.1f}  sdu {r["sd_u"]:.3f} maxsh {r["max_share"]:.2f}  geo {100*r["doc_geo"]:+6.2f}/{100*r["se_doc_geo"]:4.2f}')
print("== cross-slice")
C=[json.loads(l) for l in open("se2/cross.jsonl")]
for c in C:
    print(f'{c["part"]} {nm(c["a"])[:18]:18s} {nm(c["b"])[:18]:18s} {c["g1"][-6:]} {100*c["logr1"]:+6.2f}±{100*c["se1"]:.2f}  {c["g2"][-6:]} {100*c["logr2"]:+6.2f}±{100*c["se2"]:.2f}  z {c["z"]:+.2f}')
z=np.array([c["z"] for c in C]); print("n",len(z),"mean z",z.mean().round(2),"sd z",z.std().round(2),"rms",np.sqrt((z**2).mean()).round(2))
