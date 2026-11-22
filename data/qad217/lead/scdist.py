import torch, sys, numpy as np, collections
R="/data/q27/w214/runs"
runs=[("25A","2e+05","2.5 lr1e-3 0.25M ok"),("25A","5e+05","2.5 lr1e-3 0.5M BLOWUP"),("25A-lr3e-4","2e+05","2.5 lr3e-4 0.25M ok"),("25A-lr3e-4","5e+05","2.5 lr3e-4 0.5M ok"),
      ("30A","2e+05","3.0 lr1e-3 0.25M"),("30A","5e+05","3.0 lr1e-3 0.5M"),("30A-lr3e-4","5e+05","3.0 lr3e-4 0.5M ok"),
      ("40A-lr3e-4","2e+05","4.0 lr3e-4 0.25M ok"),("40A-lr3e-4","5e+05","4.0 lr3e-4 0.5M BLOWUP(j217 3.15)"),("40A","5e+05","4.0 lr1e-3 0.5M bad")]
def fam(k):
    k=k.split("layers.")[1]; L=int(k.split(".")[0]); n=k.split(".",1)[1]; return L,n
print(f"{'run':38s} {'rms':>7s} {'p99':>7s} {'p99.9':>7s} {'max':>7s} {'n>2%':>8s} {'n>5%':>7s}  top tensors by max|d|")
for a,b,lab in runs:
    try: d=torch.load(f"{R}/{a}/ckpt-{b}/params.pt",map_location="cpu",weights_only=False)["state"]
    except Exception as e: print(lab,"ERR",e); continue
    allv=np.concatenate([v.numpy().ravel() for v in d.values()]); av=np.abs(allv)
    per=sorted(((float(v.abs().max()),k) for k,v in d.items()),reverse=True)[:3]
    tops=", ".join(f"{k.split('layers.')[1].replace('.linear_attn','').replace('.self_attn','').replace('.mlp','')}={m:.3f}" for m,k in per)
    print(f"{lab:38s} {np.sqrt((allv**2).mean()):7.4f} {np.quantile(av,.99):7.4f} {np.quantile(av,.999):7.4f} {av.max():7.4f} {int((av>0.02).sum()):8d} {int((av>0.05).sum()):7d}  {tops}")
