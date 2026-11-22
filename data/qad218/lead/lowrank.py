import torch, json, numpy as np, os, sys
torch.set_num_threads(8)
R="/data/w218/runs/r0"; meta=json.load(open(f"{R}/g0/grad.json"))["tensors"]
pick=[k for k in meta if any(s in k for s in ("layers.8.linear_attn.in_proj_qkv","layers.23.self_attn.k_proj","layers.23.self_attn.q_proj","layers.40.mlp.down_proj","layers.55.mlp.gate_proj","layers.55.linear_attn.out_proj"))]
def load(c,k):
    m=meta[k]; return torch.from_numpy(np.fromfile(f"{R}/g{c}/{k}.g",np.float32).reshape(m["k"],m["n"]))
def cos(a,b): return float((a*b).sum()/(a.norm()*b.norm()))
print(f"{'tensor':40s} {'full cos':>8s} " + " ".join(f"r{r:<3d}cos/energy" for r in (4,16,64)))
for k in pick:
    g0,g1,g2=load(0,k),load(1,k),load(2,k)
    out=[f"{cos(g0,g1):8.3f}"]
    for r in (4,16,64):
        U,S,V=torch.svd_lowrank(g2,q=r,niter=4)   # subspace from an independent chunk
        P=lambda g: U@(U.T@g@V)@V.T
        a,b=P(g0),P(g1)
        out.append(f"{cos(a,b):7.3f}/{float(a.norm()**2/g0.norm()**2):.3f}")
    print(f"{k.replace('model.language_model.layers.',''):40s} "+" ".join(out), flush=True)
