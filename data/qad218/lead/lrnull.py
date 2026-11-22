import json, numpy as np, torch
torch.set_num_threads(8); torch.manual_seed(1)
R="/data/w218/runs/r0"; meta=json.load(open(f"{R}/g0/grad.json"))["tensors"]
def ld(c,k): m=meta[k]; return torch.from_numpy(np.fromfile(f"{R}/g{c}/{k}.g",np.float32).reshape(m["k"],m["n"]))
def cos(a,b): return float((a*b).sum()/(a.norm()*b.norm()))
for key in [k for k in meta if any(s in k for s in ("55.mlp.gate","40.mlp.down","23.self_attn.q"))]:
    G0,G1,G2=ld(0,key),ld(1,key),ld(2,key)
    U,S,V=torch.svd_lowrank(G2,q=4,niter=4)
    out=[]
    for A,B in ((U,V),)+tuple((torch.linalg.qr(torch.randn(U.shape))[0],torch.linalg.qr(torch.randn(V.shape))[0]) for _ in range(3)):
        P=lambda g: A@(A.T@g@B)@B.T; out.append(cos(P(G0),P(G1)))
    print(key.replace("model.language_model.layers.",""), f"weight-grad r4: signal {out[0]:.3f}, random subspaces " + ", ".join(f"{x:.3f}" for x in out[1:]), flush=True)
