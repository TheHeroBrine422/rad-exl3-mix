import sys,os; sys.path.insert(0,"/data/q27/w211")
import torch, dscommon as D, e3
torch.set_num_threads(1)
nd=torch.distributions.Normal(0.,1.)
def ks(x,n=2_000_000,seed=1):
    g=torch.Generator().manual_seed(seed); s=x.reshape(-1)[torch.randperm(x.numel(),generator=g)[:n]]; s=torch.sort(s)[0]
    return float((torch.arange(1,n+1)/n-nd.cdf(s)).abs().max())
for layer,e,kind in [(5,42,"down"),(5,42,"gate_up"),(35,104,"down")]:
    W=D.load_expert_w(layer,e,kind); rk=e3.had_blocks(W.shape[1],1000+layer); Wr=rk(W)
    Ws=Wr/Wr.pow(2).mean(1,keepdim=True).sqrt(); Wn=W/W.pow(2).mean(1,keepdim=True).sqrt()
    g=torch.randn(W.shape,generator=torch.Generator().manual_seed(2),dtype=torch.float64); g=g/g.pow(2).mean(1,keepdim=True).sqrt()
    print(layer,e,kind,"KS rotated",round(ks(Ws),5),"KS gaussian rows (unit-RMS)",round(ks(g),5),"KS unrotated",round(ks(Wn),5), "sample-size floor ~", round(0.87/2e6**.5,5),flush=True)
