#!/usr/bin/env python3
"""m0an.py -- M0-metric bounds for the embedding (w83, #119): eigen AM/GM of M0, and Gaussian water-filling in the
M0-whitened space vs a white coder's M0-weighted error. Paths: colo /data/work/w13/w83 (specan.py outputs)."""
import numpy as np, math, sys
sys.path.insert(0,'/data/work/w13/w83/code')
M0=np.load('/data/work/w13/w83/out/specan-M0.npy').astype(np.float64)
S=np.load('/data/work/w13/w83/out/spec.npz'); C=S['Gn']/248320
def amgm(v): v=np.maximum(v,1e-30*v.max()); return v.mean()/math.exp(np.log(v).mean())
ev=np.linalg.eigvalsh(M0); print('M0 eig AM/GM %.3f (%.3f bit), diag AM/GM %.3f'%(amgm(ev),0.5*math.log2(amgm(ev)),amgm(np.diag(M0))))
w,Q=np.linalg.eigh(M0); w=np.maximum(w,1e-12*w.max()); Mh=(Q*np.sqrt(w))@Q.T
lam=np.linalg.eigvalsh(Mh@C@Mh)
# white coder (uniform error D per coefficient in any orthonormal basis): weighted error = D*tr(M0); signal = tr(M0 C)
base_const = np.trace(M0)/len(M0) * np.trace(C)/len(C) / np.trace(M0@C) * len(C)  # rel_M0 = 2^-2R * base_const
print('white coder M0-rel const %.3f'%base_const)
def wf(lam,R):
    lam=np.sort(np.maximum(lam,1e-300))[::-1]; n=len(lam); lo,hi=math.log(lam[-1])-60,math.log(lam[0])
    for _ in range(200):
        mid=(lo+hi)/2; th=math.exp(mid); r=np.maximum(0,0.5*np.log2(lam/th)).sum()/n
        lo,hi=(mid,hi) if r>R else (lo,mid)
    th=math.exp((lo+hi)/2); return float(np.minimum(lam,th).sum()/lam.sum())
for R in (1,1.5,2,3):
    D=wf(lam,R); b=base_const*2**(-2*R); print('R %.1f wf %.5f white %.5f ratio %.3f saves %.3f bit'%(R,D,b,D/b,0.5*math.log2(b/D)))
