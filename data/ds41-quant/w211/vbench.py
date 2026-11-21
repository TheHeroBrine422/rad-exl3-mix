import sys,os,time; sys.path.insert(0,"/data/q27/w211")
import numpy as np, dscommon as D, trellis
cb=np.ascontiguousarray(trellis.codebook("mad"),dtype=np.float32)
x=np.random.default_rng(0).standard_normal((48,256)).astype(np.float32)
for K in (1,2,3,4):
    t=time.time(); trellis.vit(x,cb,K); print(K, round((time.time()-t)/48*1000,2),"ms/tile",flush=True)
