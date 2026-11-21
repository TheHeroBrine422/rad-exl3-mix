import sys,os; sys.path.insert(0,"/data/q27/w211")
import numpy as np, torch, dscommon as D, e3, trellis, time
torch.set_num_threads(1)
W0=D.load_expert_w(5,42,"down"); H0,st=D.load_expert_h(5,42,"down")
rk=e3.had_blocks(2304,1005); Wr=rk(W0[:64]); Hr=rk(rk(H0).T).T
rms=Wr.pow(2).mean(1,keepdim=True).sqrt(); Ws=Wr/rms
cb=np.ascontiguousarray(trellis.codebook("mad"),dtype=np.float32)
for K in (1,2,4):
    t=time.time(); Qa=trellis.ldlq_trellis(Ws,Hr,cb,K,0.01); ta=time.time()-t
    pre=D.prep_ldlq(Hr,0.01); t=time.time(); Qb=D.ldlq_prepared(Ws,pre,cb,K); tb=time.time()-t
    # chunked: two halves
    Qc=torch.cat([D.ldlq_prepared(Ws[:32],pre,cb,K),D.ldlq_prepared(Ws[32:],pre,cb,K)],0)
    print(K,"max|ref-prepared|",float((Qa-Qb).abs().max()),"max|ref-chunked|",float((Qa-Qc).abs().max()),"rel",e3.rel(Wr,Qa*rms,Hr),f"time {ta:.1f}s ({64//16*144} tiles)")
