import json, math, random
D='/workspace/rad-local/multi/v3b'
res={}
for l in open(f'{D}/results.jsonl'):
    r=json.loads(l)
    if r['tag'].startswith('grd183-'): res[r['tag'][7:]]=r
refs=[json.loads(l) for l in open('/workspace/rad-local/workers/lead-srch/refs-t183.jsonl')]
V2=sorted((r['bpw'],r['kld']) for r in refs if r['name'].startswith('v2'))
def ll(b):
    segs=list(zip(V2,V2[1:])); s=segs[0] if b<V2[0][0] else segs[-1] if b>V2[-1][0] else next(x for x in segs if x[0][0]<=b<=x[1][0])
    (b0,k0),(b1,k1)=s; return math.exp(math.log(k0)+(math.log(k1)-math.log(k0))*(b-b0)/(b1-b0))
pairs=json.load(open('/workspace/rad-local/workers/lead-srch/guard-pairs.json'))
out=[]
for t,r,size,lr in pairs:
    a,b=res[t],res[r]
    da=json.load(open(f'{D}/docs/grd183-{t}-t183.json'))['m']; db=json.load(open(f'{D}/docs/grd183-{r}-t183.json'))['m']
    off=-math.log(ll(a['bpw'])/ll(b['bpw']))
    idx=[i for i in range(len(da)) if da[i][0] is not None and db[i][0] is not None and da[i][1]]
    st=lambda I: math.log(sum(da[i][0] for i in I)/sum(db[i][0] for i in I))+off
    d=st(idx); rng=random.Random(1); bs=sorted(st([rng.choice(idx) for _ in idx]) for _ in range(1000))
    dm=sum(math.log((da[i][0]/da[i][1])/(db[i][0]/db[i][1])) for i in idx)/len(idx)+off
    out.append((size,lr,d,bs[25],bs[975],dm))
    print(f"{size:4} search {lr:+.3f}  t183 pooled {d:+.4f} [{bs[25]:+.4f},{bs[975]:+.4f}]  t183 per-doc {dm:+.4f}")
n=len(out); print('t183 pooled better:',sum(o[2]<0 for o in out),'/',n,'| CI clearly better:',sum(o[4]<0 for o in out),' clearly worse:',sum(o[3]>0 for o in out))
print('mean t183 pooled %+.4f  mean search %+.4f  mean t183 per-doc %+.4f'%(sum(o[2] for o in out)/n,sum(o[1] for o in out)/n,sum(o[5] for o in out)/n))
m=[o for o in out if 2<=o[0]<=6]; print('active sizes only:',sum(o[2]<0 for o in m),'/',len(m),'mean %+.4f'%(sum(o[2] for o in m)/len(m)))
