import torch, re, sys, math
class BK(float): pass
class SD(float): pass
class RW(float): pass
class EMK(float):
    def __new__(cls, x=0.0, tag=""): o = float.__new__(cls, x); o.tag = tag; return o
def load(p):
    z = torch.load(p, map_location="cpu", weights_only=False)
    def find(o):
        if isinstance(o, dict):
            if any(isinstance(k, str) and (k.endswith(":r") or k.endswith(":c")) for k in o): return o
            for v in o.values():
                r = find(v)
                if r is not None: return r
        return None
    d = find(z); return {k: v.float().reshape(-1) for k, v in d.items() if torch.is_tensor(v) and re.search(r"\.layers\.(\d+)\.", k)}
R = "/data/q27/queue/runs/"
S = {n: load(R + p + "/params.pt") for n, p in [
    ("ctl0.5M", "I25C/ckpt-5e+05"), ("ctl1M", "I25C/ckpt-1e+06"), ("ctl2M", "I25-lr3e-4-2M/ckpt-2e+06"),
    ("MF-mse1M", "MF25-s1/ckpt-1e+06"), ("MD-mse0.5M", "MD25-s1/ckpt-5e+05"), ("MD-kl1M", "MD25/ckpt-1e+06"),
    ("MJ1M", "MJ25/ckpt-1e+06"), ("MJ-2M", "MJ25-2M/ckpt-2e+06")]}
def lay(k): return int(re.search(r"\.layers\.(\d+)\.", k).group(1))
def vec(n, lo, hi): return torch.cat([v for k, v in sorted(S[n].items()) if lo <= lay(k) <= hi])
print("rms log-scale change vs release (x1e-3):  L0-15  L16-31  L32-47  L48-63")
for n in S: print(f"  {n:11s}", "  ".join(f"{vec(n,a,b).pow(2).mean().sqrt()*1e3:6.2f}" for a, b in [(0,15),(16,31),(32,47),(48,63)]))
print("cosine between runs' changes, front (L0-31) / back (L32-63):")
for a, b in [("ctl1M","ctl2M"),("ctl0.5M","ctl1M"),("MF-mse1M","ctl1M"),("MD-mse0.5M","MD-kl1M"),("MD-kl1M","ctl1M"),("MJ1M","ctl1M"),("MJ1M","MJ-2M"),("MJ-2M","ctl2M")]:
    c = [float(torch.nn.functional.cosine_similarity(vec(a,lo,hi), vec(b,lo,hi), dim=0)) for lo, hi in [(0,31),(32,63)]]
    print(f"  {a:11s} vs {b:11s}  {c[0]:+.3f} / {c[1]:+.3f}")
