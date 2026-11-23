class BK(float): pass
class SD(float): pass
class RW(float): pass
class EMK(float):
    def __new__(cls, x=0.0, tag=""): o = float.__new__(cls, x); o.tag = tag; return o
import torch, re, sys
def load(p):
    z = torch.load(p, map_location="cpu", weights_only=False)
    def find(o):
        if isinstance(o, dict):
            if any(isinstance(k, str) and (k.endswith(":r") or k.endswith(":c")) for k in o): return o
            for v in o.values():
                r = find(v)
                if r is not None: return r
    d = find(z); return {k: v.float().reshape(-1) for k, v in d.items() if torch.is_tensor(v) and re.search(r"\.layers\.(\d+)\.", k)}
def lay(k): return int(re.search(r"\.layers\.(\d+)\.", k).group(1))
run, pts = sys.argv[1], sys.argv[2].split(",")
S = [load(f"{run}/ckpt-raw-{p}/params.pt") for p in pts]
def vec(s, lo, hi): return torch.cat([v for k, v in sorted(s.items()) if lo <= lay(k) <= hi])
print("increment k = snapshot k -> k+1; cos(inc_k, inc_k+1) per group (pure noise ~0; drift > 0); |inc| x1e-3")
for g, (lo, hi) in {"L0-31": (0, 31), "L32-63": (32, 63)}.items():
    V = [vec(s, lo, hi) for s in S]; inc = [V[i + 1] - V[i] for i in range(len(V) - 1)]
    print(g, "  ".join(f"{pts[i]}→{pts[i+2]}: cos {float(torch.nn.functional.cosine_similarity(inc[i], inc[i+1], dim=0)):+.3f} |{float(inc[i+1].norm()/inc[i+1].numel()**0.5)*1e3:.2f}|" for i in range(len(inc) - 1)))
