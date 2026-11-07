import json, sys, os, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/workspace/rad-exl3-mix/proxy2b")
from common import load_model
torch.cuda.set_per_process_memory_fraction(7.2 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
gd = "/data/g08-fit1"; m = json.load(open(f"{gd}/kld.json")); toks = np.memmap(f"{gd}/tokens.i32", np.int32, "r")
off = np.concatenate([[0], np.cumsum([d["tokens"] for d in m["docs"]])])
seqs = [torch.from_numpy(np.array(toks[off[i]:off[i] + 1024], np.int64)) for i, d in enumerate(m["docs"]) if d["tokens"] >= 1024][:3]
P = json.load(open("/data/yb/w75/m08perm/perm06.json"))["layers"]
A, _, _ = load_model("/data/models/qwen35-08b")
lay = A.get_decoder().layers
mods = {int(l): lay[int(l)].self_attn.o_proj for l in P}
cols = {l: torch.tensor(P[str(l)]["cols"], device="cuda") for l in mods}
Wc = {l: mods[l].weight.data[:, cols[l]].contiguous() for l in mods}
cap = {}
def hook(l):
    def h(mod, inp, out): cap[l] = (inp[0].detach().clone(), out.detach().clone())
    return h
hs = [mods[l].register_forward_hook(hook(l)) for l in mods]
def kl(a, b):
    s = 0.0
    for i in range(0, a.shape[0], 128):
        pa, pb = F.log_softmax(a[i:i+128].float(), -1), F.log_softmax(b[i:i+128].float(), -1)
        s += float((pa.exp() * (pa - pb)).sum(-1).sum())
    return s / a.shape[0]
with torch.no_grad():
    for si, s in enumerate(seqs):
        x = s[None].cuda()
        a1 = A(x).logits[0]; c1 = dict(cap)
        a2 = A(x).logits[0]
        print(f"seq {si}: A run-to-run bitwise {torch.equal(a1, a2)}, max {float((a1.float()-a2.float()).abs().max()):.4f}, KL {kl(a1, a2):.2e}")
        for l in sorted(mods):
            xi, yo = c1[l]
            yc = F.linear(xi.index_select(-1, cols[l]), Wc[l])
            y64 = (xi.double() @ mods[l].weight.data.double().T)
            d = (yo.float() - yc.float()); 
            print(f"  L{l} o_proj same input: natural vs permuted-order bitwise {torch.equal(yo, yc)}, frac elems differ {float((yo != yc).float().mean()):.4f}, "
                  f"rel err vs f64: natural {float((yo.double()-y64).norm()/y64.norm()):.2e} permuted {float((yc.double()-y64).norm()/y64.norm()):.2e}")
        del a1, a2
