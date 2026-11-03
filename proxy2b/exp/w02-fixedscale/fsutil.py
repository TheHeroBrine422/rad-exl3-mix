"""fsutil.py -- shared pieces of w02's fixed-code scale tools (refitq.py, swiglufit.py).

Why no codes are needed: quantize_exl3 stores W_q = diag(su) . Had_k Q_r Had_n . diag(sv) (Q_r = the decoded trellis,
fixed), and refit_scales only multiplies su by r and sv by c. So any further scale fit is W_q' = diag(r) W_q diag(c)
on the dequantized weight ourq.py --save-wq wrote, and the container edit is suh *= r, svh *= c."""
import json, os, struct
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file

FAM = {"self_attn.q_proj": "attn_in", "self_attn.k_proj": "attn_in", "self_attn.v_proj": "attn_in",
       "self_attn.o_proj": "attn_o", "linear_attn.in_proj_qkv": "gdn_in", "linear_attn.in_proj_z": "gdn_in",
       "linear_attn.out_proj": "gdn_out", "mlp.gate_proj": "gate_up", "mlp.up_proj": "gate_up", "mlp.down_proj": "down"}
FAMS = ("attn_in", "attn_o", "gdn_in", "gdn_out", "gate_up", "down")
def fam(key): return FAM[".".join(key.split(".")[4:6])]
def layer(key): return int(key.split(".")[3])

def group(key):
    """ourq.py's input-Hessian name of a linear (layer_plan)"""
    p, rest = ".".join(key.split(".")[:4]), ".".join(key.split(".")[4:6])
    if rest in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "linear_attn.in_proj_qkv", "linear_attn.in_proj_z"):
        return f"{p}.input"
    return {"self_attn.o_proj": f"{p}.self_attn.o_proj.input", "linear_attn.out_proj": f"{p}.linear_attn.out_proj.input",
            "mlp.gate_proj": f"{p}.post_attention.input", "mlp.up_proj": f"{p}.post_attention.input",
            "mlp.down_proj": f"{p}.mlp.down_proj.input"}[rest]

class Ckpt:
    """bf16 checkpoint reader (index or single file)"""
    def __init__(self, d):
        ix = f"{d}/model.safetensors.index.json"
        self.d, self.map = d, (json.load(open(ix))["weight_map"] if os.path.exists(ix) else None); self.h = {}
    def get(self, name):
        f = self.map[name] if self.map else "model.safetensors"
        if f not in self.h: self.h[f] = safe_open(f"{self.d}/{f}", "pt", device="cpu")
        return self.h[f].get_tensor(name)

def hessian(path, dev):
    """the challenge layout (packed upper triangle, sum over rows) -> the mean x xᵀ, (k, k) fp32, undamped"""
    with safe_open(path, "pt", device="cpu") as f:
        up = f.get_tensor("hessian_upper"); meta = f.metadata()
    k, rows = int(meta["k"]), int(meta["rows"])
    H = torch.zeros(k, k, dtype=torch.float32, device=dev); iu = torch.triu_indices(k, k, device=dev)
    H[iu[0], iu[1]] = up.to(dev).float(); H = H + H.triu(1).T
    return H / rows

def colerr(W, Q, H, chunk=8192):
    """per output column n (layout (k, n)): e_nᵀ H e_n and w_nᵀ H w_n, float64"""
    err, ref = [], []
    for a in range(0, W.shape[1], chunk):
        E = W[:, a:a + chunk] - Q[:, a:a + chunk]; Wc = W[:, a:a + chunk]
        err.append(((H @ E) * E).sum(0).double()); ref.append(((H @ Wc) * Wc).sum(0).double())
    return torch.cat(err), torch.cat(ref)

def wq_keys(qdir):
    return sorted(f[:-len(".safetensors")] for f in os.listdir(f"{qdir}/wq") if f.endswith(".safetensors"))

def load_wq(qdir, key, dev):
    """(k, n) fp32 from the saved (out, in) f16"""
    with safe_open(f"{qdir}/wq/{key}.safetensors", "pt", device="cpu") as f: return f.get_tensor("weight").to(dev).float().T.contiguous()

def write_wq(dst, key, Q):
    save_file({"weight": Q.T.half().contiguous().cpu()}, f"{dst}/wq/{key}.safetensors.part")
    os.replace(f"{dst}/wq/{key}.safetensors.part", f"{dst}/wq/{key}.safetensors")

def link_wq(src, dst, key):
    d = f"{dst}/wq/{key}.safetensors"
    if not os.path.exists(d): os.link(f"{src}/wq/{key}.safetensors", d)

def write_scales(dst, key, r, c):
    os.makedirs(f"{dst}/scales", exist_ok=True)
    save_file({"r": r.float().contiguous().cpu(), "c": c.float().contiguous().cpu()}, f"{dst}/scales/{key}.safetensors")

def edit_shards(src, dst, meta):
    """src's q-L*.safetensors with suh *= r, svh *= c from dst/scales (the real-container form of the refit);
    config.json / index copied. Tensors without a scales file are copied unchanged."""
    import shutil
    for f in sorted(os.listdir(src)):
        if f in ("config.json", "model.safetensors.index.json"): shutil.copy(f"{src}/{f}", f"{dst}/{f}")
        if not (f.startswith("q-") and f.endswith(".safetensors")): continue
        with safe_open(f"{src}/{f}", "pt", device="cpu") as h:
            t = {k: h.get_tensor(k) for k in h.keys()}; md = dict(h.metadata() or {})
        for k in list(t):
            for side, sk in (("suh", "r"), ("svh", "c")):
                if not k.endswith("." + side): continue
                sf = f"{dst}/scales/{k[:-len(side) - 1]}.safetensors"
                if os.path.exists(sf):
                    with safe_open(sf, "pt") as s: m = s.get_tensor(sk)
                    t[k] = (t[k].float() * m).half()
        md["w02_refit"] = json.dumps(meta)
        save_file(t, f"{dst}/{f}.part", metadata=md); os.replace(f"{dst}/{f}.part", f"{dst}/{f}")

def cap_vram():
    if torch.cuda.is_available() and os.environ.get("VRAM_CAP_GB"):
        tot = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["VRAM_CAP_GB"]) / tot))

# ---- C10 (swiglufit.py): per-channel sums and the joint fit, here so the self-test can check them
def swiglu_acc(s, aa, gq, uq, grid):
    """add one block of positions: aa = silu(g) u (target), gq / uq the quantized gate / up outputs, [T, J];
    s = [S_ap (G, J), S_pp (G, J), S_aa (J)] float64"""
    s[2] += aa.pow(2).sum(0).double()
    for i in range(len(grid)):
        p = torch.nn.functional.silu(grid[i] * gq) * uq
        s[0][i] += (aa * p).sum(0).double(); s[1][i] += p.pow(2).sum(0).double()

def swiglu_fit(s, i1):
    """-> (grid index of c_g, c_u there) per channel for the joint fit, and c_u at c_g = grid[i1] (= 1): up-only"""
    sap, spp, saa = s
    cu = torch.where(spp > 1e-30, sap / spp.clamp(min=1e-30), torch.ones_like(spp))
    gi = (saa[None] - sap * cu).argmin(0)                  # sum (a - c_u p)² at the optimal c_u, per grid point
    return gi, cu.gather(0, gi[None])[0], cu[i1]

def swiglu_resid(s, gi, cu):
    """(sum over channels of sum_t (a - cu p(c_g))², sum a²) for given grid indices and c_u"""
    sap, spp, saa = s
    a_ = sap.gather(0, gi[None])[0]; p_ = spp.gather(0, gi[None])[0]
    return float((saa - 2 * cu * a_ + cu * cu * p_).sum()), float(saa.sum())
