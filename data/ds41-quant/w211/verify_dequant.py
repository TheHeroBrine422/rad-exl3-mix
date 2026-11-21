"""verify_dequant.py LAYER EXPERT -- dscommon's MXFP4 dequantizer vs the reference model code (ds41pipe ref/kernel.py + ref/model.py):
(1) w1/w2/w3 dequant equals ref.kernel.dequant_fp4_weight bit for bit; (2) the expert's output on random inputs, through the
reference Expert module (fp4 weights, its fp8 act quant + fp4_gemm) vs the same SwiGLU built from our dequantized weights."""
import sys, os, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
import dscommon as D
REF = os.environ.get("W211_REF", "/data/q27/w211/ref")
layer, e = int(sys.argv[1]), int(sys.argv[2])
sys.path.insert(0, REF)
import importlib.util
pkg = types.ModuleType("ref"); pkg.__path__ = [REF]; sys.modules["ref"] = pkg
import ref.kernel as rk
raw = {n: (D._get(f"layers.{layer}.ffn.experts.{e}.{n}.weight"), D._get(f"layers.{layer}.ffn.experts.{e}.{n}.scale")) for n in ("w1", "w2", "w3")}
ours = {n: torch.from_numpy(D.dequant_fp4(*raw[n])) for n in raw}
for n in raw:
    r = rk.dequant_fp4_weight(raw[n][0].view(torch.float4_e2m1fn_x2), raw[n][1].view(torch.float8_e8m0fnu))
    print(n, tuple(ours[n].shape), "max|ours - ref(bf16)| =", float((ours[n] - r.float()).abs().max()), "bit-equal after bf16:", bool((ours[n].bfloat16() == r).all()), "bf16-exact:", bool((ours[n].bfloat16().float() == ours[n]).all()))
import ref.model as M
cfg = json = None
torch.manual_seed(0)
dim, inter = ours["w1"].shape[1], ours["w1"].shape[0]
ex = M.Expert(dim, inter, dtype=torch.float4_e2m1fn_x2, swiglu_limit=10.0)
for n in raw:
    getattr(ex, n).weight.data = raw[n][0].view(torch.float4_e2m1fn_x2); getattr(ex, n).weight.scale.data = raw[n][1].view(torch.float8_e8m0fnu); getattr(ex, n).scale.data = getattr(ex, n).weight.scale.data
x = torch.randn(8, dim, dtype=torch.bfloat16) * 0.7
y_ref = ex(x).float()
xq = x   # our path: same SwiGLU with dequantized weights; the reference additionally rounds activations to fp8 (act_quant) before each GEMM
a_q, a_s = rk.act_quant(x, 128 if False else M.fp8_block_size, M.scale_fmt, M.scale_dtype)
def aq(z):
    zq, zs = rk.act_quant(z.contiguous(), M.fp8_block_size, M.scale_fmt, M.scale_dtype)
    return (zq.float().view(z.shape[0], -1, M.fp8_block_size) * zs.float().unsqueeze(-1)).view_as(z)
rb = lambda t: t.bfloat16().float()   # the reference GEMM returns bf16 (CPU F.linear)
gate = rb(aq(x).float() @ ours["w1"].T); up = rb(aq(x).float() @ ours["w3"].T)
gate = gate.clamp(max=10.0); up = up.clamp(-10.0, 10.0)
h = (F.silu(gate) * up).bfloat16()
y_ours = rb(aq(h).float() @ ours["w2"].T)
print("expert output rel diff (ours-dequant fp32 GEMM vs reference Expert/fp4_gemm):", float((y_ours - y_ref).norm() / y_ref.norm()), "ref norm", float(y_ref.norm()))
