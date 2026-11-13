"""mkf16ck.py MODEL QDIR PERM OUT [DONOR.rad] -- a checkpoint view for the f16 reference head (#164-02, w136): OUT holds
symlinks to MODEL's files, its index sending lm_head.weight to OUT/lm_head-f16.safetensors = QDIR/wq/lm_head.safetensors
(ourq's f16 dequantized head, rows in R16 order: stored row j = natural row PERM[j]) un-permuted to vocab order, kept f16
(no bf16 rounding, unlike the search's donors). With DONOR (a hd-u*.rad of hd129.sh): how many of its bf16 head values
equal bf16(this f16 head), i.e. that the donor is this head rounded to bf16. Prints JSON."""
import json, os, sys
import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file

M, QD, PERM, O = sys.argv[1:5]
os.makedirs(O, exist_ok=True)
for f in os.listdir(M):
    if f != "model.safetensors.index.json" and not os.path.exists(f"{O}/{f}"):
        os.symlink(f"{M}/{f}", f"{O}/{f}")
idx = json.load(open(f"{M}/model.safetensors.index.json")); idx["weight_map"]["lm_head.weight"] = "lm_head-f16.safetensors"
perm = np.load(PERM)
with safe_open(f"{QD}/wq/lm_head.safetensors", "np") as f:
    W = f.get_tensor("weight")
assert W.dtype == np.float16 and W.shape[0] == len(perm), (W.dtype, W.shape)
N = np.empty_like(W); N[perm] = W
if not os.path.exists(f"{O}/lm_head-f16.safetensors"):
    save_file({"lm_head.weight": N}, f"{O}/lm_head-f16.safetensors.part", metadata={"format": "pt"})
    os.rename(f"{O}/lm_head-f16.safetensors.part", f"{O}/lm_head-f16.safetensors")
json.dump(idx, open(f"{O}/model.safetensors.index.json", "w"), indent=2)
r = {"out": O, "rows": int(W.shape[0]), "f16_head": f"{O}/lm_head-f16.safetensors"}
if len(sys.argv) > 5:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from radents import entries   # noqa: E402
    dn = sys.argv[5]; off, nb, shape = entries(dn)["output.weight"]
    D = np.memmap(dn, np.uint16, "r", offset=off, shape=tuple(shape))
    eq = 0
    for c in range(0, N.shape[0], 16384):
        b = torch.from_numpy(N[c:c + 16384]).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        eq += int((b == D[c:c + 16384]).sum())
    r.update({"donor": dn, "donor_eq_bf16_of_f16": eq, "of": int(N.size)})
print(json.dumps(r))
