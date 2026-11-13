"""int8rel144.py EMB.safetensors OUT.jsonl -- (w144, #174-02) row error of the simple int8 host-embedding candidates
(lead note 11-13: deadcode's INT8 G128 FWHT128 style, or plain int8 per-row RTN), the measure the menu prices
(rel = sum (w - w_hat)^2 / sum w^2 over all rows, as embsrc.py; err = embprices174 floor model on it):
  int8-row       one bf16 scale a row (absmax / 127), round to nearest, codes in [-127, 127]
  int8-g128      one bf16 scale a 128 columns (absmax / 127)
  int8-g128h     orthonormal 128-point Hadamard (FWHT) on each 128 columns, then int8-g128; error in the original space
EMB: prep144.py's one-tensor checkpoint (lm_head.weight = the bf16 token table). CPU, row blocks of 4096."""
import json, sys
import numpy as np
import torch
from safetensors import safe_open

emb, out = sys.argv[1], sys.argv[2]
torch.set_num_threads(4)
with safe_open(emb, "pt") as f:
    W = f.get_tensor("lm_head.weight")
V, E = W.shape
H = torch.tensor([[1.0]])
while H.shape[0] < 128:
    H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
H = H / np.sqrt(128)                                 # orthonormal: H @ H.T = I


def rtn(x):                                          # x [..., g]: int8 RTN with a bf16 absmax scale a last-axis group
    s = (x.abs().amax(-1, keepdim=True) / 127).to(torch.bfloat16).float().clamp_min(1e-30)
    return (x / s).round().clamp(-127, 127) * s


acc = {k: 0.0 for k in ("int8-row", "int8-g128", "int8-g128h")}
ref = 0.0
for r0 in range(0, V, 4096):
    w = W[r0:r0 + 4096].float()
    ref += float((w * w).sum())
    g = w.view(-1, E // 128, 128)
    acc["int8-row"] += float(((rtn(w) - w) ** 2).sum())
    acc["int8-g128"] += float(((rtn(g) - g) ** 2).sum())
    acc["int8-g128h"] += float(((rtn(g @ H) @ H.T - g) ** 2).sum())
with open(out, "a") as f:
    for k, v in acc.items():
        f.write(json.dumps({"level": k, "rel": v / ref, "rows": V, "bytes_codes": V * E,
                            "bytes_scales": 2 * V * (1 if k == "int8-row" else E // 128)}) + "\n")
print(open(out).read())
