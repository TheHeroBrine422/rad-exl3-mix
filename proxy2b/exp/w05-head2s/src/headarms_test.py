#!/usr/bin/env python3
"""headarms_test.py WORK -- CPU self-test of headarms.py on a synthetic head (V 2048, H 64): a fake checkpoint, a dump
with document ids, arms = the reference itself (KL 0), small / large noise in vocabulary order, the small one stored
in a permuted order (must score identically to its vocabulary-order copy), and a 2-donor slab plan (must equal the
same head composed by hand)."""
import json, os, subprocess, sys
import numpy as np, torch
from safetensors.torch import save_file
W = sys.argv[1]; os.makedirs(W, exist_ok=True); torch.manual_seed(0); V, H = 2048, 64
ref = (torch.randn(V, H) * 0.5).to(torch.bfloat16)
os.makedirs(f"{W}/ck", exist_ok=True); save_file({"lm_head.weight": ref}, f"{W}/ck/h.safetensors")
json.dump({"weight_map": {"lm_head.weight": "h.safetensors"}}, open(f"{W}/ck/model.safetensors.index.json", "w"))
os.makedirs(f"{W}/dump", exist_ok=True)
X = (torch.randn(3000, H) * 2).to(torch.bfloat16); X.view(torch.int16).numpy().tofile(f"{W}/dump/336.x")
open(f"{W}/dump/336.meta", "w").write(f"{H} 1 1\n"); np.repeat(np.arange(30), 100).astype(np.int32).tofile(f"{W}/dump/doc.i32")
perm = np.random.default_rng(1).permutation(V); np.save(f"{W}/perm.npy", perm)
def put(d, w):
    os.makedirs(f"{d}/wq", exist_ok=True); save_file({"weight": w.half().contiguous()}, f"{d}/wq/lm_head.safetensors")
r32 = ref.float()
small, large = r32 + 0.02 * torch.randn(V, H), r32 + 0.08 * torch.randn(V, H)
put(f"{W}/same", r32); put(f"{W}/small", small); put(f"{W}/large", large); put(f"{W}/smallp", small[perm])
put(f"{W}/q/K3", small[perm]); put(f"{W}/q/K4", large[perm])
kb = [3.0] * 8 + [4.0] * 8; json.dump({"Ks": [3, 4], "plans": {"p": {"K_blocks": kb}}}, open(f"{W}/plan.json", "w"))
hand = torch.empty(V, H); rowK = np.repeat(kb, 128); hp = torch.where(torch.from_numpy(rowK == 3)[:, None], small[perm], large[perm])
hand[perm] = hp; put(f"{W}/hand", hand)
here = os.path.dirname(os.path.abspath(__file__))
r = subprocess.run([sys.executable, f"{here}/headarms.py", f"{W}/dump", f"{W}/ck", f"{W}/out", f"same={W}/same",
                    f"small={W}/small", f"large={W}/large", f"smallp={W}/smallp:{W}/perm.npy",
                    f"plan=plan:{W}/plan.json:p:{W}/q:{W}/perm.npy", f"hand={W}/hand", "--sens-perm", f"{W}/perm.npy"],
                   capture_output=True, text=True, env=dict(os.environ, HA_GUARD="1000"))
print(r.stdout[-1500:], r.stderr[-1500:])
rp = json.load(open(f"{W}/out/report.json"))["halves"]["all"]
kl = lambda n: np.fromfile(f"{W}/out/{n}.kl.f32", np.float32)
c = {"ref arm KL ~ 0": rp["same"]["kl"] < 1e-5,
     "small < large": rp["small"]["kl"] < rp["large"]["kl"],
     "permuted storage == vocab order": np.array_equal(kl("small"), kl("smallp")),
     "plan == hand-built head": np.array_equal(kl("plan"), kl("hand")),
     "small vs same: z large": rp["small"]["z"] > 5,
     "guard positions found": rp["small"]["guard_kl"] is not None}
for k, v in c.items(): print(f"{k}: {v}")
print("HEADARMS SELFTEST", "PASS" if all(c.values()) else "FAIL")
