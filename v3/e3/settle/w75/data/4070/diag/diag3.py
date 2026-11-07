# w75 #106-01 diagnostic (reported only): o_proj error under the quantizer's two-sided metric (c18 H_out diag x calib H_in)
# and ourq's own proxy_err, perm vs natural; per head-share of the H_in-weighted error (which heads got worse)
import json, sys, torch
sys.path.insert(0, "/workspace/rad-exl3-mix/proxy2b/exp/w02-fixedscale"); import fsutil as U
from safetensors import safe_open
O, W64 = "/data/yb/w75", "/data/yb/w64"; ck = U.Ckpt("/data/models/qwen35-08b")
P = json.load(open(f"{O}/m08perm/perm06.json"))["layers"]
pe = lambda d: {j["key"]: j["proxy_err"] for j in map(json.loads, open(f"{d}/proxy.jsonl"))}
for k in ("3", "2"):
    for s, x in ((0, ""), (1, "_s1"), (2, "_s2")):
        pn, pp = pe(f"{O}/q-on{k}{x}"), pe(f"{O}/q-op{k}{x}"); row = []
        for l in (3, 7, 11, 15, 19, 23):
            key = f"model.language_model.layers.{l}.self_attn.o_proj"
            W = ck.get(key + ".weight").double(); H = U.hessian(f"/data/h08/calib/{U.group(key)}.safetensors", "cpu").double()
            with safe_open(f"/data/yb/c18/hdiag-snap1024-dy/{key}.safetensors", "pt") as f:
                D = (f.get_tensor("hout_diag") if "hout_diag" in f.keys() else f.get_tensor("hout").diagonal()).double()
            r = {}
            for arm, d in (("nat", f"{W64}/dec-k{k}ste{x}"), ("perm", f"{O}/dec-opn{k}{x}")):
                with safe_open(f"{d}/{key}.safetensors", "pt") as f: Q = f.get_tensor("weight").double()
                E = Q - W; EH = E @ H
                r[arm] = float((D[:, None] * EH * E).sum() / (D[:, None] * (W @ H) * W).sum())
                hh = torch.stack([(EH[:, h*256:(h+1)*256] * E[:, h*256:(h+1)*256]).sum() for h in range(8)])
                r[arm + "_heads"] = (hh / hh.sum()).tolist()
            row.append((l, r["perm"] / r["nat"], pp[key] / pn[key]))
        print(f"K{k} s{s}: two-sided rel perm/nat " + " ".join(f"L{l} {a:.3f}" for l, a, _ in row)
              + " | ourq proxy_err perm/nat " + " ".join(f"{b:.3f}" for _, _, b in row))
