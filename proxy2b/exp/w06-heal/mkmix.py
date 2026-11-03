#!/usr/bin/env python3
"""mkmix.py MAP OUT K2WQ K3WQ [K2PARAMS K3PARAMS] -- an H4 mix of two uniform quants, and the transferred healing.

MAP: 'family' = MLP (gate / up / down) from the K2 quant, attention / GDN from K3; 'depth' = even layers K2, odd K3.
OUT/wq: symlinks into the two levels' --save-wq dirs; OUT/mix.json: key -> level and the mix's trunk bits per weight.
With both PARAMS (heal.py train of each uniform level): OUT/xfer.pt, the level library composed for the mix -- each
linear's r / c from its level's run; q_norm / k_norm of a layer from the level of that layer's q_proj; the final
norm the mean of the two levels' log multipliers."""
import json, os, sys
import torch
from safetensors import safe_open

def level(key, mp):
    l = int(key.split("layers.", 1)[1].split(".")[0])
    if mp == "family": return 2 if ".mlp." in key else 3
    if mp == "depth": return 2 if l % 2 == 0 else 3
    raise SystemExit(f"unknown map {mp}")

def main():
    mp, out, w2, w3 = sys.argv[1:5]; pr = sys.argv[5:7]
    src = {2: w2, 3: w3}; os.makedirs(f"{out}/wq", exist_ok=True)
    keys = sorted(f[:-12] for f in os.listdir(w3) if f.endswith(".safetensors"))
    assert keys == sorted(f[:-12] for f in os.listdir(w2) if f.endswith(".safetensors")), "level dirs differ"
    lv, nb, nw = {}, 0.0, 0
    for k in keys:
        lv[k] = level(k, mp); o = f"{out}/wq/{k}.safetensors"
        if os.path.lexists(o): os.remove(o)
        os.symlink(os.path.realpath(f"{src[lv[k]]}/{k}.safetensors"), o)
        with safe_open(o, "pt") as h: s = h.get_slice("weight").get_shape()
        nb += s[0] * s[1] * lv[k]; nw += s[0] * s[1]
    json.dump({"map": mp, "bpw": nb / nw, "level": lv}, open(f"{out}/mix.json", "w"), indent=0)
    print(f"mix {mp}: {len(keys)} linears, trunk {nb / nw:.3f} bits/weight")
    if len(pr) == 2:
        P = {2: torch.load(pr[0], weights_only=False), 3: torch.load(pr[1], weights_only=False)}
        assert P[2]["norm_form"] == P[3]["norm_form"]
        lin = {k: P[lv[k]]["lin"][k] for k in keys if k in P[lv[k]]["lin"]}
        norm = {}
        for k in P[3]["norm"]:
            if "layers." in k:
                l = k.split("layers.", 1)[1].split(".")[0]; q = k.split("layers.", 1)[0] + f"layers.{l}.self_attn.q_proj"
                norm[k] = P[lv[q]]["norm"][k]
            else:
                norm[k] = 0.5 * (P[2]["norm"][k] + P[3]["norm"][k])
        torch.save({"lin": lin, "norm": norm, "norm_form": P[3]["norm_form"], "meta": {"map": mp, "from": pr}}, f"{out}/xfer.pt")
        print(f"xfer.pt: {len(lin)} linears, {len(norm)} norms")

if __name__ == "__main__":
    main()
