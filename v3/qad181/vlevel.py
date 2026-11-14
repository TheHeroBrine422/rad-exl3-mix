#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""vlevel.py OUT --ladder ROOT (--assign ASSIGN.json | --uniform LEVEL) [--trunk-only] -- a searched mix's trunk as one
"virtual level" (#181): every trunk tensor's trellis / suh / svh / mul1 copied byte for byte from the level the mix
picked it from (ROOT/<level>/final), one q-LNN.safetensors per layer, ourq's layout (index, config, stamp with the
codebook), so the QAD student (qmodel.Student / deq.Level: K per tensor from the trellis shape) and rxq.swj.apply_level
read it as they read a uniform level. Also OUT.recipe: rad-convert's recipe for it, one exact-name rule per container
trunk weight with that weight's rate (recipe rules are first-match globs; a name without `*` matches itself).

ASSIGN = the search driver's assign file ({container weight: level name}; output.weight / token_embd.weight ignored).
The container's ffn_gate_up is the level's mlp.gate_proj + mlp.up_proj (both from the picked level)."""
import argparse, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "pipeline"))
from rxq import mix as M  # noqa: E402

L = "model.language_model.layers"
FAM = {"attn_qg": ["self_attn.q_proj"], "attn_k": ["self_attn.k_proj"], "attn_v": ["self_attn.v_proj"],
       "attn_output": ["self_attn.o_proj"], "ssm_qkv": ["linear_attn.in_proj_qkv"], "ssm_z": ["linear_attn.in_proj_z"],
       "ssm_out": ["linear_attn.out_proj"], "ffn_gate_up": ["mlp.gate_proj", "mlp.up_proj"], "ffn_down": ["mlp.down_proj"]}


def keys_of(w):
    """container trunk weight blk.N.fam.weight -> its level keys"""
    _, n, fam, _ = w.split(".")
    return [f"{L}.{n}.{s}" for s in FAM[fam]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out"); ap.add_argument("--ladder", required=True)
    ap.add_argument("--assign"); ap.add_argument("--uniform")
    a = ap.parse_args()
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    names = M.trunk_names()
    if a.assign:
        asg = json.load(open(a.assign)); pick = {w: asg[w] for w in names}
    else:
        pick = {w: a.uniform for w in names}
    levels = sorted(set(pick.values()))
    st = {}
    for lv in levels:
        e = json.load(open(f"{a.ladder}/{lv}/final/stamp.json"))[-1]
        st[lv] = (e.get("codebook") or e["args"]["codebook"], float(e["K"] if "K" in e else e["args"]["K"]), e.get("stamp"))
    cbs = {v[0] for v in st.values()}
    assert len(cbs) == 1, f"mixed codebooks {cbs}: deq.Level takes one codebook a level"
    cb = cbs.pop()
    os.makedirs(a.out, exist_ok=True)
    fh, ix = {}, {}
    def get(lv, name):
        d = f"{a.ladder}/{lv}/final"
        if lv not in ix: ix[lv] = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
        f = ix[lv][name]
        if (lv, f) not in fh: fh[(lv, f)] = safe_open(f"{d}/{f}", "pt", device="cpu")
        return fh[(lv, f)].get_tensor(name)
    wmap, rec, per_layer = {}, [], {}
    for w in names:
        lv = pick[w]
        for k in keys_of(w):
            per_layer.setdefault(int(k.split(".")[3]), []).append((k, lv))
        rec.append(f"{w} exl3i src={os.path.abspath(a.out)} codebook={M.CB[cb]} rate={M.fmt_k(st[lv][1])}\n")
    for l in sorted(per_layer):
        f = f"q-L{l:02d}.safetensors"
        t = {}
        for k, lv in per_layer[l]:
            for s in ("trellis", "suh", "svh", "mul1"):
                t[f"{k}.{s}"] = get(lv, f"{k}.{s}").clone()
            K = t[f"{k}.trellis"].shape[2] / 16
            assert K == st[lv][1], (k, lv, K, st[lv][1])
        for n in t: wmap[n] = f
        if not os.path.exists(f"{a.out}/{f}"):
            save_file(t, f"{a.out}/{f}.part", metadata={"vlevel181": json.dumps({k: lv for k, lv in per_layer[l]})})
            os.replace(f"{a.out}/{f}.part", f"{a.out}/{f}")
    json.dump({"metadata": {}, "weight_map": wmap}, open(f"{a.out}/model.safetensors.index.json", "w"), indent=1)
    shutil.copy(f"{a.ladder}/{levels[0]}/final/config.json", f"{a.out}/config.json")
    json.dump([{"codebook": cb, "K": "mixed", "args": {"K": "mixed", "codebook": cb}, "vlevel181": {
        "assign": os.path.abspath(a.assign) if a.assign else None, "uniform": a.uniform, "ladder": os.path.abspath(a.ladder),
        "levels": {lv: {"K": st[lv][1], "stamp": st[lv][2]} for lv in levels}}}], open(f"{a.out}/stamp.json", "w"), indent=1)
    open(a.out.rstrip("/") + ".recipe", "w").write("".join(rec))
    open(f"{a.out}/.done", "w").close()
    print(f"VLEVEL DONE {len(names)} trunk weights from {len(levels)} levels ({cb}) -> {a.out}")


if __name__ == "__main__":
    main()
