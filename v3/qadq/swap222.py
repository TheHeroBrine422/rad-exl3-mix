#!/usr/bin/env python3
# Copyright 2026 Hero422. Licensed under the Apache License, Version 2.0 (see LICENSE).
"""swap222.py RAD LV CKPT OUT.rad [--identity] [--verify] -- swap214.py (below) plus, when the checkpoint has them (w222: arms H and N): CKPT/scales/lm_head.safetensors (`r` on output.weight's suh plane, `c` on its svh plane, stored row order, same f16 arithmetic) and CKPT/norms.safetensors (the RMSNorm weights as the checkpoint stores them, bf16, written over the container's norm planes: the container holds the checkpoint's bf16 norm weights as they are, blk.N.attn_norm / ffn_norm / attn_q_norm / attn_k_norm / ssm_norm.weight and output_norm.weight). A checkpoint without those files gives exactly swap214.py's output.
swap214.py: -- a copy of a released container with a QAD checkpoint's scales folded in (#214):
every trunk weight's suh / svh (/ suh.1) plane becomes f16(stored f16 * multiplier), the multipliers read from CKPT/scales/<key>.safetensors
(`r` on suh, `c` on svh; rxq.swj.apply_level's format and arithmetic). Nothing else in the file changes (same size, same planes, same
offsets; checked by `--verify` against the source: only the scale planes of the trunk weights may differ).
LV = rad2level214.py's OUT (map214.json: container weight -> level keys). --identity: no checkpoint (multipliers 1): the output must equal
the source byte for byte (the packer's gate). Output written OUT.rad.part, then renamed."""
import argparse, json, os, shutil, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qad214"))
from rad214 import Rad  # noqa: E402


def mult(ck, key, which, n):
    from safetensors import safe_open
    with safe_open(f"{ck}/scales/{key}.safetensors", "pt") as s:
        m = s.get_tensor(which).float().numpy()
    assert m.shape == (n,), (key, which, m.shape, n)
    return m


def scaled(old, m):
    """f16 plane bytes' values * multiplier -> f16 (apply_level: (f16.float() * mult.float()).half())"""
    return (old.astype(np.float32) * m.astype(np.float32)).astype(np.float16)


NORM_MAP = [("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm"), ("self_attn.q_norm", "attn_q_norm"), ("self_attn.k_norm", "attn_k_norm"), ("linear_attn.norm", "ssm_norm")]


def container_norm_name(hf):
    """HF norm weight name -> the container's entry name (None: not a served norm)"""
    if hf == "model.language_model.norm.weight": return "output_norm.weight"
    import re
    m = re.match(r"model\.language_model\.layers\.(\d+)\.(.+)\.weight$", hf)
    if m:
        for a, b in NORM_MAP:
            if m.group(2) == a: return f"blk.{m.group(1)}.{b}.weight"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("rad"); ap.add_argument("lv"); ap.add_argument("ckpt"); ap.add_argument("out")
    ap.add_argument("--identity", action="store_true"); ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    rad = Rad(a.rad)
    mp = json.load(open(f"{a.lv}/map214.json"))["weights"]
    part = a.out + ".part"
    shutil.copyfile(a.rad, part)
    changed = 0; extra = set()
    with open(part, "r+b") as f:
        def put(name, role, vals):
            nonlocal changed
            po, pn = rad.entries[name].planes[role]
            assert vals.dtype == np.float16 and vals.nbytes == pn, (name, role, vals.nbytes, pn)
            f.seek(po); old = f.read(pn); new = vals.astype("<f2").tobytes()
            if old != new: changed += 1; f.seek(po); f.write(new)
        for w, d in mp.items():
            keys = d["keys"]
            suh = [np.frombuffer(rad.read(w, "suh"), "<f2"), np.frombuffer(rad.read(w, "suh.1"), "<f2")] if d["stacked"] else [np.frombuffer(rad.read(w, "suh"), "<f2")]
            svh = np.frombuffer(rad.read(w, "svh"), "<f2")
            if a.identity:
                continue
            h = d["rows"] // 2 if d["stacked"] else d["rows"]
            for j, k in enumerate(keys):
                put(w, "suh" if j == 0 else "suh.1", scaled(suh[j], mult(a.ckpt, k, "r", d["cols"])))
            svh_new = np.concatenate([scaled(svh[j * h:(j + 1) * h], mult(a.ckpt, k, "c", h)) for j, k in enumerate(keys)])
            put(w, "svh", svh_new)
        hf_ = f"{a.ckpt}/scales/lm_head.safetensors"
        if not a.identity and os.path.exists(hf_):                       # w222 (H)
            from safetensors import safe_open
            with safe_open(hf_, "pt") as z: hr, hc = z.get_tensor("r").float().numpy(), z.get_tensor("c").float().numpy()
            put("output.weight", "suh", scaled(np.frombuffer(rad.read("output.weight", "suh"), "<f2"), hr))
            put("output.weight", "svh", scaled(np.frombuffer(rad.read("output.weight", "svh"), "<f2"), hc))
            extra.update({("output.weight", "suh"), ("output.weight", "svh")})
        nf = f"{a.ckpt}/norms.safetensors"
        if not a.identity and os.path.exists(nf):                        # w222 (N): bf16 norm weights over the container's
            import torch
            from safetensors import safe_open
            with safe_open(nf, "pt") as z:
                for k in z.keys():
                    cn = container_norm_name(k); assert cn in rad.entries, (k, cn)
                    t = z.get_tensor(k); po, pn = rad.entries[cn].planes["codes"]; assert t.dtype == torch.bfloat16 and t.numel() * 2 == pn, (k, t.dtype, t.numel(), pn)
                    new = t.contiguous().view(torch.int16).numpy().astype("<i2").tobytes(); f.seek(po); old = f.read(pn)
                    if old != new: changed += 1; f.seek(po); f.write(new)
                    extra.add((cn, "codes"))
    if a.verify and not a.identity:
        iv = sorted([(po, po + pn) for w in mp for r in ("suh", "suh.1", "svh") if r in rad.entries[w].planes for po, pn in [rad.entries[w].planes[r]]] + [(rad.entries[w].planes[r][0], rad.entries[w].planes[r][0] + rad.entries[w].planes[r][1]) for w, r in extra])
        st, en = np.array([x[0] for x in iv]), np.array([x[1] for x in iv])
        bad = 0
        with open(a.rad, "rb") as f1, open(part, "rb") as f2:
            pos, CH, size = 0, 1 << 26, os.path.getsize(a.rad)
            while pos < size:
                b1, b2 = f1.read(CH), f2.read(CH)
                if b1 != b2:
                    o = pos + np.nonzero(np.frombuffer(b1, np.uint8) != np.frombuffer(b2, np.uint8))[0]
                    k = np.searchsorted(st, o, side="right") - 1
                    bad += int(((k < 0) | (o >= en[np.maximum(k, 0)])).sum())
                pos += len(b1)
        assert bad == 0, f"{bad} chunks differ outside the scale planes"
        print("verify: only trunk scale planes" + (f" and {len(extra)} head / norm planes" if extra else "") + " differ")
    if a.identity: assert open(a.rad, "rb").read(1 << 20) == open(part, "rb").read(1 << 20)
    os.replace(part, a.out)
    print(f"SWAP DONE {a.out}: {changed} scale planes changed of {sum(2 + d['stacked'] for d in mp.values())}")


if __name__ == "__main__":
    main()
