#!/usr/bin/env python3
"""patchnorms196.py BASE_IN BASE_OUT NORMS.safetensors -- #196 N: the served container with the trained RMSNorm weights.

BASE_IN is a RAD2 container (radiance's format: rad_format.h, the directory of RadFileEntry, the plane table). Its norm
tensors are plain bf16 planes (one per entry, numel x 2 bytes), named blk.N.attn_norm / ffn_norm / ssm_norm / attn_q_norm /
attn_k_norm and output_norm, with the checkpoint's own values (checked: max |diff| 0 against the bf16 checkpoint on 5
samples, 2026-11-15). BASE_OUT is a copy (reflink when the filesystem has it; box2's /data refused clones on 11-15, so a plain copy) with those planes overwritten in place: the bytes are
bf16(new weight) at each plane's own offset; nothing else changes. Read-back verification of every norm: the planes
must equal bf16(new) exactly. Names: qad196's norms.safetensors keys (HF names) -> the container names below.
Usage in eval196.py: the served mix uses BASE_OUT as its base (radmix's BASE), and the census too."""
import os, struct, subprocess, sys

import numpy as np
from safetensors.torch import load_file
import torch

HDR = struct.Struct('<IIQ' + 'Q' * 5 + 'Q' * 2 * 9 + 'Q' * 8)
ENT = struct.Struct('<QQQII' + 'q' * 6 + 'iiiI' + 'QQQ' + 'QQ')
PLN = struct.Struct('<QQ')
MAP = {"input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm", "self_attn.q_norm": "attn_q_norm",
       "self_attn.k_norm": "attn_k_norm", "linear_attn.norm": "ssm_norm"}


def container_name(hf):
    if hf == "model.language_model.norm.weight": return "output_norm.weight"
    p = hf.split(".")                                        # model.language_model.layers.N.<sub>.weight
    assert p[:3] == ["model", "language_model", "layers"], hf
    n = p[3]; sub = ".".join(p[4:-1])
    assert sub in MAP, hf
    return f"blk.{n}.{MAP[sub]}.weight"


def directory(f):
    h = HDR.unpack(f.read(HDR.size))
    str_off, str_bytes = h[8], h[9]; dir_off, dir_n = h[12], h[13]; plane_off, plane_n = h[14], h[15]
    f.seek(str_off); strs = f.read(str_bytes)
    def s(o): return strs[o:strs.index(b"\0", o)].decode()
    f.seek(dir_off); ents = {}
    for _ in range(dir_n):
        e = ENT.unpack(f.read(ENT.size)); ents[s(e[0])] = e
    f.seek(plane_off); planes = [PLN.unpack(f.read(PLN.size)) for _ in range(plane_n)]
    return ents, planes


def main():
    src, dst, norms = sys.argv[1], sys.argv[2], sys.argv[3]
    nw = load_file(norms)
    want = {container_name(k): v for k, v in nw.items()}
    if os.path.exists(dst): os.remove(dst)
    rc = subprocess.call(["cp", "--reflink=always", src, dst])
    if rc:                                              # no reflink on this filesystem (box2, 11-15): a plain copy (8 GB, deleted after the mix)
        rc = subprocess.call(["cp", src, dst])
    if rc: raise SystemExit(f"copy of the base failed (rc {rc})")
    with open(src, "rb") as f: ents, planes = directory(f)
    missing = sorted(set(want) - set(ents))
    assert not missing, f"container has no entry for {missing[:5]}"
    norm_names = [n for n in ents if n.endswith(("_norm.weight", "output_norm.weight"))]
    assert len(norm_names) == len(want) == 209, (len(norm_names), len(want))
    with open(dst, "r+b") as f:
        for n, t in want.items():
            e = ents[n]; pf = e[15]; off = planes[pf][0]; nb = planes[pf][1]
            assert e[14] == 1 and tuple(e[5:5 + e[4]]) == tuple(t.shape), (n, e[5:5 + e[4]], tuple(t.shape))
            assert nb == t.numel() * 2, (n, nb, t.numel())
            b = t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().astype("<i2").tobytes()
            f.seek(off); f.write(b)
        f.flush(); os.fsync(f.fileno())
    bad = 0
    with open(dst, "rb") as f:
        for n, t in want.items():
            pf = ents[n][15]; f.seek(planes[pf][0]); got = np.frombuffer(f.read(planes[pf][1]), dtype="<i2")
            exp = t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().astype("<i2")
            if not np.array_equal(got, exp): bad += 1
    assert bad == 0, f"{bad} norm planes did not read back"
    print(f"patched {len(want)} norm planes -> {dst} (read-back ok)")


if __name__ == "__main__":
    main()
