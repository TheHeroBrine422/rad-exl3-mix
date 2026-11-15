#!/usr/bin/env python3
"""#185 w162 (B): the trunk lines of a candidate, from a level's final/ directory or from its pack.

  final_mix.py FINAL_DIR BASE HEAD EMB OUT_PREFIX [--pack PACK.rad] [--codebook CB] [--act MODE]

writes
  OUT_PREFIX.names        the trunk's container names (one per line; the 336 exl3i projections of a level)
  OUT_PREFIX.final.mix    a manifest: `base BASE`, one `<weight> final FINAL_DIR codebook=CB` line per trunk weight,
                          and the head / embedding donor lines (HEAD, EMB) as radmix mixfile lines
  OUT_PREFIX.pack.mix     the same mix with every trunk weight from PACK (the pack path; radmix's own form), when --pack

Every trunk weight gets `act <weight> MODE` (default a8g128, as the search's mixes carry: the GDN's ssm_qkv / ssm_z / ab share
one activation mode, and the engine refuses a mix without them).

The names are derived from the final/ index (exllamav3 names; a layer's linear_attn gives ssm_qkv / ssm_z / ssm_out, its
self_attn attn_qg / attn_k / attn_v / attn_output, its mlp ffn_gate_up (stacked gate + up) and ffn_down).
"""
import json
import os
import re
import sys


def trunk_names(final_dir):
    idx = json.load(open(os.path.join(final_dir, "model.safetensors.index.json")))["weight_map"]
    layers = {}
    for k in idx:
        m = re.match(r"model\.language_model\.layers\.(\d+)\.(.+)\.trellis$", k)
        if m:
            layers.setdefault(int(m.group(1)), set()).add(m.group(2))
    out = []
    for n in sorted(layers):
        parts = layers[n]
        if "linear_attn.in_proj_qkv" in parts:
            roles = [("ssm_qkv", "linear_attn.in_proj_qkv"), ("ssm_z", "linear_attn.in_proj_z"), ("ssm_out", "linear_attn.out_proj")]
        else:
            roles = [("attn_qg", "self_attn.q_proj"), ("attn_k", "self_attn.k_proj"), ("attn_v", "self_attn.v_proj"),
                     ("attn_output", "self_attn.o_proj")]
        roles += [("ffn_gate_up", "mlp.gate_proj"), ("ffn_down", "mlp.down_proj")]
        for role, part in roles:
            if part in parts or (role == "ffn_gate_up" and "mlp.up_proj" in parts):
                out.append(f"blk.{n}.{role}.weight")
    return out


def main(argv):
    if len(argv) < 6:
        print(__doc__)
        return 2
    final_dir, base, head, emb, prefix = argv[1:6]
    pack = argv[argv.index("--pack") + 1] if "--pack" in argv else None
    cb = argv[argv.index("--codebook") + 1] if "--codebook" in argv else "mad"
    act = argv[argv.index("--act") + 1] if "--act" in argv else "a8g128"
    names = trunk_names(final_dir)
    assert len(names) == 336, f"expected 336 trunk weights, found {len(names)}"
    with open(prefix + ".names", "w") as f:
        f.write("\n".join(names) + "\n")
    tail = [f"output.weight {head}", f"token_embd.weight {emb}"]
    with open(prefix + ".final.mix", "w") as f:
        f.write(f"base {base}\n")
        for n in names:
            f.write(f"{n} final {final_dir} codebook={cb}\n")
        f.write("\n".join(tail) + "\n")
        f.write("".join(f"act {n} {act}\n" for n in names))
    if pack:
        with open(prefix + ".pack.mix", "w") as f:
            f.write(f"base {base}\n")
            for n in names:
                f.write(f"{n} {pack}\n")
            f.write("\n".join(tail) + "\n")
            f.write("".join(f"act {n} {act}\n" for n in names))
    print(f"{len(names)} trunk weights -> {prefix}.names, {prefix}.final.mix" + (f", {prefix}.pack.mix" if pack else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
