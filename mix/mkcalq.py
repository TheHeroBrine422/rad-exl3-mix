#!/usr/bin/env python3
"""mkcalq.py BASE_TAG DONOR_TAG QUEUE [BANDW] -- the proxy-calibration queue: on BASE, swap one tensor group of one
band of layers to DONOR's version (a large, measurable error step), for every group x band; plus BASE itself and
every other container's head on BASE's trunk. Mix files go to $RXM_WORK/evq/mixes/. Groups:
  attn_in  = attn_qg, attn_k, attn_v (full-attention layers, l % 4 == 3)   attn_out = attn_output
  ssm_in   = ssm_qkv, ssm_z (the GDN layers)                               ssm_out  = ssm_out
  ffn_gu   = ffn_gate_up                                                   ffn_down = ffn_down"""
import os, sys
# paths (README: Paths)
RXM_WORK = os.environ.get("RXM_WORK", "/root")
RXM_MODELS = os.environ.get("RXM_MODELS", f"{RXM_WORK}/models/final")
F = RXM_MODELS
base, donor, q = sys.argv[1:4]
bw = int(sys.argv[4]) if len(sys.argv) > 4 else 8
L = 64
groups = {
    "attn_in": (lambda l: l % 4 == 3, ["attn_qg", "attn_k", "attn_v"]),
    "attn_out": (lambda l: l % 4 == 3, ["attn_output"]),
    "ssm_in": (lambda l: l % 4 != 3, ["ssm_qkv", "ssm_z"]),
    "ssm_out": (lambda l: l % 4 != 3, ["ssm_out"]),
    "ffn_gu": (lambda l: True, ["ffn_gate_up"]),
    "ffn_down": (lambda l: True, ["ffn_down"]),
}
rad = lambda t: f"{F}/q38-27b-{t}.rad"
os.makedirs(f"{RXM_WORK}/evq/mixes", exist_ok=True)
lines = [f"base-{base} {rad(base)} -"]
for b0 in range(0, L, bw):
    for g, (sel, ts) in groups.items():
        tag = f"cal-{base}-{donor}-{g}-L{b0:02d}"
        names = [f"blk.{l}.{t}.weight" for l in range(b0, b0 + bw) if sel(l) for t in ts]
        if not names: continue
        p = f"{RXM_WORK}/evq/mixes/{tag}.mix"
        open(p, "w").write("".join(f"{n} {rad(donor)}\n" for n in names))
        lines.append(f"{tag} {rad(base)} {p}")
for t in ["sc1.40-e3", "sc2.00-e3", "sc3.00-e4", "sc5.00-e5", "sc6.00-e6", "plain2.00-e3", "plain4.00-e4", "plain6.00-e6"]:
    if t == base: continue
    tag = f"head-{base}-{t}"; p = f"{RXM_WORK}/evq/mixes/{tag}.mix"
    open(p, "w").write(f"output.weight {rad(t)}\n")
    lines.append(f"{tag} {rad(base)} {p}")
with open(q, "a") as f:
    f.write("\n".join(lines) + "\n")
print(len(lines), "queue lines ->", q)
